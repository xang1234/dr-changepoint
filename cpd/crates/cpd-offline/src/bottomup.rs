// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{
    BudgetStatus, CpdError, Diagnostics, ExecutionContext, OfflineChangePointResult,
    OfflineDetector, Penalty, Stopping, TimeSeriesView, ValidatedConstraints,
    check_missing_compatibility, penalty_value, validate_constraints, validate_stopping,
};
use cpd_costs::CostModel;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::Instant;

const DEFAULT_CANCEL_CHECK_EVERY: usize = 1000;
const AUTO_PARAMS_PER_SEGMENT: usize = 0;

/// Configuration for [`BottomUp`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct BottomUpConfig {
    pub stopping: Stopping,
    pub params_per_segment: usize,
    pub cancel_check_every: usize,
}

impl Default for BottomUpConfig {
    fn default() -> Self {
        Self {
            stopping: Stopping::Penalized(Penalty::BIC),
            params_per_segment: AUTO_PARAMS_PER_SEGMENT,
            cancel_check_every: DEFAULT_CANCEL_CHECK_EVERY,
        }
    }
}

impl BottomUpConfig {
    fn validate(&self) -> Result<(), CpdError> {
        validate_stopping(&self.stopping)?;
        Ok(())
    }

    fn normalized_cancel_check_every(&self) -> usize {
        self.cancel_check_every.max(1)
    }
}

/// Bottom-up merge-based offline detector.
#[derive(Debug)]
pub struct BottomUp<C: CostModel> {
    cost_model: C,
    config: BottomUpConfig,
}

impl<C: CostModel> BottomUp<C> {
    pub fn new(cost_model: C, config: BottomUpConfig) -> Result<Self, CpdError> {
        config.validate()?;
        Ok(Self { cost_model, config })
    }

    pub fn cost_model(&self) -> &C {
        &self.cost_model
    }

    pub fn config(&self) -> &BottomUpConfig {
        &self.config
    }
}

#[derive(Default, Clone, Copy, Debug)]
struct RuntimeStats {
    cost_evals: usize,
    candidates_considered: usize,
    merges_applied: usize,
    soft_budget_exceeded: bool,
}

#[derive(Clone, Copy, Debug)]
struct SegmentNode {
    start: usize,
    end: usize,
    prev: Option<usize>,
    next: Option<usize>,
    alive: bool,
    version: u64,
    cost: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct MergeCandidate {
    left_idx: usize,
    right_idx: usize,
    left_version: u64,
    right_version: u64,
    boundary: usize,
    delta: f64,
    merged_cost: f64,
}

impl PartialEq for MergeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.left_idx == other.left_idx
            && self.right_idx == other.right_idx
            && self.left_version == other.left_version
            && self.right_version == other.right_version
            && self.boundary == other.boundary
            && self.delta.to_bits() == other.delta.to_bits()
            && self.merged_cost.to_bits() == other.merged_cost.to_bits()
    }
}

impl Eq for MergeCandidate {}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering so BinaryHeap pops the smallest merge-cost increase first.
        other
            .delta
            .total_cmp(&self.delta)
            .then_with(|| other.boundary.cmp(&self.boundary))
            .then_with(|| other.left_idx.cmp(&self.left_idx))
            .then_with(|| other.right_idx.cmp(&self.right_idx))
    }
}

#[derive(Clone, Debug)]
struct ResolvedPenalty {
    beta: f64,
    params_per_segment: usize,
    params_source: &'static str,
}

fn checked_counter_increment(counter: &mut usize, name: &str) -> Result<(), CpdError> {
    *counter = counter
        .checked_add(1)
        .ok_or_else(|| CpdError::resource_limit(format!("{name} counter overflow")))?;
    Ok(())
}

fn resolve_penalty_params<C: CostModel>(
    model: &C,
    penalty: &Penalty,
    configured_params_per_segment: usize,
) -> (usize, &'static str) {
    match penalty {
        Penalty::BIC | Penalty::AIC => {
            if configured_params_per_segment == AUTO_PARAMS_PER_SEGMENT {
                (model.penalty_params_per_segment(), "model_default_auto")
            } else {
                (configured_params_per_segment, "config_override")
            }
        }
        Penalty::Manual(_) => {
            if configured_params_per_segment == AUTO_PARAMS_PER_SEGMENT {
                (model.penalty_params_per_segment(), "model_default_auto")
            } else {
                (configured_params_per_segment, "config")
            }
        }
    }
}

fn resolve_penalty_beta<C: CostModel>(
    model: &C,
    penalty: &Penalty,
    n: usize,
    d: usize,
    configured_params_per_segment: usize,
) -> Result<ResolvedPenalty, CpdError> {
    let (params_per_segment, params_source) =
        resolve_penalty_params(model, penalty, configured_params_per_segment);
    if params_per_segment == 0 {
        return Err(CpdError::invalid_input(
            "resolved params_per_segment must be >= 1; got 0",
        ));
    }
    let beta = penalty_value(penalty, n, d, params_per_segment)?;
    if !beta.is_finite() || beta <= 0.0 {
        return Err(CpdError::invalid_input(format!(
            "resolved penalty must be finite and > 0.0; got beta={beta}"
        )));
    }
    Ok(ResolvedPenalty {
        beta,
        params_per_segment,
        params_source,
    })
}

fn evaluate_segment_cost<C: CostModel>(
    model: &C,
    cache: &C::Cache,
    start: usize,
    end: usize,
    ctx: &ExecutionContext<'_>,
    runtime: &mut RuntimeStats,
) -> Result<f64, CpdError> {
    checked_counter_increment(&mut runtime.cost_evals, "cost_evals")?;

    match ctx.check_cost_eval_budget(runtime.cost_evals)? {
        BudgetStatus::WithinBudget => {}
        BudgetStatus::ExceededSoftDegrade => {
            runtime.soft_budget_exceeded = true;
        }
    }

    let segment_cost = model.segment_cost(cache, start, end);
    if !segment_cost.is_finite() {
        return Err(CpdError::numerical_issue(format!(
            "non-finite segment cost at [{start}, {end}): {segment_cost}"
        )));
    }
    Ok(segment_cost)
}

fn check_runtime_controls(
    iteration: usize,
    cancel_check_every: usize,
    ctx: &ExecutionContext<'_>,
    started_at: Instant,
    runtime: &mut RuntimeStats,
) -> Result<(), CpdError> {
    if iteration.is_multiple_of(cancel_check_every) {
        ctx.check_cancelled_every(iteration, 1)?;
        match ctx.check_time_budget(started_at)? {
            BudgetStatus::WithinBudget => {}
            BudgetStatus::ExceededSoftDegrade => {
                runtime.soft_budget_exceeded = true;
            }
        }
    }
    Ok(())
}

fn build_initial_breakpoints(
    validated: &ValidatedConstraints,
    n: usize,
) -> Result<Vec<usize>, CpdError> {
    let min_segment_len = validated.min_segment_len;
    if n == 0 {
        return Err(CpdError::invalid_input("BottomUp requires n >= 1; got n=0"));
    }

    let mut breakpoints = Vec::with_capacity(n / min_segment_len.max(1) + 1);
    let mut cursor = 0usize;
    while n.saturating_sub(cursor) > min_segment_len {
        let lower = cursor
            .checked_add(min_segment_len)
            .ok_or_else(|| CpdError::resource_limit("initial breakpoint lower bound overflow"))?;
        let start_idx = validated
            .effective_candidates
            .partition_point(|&split| split < lower);

        let next_split = validated.effective_candidates[start_idx..]
            .iter()
            .copied()
            .find(|&split| n.saturating_sub(split) >= min_segment_len);

        let Some(split) = next_split else {
            break;
        };
        if split <= cursor || split >= n {
            return Err(CpdError::invalid_input(format!(
                "BottomUp initialization produced invalid split={split} for n={n}, cursor={cursor}"
            )));
        }
        breakpoints.push(split);
        cursor = split;
    }

    breakpoints.push(n);
    Ok(breakpoints)
}

fn build_segment_nodes(breakpoints: &[usize]) -> Result<Vec<SegmentNode>, CpdError> {
    if breakpoints.is_empty() {
        return Err(CpdError::invalid_input(
            "BottomUp breakpoints must be non-empty at initialization",
        ));
    }

    let mut segments = Vec::with_capacity(breakpoints.len());
    let mut start = 0usize;
    for (idx, &end) in breakpoints.iter().enumerate() {
        if end <= start {
            return Err(CpdError::invalid_input(format!(
                "BottomUp initialization requires strictly increasing breakpoints; index={idx}, start={start}, end={end}"
            )));
        }

        segments.push(SegmentNode {
            start,
            end,
            prev: idx.checked_sub(1),
            next: None,
            alive: true,
            version: 0,
            cost: None,
        });

        if idx > 0 {
            segments[idx - 1].next = Some(idx);
        }

        start = end;
    }

    Ok(segments)
}

fn segment_cost_for<C: CostModel>(
    segments: &mut [SegmentNode],
    idx: usize,
    model: &C,
    cache: &C::Cache,
    ctx: &ExecutionContext<'_>,
    runtime: &mut RuntimeStats,
) -> Result<f64, CpdError> {
    if let Some(cost) = segments[idx].cost {
        return Ok(cost);
    }

    let (start, end) = {
        let segment = &segments[idx];
        (segment.start, segment.end)
    };
    let cost = evaluate_segment_cost(model, cache, start, end, ctx, runtime)?;
    segments[idx].cost = Some(cost);
    Ok(cost)
}

#[allow(clippy::too_many_arguments)]
fn make_candidate_for_left<C: CostModel>(
    segments: &mut [SegmentNode],
    left_idx: usize,
    model: &C,
    cache: &C::Cache,
    ctx: &ExecutionContext<'_>,
    cancel_check_every: usize,
    started_at: Instant,
    runtime: &mut RuntimeStats,
    iteration: &mut usize,
) -> Result<Option<MergeCandidate>, CpdError> {
    if left_idx >= segments.len() || !segments[left_idx].alive {
        return Ok(None);
    }
    let Some(right_idx) = segments[left_idx].next else {
        return Ok(None);
    };
    if right_idx >= segments.len() || !segments[right_idx].alive {
        return Ok(None);
    }
    if segments[right_idx].prev != Some(left_idx) {
        return Ok(None);
    }

    checked_counter_increment(iteration, "iteration")?;
    check_runtime_controls(*iteration, cancel_check_every, ctx, started_at, runtime)?;
    checked_counter_increment(&mut runtime.candidates_considered, "candidates_considered")?;

    let left_cost = segment_cost_for(segments, left_idx, model, cache, ctx, runtime)?;
    let right_cost = segment_cost_for(segments, right_idx, model, cache, ctx, runtime)?;

    let (merge_start, merge_end, boundary, left_version, right_version) = {
        let left = &segments[left_idx];
        let right = &segments[right_idx];
        (left.start, right.end, left.end, left.version, right.version)
    };

    let merged_cost = evaluate_segment_cost(model, cache, merge_start, merge_end, ctx, runtime)?;
    let delta = merged_cost - left_cost - right_cost;
    if !delta.is_finite() {
        return Err(CpdError::numerical_issue(format!(
            "non-finite merge gain at boundary={boundary}: merged_cost={merged_cost}, left_cost={left_cost}, right_cost={right_cost}, delta={delta}"
        )));
    }

    Ok(Some(MergeCandidate {
        left_idx,
        right_idx,
        left_version,
        right_version,
        boundary,
        delta,
        merged_cost,
    }))
}

#[allow(clippy::too_many_arguments)]
fn push_candidate_for_left<C: CostModel>(
    heap: &mut BinaryHeap<MergeCandidate>,
    segments: &mut [SegmentNode],
    left_idx: usize,
    model: &C,
    cache: &C::Cache,
    ctx: &ExecutionContext<'_>,
    cancel_check_every: usize,
    started_at: Instant,
    runtime: &mut RuntimeStats,
    iteration: &mut usize,
) -> Result<(), CpdError> {
    if let Some(candidate) = make_candidate_for_left(
        segments,
        left_idx,
        model,
        cache,
        ctx,
        cancel_check_every,
        started_at,
        runtime,
        iteration,
    )? {
        heap.push(candidate);
    }
    Ok(())
}

fn candidate_is_current(candidate: &MergeCandidate, segments: &[SegmentNode]) -> bool {
    let Some(left) = segments.get(candidate.left_idx) else {
        return false;
    };
    let Some(right) = segments.get(candidate.right_idx) else {
        return false;
    };

    left.alive
        && right.alive
        && left.next == Some(candidate.right_idx)
        && right.prev == Some(candidate.left_idx)
        && left.version == candidate.left_version
        && right.version == candidate.right_version
}

fn pop_best_current_candidate(
    heap: &mut BinaryHeap<MergeCandidate>,
    segments: &[SegmentNode],
) -> Option<MergeCandidate> {
    while let Some(candidate) = heap.pop() {
        if candidate_is_current(&candidate, segments) {
            return Some(candidate);
        }
    }
    None
}

fn merge_candidate(
    segments: &mut [SegmentNode],
    candidate: MergeCandidate,
    segment_count: &mut usize,
    runtime: &mut RuntimeStats,
) -> Result<usize, CpdError> {
    if !candidate_is_current(&candidate, segments) {
        return Err(CpdError::invalid_input(
            "internal BottomUp state error: attempted to merge stale candidate",
        ));
    }

    let left_idx = candidate.left_idx;
    let right_idx = candidate.right_idx;
    let right_next = segments[right_idx].next;

    segments[left_idx].end = segments[right_idx].end;
    segments[left_idx].next = right_next;
    segments[left_idx].cost = Some(candidate.merged_cost);
    segments[left_idx].version = segments[left_idx]
        .version
        .checked_add(1)
        .ok_or_else(|| CpdError::resource_limit("segment version overflow"))?;

    if let Some(next_idx) = right_next {
        segments[next_idx].prev = Some(left_idx);
    }

    segments[right_idx].alive = false;
    segments[right_idx].prev = None;
    segments[right_idx].next = None;
    segments[right_idx].cost = None;
    segments[right_idx].version = segments[right_idx]
        .version
        .checked_add(1)
        .ok_or_else(|| CpdError::resource_limit("segment version overflow"))?;

    *segment_count = segment_count
        .checked_sub(1)
        .ok_or_else(|| CpdError::resource_limit("segment_count underflow"))?;
    checked_counter_increment(&mut runtime.merges_applied, "merges_applied")?;

    Ok(left_idx)
}

fn collect_breakpoints(segments: &[SegmentNode]) -> Result<Vec<usize>, CpdError> {
    let mut out = Vec::new();
    let mut cursor = Some(0usize);
    let mut hops = 0usize;

    while let Some(idx) = cursor {
        hops = hops
            .checked_add(1)
            .ok_or_else(|| CpdError::resource_limit("segment traversal hop overflow"))?;
        if hops > segments.len().saturating_add(1) {
            return Err(CpdError::invalid_input(
                "internal BottomUp state error: cycle detected while collecting breakpoints",
            ));
        }

        let segment = segments.get(idx).ok_or_else(|| {
            CpdError::invalid_input("internal BottomUp state error: invalid segment index")
        })?;
        if !segment.alive {
            return Err(CpdError::invalid_input(
                "internal BottomUp state error: dead segment in active chain",
            ));
        }
        out.push(segment.end);
        cursor = segment.next;
    }

    Ok(out)
}

impl<C: CostModel> OfflineDetector for BottomUp<C> {
    fn detect(
        &self,
        x: &TimeSeriesView<'_>,
        ctx: &ExecutionContext<'_>,
    ) -> Result<OfflineChangePointResult, CpdError> {
        self.config.validate()?;
        let validated = validate_constraints(ctx.constraints, x.n)?;

        self.cost_model.validate(x)?;
        check_missing_compatibility(x.missing, self.cost_model.missing_support())?;
        let cache = self.cost_model.precompute(x, &validated.cache_policy)?;

        let started_at = Instant::now();
        let cancel_check_every = self.config.normalized_cancel_check_every();
        let mut runtime = RuntimeStats::default();
        let mut iteration = 0usize;
        let mut notes = vec![];
        let mut warnings = vec![];

        if let Some(max_depth) = validated.max_depth {
            warnings.push(format!(
                "constraints.max_depth={max_depth} is ignored by BottomUp"
            ));
        }

        let initial_breakpoints = build_initial_breakpoints(&validated, x.n)?;
        let mut segments = build_segment_nodes(initial_breakpoints.as_slice())?;
        let mut segment_count = segments.len();
        let initial_change_count = segment_count.saturating_sub(1);

        let mut heap = BinaryHeap::new();
        if segment_count >= 2 {
            for left_idx in 0..(segment_count - 1) {
                push_candidate_for_left(
                    &mut heap,
                    &mut segments,
                    left_idx,
                    &self.cost_model,
                    &cache,
                    ctx,
                    cancel_check_every,
                    started_at,
                    &mut runtime,
                    &mut iteration,
                )?;
            }
        }

        match &self.config.stopping {
            Stopping::KnownK(k) => {
                if let Some(max_change_points) = validated.max_change_points
                    && max_change_points < *k
                {
                    return Err(CpdError::invalid_input(format!(
                        "KnownK={k} exceeds constraints.max_change_points={max_change_points}"
                    )));
                }

                let current_change_count = segment_count.saturating_sub(1);
                if current_change_count < *k {
                    return Err(CpdError::invalid_input(format!(
                        "KnownK exact solution unreachable: requested k={k}, but BottomUp initialization produced only {current_change_count} changes under constraints"
                    )));
                }

                let merges_needed = current_change_count.saturating_sub(*k);
                for merge_idx in 0..merges_needed {
                    let best = pop_best_current_candidate(&mut heap, segments.as_slice()).ok_or_else(
                        || {
                            CpdError::invalid_input(format!(
                                "KnownK exact solution unreachable: requested k={k}, but BottomUp frontier exhausted after {} merges",
                                runtime.merges_applied
                            ))
                        },
                    )?;
                    let merged_left_idx =
                        merge_candidate(&mut segments, best, &mut segment_count, &mut runtime)?;

                    if let Some(prev_idx) = segments[merged_left_idx].prev {
                        push_candidate_for_left(
                            &mut heap,
                            &mut segments,
                            prev_idx,
                            &self.cost_model,
                            &cache,
                            ctx,
                            cancel_check_every,
                            started_at,
                            &mut runtime,
                            &mut iteration,
                        )?;
                    }
                    push_candidate_for_left(
                        &mut heap,
                        &mut segments,
                        merged_left_idx,
                        &self.cost_model,
                        &cache,
                        ctx,
                        cancel_check_every,
                        started_at,
                        &mut runtime,
                        &mut iteration,
                    )?;

                    if merges_needed > 0 {
                        ctx.report_progress((merge_idx + 1) as f32 / merges_needed as f32);
                    }
                }

                notes.push(format!(
                    "stopping=KnownK({k}), initial_change_count={initial_change_count}, merges_needed={merges_needed}"
                ));
            }
            Stopping::Penalized(penalty) => {
                let resolved = resolve_penalty_beta(
                    &self.cost_model,
                    penalty,
                    x.n,
                    x.d,
                    self.config.params_per_segment,
                )?;
                notes.push(format!(
                    "stopping=Penalized({penalty:?}), beta={}, params_per_segment={} ({})",
                    resolved.beta, resolved.params_per_segment, resolved.params_source
                ));

                if let Some(max_change_points) = validated.max_change_points {
                    let mut mandatory_merges = 0usize;
                    while segment_count.saturating_sub(1) > max_change_points {
                        let best =
                            pop_best_current_candidate(&mut heap, segments.as_slice()).ok_or_else(
                                || {
                                    CpdError::invalid_input(format!(
                                        "BottomUp cannot satisfy constraints.max_change_points={max_change_points} under current candidate splits"
                                    ))
                                },
                            )?;
                        let merged_left_idx =
                            merge_candidate(&mut segments, best, &mut segment_count, &mut runtime)?;
                        checked_counter_increment(&mut mandatory_merges, "mandatory_merges")?;

                        if let Some(prev_idx) = segments[merged_left_idx].prev {
                            push_candidate_for_left(
                                &mut heap,
                                &mut segments,
                                prev_idx,
                                &self.cost_model,
                                &cache,
                                ctx,
                                cancel_check_every,
                                started_at,
                                &mut runtime,
                                &mut iteration,
                            )?;
                        }
                        push_candidate_for_left(
                            &mut heap,
                            &mut segments,
                            merged_left_idx,
                            &self.cost_model,
                            &cache,
                            ctx,
                            cancel_check_every,
                            started_at,
                            &mut runtime,
                            &mut iteration,
                        )?;
                    }

                    if mandatory_merges > 0 {
                        notes.push(format!(
                            "applied {mandatory_merges} mandatory merges to satisfy constraints.max_change_points={max_change_points}"
                        ));
                    }
                }

                let mut penalized_merges = 0usize;
                loop {
                    let Some(best) = pop_best_current_candidate(&mut heap, segments.as_slice())
                    else {
                        break;
                    };
                    if best.delta > resolved.beta {
                        notes.push(format!(
                            "penalized_stop_delta={}, beta={}",
                            best.delta, resolved.beta
                        ));
                        break;
                    }

                    let merged_left_idx =
                        merge_candidate(&mut segments, best, &mut segment_count, &mut runtime)?;
                    checked_counter_increment(&mut penalized_merges, "penalized_merges")?;

                    if let Some(prev_idx) = segments[merged_left_idx].prev {
                        push_candidate_for_left(
                            &mut heap,
                            &mut segments,
                            prev_idx,
                            &self.cost_model,
                            &cache,
                            ctx,
                            cancel_check_every,
                            started_at,
                            &mut runtime,
                            &mut iteration,
                        )?;
                    }
                    push_candidate_for_left(
                        &mut heap,
                        &mut segments,
                        merged_left_idx,
                        &self.cost_model,
                        &cache,
                        ctx,
                        cancel_check_every,
                        started_at,
                        &mut runtime,
                        &mut iteration,
                    )?;

                    let progress_denom = (penalized_merges + heap.len() + 1) as f32;
                    if progress_denom.is_finite() && progress_denom > 0.0 {
                        ctx.report_progress(penalized_merges as f32 / progress_denom);
                    }
                }

                notes.push(format!("penalized_merges={penalized_merges}"));
            }
            Stopping::PenaltyPath(path) => {
                return Err(CpdError::not_supported(format!(
                    "BottomUp penalty sweep is deferred for this issue; got PenaltyPath of length {}",
                    path.len()
                )));
            }
        }

        if runtime.soft_budget_exceeded {
            warnings.push(
                "budget exceeded under SoftDegrade mode; run continued without algorithm fallback"
                    .to_string(),
            );
        }

        let runtime_ms = u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX);

        ctx.record_scalar("offline.bottomup.cost_evals", runtime.cost_evals as f64);
        ctx.record_scalar(
            "offline.bottomup.candidates_considered",
            runtime.candidates_considered as f64,
        );
        ctx.record_scalar(
            "offline.bottomup.merges_applied",
            runtime.merges_applied as f64,
        );
        ctx.record_scalar("offline.bottomup.runtime_ms", runtime_ms as f64);
        ctx.report_progress(1.0);

        notes.push(format!(
            "initial_segments={}, initial_change_count={}, final_change_count={}, merges_applied={}, cost_evals={}, candidates_considered={}",
            initial_breakpoints.len(),
            initial_change_count,
            segment_count.saturating_sub(1),
            runtime.merges_applied,
            runtime.cost_evals,
            runtime.candidates_considered
        ));

        let diagnostics = Diagnostics {
            n: x.n,
            d: x.d,
            runtime_ms: Some(runtime_ms),
            notes,
            warnings,
            algorithm: Cow::Borrowed("bottomup"),
            cost_model: Cow::Borrowed(self.cost_model.name()),
            repro_mode: ctx.repro_mode,
            ..Diagnostics::default()
        };

        let breakpoints = collect_breakpoints(segments.as_slice())?;
        OfflineChangePointResult::new(x.n, breakpoints, diagnostics)
    }
}

#[cfg(test)]
mod tests {
    use super::{BottomUp, BottomUpConfig};
    use cpd_core::{
        Constraints, DTypeView, ExecutionContext, MemoryLayout, MissingPolicy, OfflineDetector,
        Penalty, Stopping, TimeIndex, TimeSeriesView,
    };
    use cpd_costs::{CostL2Mean, CostNormalMeanVar};

    fn make_f64_view<'a>(
        values: &'a [f64],
        n: usize,
        d: usize,
        layout: MemoryLayout,
        missing: MissingPolicy,
    ) -> TimeSeriesView<'a> {
        TimeSeriesView::new(
            DTypeView::F64(values),
            n,
            d,
            layout,
            None,
            TimeIndex::None,
            missing,
        )
        .expect("test view should be valid")
    }

    fn constraints_with_min_segment_len(min_segment_len: usize) -> Constraints {
        Constraints {
            min_segment_len,
            ..Constraints::default()
        }
    }

    #[test]
    fn config_defaults_and_validation() {
        let default_cfg = BottomUpConfig::default();
        assert_eq!(default_cfg.stopping, Stopping::Penalized(Penalty::BIC));
        assert_eq!(default_cfg.params_per_segment, 0);
        assert_eq!(default_cfg.cancel_check_every, 1000);

        let ok = BottomUp::new(CostL2Mean::default(), default_cfg.clone())
            .expect("default config should be valid");
        assert_eq!(ok.config(), &default_cfg);
    }

    #[test]
    fn cancel_check_every_zero_is_normalized() {
        let detector = BottomUp::new(
            CostL2Mean::default(),
            BottomUpConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.5)),
                params_per_segment: 2,
                cancel_check_every: 0,
            },
        )
        .expect("config with zero cadence should normalize");

        let values = vec![0.0, 0.0, 10.0, 10.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = constraints_with_min_segment_len(1);
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");
        assert_eq!(result.breakpoints.last().copied(), Some(values.len()));
    }

    #[test]
    fn known_small_example_two_changes_l2() {
        let detector = BottomUp::new(
            CostL2Mean::default(),
            BottomUpConfig {
                stopping: Stopping::KnownK(2),
                params_per_segment: 2,
                cancel_check_every: 8,
            },
        )
        .expect("config should be valid");

        let values = vec![
            0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, -4.0, -4.0, -4.0, -4.0,
        ];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let constraints = constraints_with_min_segment_len(2);
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");
        assert_eq!(result.breakpoints, vec![4, 8, 12]);
    }

    #[test]
    fn penalized_detects_single_change_l2() {
        let detector = BottomUp::new(
            CostL2Mean::default(),
            BottomUpConfig {
                stopping: Stopping::Penalized(Penalty::Manual(1.0)),
                params_per_segment: 2,
                cancel_check_every: 4,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let constraints = constraints_with_min_segment_len(2);
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");
        assert_eq!(result.breakpoints, vec![4, 8]);
    }

    #[test]
    fn normal_cost_detects_variance_change() {
        let detector = BottomUp::new(
            CostNormalMeanVar::default(),
            BottomUpConfig {
                stopping: Stopping::KnownK(1),
                params_per_segment: 3,
                cancel_check_every: 2,
            },
        )
        .expect("config should be valid");

        let mut values = Vec::with_capacity(20);
        for _ in 0..5 {
            values.push(-1.0);
            values.push(1.0);
        }
        for _ in 0..5 {
            values.push(-5.0);
            values.push(5.0);
        }

        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let constraints = Constraints {
            min_segment_len: 2,
            max_change_points: Some(1),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");
        assert_eq!(result.breakpoints, vec![10, 20]);
    }

    #[test]
    fn known_k_exact_unreachable_is_clear_error() {
        let detector = BottomUp::new(
            CostL2Mean::default(),
            BottomUpConfig {
                stopping: Stopping::KnownK(2),
                params_per_segment: 2,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 0.0, 10.0, 10.0, 0.0, 0.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 1,
            candidate_splits: Some(vec![3]),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let err = detector
            .detect(&view, &ctx)
            .expect_err("known-k should be unreachable");
        assert!(
            err.to_string()
                .contains("KnownK exact solution unreachable")
        );
    }

    #[test]
    fn constraints_jump_and_min_segment_len_are_enforced() {
        let detector = BottomUp::new(
            CostL2Mean::default(),
            BottomUpConfig {
                stopping: Stopping::KnownK(2),
                params_per_segment: 2,
                cancel_check_every: 4,
            },
        )
        .expect("config should be valid");

        let n = 24;
        let values: Vec<f64> = (0..n)
            .map(|idx| {
                if idx < 8 {
                    0.0
                } else if idx < 16 {
                    5.0
                } else {
                    10.0
                }
            })
            .collect();
        let view = make_f64_view(
            &values,
            n,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let constraints = Constraints {
            min_segment_len: 4,
            jump: 4,
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");

        for &cp in &result.change_points {
            assert_eq!(cp % 4, 0, "change point must respect jump=4");
        }
        let mut start = 0usize;
        for &end in &result.breakpoints {
            assert!(
                end - start >= 4,
                "segment [{start}, {end}) violates min_segment_len=4"
            );
            start = end;
        }
    }

    #[test]
    fn penalty_path_returns_not_supported() {
        let detector = BottomUp::new(
            CostL2Mean::default(),
            BottomUpConfig {
                stopping: Stopping::PenaltyPath(vec![Penalty::Manual(1.0)]),
                params_per_segment: 2,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 1.0, 2.0, 3.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = constraints_with_min_segment_len(1);
        let ctx = ExecutionContext::new(&constraints);
        let err = detector
            .detect(&view, &ctx)
            .expect_err("PenaltyPath should be deferred");
        assert!(matches!(err, cpd_core::CpdError::NotSupported(_)));
    }
}
