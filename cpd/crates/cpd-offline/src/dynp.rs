// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{
    BudgetStatus, CpdError, Diagnostics, ExecutionContext, OfflineChangePointResult,
    OfflineDetector, Penalty, Stopping, TimeSeriesView, ValidatedConstraints,
    check_missing_compatibility, penalty_value_from_effective_params, validate_constraints,
    validate_stopping,
};
use cpd_costs::CostModel;
use std::borrow::Cow;
use std::mem::size_of;
use std::time::Instant;

const DEFAULT_CANCEL_CHECK_EVERY: usize = 1000;

/// Configuration for [`Dynp`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct DynpConfig {
    pub stopping: Stopping,
    pub cancel_check_every: usize,
}

impl Default for DynpConfig {
    fn default() -> Self {
        Self {
            stopping: Stopping::KnownK(1),
            cancel_check_every: DEFAULT_CANCEL_CHECK_EVERY,
        }
    }
}

impl DynpConfig {
    fn validate(&self) -> Result<(), CpdError> {
        validate_stopping(&self.stopping)?;
        Ok(())
    }

    fn normalized_cancel_check_every(&self) -> usize {
        self.cancel_check_every.max(1)
    }
}

/// Exact dynamic-programming offline detector (optimal partitioning for fixed K).
#[derive(Debug)]
pub struct Dynp<C: CostModel> {
    cost_model: C,
    config: DynpConfig,
}

impl<C: CostModel> Dynp<C> {
    pub fn new(cost_model: C, config: DynpConfig) -> Result<Self, CpdError> {
        config.validate()?;
        Ok(Self { cost_model, config })
    }

    pub fn cost_model(&self) -> &C {
        &self.cost_model
    }

    pub fn config(&self) -> &DynpConfig {
        &self.config
    }
}

#[derive(Default, Clone, Copy, Debug)]
struct RuntimeStats {
    cost_evals: usize,
    candidates_considered: usize,
    soft_budget_exceeded: bool,
}

#[derive(Clone, Debug)]
struct DynpKernelResult {
    breakpoints: Vec<usize>,
    objective: f64,
    change_count: usize,
}

#[derive(Clone, Debug)]
struct DynpSweepResult {
    endpoints: Vec<usize>,
    backpointers: Vec<Vec<usize>>,
    objective_by_segment_count: Vec<f64>,
}

#[derive(Clone, Debug)]
struct ResolvedPenalty {
    penalty: Penalty,
    beta: f64,
    params_per_segment: usize,
}

#[derive(Clone, Debug)]
struct PenalizedSelection {
    kernel: DynpKernelResult,
    penalized_objective: f64,
}

fn resolve_penalty_beta<C: CostModel>(
    model: &C,
    penalty: &Penalty,
    n: usize,
    d: usize,
) -> Result<ResolvedPenalty, CpdError> {
    let params_per_segment = model.penalty_params_per_segment();
    if params_per_segment == 0 {
        return Err(CpdError::invalid_input(
            "resolved params_per_segment must be >= 1; got 0",
        ));
    }

    let beta = match penalty {
        Penalty::Manual(value) => *value,
        Penalty::BIC | Penalty::AIC => {
            let effective_params = model.penalty_effective_params(d).ok_or_else(|| {
                CpdError::invalid_input(format!(
                    "model_default effective-params overflow for d={d}, model={}",
                    model.name()
                ))
            })?;
            penalty_value_from_effective_params(penalty, n, effective_params)?
        }
    };
    if !beta.is_finite() || beta <= 0.0 {
        return Err(CpdError::invalid_input(format!(
            "resolved penalty must be finite and > 0.0; got beta={beta}"
        )));
    }

    Ok(ResolvedPenalty {
        penalty: penalty.clone(),
        beta,
        params_per_segment,
    })
}

fn checked_counter_increment(counter: &mut usize, name: &str) -> Result<(), CpdError> {
    *counter = counter
        .checked_add(1)
        .ok_or_else(|| CpdError::resource_limit(format!("{name} counter overflow")))?;
    Ok(())
}

fn checked_usize_mul(lhs: usize, rhs: usize, context: &str) -> Result<usize, CpdError> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| CpdError::resource_limit(format!("{context} overflow")))
}

fn checked_usize_add(lhs: usize, rhs: usize, context: &str) -> Result<usize, CpdError> {
    lhs.checked_add(rhs)
        .ok_or_else(|| CpdError::resource_limit(format!("{context} overflow")))
}

fn estimate_dynp_sweep_state_bytes(
    endpoints_len: usize,
    segments: usize,
) -> Result<usize, CpdError> {
    let backpointer_rows = checked_usize_add(segments, 1, "dynp backpointer row count")?;
    let backpointer_cells = checked_usize_mul(
        backpointer_rows,
        endpoints_len,
        "dynp backpointer cell count",
    )?;
    let backpointer_bytes = checked_usize_mul(
        backpointer_cells,
        size_of::<usize>(),
        "dynp backpointer bytes",
    )?;

    let objective_entries = checked_usize_add(segments, 1, "dynp objective entry count")?;
    let objective_bytes =
        checked_usize_mul(objective_entries, size_of::<f64>(), "dynp objective bytes")?;

    let dp_entries = checked_usize_mul(endpoints_len, 2, "dynp dp entry count")?;
    let dp_bytes = checked_usize_mul(dp_entries, size_of::<f64>(), "dynp dp bytes")?;

    let endpoint_bytes =
        checked_usize_mul(endpoints_len, size_of::<usize>(), "dynp endpoint bytes")?;

    let base = checked_usize_add(backpointer_bytes, objective_bytes, "dynp state bytes")?;
    let base = checked_usize_add(base, dp_bytes, "dynp state bytes")?;
    checked_usize_add(base, endpoint_bytes, "dynp state bytes")
}

fn enforce_memory_budget(
    validated: &ValidatedConstraints,
    required_bytes: usize,
) -> Result<(), CpdError> {
    if let Some(limit_bytes) = validated.memory_budget_bytes
        && required_bytes > limit_bytes
    {
        return Err(CpdError::resource_limit(format!(
            "constraints.memory_budget_bytes exceeded for dynp state: required_bytes={required_bytes}, limit_bytes={limit_bytes}; increase constraints.memory_budget_bytes, reduce constraints.max_change_points, or increase constraints.jump"
        )));
    }
    Ok(())
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

fn build_endpoints(validated: &ValidatedConstraints, n: usize) -> Vec<usize> {
    let mut endpoints = Vec::with_capacity(validated.effective_candidates.len() + 2);
    endpoints.push(0);
    endpoints.extend(validated.effective_candidates.iter().copied());
    if endpoints.last().copied() != Some(n) {
        endpoints.push(n);
    }
    endpoints
}

#[allow(clippy::too_many_arguments)]
fn run_dynp_sweep<C: CostModel>(
    model: &C,
    cache: &C::Cache,
    x: &TimeSeriesView<'_>,
    validated: &ValidatedConstraints,
    max_k: usize,
    cancel_check_every: usize,
    ctx: &ExecutionContext<'_>,
    started_at: Instant,
    runtime: &mut RuntimeStats,
) -> Result<DynpSweepResult, CpdError> {
    let candidate_count = validated.effective_candidates.len();
    if max_k > candidate_count {
        return Err(CpdError::invalid_input(format!(
            "max_k={max_k} exceeds available candidate split positions={candidate_count}"
        )));
    }

    let segments = max_k
        .checked_add(1)
        .ok_or_else(|| CpdError::resource_limit("segments overflow"))?;
    let endpoints = build_endpoints(validated, x.n);
    let estimated_state_bytes = estimate_dynp_sweep_state_bytes(endpoints.len(), segments)?;
    enforce_memory_budget(validated, estimated_state_bytes)?;
    let target_idx = endpoints.len().saturating_sub(1);
    let inf = f64::INFINITY;
    let mut backpointers = vec![vec![usize::MAX; endpoints.len()]; segments + 1];
    let mut objective_by_segment_count = vec![inf; segments + 1];
    let mut dp_prev = vec![inf; endpoints.len()];
    dp_prev[0] = 0.0;
    let mut iteration = 0usize;

    for (segment_count, backpointer_row) in backpointers
        .iter_mut()
        .enumerate()
        .take(segments + 1)
        .skip(1)
    {
        let mut dp_curr = vec![inf; endpoints.len()];

        for end_idx in 1..endpoints.len() {
            checked_counter_increment(&mut iteration, "iteration")?;
            check_runtime_controls(iteration, cancel_check_every, ctx, started_at, runtime)?;

            if end_idx < segment_count {
                continue;
            }

            let end = endpoints[end_idx];
            let mut best_objective = inf;
            let mut best_prev_idx = usize::MAX;

            for start_idx in 0..end_idx {
                checked_counter_increment(&mut iteration, "iteration")?;
                check_runtime_controls(iteration, cancel_check_every, ctx, started_at, runtime)?;

                let start = endpoints[start_idx];
                let segment_len = end.saturating_sub(start);
                if segment_len < validated.min_segment_len {
                    continue;
                }
                if !dp_prev[start_idx].is_finite() {
                    continue;
                }

                checked_counter_increment(
                    &mut runtime.candidates_considered,
                    "candidates_considered",
                )?;
                let segment_cost = evaluate_segment_cost(model, cache, start, end, ctx, runtime)?;
                let objective = dp_prev[start_idx] + segment_cost;
                if !objective.is_finite() {
                    return Err(CpdError::numerical_issue(format!(
                        "non-finite dynp objective at segment_count={segment_count}, start={start}, end={end}"
                    )));
                }

                let is_better = objective < best_objective
                    || (objective == best_objective && start_idx < best_prev_idx);
                if is_better {
                    best_objective = objective;
                    best_prev_idx = start_idx;
                }
            }

            if best_prev_idx != usize::MAX {
                dp_curr[end_idx] = best_objective;
                backpointer_row[end_idx] = best_prev_idx;
            }
        }

        objective_by_segment_count[segment_count] = dp_curr[target_idx];
        dp_prev = dp_curr;
        ctx.report_progress(segment_count as f32 / segments as f32);
    }

    Ok(DynpSweepResult {
        endpoints,
        backpointers,
        objective_by_segment_count,
    })
}

fn reconstruct_breakpoints_for_segment_count(
    sweep: &DynpSweepResult,
    segment_count: usize,
    n: usize,
) -> Result<Vec<usize>, CpdError> {
    if segment_count == 0 || segment_count >= sweep.backpointers.len() {
        return Err(CpdError::invalid_input(format!(
            "invalid segment_count={segment_count} for backtracking"
        )));
    }

    if sweep.endpoints.last().copied() != Some(n) {
        return Err(CpdError::invalid_input(format!(
            "invalid dynp endpoint state: expected terminal endpoint n={n}"
        )));
    }

    if segment_count == 1 {
        return Ok(vec![n]);
    }

    let target_idx = sweep.endpoints.len().saturating_sub(1);
    let mut split_points_reversed = Vec::with_capacity(segment_count - 1);
    let mut cursor = target_idx;
    for current_segment_count in (2..=segment_count).rev() {
        let prev_idx = sweep.backpointers[current_segment_count][cursor];
        if prev_idx == usize::MAX {
            return Err(CpdError::invalid_input(format!(
                "backtracking failed at segment_count={current_segment_count}, endpoint={}",
                sweep.endpoints[cursor]
            )));
        }
        let split = sweep.endpoints[prev_idx];
        if split == 0 || split >= n {
            return Err(CpdError::invalid_input(format!(
                "invalid split during backtracking at split={split}"
            )));
        }
        split_points_reversed.push(split);
        cursor = prev_idx;
    }
    split_points_reversed.reverse();

    let mut breakpoints = split_points_reversed;
    breakpoints.push(n);

    Ok(breakpoints)
}

fn max_change_points_to_search(validated: &ValidatedConstraints) -> usize {
    let candidate_count = validated.effective_candidates.len();
    match validated.max_change_points {
        Some(max_change_points) => max_change_points.min(candidate_count),
        None => candidate_count,
    }
}

fn select_penalized_kernel(
    sweep: &DynpSweepResult,
    x: &TimeSeriesView<'_>,
    beta: f64,
    context: &str,
) -> Result<PenalizedSelection, CpdError> {
    if !beta.is_finite() || beta <= 0.0 {
        return Err(CpdError::invalid_input(format!(
            "{context} requires finite beta > 0.0; got {beta}"
        )));
    }

    let mut best_k = usize::MAX;
    let mut best_objective = f64::INFINITY;
    let mut best_penalized_objective = f64::INFINITY;

    for segment_count in 1..sweep.objective_by_segment_count.len() {
        let objective = sweep.objective_by_segment_count[segment_count];
        if !objective.is_finite() {
            continue;
        }

        let k = segment_count - 1;
        let penalized_objective = objective + beta * (k as f64);
        if !penalized_objective.is_finite() {
            return Err(CpdError::numerical_issue(format!(
                "{context} produced non-finite objective for k={k}: objective={objective}, beta={beta}"
            )));
        }

        let is_better = penalized_objective < best_penalized_objective
            || (penalized_objective == best_penalized_objective && k < best_k);
        if is_better {
            best_k = k;
            best_objective = objective;
            best_penalized_objective = penalized_objective;
        }
    }

    if best_k == usize::MAX {
        return Err(CpdError::invalid_input(format!(
            "{context} found no feasible segmentation under current constraints"
        )));
    }

    let segment_count = best_k
        .checked_add(1)
        .ok_or_else(|| CpdError::resource_limit("segment_count overflow"))?;
    let breakpoints = reconstruct_breakpoints_for_segment_count(sweep, segment_count, x.n)
        .map_err(|err| CpdError::invalid_input(format!("{context} backtracking failed: {err}")))?;

    Ok(PenalizedSelection {
        kernel: DynpKernelResult {
            breakpoints,
            objective: best_objective,
            change_count: best_k,
        },
        penalized_objective: best_penalized_objective,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_dynp_known_k<C: CostModel>(
    model: &C,
    cache: &C::Cache,
    x: &TimeSeriesView<'_>,
    validated: &ValidatedConstraints,
    k: usize,
    cancel_check_every: usize,
    ctx: &ExecutionContext<'_>,
    started_at: Instant,
    runtime: &mut RuntimeStats,
) -> Result<DynpKernelResult, CpdError> {
    if let Some(max_change_points) = validated.max_change_points
        && max_change_points < k
    {
        return Err(CpdError::invalid_input(format!(
            "KnownK={k} exceeds constraints.max_change_points={max_change_points}"
        )));
    }

    let candidate_count = validated.effective_candidates.len();
    if candidate_count < k {
        return Err(CpdError::invalid_input(format!(
            "KnownK exact solution unreachable: requested k={k}, but only {candidate_count} candidate split positions are available under constraints"
        )));
    }

    let sweep = run_dynp_sweep(
        model,
        cache,
        x,
        validated,
        k,
        cancel_check_every,
        ctx,
        started_at,
        runtime,
    )?;

    let segment_count = k
        .checked_add(1)
        .ok_or_else(|| CpdError::resource_limit("segment_count overflow"))?;
    let objective = sweep.objective_by_segment_count[segment_count];
    if !objective.is_finite() {
        return Err(CpdError::invalid_input(format!(
            "KnownK exact solution unreachable: requested k={k} under current constraints"
        )));
    }

    let breakpoints = reconstruct_breakpoints_for_segment_count(&sweep, segment_count, x.n)
        .map_err(|err| {
            CpdError::invalid_input(format!("KnownK exact solution unreachable: {err}"))
        })?;

    Ok(DynpKernelResult {
        objective,
        change_count: k,
        breakpoints,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_dynp_penalized<C: CostModel>(
    model: &C,
    cache: &C::Cache,
    x: &TimeSeriesView<'_>,
    validated: &ValidatedConstraints,
    beta: f64,
    cancel_check_every: usize,
    ctx: &ExecutionContext<'_>,
    started_at: Instant,
    runtime: &mut RuntimeStats,
) -> Result<PenalizedSelection, CpdError> {
    let max_k = max_change_points_to_search(validated);
    let sweep = run_dynp_sweep(
        model,
        cache,
        x,
        validated,
        max_k,
        cancel_check_every,
        ctx,
        started_at,
        runtime,
    )?;
    select_penalized_kernel(&sweep, x, beta, "Penalized selection")
}

#[allow(clippy::too_many_arguments)]
fn run_dynp_penalty_path<C: CostModel>(
    model: &C,
    cache: &C::Cache,
    x: &TimeSeriesView<'_>,
    validated: &ValidatedConstraints,
    betas: &[f64],
    cancel_check_every: usize,
    ctx: &ExecutionContext<'_>,
    started_at: Instant,
    runtime: &mut RuntimeStats,
) -> Result<Vec<PenalizedSelection>, CpdError> {
    if betas.is_empty() {
        return Err(CpdError::invalid_input(
            "run_dynp_penalty_path requires at least one beta",
        ));
    }

    for (idx, &beta) in betas.iter().enumerate() {
        if !beta.is_finite() || beta <= 0.0 {
            return Err(CpdError::invalid_input(format!(
                "run_dynp_penalty_path beta[{idx}] must be finite and > 0; got {beta}"
            )));
        }
    }

    let max_k = max_change_points_to_search(validated);
    let sweep = run_dynp_sweep(
        model,
        cache,
        x,
        validated,
        max_k,
        cancel_check_every,
        ctx,
        started_at,
        runtime,
    )?;

    let mut out = Vec::with_capacity(betas.len());
    for (idx, &beta) in betas.iter().enumerate() {
        let context = format!("PenaltyPath[{idx}]");
        let selection = select_penalized_kernel(&sweep, x, beta, context.as_str())?;
        out.push(selection);
    }

    Ok(out)
}

impl<C: CostModel> OfflineDetector for Dynp<C> {
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
        let mut notes = vec![];
        let mut warnings = vec![];

        if let Some(max_depth) = validated.max_depth {
            warnings.push(format!(
                "constraints.max_depth={max_depth} is ignored by Dynp"
            ));
        }

        let kernel = match &self.config.stopping {
            Stopping::KnownK(k) => {
                notes.push(format!("stopping=KnownK({k})"));
                run_dynp_known_k(
                    &self.cost_model,
                    &cache,
                    x,
                    &validated,
                    *k,
                    cancel_check_every,
                    ctx,
                    started_at,
                    &mut runtime,
                )?
            }
            Stopping::Penalized(penalty) => {
                let resolved = resolve_penalty_beta(&self.cost_model, penalty, x.n, x.d)?;
                notes.push(format!(
                    "stopping=Penalized({penalty:?}), beta={}, params_per_segment={} (model_default)",
                    resolved.beta, resolved.params_per_segment
                ));
                let selection = run_dynp_penalized(
                    &self.cost_model,
                    &cache,
                    x,
                    &validated,
                    resolved.beta,
                    cancel_check_every,
                    ctx,
                    started_at,
                    &mut runtime,
                )?;
                notes.push(format!(
                    "selected_penalized_objective={}",
                    selection.penalized_objective
                ));
                selection.kernel
            }
            Stopping::PenaltyPath(path) => {
                let mut resolved_path = Vec::with_capacity(path.len());
                let mut betas = Vec::with_capacity(path.len());
                for penalty in path {
                    let resolved = resolve_penalty_beta(&self.cost_model, penalty, x.n, x.d)?;
                    betas.push(resolved.beta);
                    resolved_path.push(resolved);
                }

                notes.push(format!(
                    "stopping=PenaltyPath(len={}), primary_index=0",
                    resolved_path.len()
                ));

                let selections = run_dynp_penalty_path(
                    &self.cost_model,
                    &cache,
                    x,
                    &validated,
                    betas.as_slice(),
                    cancel_check_every,
                    ctx,
                    started_at,
                    &mut runtime,
                )?;

                for (idx, (resolved, selection)) in
                    resolved_path.iter().zip(selections.iter()).enumerate()
                {
                    notes.push(format!(
                        "penalty_path[{idx}]: penalty={:?}, beta={}, params_per_segment={} (model_default), change_count={}, objective={}, penalized_objective={}",
                        resolved.penalty,
                        resolved.beta,
                        resolved.params_per_segment,
                        selection.kernel.change_count,
                        selection.kernel.objective,
                        selection.penalized_objective
                    ));
                }

                selections
                    .into_iter()
                    .next()
                    .expect("PenaltyPath validated to be non-empty")
                    .kernel
            }
        };

        if runtime.soft_budget_exceeded {
            warnings.push(
                "budget exceeded under SoftDegrade mode; run continued without algorithm fallback"
                    .to_string(),
            );
        }

        let runtime_ms = u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX);

        ctx.record_scalar("offline.dynp.cost_evals", runtime.cost_evals as f64);
        ctx.record_scalar(
            "offline.dynp.candidates_considered",
            runtime.candidates_considered as f64,
        );
        ctx.record_scalar("offline.dynp.runtime_ms", runtime_ms as f64);
        ctx.report_progress(1.0);

        notes.push(format!(
            "final_objective={}, change_count={}",
            kernel.objective, kernel.change_count
        ));
        notes.push(format!(
            "cost_evals={}, candidates_considered={}",
            runtime.cost_evals, runtime.candidates_considered
        ));

        let diagnostics = Diagnostics {
            n: x.n,
            d: x.d,
            runtime_ms: Some(runtime_ms),
            notes,
            warnings,
            algorithm: Cow::Borrowed("dynp"),
            cost_model: Cow::Borrowed(self.cost_model.name()),
            repro_mode: ctx.repro_mode,
            ..Diagnostics::default()
        };

        OfflineChangePointResult::new(x.n, kernel.breakpoints, diagnostics)
    }
}

#[cfg(test)]
mod tests {
    use super::{Dynp, DynpConfig};
    use crate::{Pelt, PeltConfig};
    use cpd_core::{
        BudgetMode, CancelToken, Constraints, CpdError, DTypeView, ExecutionContext, MemoryLayout,
        MissingPolicy, OfflineDetector, Penalty, ReproMode, Stopping, TimeIndex, TimeSeriesView,
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
        let default_cfg = DynpConfig::default();
        assert_eq!(default_cfg.stopping, Stopping::KnownK(1));
        assert_eq!(default_cfg.cancel_check_every, 1000);

        let ok = Dynp::new(CostL2Mean::default(), default_cfg.clone())
            .expect("default config should be valid");
        assert_eq!(ok.config(), &default_cfg);
    }

    #[test]
    fn known_small_example_two_changes_l2() {
        let detector = Dynp::new(
            CostL2Mean::default(),
            DynpConfig {
                stopping: Stopping::KnownK(2),
                cancel_check_every: 4,
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
        let result = detector
            .detect(&view, &ctx)
            .expect("dynp detect should succeed");
        assert_eq!(result.breakpoints, vec![4, 8, 12]);
    }

    #[test]
    fn known_k_zero_reports_clear_error() {
        let err = Dynp::new(
            CostL2Mean::default(),
            DynpConfig {
                stopping: Stopping::KnownK(0),
                cancel_check_every: 4,
            },
        )
        .expect_err("k=0 should be rejected during config validation");
        match err {
            CpdError::InvalidInput(msg) => assert!(msg.contains("KnownK")),
            _ => panic!("expected InvalidInput for k=0"),
        }
    }

    #[test]
    fn known_k_greater_than_n_reports_clear_error() {
        let detector = Dynp::new(
            CostL2Mean::default(),
            DynpConfig {
                stopping: Stopping::KnownK(7),
                cancel_check_every: 4,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 0.0, 10.0, 10.0, -3.0, -3.0];
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
            .expect_err("k>=n should be unreachable");

        let message = err.to_string();
        assert!(message.contains("KnownK exact solution unreachable"));
        assert!(message.contains("requested k=7"));
    }

    #[test]
    fn known_k_tie_breaking_is_deterministic_across_repro_modes() {
        let detector = Dynp::new(
            CostL2Mean::default(),
            DynpConfig {
                stopping: Stopping::KnownK(1),
                cancel_check_every: 4,
            },
        )
        .expect("config should be valid");

        let values = vec![5.0; 8];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = constraints_with_min_segment_len(2);

        let balanced_result = detector
            .detect(
                &view,
                &ExecutionContext::new(&constraints).with_repro_mode(ReproMode::Balanced),
            )
            .expect("balanced mode should succeed");
        let fast_result = detector
            .detect(
                &view,
                &ExecutionContext::new(&constraints).with_repro_mode(ReproMode::Fast),
            )
            .expect("fast mode should succeed");
        let strict_result = detector
            .detect(
                &view,
                &ExecutionContext::new(&constraints).with_repro_mode(ReproMode::Strict),
            )
            .expect("strict mode should succeed");

        assert_eq!(balanced_result.breakpoints, vec![2, 8]);
        assert_eq!(fast_result.breakpoints, balanced_result.breakpoints);
        assert_eq!(strict_result.breakpoints, balanced_result.breakpoints);
    }

    #[test]
    fn known_k_respects_max_change_points() {
        let detector = Dynp::new(
            CostL2Mean::default(),
            DynpConfig {
                stopping: Stopping::KnownK(2),
                cancel_check_every: 4,
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
            max_change_points: Some(1),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let err = detector
            .detect(&view, &ctx)
            .expect_err("KnownK > max_change_points must fail");
        assert!(
            err.to_string()
                .contains("exceeds constraints.max_change_points")
        );
    }

    #[test]
    fn known_k_exact_unreachable_is_clear_error() {
        let detector = Dynp::new(
            CostL2Mean::default(),
            DynpConfig {
                stopping: Stopping::KnownK(2),
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
        let constraints = Constraints {
            min_segment_len: 2,
            candidate_splits: Some(vec![4]),
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
    fn known_k_matches_pelt_when_pelt_is_optimal() {
        let values = vec![
            -2.0, -2.0, -2.0, -2.0, 6.0, 6.0, 6.0, 6.0, -4.0, -4.0, -4.0, -4.0,
        ];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 2,
            max_change_points: Some(2),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);

        let dynp = Dynp::new(
            CostNormalMeanVar::default(),
            DynpConfig {
                stopping: Stopping::KnownK(2),
                cancel_check_every: 8,
            },
        )
        .expect("dynp config should be valid");
        let pelt = Pelt::new(
            CostNormalMeanVar::default(),
            PeltConfig {
                stopping: Stopping::KnownK(2),
                params_per_segment: 3,
                cancel_check_every: 8,
            },
        )
        .expect("pelt config should be valid");

        let dynp_result = dynp
            .detect(&view, &ctx)
            .expect("dynp known-k should succeed");
        let pelt_result = pelt
            .detect(&view, &ctx)
            .expect("pelt known-k should succeed");

        assert_eq!(dynp_result.breakpoints, pelt_result.breakpoints);
    }

    #[test]
    fn penalized_matches_pelt_for_manual_penalty() {
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

        let dynp = Dynp::new(
            CostL2Mean::default(),
            DynpConfig {
                stopping: Stopping::Penalized(Penalty::Manual(1.0)),
                cancel_check_every: 16,
            },
        )
        .expect("penalized config should be valid");
        let pelt = Pelt::new(
            CostL2Mean::default(),
            PeltConfig {
                stopping: Stopping::Penalized(Penalty::Manual(1.0)),
                params_per_segment: 2,
                cancel_check_every: 16,
            },
        )
        .expect("pelt config should be valid");

        let dynp_result = dynp
            .detect(&view, &ctx)
            .expect("dynp penalized should succeed");
        let pelt_result = pelt
            .detect(&view, &ctx)
            .expect("pelt penalized should succeed");
        assert_eq!(dynp_result.breakpoints, pelt_result.breakpoints);
    }

    #[test]
    fn penalized_respects_max_change_points_cap() {
        let detector = Dynp::new(
            CostL2Mean::default(),
            DynpConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.1)),
                cancel_check_every: 8,
            },
        )
        .expect("penalized config should be valid");

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
        let constraints = Constraints {
            min_segment_len: 2,
            max_change_points: Some(1),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let result = detector
            .detect(&view, &ctx)
            .expect("dynp penalized should succeed");

        assert!(result.change_points.len() <= 1);
    }

    #[test]
    fn penalty_path_uses_primary_index_zero_and_records_notes() {
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

        let primary_detector = Dynp::new(
            CostL2Mean::default(),
            DynpConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.1)),
                cancel_check_every: 8,
            },
        )
        .expect("primary penalized config should be valid");
        let path_detector = Dynp::new(
            CostL2Mean::default(),
            DynpConfig {
                stopping: Stopping::PenaltyPath(vec![Penalty::Manual(0.1), Penalty::Manual(80.0)]),
                cancel_check_every: 8,
            },
        )
        .expect("penalty path config should be valid");

        let primary_result = primary_detector
            .detect(&view, &ctx)
            .expect("primary run should succeed");
        let path_result = path_detector
            .detect(&view, &ctx)
            .expect("path run should succeed");

        assert_eq!(path_result.breakpoints, primary_result.breakpoints);
        assert!(
            path_result
                .diagnostics
                .notes
                .iter()
                .any(|note| note == "stopping=PenaltyPath(len=2), primary_index=0")
        );
        assert!(
            path_result
                .diagnostics
                .notes
                .iter()
                .any(|note| note.contains("penalty_path[0]:"))
        );
        assert!(
            path_result
                .diagnostics
                .notes
                .iter()
                .any(|note| note.contains("penalty_path[1]:"))
        );
    }

    #[test]
    fn cancellation_mid_run_returns_cancelled() {
        let detector = Dynp::new(
            CostL2Mean::default(),
            DynpConfig {
                stopping: Stopping::KnownK(2),
                cancel_check_every: 1,
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
        let cancel = CancelToken::new();
        cancel.cancel();
        let ctx = ExecutionContext::new(&constraints).with_cancel(&cancel);

        let err = detector
            .detect(&view, &ctx)
            .expect_err("cancelled token must stop detect");
        assert_eq!(err.to_string(), "cancelled");
    }

    #[test]
    fn cost_eval_budget_exceeded_hard_fail() {
        let detector = Dynp::new(
            CostL2Mean::default(),
            DynpConfig {
                stopping: Stopping::KnownK(2),
                cancel_check_every: 1,
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
        let constraints = Constraints {
            min_segment_len: 2,
            max_cost_evals: Some(1),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints).with_budget_mode(BudgetMode::HardFail);

        let err = detector
            .detect(&view, &ctx)
            .expect_err("hard budget should fail");
        assert!(err.to_string().contains("max_cost_evals"));
    }

    #[test]
    fn time_budget_exceeded_hard_fail() {
        let detector = Dynp::new(
            CostL2Mean::default(),
            DynpConfig {
                stopping: Stopping::KnownK(8),
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let mut values = vec![0.0; 4_096];
        for item in values.iter_mut().take(2_048).skip(1_024) {
            *item = 5.0;
        }
        for item in values.iter_mut().skip(2_048) {
            *item = -3.0;
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
            time_budget_ms: Some(1),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints).with_budget_mode(BudgetMode::HardFail);

        let err = detector
            .detect(&view, &ctx)
            .expect_err("hard time budget should fail");
        assert!(err.to_string().contains("time_budget_ms"));
    }

    #[test]
    fn diagnostics_include_soft_budget_warning() {
        let detector = Dynp::new(
            CostL2Mean::default(),
            DynpConfig {
                stopping: Stopping::KnownK(2),
                cancel_check_every: 1,
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
        let constraints = Constraints {
            min_segment_len: 2,
            max_cost_evals: Some(1),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints).with_budget_mode(BudgetMode::SoftDegrade);

        let result = detector
            .detect(&view, &ctx)
            .expect("soft budget mode should still return a result");
        assert!(
            result
                .diagnostics
                .warnings
                .iter()
                .any(|warning| warning.contains("SoftDegrade"))
        );
    }

    #[test]
    fn memory_budget_exceeded_returns_resource_limit() {
        let detector = Dynp::new(
            CostL2Mean::default(),
            DynpConfig {
                stopping: Stopping::KnownK(2),
                cancel_check_every: 1,
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
        let constraints = Constraints {
            min_segment_len: 2,
            memory_budget_bytes: Some(64),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);

        let err = detector
            .detect(&view, &ctx)
            .expect_err("insufficient memory budget should fail");
        let message = err.to_string();
        assert!(message.contains("memory_budget_bytes"));
        assert!(message.contains("required_bytes"));
    }
}
