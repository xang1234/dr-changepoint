// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{
    BudgetStatus, CpdError, Diagnostics, ExecutionContext, OfflineChangePointResult,
    OfflineDetector, Penalty, Stopping, TimeSeriesView, ValidatedConstraints,
    check_missing_compatibility, penalty_value, validate_constraints, validate_stopping,
};
use cpd_costs::CostModel;
use std::borrow::Cow;
use std::time::Instant;

const DEFAULT_CANCEL_CHECK_EVERY: usize = 1000;
const AUTO_PARAMS_PER_SEGMENT: usize = 0;
const DEFAULT_WINDOW_WIDTH: usize = 20;

/// Configuration for [`SlidingWindow`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SlidingWindowConfig {
    pub stopping: Stopping,
    pub window_width: usize,
    pub params_per_segment: usize,
    pub cancel_check_every: usize,
}

impl Default for SlidingWindowConfig {
    fn default() -> Self {
        Self {
            stopping: Stopping::Penalized(Penalty::BIC),
            window_width: DEFAULT_WINDOW_WIDTH,
            params_per_segment: AUTO_PARAMS_PER_SEGMENT,
            cancel_check_every: DEFAULT_CANCEL_CHECK_EVERY,
        }
    }
}

impl SlidingWindowConfig {
    fn validate(&self) -> Result<(), CpdError> {
        validate_stopping(&self.stopping)?;

        if self.window_width < 2 {
            return Err(CpdError::invalid_input(format!(
                "SlidingWindowConfig.window_width must be >= 2; got {}",
                self.window_width
            )));
        }
        if !self.window_width.is_multiple_of(2) {
            return Err(CpdError::invalid_input(format!(
                "SlidingWindowConfig.window_width must be even; got {}",
                self.window_width
            )));
        }
        Ok(())
    }

    fn normalized_cancel_check_every(&self) -> usize {
        self.cancel_check_every.max(1)
    }

    fn half_window(&self) -> usize {
        self.window_width / 2
    }
}

/// Sliding-window local discrepancy offline detector.
#[derive(Debug)]
pub struct SlidingWindow<C: CostModel> {
    cost_model: C,
    config: SlidingWindowConfig,
}

impl<C: CostModel> SlidingWindow<C> {
    pub fn new(cost_model: C, config: SlidingWindowConfig) -> Result<Self, CpdError> {
        config.validate()?;
        Ok(Self { cost_model, config })
    }

    pub fn cost_model(&self) -> &C {
        &self.cost_model
    }

    pub fn config(&self) -> &SlidingWindowConfig {
        &self.config
    }
}

#[derive(Default, Clone, Copy, Debug)]
struct RuntimeStats {
    cost_evals: usize,
    candidates_considered: usize,
    windows_considered: usize,
    peaks_considered: usize,
    soft_budget_exceeded: bool,
}

#[derive(Clone, Copy, Debug)]
struct CandidateScore {
    split: usize,
    score: f64,
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

fn window_bounds(
    split: usize,
    half_window: usize,
    n: usize,
    min_segment_len: usize,
) -> Result<Option<(usize, usize)>, CpdError> {
    if half_window < min_segment_len || split < half_window {
        return Ok(None);
    }

    let start = split
        .checked_sub(half_window)
        .ok_or_else(|| CpdError::resource_limit("window start underflow"))?;
    let end = split
        .checked_add(half_window)
        .ok_or_else(|| CpdError::resource_limit("window end overflow"))?;
    if end > n {
        return Ok(None);
    }

    if split.saturating_sub(start) < min_segment_len || end.saturating_sub(split) < min_segment_len
    {
        return Ok(None);
    }

    Ok(Some((start, end)))
}

#[allow(clippy::too_many_arguments)]
fn compute_candidate_scores<C: CostModel>(
    model: &C,
    cache: &C::Cache,
    candidates: &[usize],
    validated: &ValidatedConstraints,
    half_window: usize,
    ctx: &ExecutionContext<'_>,
    cancel_check_every: usize,
    started_at: Instant,
    runtime: &mut RuntimeStats,
    iteration: &mut usize,
) -> Result<Vec<CandidateScore>, CpdError> {
    let mut out = Vec::with_capacity(candidates.len());

    for &split in candidates {
        checked_counter_increment(iteration, "iteration")?;
        check_runtime_controls(*iteration, cancel_check_every, ctx, started_at, runtime)?;
        checked_counter_increment(&mut runtime.candidates_considered, "candidates_considered")?;

        let Some((start, end)) =
            window_bounds(split, half_window, validated.n, validated.min_segment_len)?
        else {
            continue;
        };
        checked_counter_increment(&mut runtime.windows_considered, "windows_considered")?;

        let whole_cost = evaluate_segment_cost(model, cache, start, end, ctx, runtime)?;
        let left_cost = evaluate_segment_cost(model, cache, start, split, ctx, runtime)?;
        let right_cost = evaluate_segment_cost(model, cache, split, end, ctx, runtime)?;
        let score = whole_cost - left_cost - right_cost;
        if !score.is_finite() {
            return Err(CpdError::numerical_issue(format!(
                "non-finite score at split={split}, window=[{start}, {end}): whole={whole_cost}, left={left_cost}, right={right_cost}, score={score}"
            )));
        }

        out.push(CandidateScore { split, score });
    }

    Ok(out)
}

fn extract_peaks(scores: &[CandidateScore]) -> Vec<CandidateScore> {
    if scores.is_empty() {
        return vec![];
    }

    let mut peaks = Vec::with_capacity(scores.len());
    for idx in 0..scores.len() {
        let current = scores[idx];
        let left_ok = idx == 0 || current.score > scores[idx - 1].score;
        let right_ok = idx + 1 == scores.len() || current.score >= scores[idx + 1].score;
        if left_ok && right_ok {
            peaks.push(current);
        }
    }

    if peaks.is_empty() {
        scores.to_vec()
    } else {
        peaks
    }
}

fn rank_candidates(scores: &mut [CandidateScore]) {
    scores.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.split.cmp(&right.split))
    });
}

fn can_insert_split(selected: &[usize], split: usize, min_segment_len: usize, n: usize) -> bool {
    match selected.binary_search(&split) {
        Ok(_) => false,
        Err(insert_idx) => {
            let prev = if insert_idx == 0 {
                0
            } else {
                selected[insert_idx - 1]
            };
            let next = if insert_idx == selected.len() {
                n
            } else {
                selected[insert_idx]
            };
            split.saturating_sub(prev) >= min_segment_len
                && next.saturating_sub(split) >= min_segment_len
        }
    }
}

fn insert_sorted_unique(values: &mut Vec<usize>, value: usize) -> Result<(), CpdError> {
    match values.binary_search(&value) {
        Ok(_) => Err(CpdError::invalid_input(format!(
            "duplicate split selected at {value}; internal SlidingWindow state is inconsistent"
        ))),
        Err(idx) => {
            values.insert(idx, value);
            Ok(())
        }
    }
}

fn select_known_k(
    ranked_candidates: &[CandidateScore],
    validated: &ValidatedConstraints,
    n: usize,
    k: usize,
) -> Result<Vec<usize>, CpdError> {
    let mut selected = Vec::with_capacity(k);
    for candidate in ranked_candidates {
        if can_insert_split(&selected, candidate.split, validated.min_segment_len, n) {
            insert_sorted_unique(&mut selected, candidate.split)?;
            if selected.len() == k {
                break;
            }
        }
    }

    if selected.len() != k {
        return Err(CpdError::invalid_input(format!(
            "KnownK exact solution unreachable: requested k={k}, accepted={} after ranking {} peak candidates",
            selected.len(),
            ranked_candidates.len()
        )));
    }

    Ok(selected)
}

fn select_penalized(
    ranked_candidates: &[CandidateScore],
    validated: &ValidatedConstraints,
    n: usize,
    beta: f64,
) -> Result<Vec<usize>, CpdError> {
    let mut selected = vec![];
    for candidate in ranked_candidates {
        if candidate.score <= beta {
            break;
        }
        if let Some(max_change_points) = validated.max_change_points
            && selected.len() >= max_change_points
        {
            break;
        }
        if can_insert_split(&selected, candidate.split, validated.min_segment_len, n) {
            insert_sorted_unique(&mut selected, candidate.split)?;
        }
    }
    Ok(selected)
}

fn build_result_breakpoints(n: usize, change_points: Vec<usize>) -> Vec<usize> {
    let mut breakpoints = change_points;
    breakpoints.push(n);
    breakpoints
}

impl<C: CostModel> OfflineDetector for SlidingWindow<C> {
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
        let half_window = self.config.half_window();
        let mut runtime = RuntimeStats::default();
        let mut iteration = 0usize;
        let mut notes = vec![format!(
            "window_width={}, half_window={half_window}",
            self.config.window_width
        )];
        let mut warnings = vec![
            "SlidingWindow is local and may under-detect diffuse long-range shifts".to_string(),
        ];

        if let Some(max_depth) = validated.max_depth {
            warnings.push(format!(
                "constraints.max_depth={max_depth} is ignored by SlidingWindow"
            ));
        }

        let mut ranked_candidates = extract_peaks(&compute_candidate_scores(
            &self.cost_model,
            &cache,
            &validated.effective_candidates,
            &validated,
            half_window,
            ctx,
            cancel_check_every,
            started_at,
            &mut runtime,
            &mut iteration,
        )?);
        runtime.peaks_considered = ranked_candidates.len();
        rank_candidates(&mut ranked_candidates);

        let accepted_splits = match &self.config.stopping {
            Stopping::KnownK(k) => {
                if let Some(max_change_points) = validated.max_change_points
                    && max_change_points < *k
                {
                    return Err(CpdError::invalid_input(format!(
                        "KnownK={k} exceeds constraints.max_change_points={max_change_points}"
                    )));
                }
                notes.push(format!("stopping=KnownK({k})"));
                select_known_k(&ranked_candidates, &validated, x.n, *k)?
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
                select_penalized(&ranked_candidates, &validated, x.n, resolved.beta)?
            }
            Stopping::PenaltyPath(path) => {
                return Err(CpdError::not_supported(format!(
                    "SlidingWindow penalty sweep is deferred for this issue; got PenaltyPath of length {}",
                    path.len()
                )));
            }
        };

        if runtime.soft_budget_exceeded {
            warnings.push(
                "budget exceeded under SoftDegrade mode; run continued without algorithm fallback"
                    .to_string(),
            );
        }

        let runtime_ms = u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX);
        ctx.record_scalar(
            "offline.sliding_window.cost_evals",
            runtime.cost_evals as f64,
        );
        ctx.record_scalar(
            "offline.sliding_window.candidates_considered",
            runtime.candidates_considered as f64,
        );
        ctx.record_scalar(
            "offline.sliding_window.windows_considered",
            runtime.windows_considered as f64,
        );
        ctx.record_scalar(
            "offline.sliding_window.peaks_considered",
            runtime.peaks_considered as f64,
        );
        ctx.record_scalar("offline.sliding_window.runtime_ms", runtime_ms as f64);
        ctx.report_progress(1.0);

        notes.push(format!(
            "final_change_count={}, cost_evals={}, candidates_considered={}, windows_considered={}, peak_candidates={}",
            accepted_splits.len(),
            runtime.cost_evals,
            runtime.candidates_considered,
            runtime.windows_considered,
            runtime.peaks_considered
        ));

        let diagnostics = Diagnostics {
            n: x.n,
            d: x.d,
            runtime_ms: Some(runtime_ms),
            notes,
            warnings,
            algorithm: Cow::Borrowed("sliding_window"),
            cost_model: Cow::Borrowed(self.cost_model.name()),
            repro_mode: ctx.repro_mode,
            ..Diagnostics::default()
        };

        let breakpoints = build_result_breakpoints(x.n, accepted_splits);
        OfflineChangePointResult::new(x.n, breakpoints, diagnostics)
    }
}

#[cfg(test)]
mod tests {
    use super::{SlidingWindow, SlidingWindowConfig};
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

    fn assert_breakpoint_near(breakpoints: &[usize], expected: usize, tolerance: usize) {
        assert!(
            breakpoints
                .iter()
                .any(|&breakpoint| breakpoint.abs_diff(expected) <= tolerance),
            "expected a breakpoint within +/-{tolerance} of {expected}; got {breakpoints:?}",
        );
    }

    #[test]
    fn config_defaults_and_validation() {
        let default_cfg = SlidingWindowConfig::default();
        assert_eq!(default_cfg.stopping, Stopping::Penalized(Penalty::BIC));
        assert_eq!(default_cfg.window_width, 20);
        assert_eq!(default_cfg.params_per_segment, 0);
        assert_eq!(default_cfg.cancel_check_every, 1000);

        let ok = SlidingWindow::new(CostL2Mean::default(), default_cfg.clone())
            .expect("default config should be valid");
        assert_eq!(ok.config(), &default_cfg);

        let err_small = SlidingWindow::new(
            CostL2Mean::default(),
            SlidingWindowConfig {
                stopping: Stopping::Penalized(Penalty::Manual(1.0)),
                window_width: 1,
                params_per_segment: 2,
                cancel_check_every: 100,
            },
        )
        .expect_err("window_width < 2 must fail");
        assert!(err_small.to_string().contains("window_width"));

        let err_odd = SlidingWindow::new(
            CostL2Mean::default(),
            SlidingWindowConfig {
                stopping: Stopping::Penalized(Penalty::Manual(1.0)),
                window_width: 7,
                params_per_segment: 2,
                cancel_check_every: 100,
            },
        )
        .expect_err("odd window_width must fail");
        assert!(err_odd.to_string().contains("even"));
    }

    #[test]
    fn cancel_check_every_zero_is_normalized() {
        let detector = SlidingWindow::new(
            CostL2Mean::default(),
            SlidingWindowConfig {
                stopping: Stopping::KnownK(1),
                window_width: 4,
                params_per_segment: 2,
                cancel_check_every: 0,
            },
        )
        .expect("config with zero cadence should normalize");

        let values = vec![0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0];
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
    fn known_small_example_one_change_l2() {
        let detector = SlidingWindow::new(
            CostL2Mean::default(),
            SlidingWindowConfig {
                stopping: Stopping::KnownK(1),
                window_width: 8,
                params_per_segment: 2,
                cancel_check_every: 8,
            },
        )
        .expect("config should be valid");

        let values = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0,
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
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
        assert_eq!(result.breakpoints.len(), 2);
        assert_eq!(result.breakpoints.last().copied(), Some(values.len()));
        assert_breakpoint_near(&result.breakpoints, 12, 1);
    }

    #[test]
    fn known_small_example_two_changes_l2() {
        let detector = SlidingWindow::new(
            CostL2Mean::default(),
            SlidingWindowConfig {
                stopping: Stopping::KnownK(2),
                window_width: 6,
                params_per_segment: 2,
                cancel_check_every: 4,
            },
        )
        .expect("config should be valid");

        let values = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0,
            -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0,
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
        assert_eq!(result.breakpoints.len(), 3);
        assert_eq!(result.breakpoints.last().copied(), Some(values.len()));
        assert_breakpoint_near(&result.breakpoints, 8, 1);
        assert_breakpoint_near(&result.breakpoints, 16, 1);
    }

    #[test]
    fn normal_cost_detects_variance_change() {
        let detector = SlidingWindow::new(
            CostNormalMeanVar::default(),
            SlidingWindowConfig {
                stopping: Stopping::KnownK(1),
                window_width: 8,
                params_per_segment: 3,
                cancel_check_every: 2,
            },
        )
        .expect("config should be valid");

        let mut values = Vec::with_capacity(40);
        for _ in 0..10 {
            values.push(-1.0);
            values.push(1.0);
        }
        for _ in 0..10 {
            values.push(-6.0);
            values.push(6.0);
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
        assert_eq!(result.breakpoints.len(), 2);
        assert_eq!(result.breakpoints.last().copied(), Some(values.len()));
        assert_breakpoint_near(&result.breakpoints, 20, 2);
    }

    #[test]
    fn penalized_high_threshold_returns_no_change_points() {
        let detector = SlidingWindow::new(
            CostL2Mean::default(),
            SlidingWindowConfig {
                stopping: Stopping::Penalized(Penalty::Manual(1.0e12)),
                window_width: 8,
                params_per_segment: 2,
                cancel_check_every: 16,
            },
        )
        .expect("config should be valid");

        let values = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
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
        assert_eq!(result.breakpoints, vec![values.len()]);
    }

    #[test]
    fn known_k_unreachable_is_clear_error_when_window_too_wide() {
        let detector = SlidingWindow::new(
            CostL2Mean::default(),
            SlidingWindowConfig {
                stopping: Stopping::KnownK(1),
                window_width: 20,
                params_per_segment: 2,
                cancel_check_every: 2,
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
        let err = detector
            .detect(&view, &ctx)
            .expect_err("known-k should be unreachable");
        assert!(
            err.to_string()
                .contains("KnownK exact solution unreachable")
        );
    }

    #[test]
    fn known_k_fails_when_max_change_points_below_k() {
        let detector = SlidingWindow::new(
            CostL2Mean::default(),
            SlidingWindowConfig {
                stopping: Stopping::KnownK(2),
                window_width: 4,
                params_per_segment: 2,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 0.0, 5.0, 5.0, 10.0, 10.0, 15.0, 15.0];
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
            .expect_err("KnownK must reject when max_change_points < k");
        assert!(err.to_string().contains("max_change_points"));
    }

    #[test]
    fn penalty_path_returns_not_supported() {
        let detector = SlidingWindow::new(
            CostL2Mean::default(),
            SlidingWindowConfig {
                stopping: Stopping::PenaltyPath(vec![Penalty::Manual(1.0)]),
                window_width: 4,
                params_per_segment: 2,
                cancel_check_every: 16,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
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
