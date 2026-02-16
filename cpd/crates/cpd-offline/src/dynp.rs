// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{
    BudgetStatus, CpdError, Diagnostics, ExecutionContext, OfflineChangePointResult,
    OfflineDetector, Stopping, TimeSeriesView, ValidatedConstraints, check_missing_compatibility,
    validate_constraints, validate_stopping,
};
use cpd_costs::CostModel;
use std::borrow::Cow;
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

fn checked_counter_increment(counter: &mut usize, name: &str) -> Result<(), CpdError> {
    *counter = counter
        .checked_add(1)
        .ok_or_else(|| CpdError::resource_limit(format!("{name} counter overflow")))?;
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

    let segments = k
        .checked_add(1)
        .ok_or_else(|| CpdError::resource_limit("segments overflow"))?;
    let endpoints = build_endpoints(validated, x.n);
    let target_idx = endpoints.len().saturating_sub(1);
    let inf = f64::INFINITY;
    let mut backpointers = vec![vec![usize::MAX; endpoints.len()]; segments + 1];
    let mut dp_prev = vec![inf; endpoints.len()];
    dp_prev[0] = 0.0;
    let mut iteration = 0usize;

    for segment_count in 1..=segments {
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
                backpointers[segment_count][end_idx] = best_prev_idx;
            }
        }

        dp_prev = dp_curr;
        ctx.report_progress(segment_count as f32 / segments as f32);
    }

    let objective = dp_prev[target_idx];
    if !objective.is_finite() {
        return Err(CpdError::invalid_input(format!(
            "KnownK exact solution unreachable: requested k={k} under current constraints"
        )));
    }

    let mut split_points_reversed = Vec::with_capacity(k);
    let mut cursor = target_idx;
    for segment_count in (2..=segments).rev() {
        let prev_idx = backpointers[segment_count][cursor];
        if prev_idx == usize::MAX {
            return Err(CpdError::invalid_input(format!(
                "KnownK exact solution unreachable: backtracking failed at segment_count={segment_count}, endpoint={}",
                endpoints[cursor]
            )));
        }
        let split = endpoints[prev_idx];
        if split == 0 || split >= x.n {
            return Err(CpdError::invalid_input(format!(
                "KnownK exact solution unreachable: invalid split during backtracking at split={split}"
            )));
        }
        split_points_reversed.push(split);
        cursor = prev_idx;
    }
    split_points_reversed.reverse();

    let mut breakpoints = split_points_reversed;
    breakpoints.push(x.n);

    Ok(DynpKernelResult {
        objective,
        change_count: k,
        breakpoints,
    })
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
                return Err(CpdError::not_supported(format!(
                    "Dynp currently supports only KnownK stopping; got Penalized({penalty:?})"
                )));
            }
            Stopping::PenaltyPath(path) => {
                return Err(CpdError::not_supported(format!(
                    "Dynp currently supports only KnownK stopping; got PenaltyPath of length {}",
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
    fn penalized_stopping_is_not_supported() {
        let detector = Dynp::new(
            CostL2Mean::default(),
            DynpConfig {
                stopping: Stopping::Penalized(Penalty::Manual(1.0)),
                cancel_check_every: 16,
            },
        )
        .expect("config should be valid");

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
        let err = detector
            .detect(&view, &ctx)
            .expect_err("penalized stopping should be rejected");
        assert!(err.to_string().contains("supports only KnownK"));
    }
}
