// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{
    BudgetStatus, CpdError, Diagnostics, ExecutionContext, OfflineChangePointResult,
    OfflineDetector, Penalty, PruningStats, Stopping, TimeSeriesView, ValidatedConstraints,
    check_missing_compatibility, penalty_value, validate_constraints, validate_stopping,
};
use cpd_costs::CostModel;
use std::borrow::Cow;
use std::time::Instant;

const DEFAULT_CANCEL_CHECK_EVERY: usize = 1000;
const KNOWN_K_MAX_DOUBLINGS: usize = 80;
const KNOWN_K_MAX_BISECTION_ITERS: usize = 64;

/// Configuration for [`Pelt`].
#[derive(Clone, Debug, PartialEq)]
pub struct PeltConfig {
    pub stopping: Stopping,
    pub params_per_segment: usize,
    pub cancel_check_every: usize,
}

impl Default for PeltConfig {
    fn default() -> Self {
        Self {
            stopping: Stopping::Penalized(Penalty::BIC),
            params_per_segment: 2,
            cancel_check_every: DEFAULT_CANCEL_CHECK_EVERY,
        }
    }
}

impl PeltConfig {
    fn validate(&self) -> Result<(), CpdError> {
        validate_stopping(&self.stopping)?;

        if self.params_per_segment == 0 {
            return Err(CpdError::invalid_input(
                "PeltConfig.params_per_segment must be >= 1; got 0",
            ));
        }

        Ok(())
    }

    fn normalized_cancel_check_every(&self) -> usize {
        self.cancel_check_every.max(1)
    }
}

/// Pruned Exact Linear Time offline detector.
#[derive(Debug)]
pub struct Pelt<C: CostModel> {
    cost_model: C,
    config: PeltConfig,
}

impl<C: CostModel> Pelt<C> {
    pub fn new(cost_model: C, config: PeltConfig) -> Result<Self, CpdError> {
        config.validate()?;
        Ok(Self { cost_model, config })
    }

    pub fn cost_model(&self) -> &C {
        &self.cost_model
    }

    pub fn config(&self) -> &PeltConfig {
        &self.config
    }
}

#[derive(Clone, Debug)]
struct KernelResult {
    breakpoints: Vec<usize>,
    change_count: usize,
    objective: f64,
    cost_evals: usize,
    candidates_considered: usize,
    candidates_pruned: usize,
}

#[derive(Default, Clone, Copy, Debug)]
struct RuntimeStats {
    cost_evals: usize,
    candidates_considered: usize,
    candidates_pruned: usize,
    soft_budget_exceeded: bool,
}

#[derive(Clone, Debug)]
struct KnownKSearchResult {
    kernel: KernelResult,
    selected_beta: f64,
    iterations: usize,
}

fn checked_counter_increment(counter: &mut usize, name: &str) -> Result<(), CpdError> {
    *counter = counter
        .checked_add(1)
        .ok_or_else(|| CpdError::resource_limit(format!("{name} counter overflow")))?;
    Ok(())
}

fn resolve_penalty_beta(
    penalty: &Penalty,
    n: usize,
    d: usize,
    params_per_segment: usize,
) -> Result<f64, CpdError> {
    let beta = penalty_value(penalty, n, d, params_per_segment)?;
    if !beta.is_finite() || beta <= 0.0 {
        return Err(CpdError::invalid_input(format!(
            "resolved penalty must be finite and > 0.0; got beta={beta}"
        )));
    }
    Ok(beta)
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

fn build_targets(validated: &ValidatedConstraints, n: usize) -> Vec<usize> {
    let mut targets = validated.effective_candidates.clone();
    if targets.last().copied() != Some(n) {
        targets.push(n);
    }
    targets
}

fn reconstruct_breakpoints(n: usize, last_cp: &[usize]) -> Result<(Vec<usize>, usize), CpdError> {
    let mut reverse = vec![n];
    let mut cursor = n;
    let mut hops = 0usize;

    while cursor > 0 {
        hops = hops
            .checked_add(1)
            .ok_or_else(|| CpdError::resource_limit("breakpoint backtrack hop overflow"))?;
        if hops > n + 1 {
            return Err(CpdError::invalid_input(
                "invalid DP backtrack state: cycle detected",
            ));
        }

        let tau = last_cp[cursor];
        if tau == usize::MAX {
            return Err(CpdError::invalid_input(format!(
                "invalid DP backtrack state: missing predecessor at t={cursor}"
            )));
        }
        if tau >= cursor {
            return Err(CpdError::invalid_input(format!(
                "invalid DP backtrack state: predecessor tau={tau} is not < t={cursor}"
            )));
        }
        if tau == 0 {
            break;
        }
        reverse.push(tau);
        cursor = tau;
    }

    reverse.reverse();
    let change_count = reverse.len().saturating_sub(1);
    Ok((reverse, change_count))
}

fn run_pelt_penalized<C: CostModel>(
    model: &C,
    cache: &C::Cache,
    x: &TimeSeriesView<'_>,
    validated: &ValidatedConstraints,
    beta: f64,
    prune_candidates: bool,
    cancel_check_every: usize,
    ctx: &ExecutionContext<'_>,
    started_at: Instant,
    runtime: &mut RuntimeStats,
) -> Result<KernelResult, CpdError> {
    if !beta.is_finite() || beta <= 0.0 {
        return Err(CpdError::invalid_input(format!(
            "run_pelt_penalized requires finite beta > 0; got {beta}"
        )));
    }

    let targets = build_targets(validated, x.n);
    let total_targets = targets.len().max(1);
    let min_segment_len = validated.min_segment_len;

    let mut f = vec![f64::INFINITY; x.n + 1];
    let mut last_cp = vec![usize::MAX; x.n + 1];
    let mut changes = vec![usize::MAX; x.n + 1];

    f[0] = -beta;
    last_cp[0] = 0;
    changes[0] = 0;

    let mut candidate_set = vec![0usize];
    let mut run_cost_evals = 0usize;
    let mut run_considered = 0usize;
    let mut run_pruned = 0usize;

    for (target_idx, &t) in targets.iter().enumerate() {
        if target_idx % cancel_check_every == 0 {
            ctx.check_cancelled_every(target_idx, 1)?;
            match ctx.check_time_budget(started_at)? {
                BudgetStatus::WithinBudget => {}
                BudgetStatus::ExceededSoftDegrade => {
                    runtime.soft_budget_exceeded = true;
                }
            }
        }

        let mut scored = vec![None; candidate_set.len()];
        let mut best_cost = f64::INFINITY;
        let mut best_tau = usize::MAX;
        let mut best_changes = usize::MAX;

        for (idx, &tau) in candidate_set.iter().enumerate() {
            if t <= tau || t - tau < min_segment_len {
                continue;
            }
            if !f[tau].is_finite() {
                continue;
            }

            let proposed_changes = if tau == 0 {
                changes[tau]
            } else {
                changes[tau].saturating_add(1)
            };

            if let Some(max_change_points) = validated.max_change_points
                && proposed_changes > max_change_points
            {
                continue;
            }

            let segment_cost = evaluate_segment_cost(model, cache, tau, t, ctx, runtime)?;
            checked_counter_increment(&mut run_cost_evals, "run_cost_evals")?;
            checked_counter_increment(&mut runtime.candidates_considered, "candidates_considered")?;
            checked_counter_increment(&mut run_considered, "run_candidates_considered")?;

            let score_no_penalty = f[tau] + segment_cost;
            if !score_no_penalty.is_finite() {
                return Err(CpdError::numerical_issue(format!(
                    "non-finite score without penalty at t={t}, tau={tau}: F(tau)={}, segment_cost={segment_cost}, score_no_penalty={score_no_penalty}",
                    f[tau]
                )));
            }

            let candidate = score_no_penalty + beta;
            if !candidate.is_finite() {
                return Err(CpdError::numerical_issue(format!(
                    "non-finite objective at t={t}, tau={tau}: F(tau)={}, segment_cost={segment_cost}, beta={beta}, candidate={candidate}",
                    f[tau]
                )));
            }

            scored[idx] = Some((score_no_penalty, proposed_changes));

            if candidate < best_cost || (candidate == best_cost && tau < best_tau) {
                best_cost = candidate;
                best_tau = tau;
                best_changes = proposed_changes;
            }
        }

        if best_tau == usize::MAX {
            return Err(CpdError::invalid_input(format!(
                "no feasible segmentation under constraints at t={t}; check min_segment_len, candidate_splits/jump, and max_change_points"
            )));
        }

        f[t] = best_cost;
        last_cp[t] = best_tau;
        changes[t] = best_changes;

        let mut next_candidate_set = Vec::with_capacity(candidate_set.len() + 1);
        if prune_candidates {
            for (idx, &tau) in candidate_set.iter().enumerate() {
                if let Some((score_no_penalty, _)) = scored[idx] {
                    if score_no_penalty < best_cost {
                        next_candidate_set.push(tau);
                    } else {
                        checked_counter_increment(
                            &mut runtime.candidates_pruned,
                            "candidates_pruned",
                        )?;
                        checked_counter_increment(&mut run_pruned, "run_candidates_pruned")?;
                    }
                } else {
                    next_candidate_set.push(tau);
                }
            }
        } else {
            next_candidate_set.extend_from_slice(&candidate_set);
        }

        if t < x.n {
            next_candidate_set.push(t);
        }
        candidate_set = next_candidate_set;

        ctx.report_progress((target_idx + 1) as f32 / total_targets as f32);
    }

    if !f[x.n].is_finite() {
        return Err(CpdError::invalid_input(
            "no feasible segmentation reached terminal index n",
        ));
    }

    let (breakpoints, change_count) = reconstruct_breakpoints(x.n, &last_cp)?;
    Ok(KernelResult {
        breakpoints,
        change_count,
        objective: f[x.n],
        cost_evals: run_cost_evals,
        candidates_considered: run_considered,
        candidates_pruned: run_pruned,
    })
}

fn run_known_k_search<C: CostModel>(
    model: &C,
    cache: &C::Cache,
    x: &TimeSeriesView<'_>,
    validated: &ValidatedConstraints,
    k: usize,
    cancel_check_every: usize,
    ctx: &ExecutionContext<'_>,
    started_at: Instant,
    runtime: &mut RuntimeStats,
) -> Result<KnownKSearchResult, CpdError> {
    if let Some(max_change_points) = validated.max_change_points
        && max_change_points < k
    {
        return Err(CpdError::invalid_input(format!(
            "KnownK={k} exceeds constraints.max_change_points={max_change_points}"
        )));
    }

    // KnownK search needs unconstrained cardinality in the inner beta evaluations.
    // The outer search enforces exact-k, and user-facing max_change_points validation
    // is still honored via the guard above.
    let mut search_constraints = validated.clone();
    search_constraints.max_change_points = None;

    let mut iterations = 0usize;
    let mut low_beta = f64::EPSILON;
    let low_kernel = run_pelt_penalized(
        model,
        cache,
        x,
        &search_constraints,
        low_beta,
        false,
        cancel_check_every,
        ctx,
        started_at,
        runtime,
    )?;
    checked_counter_increment(&mut iterations, "known_k_iterations")?;

    if low_kernel.change_count == k {
        return Ok(KnownKSearchResult {
            kernel: low_kernel,
            selected_beta: low_beta,
            iterations,
        });
    }

    if low_kernel.change_count < k {
        return Err(CpdError::invalid_input(format!(
            "KnownK exact solution unreachable: requested k={k}, but smallest tested penalty beta={low_beta} produced only {} changes",
            low_kernel.change_count
        )));
    }

    let mut high_beta = low_beta;
    let mut high_kernel = low_kernel.clone();
    let mut low_kernel = low_kernel;

    for _ in 0..KNOWN_K_MAX_DOUBLINGS {
        if high_kernel.change_count <= k {
            break;
        }
        high_beta *= 2.0;
        if !high_beta.is_finite() {
            return Err(CpdError::invalid_input(format!(
                "KnownK exact solution unreachable: penalty bracketing overflowed while searching for k={k}"
            )));
        }
        high_kernel = run_pelt_penalized(
            model,
            cache,
            x,
            &search_constraints,
            high_beta,
            false,
            cancel_check_every,
            ctx,
            started_at,
            runtime,
        )?;
        checked_counter_increment(&mut iterations, "known_k_iterations")?;

        if high_kernel.change_count == k {
            return Ok(KnownKSearchResult {
                kernel: high_kernel,
                selected_beta: high_beta,
                iterations,
            });
        }
    }

    if high_kernel.change_count > k {
        return Err(CpdError::invalid_input(format!(
            "KnownK exact solution unreachable: failed to bracket requested k={k} within {KNOWN_K_MAX_DOUBLINGS} penalty doublings"
        )));
    }

    for _ in 0..KNOWN_K_MAX_BISECTION_ITERS {
        let mid_beta = low_beta + (high_beta - low_beta) * 0.5;
        if !(mid_beta > low_beta && mid_beta < high_beta) {
            break;
        }

        let mid_kernel = run_pelt_penalized(
            model,
            cache,
            x,
            &search_constraints,
            mid_beta,
            false,
            cancel_check_every,
            ctx,
            started_at,
            runtime,
        )?;
        checked_counter_increment(&mut iterations, "known_k_iterations")?;

        if mid_kernel.change_count == k {
            return Ok(KnownKSearchResult {
                kernel: mid_kernel,
                selected_beta: mid_beta,
                iterations,
            });
        }

        if mid_kernel.change_count > k {
            low_beta = mid_beta;
            low_kernel = mid_kernel;
        } else {
            high_beta = mid_beta;
            high_kernel = mid_kernel;
        }
    }

    Err(CpdError::invalid_input(format!(
        "KnownK exact solution unreachable: requested k={k}, bracketed counts were low_beta={low_beta} -> {} changes and high_beta={high_beta} -> {} changes",
        low_kernel.change_count, high_kernel.change_count
    )))
}

impl<C: CostModel> OfflineDetector for Pelt<C> {
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
                "constraints.max_depth={max_depth} is ignored by PELT"
            ));
        }

        let kernel = match &self.config.stopping {
            Stopping::Penalized(penalty) => {
                let beta = resolve_penalty_beta(penalty, x.n, x.d, self.config.params_per_segment)?;
                notes.push(format!("stopping=Penalized({penalty:?}), beta={beta}"));
                run_pelt_penalized(
                    &self.cost_model,
                    &cache,
                    x,
                    &validated,
                    beta,
                    true,
                    cancel_check_every,
                    ctx,
                    started_at,
                    &mut runtime,
                )?
            }
            Stopping::KnownK(k) => {
                let known_k = run_known_k_search(
                    &self.cost_model,
                    &cache,
                    x,
                    &validated,
                    *k,
                    cancel_check_every,
                    ctx,
                    started_at,
                    &mut runtime,
                )?;
                notes.push(format!(
                    "stopping=KnownK({k}), selected_beta={}, search_iterations={}",
                    known_k.selected_beta, known_k.iterations
                ));
                known_k.kernel
            }
            Stopping::PenaltyPath(path) => {
                return Err(CpdError::not_supported(format!(
                    "PELT penalty sweep is deferred for this issue; got PenaltyPath of length {}",
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

        let runtime_ms = match u64::try_from(started_at.elapsed().as_millis()) {
            Ok(ms) => ms,
            Err(_) => u64::MAX,
        };

        ctx.record_scalar("offline.pelt.cost_evals", runtime.cost_evals as f64);
        ctx.record_scalar(
            "offline.pelt.candidates_considered",
            runtime.candidates_considered as f64,
        );
        ctx.record_scalar(
            "offline.pelt.candidates_pruned",
            runtime.candidates_pruned as f64,
        );
        ctx.record_scalar("offline.pelt.runtime_ms", runtime_ms as f64);
        ctx.report_progress(1.0);

        notes.push(format!(
            "final_objective={}, change_count={}",
            kernel.objective, kernel.change_count
        ));
        notes.push(format!("run_cost_evals={}", kernel.cost_evals));
        notes.push(format!(
            "run_candidates_considered={}, run_candidates_pruned={}",
            kernel.candidates_considered, kernel.candidates_pruned
        ));

        let diagnostics = Diagnostics {
            n: x.n,
            d: x.d,
            runtime_ms: Some(runtime_ms),
            notes,
            warnings,
            algorithm: Cow::Borrowed("pelt"),
            cost_model: Cow::Borrowed(self.cost_model.name()),
            repro_mode: ctx.repro_mode,
            pruning_stats: Some(PruningStats {
                candidates_considered: runtime.candidates_considered,
                candidates_pruned: runtime.candidates_pruned,
            }),
            ..Diagnostics::default()
        };

        OfflineChangePointResult::new(x.n, kernel.breakpoints, diagnostics)
    }
}

#[cfg(test)]
mod tests {
    use super::{Pelt, PeltConfig};
    use cpd_core::{
        BudgetMode, Constraints, DTypeView, ExecutionContext, MemoryLayout, MissingPolicy,
        OfflineDetector, Penalty, ProgressSink, ReproMode, Stopping, TimeIndex, TimeSeriesView,
    };
    use cpd_costs::{CostL2Mean, CostNormalMeanVar};
    use std::thread;
    use std::time::Duration;

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

    fn assert_strictly_increasing(values: &[usize]) {
        for window in values.windows(2) {
            assert!(window[0] < window[1], "not strictly increasing: {values:?}");
        }
    }

    #[test]
    fn config_defaults_and_validation() {
        let default_cfg = PeltConfig::default();
        assert_eq!(default_cfg.stopping, Stopping::Penalized(Penalty::BIC));
        assert_eq!(default_cfg.params_per_segment, 2);
        assert_eq!(default_cfg.cancel_check_every, 1000);

        let ok = Pelt::new(CostL2Mean::default(), default_cfg.clone())
            .expect("default config should be valid");
        assert_eq!(ok.config(), &default_cfg);

        let err = Pelt::new(
            CostL2Mean::default(),
            PeltConfig {
                params_per_segment: 0,
                ..default_cfg
            },
        )
        .expect_err("params_per_segment=0 must fail");
        assert!(err.to_string().contains("params_per_segment"));
    }

    #[test]
    fn cancel_check_every_zero_is_normalized() {
        let detector = Pelt::new(
            CostL2Mean::default(),
            PeltConfig {
                stopping: Stopping::Penalized(Penalty::Manual(1.0)),
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
    fn known_small_example_one_change_l2() {
        let detector = Pelt::new(
            CostL2Mean::default(),
            PeltConfig {
                stopping: Stopping::Penalized(Penalty::Manual(1.0)),
                params_per_segment: 2,
                cancel_check_every: 8,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let constraints = constraints_with_min_segment_len(2);
        let ctx = ExecutionContext::new(&constraints).with_repro_mode(ReproMode::Balanced);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");
        assert_eq!(result.breakpoints, vec![5, 10]);
        assert_eq!(result.change_points, vec![5]);
    }

    #[test]
    fn known_small_example_two_changes_l2() {
        let detector = Pelt::new(
            CostL2Mean::default(),
            PeltConfig {
                stopping: Stopping::Penalized(Penalty::Manual(1.0)),
                params_per_segment: 2,
                cancel_check_every: 4,
            },
        )
        .expect("config should be valid");

        let values = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
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
        assert_eq!(result.breakpoints, vec![5, 10, 15]);
    }

    #[test]
    fn normal_cost_detects_variance_change() {
        let detector = Pelt::new(
            CostNormalMeanVar::default(),
            PeltConfig {
                stopping: Stopping::Penalized(Penalty::Manual(1.0)),
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
    fn tie_breaking_prefers_leftmost_split() {
        let detector = Pelt::new(
            CostL2Mean::default(),
            PeltConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.1)),
                params_per_segment: 2,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
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
            candidate_splits: Some(vec![2, 4]),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");
        assert_eq!(result.breakpoints, vec![2, 6]);
    }

    #[test]
    fn constraints_min_segment_len_and_jump_are_enforced() {
        let detector = Pelt::new(
            CostL2Mean::default(),
            PeltConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.5)),
                params_per_segment: 2,
                cancel_check_every: 3,
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
        assert_eq!(result.breakpoints.last().copied(), Some(n));
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
    fn explicit_candidate_splits_and_max_change_points_are_enforced() {
        let detector = Pelt::new(
            CostL2Mean::default(),
            PeltConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.1)),
                params_per_segment: 2,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let values = vec![
            0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0,
        ];
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
            candidate_splits: Some(vec![4, 8]),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");

        assert!(result.change_points.len() <= 1);
        for cp in result.change_points {
            assert!(cp == 4 || cp == 8);
        }
    }

    #[test]
    fn constant_series_has_no_changes_with_reasonable_penalty() {
        let detector = Pelt::new(
            CostL2Mean::default(),
            PeltConfig {
                stopping: Stopping::Penalized(Penalty::Manual(1.0)),
                params_per_segment: 2,
                cancel_check_every: 16,
            },
        )
        .expect("config should be valid");

        let values = vec![3.0; 64];
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
        assert_eq!(result.breakpoints, vec![64]);
        assert_eq!(result.change_points, vec![]);
    }

    #[test]
    fn cancellation_mid_run_returns_cancelled() {
        let detector = Pelt::new(
            CostL2Mean::default(),
            PeltConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.1)),
                params_per_segment: 2,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let n = 5_000;
        let values: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let view = make_f64_view(
            &values,
            n,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let constraints = constraints_with_min_segment_len(1);
        let cancel = cpd_core::CancelToken::new();
        cancel.cancel();
        let ctx = ExecutionContext::new(&constraints).with_cancel(&cancel);
        let err = detector
            .detect(&view, &ctx)
            .expect_err("cancelled token must stop detect");
        assert_eq!(err.to_string(), "cancelled");
    }

    #[test]
    fn cost_eval_budget_exceeded_hard_fail() {
        let detector = Pelt::new(
            CostL2Mean::default(),
            PeltConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.1)),
                params_per_segment: 2,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 1,
            max_cost_evals: Some(1),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints).with_budget_mode(BudgetMode::HardFail);
        let err = detector
            .detect(&view, &ctx)
            .expect_err("hard budget should fail");
        assert!(err.to_string().contains("max_cost_evals"));
    }

    struct SlowProgressSink;

    impl ProgressSink for SlowProgressSink {
        fn on_progress(&self, _fraction: f32) {
            thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn time_budget_exceeded_hard_fail() {
        let detector = Pelt::new(
            CostL2Mean::default(),
            PeltConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.1)),
                params_per_segment: 2,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 1,
            time_budget_ms: Some(1),
            ..Constraints::default()
        };
        let slow_progress = SlowProgressSink;
        let ctx = ExecutionContext::new(&constraints)
            .with_budget_mode(BudgetMode::HardFail)
            .with_progress_sink(&slow_progress);
        let err = detector
            .detect(&view, &ctx)
            .expect_err("hard time budget should fail");
        assert!(err.to_string().contains("time_budget_ms"));
    }

    #[test]
    fn known_k_exact_success() {
        let detector = Pelt::new(
            CostL2Mean::default(),
            PeltConfig {
                stopping: Stopping::KnownK(2),
                params_per_segment: 2,
                cancel_check_every: 8,
            },
        )
        .expect("config should be valid");

        let values = vec![
            0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, -5.0, -5.0, -5.0, -5.0,
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
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let result = detector
            .detect(&view, &ctx)
            .expect("known-k should succeed");
        assert_eq!(result.change_points.len(), 2);
        assert_eq!(result.breakpoints, vec![4, 8, 12]);
    }

    #[test]
    fn known_k_normal_piecewise_constant_exact_k_is_feasible() {
        let detector = Pelt::new(
            CostNormalMeanVar::default(),
            PeltConfig {
                stopping: Stopping::KnownK(2),
                params_per_segment: 3,
                cancel_check_every: 8,
            },
        )
        .expect("config should be valid");

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
        let result = detector
            .detect(&view, &ctx)
            .expect("known-k normal should be feasible");

        assert_eq!(result.change_points.len(), 2);
        assert_eq!(result.breakpoints, vec![4, 8, 12]);
        assert_strictly_increasing(&result.breakpoints);
    }

    #[test]
    fn known_k_normal_with_max_change_points_equal_k_does_not_fail_mid_dp() {
        let detector = Pelt::new(
            CostNormalMeanVar::default(),
            PeltConfig {
                stopping: Stopping::KnownK(2),
                params_per_segment: 3,
                cancel_check_every: 4,
            },
        )
        .expect("config should be valid");

        let values = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, -3.0, -3.0, -3.0, -3.0,
            -3.0, -3.0,
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
        let result = detector
            .detect(&view, &ctx)
            .expect("known-k normal should not fail mid-dp under max_change_points=k");

        assert_eq!(result.change_points.len(), 2);
        assert_eq!(result.breakpoints, vec![6, 12, 18]);
        assert_eq!(result.breakpoints.last().copied(), Some(values.len()));
        assert_strictly_increasing(&result.breakpoints);
    }

    #[test]
    fn known_k_normal_replication_keeps_scaled_breakpoints() {
        let detector = Pelt::new(
            CostNormalMeanVar::default(),
            PeltConfig {
                stopping: Stopping::KnownK(2),
                params_per_segment: 3,
                cancel_check_every: 8,
            },
        )
        .expect("config should be valid");

        let base = vec![
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, -2.0, -2.0, -2.0,
            -2.0, -2.0, -2.0,
        ];
        let replicate_factor = 3usize;
        let mut replicated = Vec::with_capacity(base.len() * replicate_factor);
        for &value in &base {
            replicated.extend(std::iter::repeat_n(value, replicate_factor));
        }

        let constraints = Constraints {
            min_segment_len: 2,
            max_change_points: Some(2),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);

        let base_view = make_f64_view(
            &base,
            base.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let replicated_view = make_f64_view(
            &replicated,
            replicated.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let base_result = detector
            .detect(&base_view, &ctx)
            .expect("base known-k normal should succeed");
        let replicated_result = detector
            .detect(&replicated_view, &ctx)
            .expect("replicated known-k normal should succeed");

        let scaled_change_points: Vec<usize> = base_result
            .change_points
            .iter()
            .map(|cp| cp * replicate_factor)
            .collect();

        assert_eq!(base_result.change_points.len(), 2);
        assert_eq!(replicated_result.change_points.len(), 2);
        assert_eq!(replicated_result.change_points, scaled_change_points);
        assert_eq!(
            replicated_result.breakpoints.last().copied(),
            Some(replicated.len())
        );
    }

    #[test]
    fn known_k_exact_unreachable_is_clear_error() {
        let detector = Pelt::new(
            CostL2Mean::default(),
            PeltConfig {
                stopping: Stopping::KnownK(2),
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
    fn penalty_path_returns_not_supported() {
        let detector = Pelt::new(
            CostL2Mean::default(),
            PeltConfig {
                stopping: Stopping::PenaltyPath(vec![Penalty::Manual(1.0)]),
                params_per_segment: 2,
                cancel_check_every: 16,
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

    #[test]
    fn diagnostics_include_pruning_and_soft_budget_and_max_depth_warning() {
        let detector = Pelt::new(
            CostL2Mean::default(),
            PeltConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.1)),
                params_per_segment: 2,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 1.0, 0.0, 1.0, 5.0, 6.0, 5.0, 6.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 1,
            max_depth: Some(4),
            max_cost_evals: Some(2),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints).with_budget_mode(BudgetMode::SoftDegrade);
        let result = detector
            .detect(&view, &ctx)
            .expect("soft degrade should continue");
        let diagnostics = result.diagnostics;
        assert_eq!(diagnostics.algorithm, "pelt");
        assert_eq!(diagnostics.cost_model, "l2_mean");
        assert!(diagnostics.pruning_stats.is_some());
        assert!(
            diagnostics
                .warnings
                .iter()
                .any(|w| w.contains("max_depth") && w.contains("ignored"))
        );
        assert!(
            diagnostics
                .warnings
                .iter()
                .any(|w| w.contains("SoftDegrade"))
        );
    }

    #[test]
    fn large_n_regression_smoke() {
        let detector = Pelt::new(
            CostL2Mean::default(),
            PeltConfig {
                stopping: Stopping::Penalized(Penalty::BIC),
                params_per_segment: 2,
                cancel_check_every: 1024,
            },
        )
        .expect("config should be valid");

        let n = 100_000;
        let mut values = vec![0.0; n];
        for v in values.iter_mut().skip(n / 2) {
            *v = 2.0;
        }
        let view = make_f64_view(
            &values,
            n,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 50,
            jump: 50,
            max_change_points: Some(8),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let result = detector
            .detect(&view, &ctx)
            .expect("large-n smoke should pass");

        assert_eq!(result.breakpoints.last().copied(), Some(n));
        assert_strictly_increasing(&result.breakpoints);
        assert!(result.change_points.len() <= 8);
    }
}
