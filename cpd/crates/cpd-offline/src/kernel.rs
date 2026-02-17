// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{
    BudgetStatus, CpdError, Diagnostics, ExecutionContext, MemoryLayout, MissingPolicy,
    OfflineChangePointResult, OfflineDetector, Penalty, PruningStats, Stopping, TimeSeriesView,
    ValidatedConstraints, check_missing_compatibility, checked_effective_params,
    compute_missing_run_stats, penalty_value_from_effective_params, validate_constraints,
    validate_stopping,
};
use std::borrow::Cow;
use std::time::Instant;

const DEFAULT_CANCEL_CHECK_EVERY: usize = 1000;
const DEFAULT_PARAMS_PER_SEGMENT: usize = 1;
const LARGE_GRAM_WARNING_BYTES: usize = 256 * 1024 * 1024;

/// Kernel choices for [`KernelCpd`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub enum KernelSpec {
    /// Radial basis function kernel.
    ///
    /// If `gamma` is `None`, a data-driven heuristic is resolved at runtime.
    Rbf { gamma: Option<f64> },
    /// Linear dot-product kernel.
    Linear,
}

impl Default for KernelSpec {
    fn default() -> Self {
        Self::Rbf { gamma: None }
    }
}

/// Configuration for [`KernelCpd`].
///
/// This detector is intentionally expensive and opt-in. It precomputes an
/// `n x n` Gram matrix, so memory grows as `O(n^2)`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct KernelCpdConfig {
    pub stopping: Stopping,
    pub kernel: KernelSpec,
    pub params_per_segment: usize,
    pub cancel_check_every: usize,
}

impl Default for KernelCpdConfig {
    fn default() -> Self {
        Self {
            stopping: Stopping::Penalized(Penalty::BIC),
            kernel: KernelSpec::default(),
            params_per_segment: DEFAULT_PARAMS_PER_SEGMENT,
            cancel_check_every: DEFAULT_CANCEL_CHECK_EVERY,
        }
    }
}

impl KernelCpdConfig {
    fn validate(&self) -> Result<(), CpdError> {
        validate_stopping(&self.stopping)?;
        if self.params_per_segment == 0 {
            return Err(CpdError::invalid_input(
                "KernelCpdConfig.params_per_segment must be >= 1; got 0",
            ));
        }
        if let KernelSpec::Rbf { gamma: Some(gamma) } = self.kernel
            && (!gamma.is_finite() || gamma <= 0.0)
        {
            return Err(CpdError::invalid_input(format!(
                "KernelSpec::Rbf gamma must be finite and > 0; got {gamma}"
            )));
        }
        Ok(())
    }

    fn normalized_cancel_check_every(&self) -> usize {
        self.cancel_check_every.max(1)
    }
}

#[derive(Clone, Debug)]
enum ResolvedKernel {
    Rbf { gamma: f64 },
    Linear,
}

impl ResolvedKernel {
    fn label(&self) -> &'static str {
        match self {
            Self::Rbf { .. } => "rbf",
            Self::Linear => "linear",
        }
    }
}

/// Exact kernel CPD detector with `O(n^2)` kernel precomputation.
///
/// The objective is a kernelized within-segment dispersion. Breakpoints are
/// selected by dynamic programming under the configured stopping policy.
#[derive(Debug)]
pub struct KernelCpd {
    config: KernelCpdConfig,
}

impl KernelCpd {
    pub fn new(config: KernelCpdConfig) -> Result<Self, CpdError> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn config(&self) -> &KernelCpdConfig {
        &self.config
    }
}

#[derive(Default, Clone, Copy, Debug)]
struct RuntimeStats {
    gram_evals: usize,
    segment_cost_evals: usize,
    candidate_evals: usize,
    soft_budget_exceeded: bool,
}

#[derive(Clone, Debug)]
struct SegmentationResult {
    breakpoints: Vec<usize>,
    objective: f64,
    change_count: usize,
}

#[derive(Clone, Debug)]
struct ResolvedPenalty {
    penalty: Penalty,
    beta: f64,
    params_per_segment: usize,
}

#[inline]
fn value_at(x: &TimeSeriesView<'_>, index: usize) -> f64 {
    match x.values {
        cpd_core::DTypeView::F32(values) => f64::from(values[index]),
        cpd_core::DTypeView::F64(values) => values[index],
    }
}

fn source_index(
    layout: MemoryLayout,
    n: usize,
    d: usize,
    row: usize,
    col: usize,
) -> Result<usize, CpdError> {
    match layout {
        MemoryLayout::CContiguous => row
            .checked_mul(d)
            .and_then(|base| base.checked_add(col))
            .ok_or_else(|| CpdError::resource_limit("index overflow in C-contiguous layout")),
        MemoryLayout::FContiguous => col
            .checked_mul(n)
            .and_then(|base| base.checked_add(row))
            .ok_or_else(|| CpdError::resource_limit("index overflow in F-contiguous layout")),
        MemoryLayout::Strided {
            row_stride,
            col_stride,
        } => {
            let row_isize = isize::try_from(row).map_err(|_| {
                CpdError::invalid_input(format!(
                    "row index {row} does not fit into isize for strided access"
                ))
            })?;
            let col_isize = isize::try_from(col).map_err(|_| {
                CpdError::invalid_input(format!(
                    "column index {col} does not fit into isize for strided access"
                ))
            })?;
            let idx = row_isize
                .checked_mul(row_stride)
                .and_then(|left| {
                    col_isize
                        .checked_mul(col_stride)
                        .and_then(|right| left.checked_add(right))
                })
                .ok_or_else(|| {
                    CpdError::resource_limit(format!(
                        "strided index overflow at row={row}, col={col}, row_stride={row_stride}, col_stride={col_stride}"
                    ))
                })?;
            usize::try_from(idx).map_err(|_| {
                CpdError::invalid_input(format!(
                    "strided index is negative at row={row}, col={col}: idx={idx}"
                ))
            })
        }
    }
}

fn materialize_rows(x: &TimeSeriesView<'_>) -> Result<Vec<Vec<f64>>, CpdError> {
    if matches!(x.missing, MissingPolicy::Ignore) {
        return Err(CpdError::invalid_input(
            "KernelCpd does not support MissingPolicy::Ignore; use Error/ImputeZero/ImputeLast",
        ));
    }

    let mut rows = vec![vec![0.0; x.d]; x.n];
    let mut carry = vec![0.0; x.d];
    let mut carry_ready = vec![false; x.d];
    let total_len = match x.values {
        cpd_core::DTypeView::F32(values) => values.len(),
        cpd_core::DTypeView::F64(values) => values.len(),
    };

    for (row_idx, row) in rows.iter_mut().enumerate() {
        for dim in 0..x.d {
            let idx = source_index(x.layout, x.n, x.d, row_idx, dim)?;
            if idx >= total_len {
                return Err(CpdError::invalid_input(format!(
                    "source index out of bounds at row={row_idx}, dim={dim}: idx={idx}, len={total_len}"
                )));
            }

            let mut value = value_at(x, idx);
            let masked_missing = x.missing_mask.is_some_and(|mask| mask[idx] == 1);
            let missing = masked_missing || value.is_nan();

            if missing {
                value = match x.missing {
                    MissingPolicy::Error => {
                        return Err(CpdError::invalid_input(format!(
                            "missing value encountered at row={row_idx}, dim={dim} with MissingPolicy::Error"
                        )));
                    }
                    MissingPolicy::ImputeZero => 0.0,
                    MissingPolicy::ImputeLast => {
                        if carry_ready[dim] {
                            carry[dim]
                        } else {
                            0.0
                        }
                    }
                    MissingPolicy::Ignore => unreachable!("handled above"),
                };
            }

            row[dim] = value;
            carry[dim] = value;
            carry_ready[dim] = true;
        }
    }

    Ok(rows)
}

fn resolve_kernel(
    spec: &KernelSpec,
    rows: &[Vec<f64>],
) -> Result<(ResolvedKernel, Vec<String>), CpdError> {
    let mut notes = vec![];
    match spec {
        KernelSpec::Linear => Ok((ResolvedKernel::Linear, notes)),
        KernelSpec::Rbf { gamma } => {
            if let Some(gamma) = gamma {
                if !gamma.is_finite() || *gamma <= 0.0 {
                    return Err(CpdError::invalid_input(format!(
                        "KernelSpec::Rbf gamma must be finite and > 0; got {gamma}"
                    )));
                }
                return Ok((ResolvedKernel::Rbf { gamma: *gamma }, notes));
            }

            let mut sum_sq = 0.0;
            let mut count = 0usize;
            for left in 0..rows.len() {
                for right in left + 1..rows.len() {
                    let mut dist_sq = 0.0;
                    for dim in 0..rows[left].len() {
                        let delta = rows[left][dim] - rows[right][dim];
                        dist_sq += delta * delta;
                    }
                    if dist_sq.is_finite() && dist_sq > 0.0 {
                        sum_sq += dist_sq;
                        count = count.saturating_add(1);
                    }
                }
            }

            let fallback = (rows.first().map_or(1, Vec::len) as f64).max(1.0);
            let gamma = if count > 0 {
                let avg_sq = sum_sq / count as f64;
                if avg_sq > 0.0 {
                    1.0 / (2.0 * avg_sq)
                } else {
                    1.0 / (2.0 * fallback)
                }
            } else {
                1.0 / (2.0 * fallback)
            };
            notes.push(format!("kernel.rbf.gamma_auto={gamma}"));
            Ok((ResolvedKernel::Rbf { gamma }, notes))
        }
    }
}

fn kernel_value(kernel: &ResolvedKernel, left: &[f64], right: &[f64]) -> f64 {
    match kernel {
        ResolvedKernel::Linear => left
            .iter()
            .zip(right.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>(),
        ResolvedKernel::Rbf { gamma } => {
            let mut dist_sq = 0.0;
            for (a, b) in left.iter().zip(right.iter()) {
                let delta = *a - *b;
                dist_sq += delta * delta;
            }
            (-gamma * dist_sq).exp()
        }
    }
}

fn poll_runtime(
    iteration: usize,
    cancel_check_every: usize,
    started_at: Instant,
    ctx: &ExecutionContext<'_>,
    runtime: &mut RuntimeStats,
) -> Result<(), CpdError> {
    if !iteration.is_multiple_of(cancel_check_every) {
        return Ok(());
    }

    ctx.check_cancelled_every(iteration, 1)?;
    match ctx.check_time_budget(started_at)? {
        BudgetStatus::WithinBudget => {}
        BudgetStatus::ExceededSoftDegrade => {
            runtime.soft_budget_exceeded = true;
        }
    }
    Ok(())
}

fn compute_gram(
    rows: &[Vec<f64>],
    kernel: &ResolvedKernel,
    cancel_check_every: usize,
    started_at: Instant,
    ctx: &ExecutionContext<'_>,
    runtime: &mut RuntimeStats,
) -> Result<Vec<f64>, CpdError> {
    let n = rows.len();
    let mut gram = vec![0.0; n * n];
    let mut iteration = 0usize;

    for left in 0..n {
        for right in left..n {
            iteration = iteration.saturating_add(1);
            poll_runtime(iteration, cancel_check_every, started_at, ctx, runtime)?;

            runtime.gram_evals = runtime.gram_evals.saturating_add(1);
            let value = kernel_value(kernel, rows[left].as_slice(), rows[right].as_slice());
            if !value.is_finite() {
                return Err(CpdError::numerical_issue(format!(
                    "non-finite kernel value at ({left}, {right})"
                )));
            }
            gram[left * n + right] = value;
            gram[right * n + left] = value;
        }
    }

    Ok(gram)
}

fn compute_prefix_sums(gram: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut prefix = vec![0.0; (n + 1) * (n + 1)];
    for row in 0..n {
        for col in 0..n {
            let idx = (row + 1) * (n + 1) + (col + 1);
            prefix[idx] = gram[row * n + col]
                + prefix[row * (n + 1) + (col + 1)]
                + prefix[(row + 1) * (n + 1) + col]
                - prefix[row * (n + 1) + col];
        }
    }

    let mut diag_prefix = vec![0.0; n + 1];
    for i in 0..n {
        diag_prefix[i + 1] = diag_prefix[i] + gram[i * n + i];
    }

    (prefix, diag_prefix)
}

fn block_sum(prefix: &[f64], n: usize, start: usize, end: usize) -> f64 {
    prefix[end * (n + 1) + end] - prefix[start * (n + 1) + end] - prefix[end * (n + 1) + start]
        + prefix[start * (n + 1) + start]
}

fn segment_cost(
    start: usize,
    end: usize,
    n: usize,
    prefix: &[f64],
    diag_prefix: &[f64],
    ctx: &ExecutionContext<'_>,
    runtime: &mut RuntimeStats,
) -> Result<f64, CpdError> {
    if end <= start {
        return Err(CpdError::invalid_input(format!(
            "invalid segment bounds: start={start}, end={end}"
        )));
    }

    runtime.segment_cost_evals = runtime.segment_cost_evals.saturating_add(1);
    match ctx.check_cost_eval_budget(runtime.segment_cost_evals)? {
        BudgetStatus::WithinBudget => {}
        BudgetStatus::ExceededSoftDegrade => {
            runtime.soft_budget_exceeded = true;
        }
    }

    let len = (end - start) as f64;
    let diag_sum = diag_prefix[end] - diag_prefix[start];
    let gram_sum = block_sum(prefix, n, start, end);
    let mut cost = diag_sum - gram_sum / len;
    if cost < 0.0 && cost > -1.0e-9 {
        cost = 0.0;
    }
    if !cost.is_finite() {
        return Err(CpdError::numerical_issue(format!(
            "non-finite segment cost at [{start}, {end})"
        )));
    }
    Ok(cost)
}

fn candidate_positions(validated: &ValidatedConstraints) -> Vec<usize> {
    let mut positions = Vec::with_capacity(validated.effective_candidates.len() + 2);
    positions.push(0);
    positions.extend(validated.effective_candidates.iter().copied());
    positions.push(validated.n);
    positions
}

fn resolve_penalty(
    penalty: &Penalty,
    n: usize,
    d: usize,
    params_per_segment: usize,
) -> Result<ResolvedPenalty, CpdError> {
    let effective_params = checked_effective_params(d, params_per_segment)?;
    let beta = penalty_value_from_effective_params(penalty, n, effective_params)?;
    if !beta.is_finite() || beta <= 0.0 {
        return Err(CpdError::invalid_input(format!(
            "resolved penalty beta must be finite and > 0; got {beta}"
        )));
    }
    Ok(ResolvedPenalty {
        penalty: penalty.clone(),
        beta,
        params_per_segment,
    })
}

fn run_known_k(
    k: usize,
    positions: &[usize],
    validated: &ValidatedConstraints,
    n: usize,
    prefix: &[f64],
    diag_prefix: &[f64],
    cancel_check_every: usize,
    started_at: Instant,
    ctx: &ExecutionContext<'_>,
    runtime: &mut RuntimeStats,
) -> Result<SegmentationResult, CpdError> {
    if k == 0 {
        return Err(CpdError::invalid_input("Stopping::KnownK requires k >= 1"));
    }
    if let Some(max_change_points) = validated.max_change_points
        && k > max_change_points
    {
        return Err(CpdError::invalid_input(format!(
            "KnownK={k} exceeds constraints.max_change_points={max_change_points}"
        )));
    }

    let segments = k.saturating_add(1);
    let p = positions.len();
    let target = p.saturating_sub(1);
    let inf = f64::INFINITY;
    let mut dp = vec![vec![inf; p]; segments + 1];
    let mut back = vec![vec![None; p]; segments + 1];
    dp[0][0] = 0.0;

    let mut iteration = 0usize;
    for seg_count in 1..=segments {
        for idx in 1..p {
            for prev in 0..idx {
                iteration = iteration.saturating_add(1);
                poll_runtime(iteration, cancel_check_every, started_at, ctx, runtime)?;

                if !dp[seg_count - 1][prev].is_finite() {
                    continue;
                }
                let seg_len = positions[idx].saturating_sub(positions[prev]);
                if seg_len < validated.min_segment_len {
                    continue;
                }

                runtime.candidate_evals = runtime.candidate_evals.saturating_add(1);
                let seg_cost = segment_cost(
                    positions[prev],
                    positions[idx],
                    n,
                    prefix,
                    diag_prefix,
                    ctx,
                    runtime,
                )?;
                let candidate = dp[seg_count - 1][prev] + seg_cost;
                if candidate < dp[seg_count][idx] {
                    dp[seg_count][idx] = candidate;
                    back[seg_count][idx] = Some(prev);
                }
            }
        }
    }

    let objective = dp[segments][target];
    if !objective.is_finite() {
        return Err(CpdError::invalid_input(format!(
            "KnownK({k}) is infeasible under current constraints"
        )));
    }

    let mut boundaries = Vec::with_capacity(segments);
    let mut idx = target;
    let mut seg_count = segments;
    while seg_count > 0 {
        boundaries.push(positions[idx]);
        let prev = back[seg_count][idx].ok_or_else(|| {
            CpdError::resource_limit("internal backtrace failure in KernelCpd KnownK")
        })?;
        idx = prev;
        seg_count -= 1;
    }
    boundaries.reverse();

    Ok(SegmentationResult {
        change_count: boundaries.len().saturating_sub(1),
        breakpoints: boundaries,
        objective,
    })
}

fn run_penalized(
    beta: f64,
    positions: &[usize],
    validated: &ValidatedConstraints,
    n: usize,
    prefix: &[f64],
    diag_prefix: &[f64],
    cancel_check_every: usize,
    started_at: Instant,
    ctx: &ExecutionContext<'_>,
    runtime: &mut RuntimeStats,
) -> Result<SegmentationResult, CpdError> {
    let p = positions.len();
    let target = p.saturating_sub(1);
    let inf = f64::INFINITY;
    let mut best = vec![inf; p];
    let mut back = vec![None; p];
    let mut change_counts = vec![usize::MAX; p];
    best[0] = 0.0;
    change_counts[0] = 0;

    let max_change_points = validated.max_change_points.unwrap_or(usize::MAX);
    let mut iteration = 0usize;

    for idx in 1..p {
        for prev in 0..idx {
            iteration = iteration.saturating_add(1);
            poll_runtime(iteration, cancel_check_every, started_at, ctx, runtime)?;

            if !best[prev].is_finite() {
                continue;
            }
            let seg_len = positions[idx].saturating_sub(positions[prev]);
            if seg_len < validated.min_segment_len {
                continue;
            }

            let additional_change = usize::from(prev > 0);
            let next_changes = change_counts[prev].saturating_add(additional_change);
            if next_changes > max_change_points {
                continue;
            }

            runtime.candidate_evals = runtime.candidate_evals.saturating_add(1);
            let seg_cost = segment_cost(
                positions[prev],
                positions[idx],
                n,
                prefix,
                diag_prefix,
                ctx,
                runtime,
            )?;
            let penalty = if prev > 0 { beta } else { 0.0 };
            let candidate = best[prev] + seg_cost + penalty;

            let improve = candidate < best[idx]
                || (candidate == best[idx] && next_changes < change_counts[idx]);
            if improve {
                best[idx] = candidate;
                back[idx] = Some(prev);
                change_counts[idx] = next_changes;
            }
        }
    }

    let objective = best[target];
    if !objective.is_finite() {
        return Err(CpdError::invalid_input(
            "penalized segmentation is infeasible under current constraints",
        ));
    }

    let mut boundaries = Vec::new();
    let mut idx = target;
    loop {
        boundaries.push(positions[idx]);
        if idx == 0 {
            break;
        }
        idx = back[idx].ok_or_else(|| {
            CpdError::resource_limit("internal backtrace failure in KernelCpd penalized run")
        })?;
    }
    boundaries.reverse();

    Ok(SegmentationResult {
        change_count: boundaries.len().saturating_sub(1),
        breakpoints: boundaries,
        objective,
    })
}

impl OfflineDetector for KernelCpd {
    fn detect(
        &self,
        x: &TimeSeriesView<'_>,
        ctx: &ExecutionContext<'_>,
    ) -> Result<OfflineChangePointResult, CpdError> {
        check_missing_compatibility(x.missing, cpd_core::MissingSupport::Reject)?;
        let validated = validate_constraints(ctx.constraints, x.n)?;
        let started_at = Instant::now();
        let cancel_check_every = self.config.normalized_cancel_check_every();
        let mut runtime = RuntimeStats::default();
        let mut notes = vec![
            "complexity=time:O(n^2),memory:O(n^2); feature=kernel".to_string(),
            "kernel_cpd=exact_opt_in".to_string(),
        ];
        let mut warnings = vec![];

        let gram_bytes =
            x.n.checked_mul(x.n)
                .and_then(|cells| cells.checked_mul(std::mem::size_of::<f64>()))
                .ok_or_else(|| CpdError::resource_limit("Gram matrix size overflow"))?;
        notes.push(format!("gram_matrix_bytes={gram_bytes}"));
        if gram_bytes >= LARGE_GRAM_WARNING_BYTES {
            warnings.push(format!(
                "kernel detector allocated a large Gram matrix ({} MiB)",
                gram_bytes / (1024 * 1024)
            ));
        }

        let rows = materialize_rows(x)?;
        let (resolved_kernel, mut kernel_notes) = resolve_kernel(&self.config.kernel, &rows)?;
        notes.append(&mut kernel_notes);
        notes.push(format!("kernel={}", resolved_kernel.label()));

        let gram = compute_gram(
            rows.as_slice(),
            &resolved_kernel,
            cancel_check_every,
            started_at,
            ctx,
            &mut runtime,
        )?;
        let (prefix, diag_prefix) = compute_prefix_sums(gram.as_slice(), x.n);
        let positions = candidate_positions(&validated);

        let selection = match &self.config.stopping {
            Stopping::KnownK(k) => {
                notes.push(format!("stopping=KnownK({k})"));
                run_known_k(
                    *k,
                    positions.as_slice(),
                    &validated,
                    x.n,
                    prefix.as_slice(),
                    diag_prefix.as_slice(),
                    cancel_check_every,
                    started_at,
                    ctx,
                    &mut runtime,
                )?
            }
            Stopping::Penalized(penalty) => {
                let resolved = resolve_penalty(penalty, x.n, x.d, self.config.params_per_segment)?;
                notes.push(format!(
                    "stopping=Penalized({:?}), beta={}, params_per_segment={}",
                    resolved.penalty, resolved.beta, resolved.params_per_segment
                ));
                run_penalized(
                    resolved.beta,
                    positions.as_slice(),
                    &validated,
                    x.n,
                    prefix.as_slice(),
                    diag_prefix.as_slice(),
                    cancel_check_every,
                    started_at,
                    ctx,
                    &mut runtime,
                )?
            }
            Stopping::PenaltyPath(path) => {
                notes.push(format!("stopping=PenaltyPath(len={})", path.len()));
                let mut primary = None;
                for (idx, penalty) in path.iter().enumerate() {
                    let resolved =
                        resolve_penalty(penalty, x.n, x.d, self.config.params_per_segment)?;
                    let run = run_penalized(
                        resolved.beta,
                        positions.as_slice(),
                        &validated,
                        x.n,
                        prefix.as_slice(),
                        diag_prefix.as_slice(),
                        cancel_check_every,
                        started_at,
                        ctx,
                        &mut runtime,
                    )?;
                    notes.push(format!(
                        "penalty_path[{idx}]: penalty={:?}, beta={}, change_count={}, objective={}",
                        resolved.penalty, resolved.beta, run.change_count, run.objective
                    ));
                    if primary.is_none() {
                        primary = Some(run);
                    }
                }
                primary
                    .ok_or_else(|| CpdError::invalid_input("PenaltyPath requires non-empty path"))?
            }
        };

        if runtime.soft_budget_exceeded {
            warnings
                .push("budget exceeded under SoftDegrade mode; kernel run continued".to_string());
        }

        let runtime_ms = u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX);
        ctx.record_scalar("offline.kernel_cpd.gram_evals", runtime.gram_evals as f64);
        ctx.record_scalar(
            "offline.kernel_cpd.segment_cost_evals",
            runtime.segment_cost_evals as f64,
        );
        ctx.record_scalar(
            "offline.kernel_cpd.candidate_evals",
            runtime.candidate_evals as f64,
        );
        ctx.record_scalar("offline.kernel_cpd.runtime_ms", runtime_ms as f64);
        ctx.report_progress(1.0);

        notes.push(format!(
            "final_objective={}, change_count={}",
            selection.objective, selection.change_count
        ));
        notes.push(format!(
            "run_segment_cost_evals={}",
            runtime.segment_cost_evals
        ));
        notes.push(format!("run_candidate_evals={}", runtime.candidate_evals));

        let missing_stats = compute_missing_run_stats(x.n * x.d, x.n_missing(), x.missing);

        let diagnostics = Diagnostics {
            n: x.n,
            d: x.d,
            runtime_ms: Some(runtime_ms),
            notes,
            warnings,
            algorithm: Cow::Borrowed("kernel-cpd"),
            cost_model: Cow::Borrowed(match resolved_kernel {
                ResolvedKernel::Rbf { .. } => "kernel-rbf",
                ResolvedKernel::Linear => "kernel-linear",
            }),
            repro_mode: ctx.repro_mode,
            pruning_stats: Some(PruningStats {
                candidates_considered: runtime.candidate_evals,
                candidates_pruned: 0,
            }),
            missing_policy_applied: Some(missing_stats.missing_policy_applied.to_string()),
            missing_fraction: Some(missing_stats.missing_fraction),
            effective_sample_count: Some(missing_stats.effective_sample_count),
            ..Diagnostics::default()
        };

        OfflineChangePointResult::new(x.n, selection.breakpoints, diagnostics)
    }
}

#[cfg(test)]
mod tests {
    use super::{KernelCpd, KernelCpdConfig, KernelSpec};
    use cpd_core::{
        Constraints, ExecutionContext, MemoryLayout, MissingPolicy, OfflineDetector, Penalty,
        Stopping, TimeIndex, TimeSeriesView,
    };

    fn make_view(values: &[f64], n: usize, d: usize) -> TimeSeriesView<'_> {
        TimeSeriesView::from_f64(
            values,
            n,
            d,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect("test view should be valid")
    }

    #[test]
    fn config_defaults_match_expectations() {
        let cfg = KernelCpdConfig::default();
        assert_eq!(cfg.stopping, Stopping::Penalized(Penalty::BIC));
        assert_eq!(cfg.params_per_segment, 1);
        assert_eq!(cfg.cancel_check_every, 1000);
        assert!(matches!(cfg.kernel, KernelSpec::Rbf { gamma: None }));
    }

    #[test]
    fn kernel_detects_distributional_change() {
        let n = 180usize;
        let split = 90usize;
        let mut values = Vec::with_capacity(n);
        for idx in 0..split {
            values.push(if idx % 2 == 0 { -1.0 } else { 1.0 });
        }
        let pattern = [-2.0, -0.2, 0.2, 2.0];
        for idx in split..n {
            values.push(pattern[idx % pattern.len()]);
        }

        let view = make_view(values.as_slice(), n, 1);
        let constraints = Constraints {
            min_segment_len: 8,
            max_change_points: Some(1),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);

        let config = KernelCpdConfig {
            stopping: Stopping::KnownK(1),
            ..KernelCpdConfig::default()
        };
        let result = KernelCpd::new(config)
            .expect("kernel detector should build")
            .detect(&view, &ctx)
            .expect("kernel detection should succeed");

        assert_eq!(result.breakpoints.len(), 2);
        let cp = result.breakpoints[0];
        assert!(
            cp.abs_diff(split) <= 12,
            "expected split near {split}, got {cp}"
        );
        assert_eq!(result.breakpoints[1], n);
    }

    #[test]
    fn linear_kernel_path_executes() {
        let n = 120usize;
        let split = 60usize;
        let mut values = Vec::with_capacity(n);
        for idx in 0..split {
            values.push((idx as f64 * 0.1).sin());
        }
        for idx in split..n {
            values.push(3.0 + (idx as f64 * 0.1).sin());
        }
        let view = make_view(values.as_slice(), n, 1);
        let constraints = Constraints {
            min_segment_len: 6,
            max_change_points: Some(1),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);

        let config = KernelCpdConfig {
            stopping: Stopping::KnownK(1),
            kernel: KernelSpec::Linear,
            ..KernelCpdConfig::default()
        };
        let result = KernelCpd::new(config)
            .expect("kernel detector should build")
            .detect(&view, &ctx)
            .expect("kernel detection should succeed");
        assert_eq!(result.breakpoints.len(), 2);
        assert_eq!(result.breakpoints[1], n);
    }
}
