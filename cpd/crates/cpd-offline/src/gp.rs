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
use std::collections::HashMap;
use std::time::Instant;

const DEFAULT_CANCEL_CHECK_EVERY: usize = 1000;
const DEFAULT_GP_PARAMS_PER_SEGMENT: usize = 3;
const LOG_2PI: f64 = 1.8378770664093453;

/// Kernel family for GP-based offline segmentation.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub enum GpKernel {
    /// Squared-exponential kernel.
    Rbf { length_scale: f64, variance: f64 },
    /// Matern 3/2 kernel.
    Matern32 { length_scale: f64, variance: f64 },
}

impl Default for GpKernel {
    fn default() -> Self {
        Self::Rbf {
            length_scale: 8.0,
            variance: 1.0,
        }
    }
}

impl GpKernel {
    fn validate(&self) -> Result<(), CpdError> {
        let (length_scale, variance) = match self {
            Self::Rbf {
                length_scale,
                variance,
            }
            | Self::Matern32 {
                length_scale,
                variance,
            } => (*length_scale, *variance),
        };

        if !length_scale.is_finite() || length_scale <= 0.0 {
            return Err(CpdError::invalid_input(format!(
                "GP kernel length_scale must be finite and > 0; got {length_scale}"
            )));
        }
        if !variance.is_finite() || variance <= 0.0 {
            return Err(CpdError::invalid_input(format!(
                "GP kernel variance must be finite and > 0; got {variance}"
            )));
        }
        Ok(())
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Rbf { .. } => "rbf",
            Self::Matern32 { .. } => "matern32",
        }
    }

    fn covariance(&self, lag: usize) -> f64 {
        let x = lag as f64;
        match self {
            Self::Rbf {
                length_scale,
                variance,
            } => {
                let z = x / *length_scale;
                *variance * (-0.5 * z * z).exp()
            }
            Self::Matern32 {
                length_scale,
                variance,
            } => {
                let root3 = 3.0_f64.sqrt();
                let z = root3 * x / *length_scale;
                *variance * (1.0 + z) * (-z).exp()
            }
        }
    }
}

/// Configuration for [`GpCpd`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct GpConfig {
    pub stopping: Stopping,
    pub kernel: GpKernel,
    pub noise_variance: f64,
    pub params_per_segment: usize,
    pub cancel_check_every: usize,
}

impl Default for GpConfig {
    fn default() -> Self {
        Self {
            stopping: Stopping::Penalized(Penalty::BIC),
            kernel: GpKernel::default(),
            noise_variance: 1.0e-3,
            params_per_segment: DEFAULT_GP_PARAMS_PER_SEGMENT,
            cancel_check_every: DEFAULT_CANCEL_CHECK_EVERY,
        }
    }
}

impl GpConfig {
    fn validate(&self) -> Result<(), CpdError> {
        validate_stopping(&self.stopping)?;
        self.kernel.validate()?;
        if !self.noise_variance.is_finite() || self.noise_variance <= 0.0 {
            return Err(CpdError::invalid_input(format!(
                "GpConfig.noise_variance must be finite and > 0; got {}",
                self.noise_variance
            )));
        }
        if self.params_per_segment == 0 {
            return Err(CpdError::invalid_input(
                "GpConfig.params_per_segment must be >= 1; got 0",
            ));
        }
        Ok(())
    }

    fn normalized_cancel_check_every(&self) -> usize {
        self.cancel_check_every.max(1)
    }
}

/// Configuration for [`ArgpCpd`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct ArgpConfig {
    pub gp: GpConfig,
    pub ar_weight: f64,
    pub ar_decay: f64,
}

impl Default for ArgpConfig {
    fn default() -> Self {
        Self {
            gp: GpConfig::default(),
            ar_weight: 0.75,
            ar_decay: 3.0,
        }
    }
}

impl ArgpConfig {
    fn validate(&self) -> Result<(), CpdError> {
        self.gp.validate()?;
        if !self.ar_weight.is_finite() || self.ar_weight < 0.0 {
            return Err(CpdError::invalid_input(format!(
                "ArgpConfig.ar_weight must be finite and >= 0; got {}",
                self.ar_weight
            )));
        }
        if !self.ar_decay.is_finite() || self.ar_decay <= 0.0 {
            return Err(CpdError::invalid_input(format!(
                "ArgpConfig.ar_decay must be finite and > 0; got {}",
                self.ar_decay
            )));
        }
        Ok(())
    }
}

/// Gaussian-process Bayesian offline detector (experimental).
///
/// This detector is intentionally expensive. Segment scoring uses GP marginal
/// likelihood and can be much slower than core baseline detectors.
#[derive(Debug)]
pub struct GpCpd {
    config: GpConfig,
}

impl GpCpd {
    pub fn new(config: GpConfig) -> Result<Self, CpdError> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn config(&self) -> &GpConfig {
        &self.config
    }
}

/// Autoregressive GP Bayesian offline detector (experimental).
///
/// Adds an AR-like exponential correlation component to the base GP kernel.
#[derive(Debug)]
pub struct ArgpCpd {
    config: ArgpConfig,
}

impl ArgpCpd {
    pub fn new(config: ArgpConfig) -> Result<Self, CpdError> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn config(&self) -> &ArgpConfig {
        &self.config
    }
}

#[derive(Default, Clone, Copy, Debug)]
struct RuntimeStats {
    candidate_evals: usize,
    segment_cost_evals: usize,
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

#[derive(Clone, Copy, Debug)]
struct Segment {
    start: usize,
    end: usize,
}

#[derive(Clone, Copy, Debug)]
struct BestSplit {
    segment_index: usize,
    split: usize,
    gain: f64,
}

#[derive(Clone, Copy, Debug)]
struct ArTerm {
    weight: f64,
    decay: f64,
}

#[derive(Debug)]
struct SegmentScorer<'a> {
    values: &'a [f64],
    kernel: &'a GpKernel,
    noise_variance: f64,
    ar_term: Option<ArTerm>,
    cache: HashMap<(usize, usize), f64>,
}

impl<'a> SegmentScorer<'a> {
    fn new(
        values: &'a [f64],
        kernel: &'a GpKernel,
        noise_variance: f64,
        ar_term: Option<ArTerm>,
    ) -> Self {
        Self {
            values,
            kernel,
            noise_variance,
            ar_term,
            cache: HashMap::new(),
        }
    }

    fn segment_cost(
        &mut self,
        start: usize,
        end: usize,
        ctx: &ExecutionContext<'_>,
        runtime: &mut RuntimeStats,
    ) -> Result<f64, CpdError> {
        if end <= start {
            return Err(CpdError::invalid_input(format!(
                "invalid segment bounds: start={start}, end={end}"
            )));
        }

        if let Some(cost) = self.cache.get(&(start, end)) {
            return Ok(*cost);
        }

        runtime.segment_cost_evals = runtime.segment_cost_evals.saturating_add(1);
        match ctx.check_cost_eval_budget(runtime.segment_cost_evals)? {
            BudgetStatus::WithinBudget => {}
            BudgetStatus::ExceededSoftDegrade => {
                runtime.soft_budget_exceeded = true;
            }
        }

        let cost = gp_segment_nll(
            self.values,
            start,
            end,
            self.kernel,
            self.noise_variance,
            self.ar_term,
        )?;
        if !cost.is_finite() {
            return Err(CpdError::numerical_issue(format!(
                "non-finite segment likelihood at [{start}, {end})"
            )));
        }
        self.cache.insert((start, end), cost);
        Ok(cost)
    }
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

fn materialize_univariate(x: &TimeSeriesView<'_>) -> Result<Vec<f64>, CpdError> {
    if x.d != 1 {
        return Err(CpdError::invalid_input(format!(
            "GP detectors currently support univariate data only (d=1); got d={}",
            x.d
        )));
    }
    if matches!(x.missing, MissingPolicy::Ignore) {
        return Err(CpdError::invalid_input(
            "GP detectors do not support MissingPolicy::Ignore; use Error/ImputeZero/ImputeLast",
        ));
    }

    let total_len = match x.values {
        cpd_core::DTypeView::F32(values) => values.len(),
        cpd_core::DTypeView::F64(values) => values.len(),
    };
    let mut out = vec![0.0; x.n];
    let mut carry = 0.0;
    let mut carry_ready = false;

    for (row, slot) in out.iter_mut().enumerate() {
        let idx = source_index(x.layout, x.n, x.d, row, 0)?;
        if idx >= total_len {
            return Err(CpdError::invalid_input(format!(
                "source index out of bounds at row={row}: idx={idx}, len={total_len}"
            )));
        }

        let mut value = value_at(x, idx);
        let masked_missing = x.missing_mask.is_some_and(|mask| mask[idx] == 1);
        let missing = masked_missing || value.is_nan();

        if missing {
            value = match x.missing {
                MissingPolicy::Error => {
                    return Err(CpdError::invalid_input(format!(
                        "missing value encountered at row={row} with MissingPolicy::Error"
                    )));
                }
                MissingPolicy::ImputeZero => 0.0,
                MissingPolicy::ImputeLast => {
                    if carry_ready {
                        carry
                    } else {
                        0.0
                    }
                }
                MissingPolicy::Ignore => unreachable!("handled above"),
            };
        }

        *slot = value;
        carry = value;
        carry_ready = true;
    }

    Ok(out)
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

fn candidate_window(candidates: &[usize], lower: usize, upper: usize) -> Option<(usize, usize)> {
    if lower > upper {
        return None;
    }
    let start = candidates.partition_point(|&split| split < lower);
    let end = candidates.partition_point(|&split| split <= upper);
    if start >= end {
        None
    } else {
        Some((start, end))
    }
}

fn insert_sorted_unique(values: &mut Vec<usize>, value: usize) {
    if let Err(position) = values.binary_search(&value) {
        values.insert(position, value);
    }
}

fn segment_bounds_from_breakpoints(n: usize, breakpoints: &[usize]) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(breakpoints.len());
    let mut start = 0usize;
    for &end in breakpoints {
        out.push((start, end));
        start = end;
    }
    if out.is_empty() {
        out.push((0, n));
    }
    out
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

#[allow(clippy::too_many_arguments)]
fn best_split_for_segment(
    segment: Segment,
    candidates: &[usize],
    validated: &ValidatedConstraints,
    scorer: &mut SegmentScorer<'_>,
    cancel_check_every: usize,
    started_at: Instant,
    ctx: &ExecutionContext<'_>,
    runtime: &mut RuntimeStats,
    iteration: &mut usize,
) -> Result<Option<BestSplit>, CpdError> {
    if segment.end <= segment.start {
        return Ok(None);
    }
    if segment.end - segment.start < validated.min_segment_len.saturating_mul(2) {
        return Ok(None);
    }

    let lower = segment.start.saturating_add(validated.min_segment_len);
    let upper = segment.end.saturating_sub(validated.min_segment_len);
    let Some((start_idx, end_idx)) = candidate_window(candidates, lower, upper) else {
        return Ok(None);
    };

    let whole_cost = scorer.segment_cost(segment.start, segment.end, ctx, runtime)?;
    let mut best_gain = f64::NEG_INFINITY;
    let mut best_split = None;

    for &split in &candidates[start_idx..end_idx] {
        *iteration = iteration.saturating_add(1);
        poll_runtime(*iteration, cancel_check_every, started_at, ctx, runtime)?;

        runtime.candidate_evals = runtime.candidate_evals.saturating_add(1);
        let left = scorer.segment_cost(segment.start, split, ctx, runtime)?;
        let right = scorer.segment_cost(split, segment.end, ctx, runtime)?;
        let gain = whole_cost - left - right;
        if !gain.is_finite() {
            return Err(CpdError::numerical_issue(format!(
                "non-finite gain at segment=[{}, {}), split={split}",
                segment.start, segment.end
            )));
        }
        if gain > best_gain || (gain == best_gain && best_split.is_none_or(|curr| split < curr)) {
            best_gain = gain;
            best_split = Some(split);
        }
    }

    Ok(best_split.map(|split| BestSplit {
        segment_index: usize::MAX,
        split,
        gain: best_gain,
    }))
}

#[allow(clippy::too_many_arguments)]
fn run_binary_segmentation(
    n: usize,
    candidates: &[usize],
    validated: &ValidatedConstraints,
    scorer: &mut SegmentScorer<'_>,
    known_k: Option<usize>,
    beta: Option<f64>,
    cancel_check_every: usize,
    started_at: Instant,
    ctx: &ExecutionContext<'_>,
    runtime: &mut RuntimeStats,
) -> Result<SegmentationResult, CpdError> {
    let max_change_points = validated.max_change_points.unwrap_or(usize::MAX);
    if let Some(k) = known_k {
        if k == 0 {
            return Err(CpdError::invalid_input("Stopping::KnownK requires k >= 1"));
        }
        if k > max_change_points {
            return Err(CpdError::invalid_input(format!(
                "KnownK={k} exceeds constraints.max_change_points={max_change_points}"
            )));
        }
    }

    let mut segments = vec![Segment { start: 0, end: n }];
    let mut breakpoints = Vec::<usize>::new();
    let mut iteration = 0usize;

    loop {
        if let Some(k) = known_k
            && breakpoints.len() >= k
        {
            break;
        }
        if breakpoints.len() >= max_change_points {
            break;
        }

        let mut best_global: Option<BestSplit> = None;
        for (segment_index, segment) in segments.iter().copied().enumerate() {
            let Some(mut local_best) = best_split_for_segment(
                segment,
                candidates,
                validated,
                scorer,
                cancel_check_every,
                started_at,
                ctx,
                runtime,
                &mut iteration,
            )?
            else {
                continue;
            };
            local_best.segment_index = segment_index;
            match best_global {
                None => best_global = Some(local_best),
                Some(best) => {
                    if local_best.gain > best.gain
                        || (local_best.gain == best.gain && local_best.split < best.split)
                    {
                        best_global = Some(local_best);
                    }
                }
            }
        }

        let Some(best) = best_global else {
            break;
        };

        if let Some(beta) = beta
            && best.gain <= beta
        {
            break;
        }

        let chosen = segments.remove(best.segment_index);
        insert_sorted_unique(&mut breakpoints, best.split);
        segments.push(Segment {
            start: chosen.start,
            end: best.split,
        });
        segments.push(Segment {
            start: best.split,
            end: chosen.end,
        });
    }

    if let Some(k) = known_k
        && breakpoints.len() < k
    {
        return Err(CpdError::invalid_input(format!(
            "KnownK({k}) is infeasible under current constraints"
        )));
    }

    let mut full_breakpoints = breakpoints;
    full_breakpoints.push(n);
    full_breakpoints.sort_unstable();
    full_breakpoints.dedup();

    let mut objective = 0.0;
    for (start, end) in segment_bounds_from_breakpoints(n, full_breakpoints.as_slice()) {
        objective += scorer.segment_cost(start, end, ctx, runtime)?;
    }
    let change_count = full_breakpoints.len().saturating_sub(1);
    if let Some(beta) = beta {
        objective += beta * change_count as f64;
    }

    Ok(SegmentationResult {
        breakpoints: full_breakpoints,
        objective,
        change_count,
    })
}

fn gp_segment_nll(
    values: &[f64],
    start: usize,
    end: usize,
    kernel: &GpKernel,
    noise_variance: f64,
    ar_term: Option<ArTerm>,
) -> Result<f64, CpdError> {
    let len = end - start;
    if len == 0 {
        return Err(CpdError::invalid_input("segment length must be >= 1"));
    }

    let mut base = vec![0.0; len * len];
    for i in 0..len {
        for j in 0..=i {
            let lag = i.abs_diff(j);
            let mut cov = kernel.covariance(lag);
            if let Some(ar_term) = ar_term {
                cov += ar_term.weight * (-(lag as f64) / ar_term.decay).exp();
            }
            if i == j {
                cov += noise_variance;
            }
            base[i * len + j] = cov;
            base[j * len + i] = cov;
        }
    }

    let y = &values[start..end];
    let mut jitter = 0.0;
    let base_jitter = (noise_variance * 1.0e-6).max(1.0e-10);
    for _attempt in 0..6 {
        let mut cov = base.clone();
        if jitter > 0.0 {
            for i in 0..len {
                cov[i * len + i] += jitter;
            }
        }
        if let Ok(nll) = gp_nll_from_cov(cov.as_mut_slice(), y, len) {
            return Ok(nll);
        }
        jitter = if jitter == 0.0 {
            base_jitter
        } else {
            jitter * 10.0
        };
    }

    Err(CpdError::numerical_issue(format!(
        "failed GP Cholesky decomposition for segment [{start}, {end}) even after jitter retries"
    )))
}

fn gp_nll_from_cov(cov: &mut [f64], y: &[f64], n: usize) -> Result<f64, CpdError> {
    cholesky_in_place(cov, n)?;

    let mut z = vec![0.0; n];
    for i in 0..n {
        let mut sum = y[i];
        for k in 0..i {
            sum -= cov[i * n + k] * z[k];
        }
        let diag = cov[i * n + i];
        z[i] = sum / diag;
    }

    let mut alpha = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = z[i];
        for k in i + 1..n {
            sum -= cov[k * n + i] * alpha[k];
        }
        let diag = cov[i * n + i];
        alpha[i] = sum / diag;
    }

    let quad = y
        .iter()
        .zip(alpha.iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum::<f64>();
    let log_det = (0..n).map(|i| cov[i * n + i].ln()).sum::<f64>() * 2.0;
    let nll = 0.5 * (quad + log_det + n as f64 * LOG_2PI);

    if !nll.is_finite() {
        return Err(CpdError::numerical_issue(
            "non-finite GP marginal likelihood",
        ));
    }
    Ok(nll)
}

fn cholesky_in_place(matrix: &mut [f64], n: usize) -> Result<(), CpdError> {
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i * n + j];
            for k in 0..j {
                sum -= matrix[i * n + k] * matrix[j * n + k];
            }

            if i == j {
                if !sum.is_finite() || sum <= 0.0 {
                    return Err(CpdError::numerical_issue(
                        "covariance is not positive definite",
                    ));
                }
                matrix[i * n + i] = sum.sqrt();
            } else {
                matrix[i * n + j] = sum / matrix[j * n + j];
            }
        }

        for j in i + 1..n {
            matrix[i * n + j] = 0.0;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn detect_with_gp_mode(
    x: &TimeSeriesView<'_>,
    ctx: &ExecutionContext<'_>,
    config: &GpConfig,
    ar_term: Option<ArTerm>,
    algorithm_label: &'static str,
) -> Result<OfflineChangePointResult, CpdError> {
    check_missing_compatibility(x.missing, cpd_core::MissingSupport::Reject)?;
    let validated = validate_constraints(ctx.constraints, x.n)?;
    let started_at = Instant::now();
    let cancel_check_every = config.normalized_cancel_check_every();
    let mut runtime = RuntimeStats::default();
    let mut notes = vec![
        "feature=gp; complexity=expensive_segment_likelihood".to_string(),
        "computational_requirements=expect_high_runtime_and_memory_for_large_n".to_string(),
    ];
    let mut warnings = vec![];

    let values = materialize_univariate(x)?;
    let mut scorer = SegmentScorer::new(
        values.as_slice(),
        &config.kernel,
        config.noise_variance,
        ar_term,
    );
    if x.n >= 512 {
        warnings.push(format!(
            "{algorithm_label} can be very expensive for n >= 512 (got n={})",
            x.n
        ));
    }

    notes.push(format!("kernel={}", config.kernel.label()));
    notes.push(format!("noise_variance={}", config.noise_variance));
    if let Some(ar_term) = ar_term {
        notes.push(format!(
            "ar_weight={}, ar_decay={}",
            ar_term.weight, ar_term.decay
        ));
    }

    let selection = match &config.stopping {
        Stopping::KnownK(k) => {
            notes.push(format!("stopping=KnownK({k})"));
            run_binary_segmentation(
                x.n,
                validated.effective_candidates.as_slice(),
                &validated,
                &mut scorer,
                Some(*k),
                None,
                cancel_check_every,
                started_at,
                ctx,
                &mut runtime,
            )?
        }
        Stopping::Penalized(penalty) => {
            let resolved = resolve_penalty(penalty, x.n, x.d, config.params_per_segment)?;
            notes.push(format!(
                "stopping=Penalized({:?}), beta={}, params_per_segment={}",
                resolved.penalty, resolved.beta, resolved.params_per_segment
            ));
            run_binary_segmentation(
                x.n,
                validated.effective_candidates.as_slice(),
                &validated,
                &mut scorer,
                None,
                Some(resolved.beta),
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
                let resolved = resolve_penalty(penalty, x.n, x.d, config.params_per_segment)?;
                let run = run_binary_segmentation(
                    x.n,
                    validated.effective_candidates.as_slice(),
                    &validated,
                    &mut scorer,
                    None,
                    Some(resolved.beta),
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
            primary.ok_or_else(|| CpdError::invalid_input("PenaltyPath requires non-empty path"))?
        }
    };

    if runtime.soft_budget_exceeded {
        warnings.push("budget exceeded under SoftDegrade mode; GP run continued".to_string());
    }

    let runtime_ms = u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX);
    ctx.record_scalar(
        "offline.gp.segment_cost_evals",
        runtime.segment_cost_evals as f64,
    );
    ctx.record_scalar("offline.gp.candidate_evals", runtime.candidate_evals as f64);
    ctx.record_scalar("offline.gp.runtime_ms", runtime_ms as f64);
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
        algorithm: Cow::Borrowed(algorithm_label),
        cost_model: Cow::Borrowed("gp-marginal-likelihood"),
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

impl OfflineDetector for GpCpd {
    fn detect(
        &self,
        x: &TimeSeriesView<'_>,
        ctx: &ExecutionContext<'_>,
    ) -> Result<OfflineChangePointResult, CpdError> {
        detect_with_gp_mode(x, ctx, &self.config, None, "gp-cpd")
    }
}

impl OfflineDetector for ArgpCpd {
    fn detect(
        &self,
        x: &TimeSeriesView<'_>,
        ctx: &ExecutionContext<'_>,
    ) -> Result<OfflineChangePointResult, CpdError> {
        detect_with_gp_mode(
            x,
            ctx,
            &self.config.gp,
            Some(ArTerm {
                weight: self.config.ar_weight,
                decay: self.config.ar_decay,
            }),
            "argp-cpd",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{ArgpConfig, ArgpCpd, GpConfig, GpCpd};
    use crate::{BinSeg, BinSegConfig};
    use cpd_core::{
        Constraints, ExecutionContext, MemoryLayout, MissingPolicy, OfflineDetector, Stopping,
        TimeIndex, TimeSeriesView,
    };
    use cpd_costs::CostL2Mean;

    fn make_view(values: &[f64], n: usize) -> TimeSeriesView<'_> {
        TimeSeriesView::from_f64(
            values,
            n,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect("test view should be valid")
    }

    fn distance_to_split(result_cp: usize, split: usize) -> usize {
        result_cp.abs_diff(split)
    }

    #[test]
    fn gp_detects_correlation_regime_change_and_compares_to_l2_binseg() {
        let n = 180usize;
        let split = 90usize;
        let mut values = Vec::with_capacity(n);
        for t in 0..split {
            let x = t as f64;
            values.push((0.08 * x).sin() + 0.12 * (0.03 * x).cos());
        }
        for t in split..n {
            let x = t as f64;
            values.push((0.42 * x).sin() + 0.12 * (0.03 * x).cos());
        }

        let view = make_view(values.as_slice(), n);
        let constraints = Constraints {
            min_segment_len: 10,
            max_change_points: Some(1),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);

        let gp_cfg = GpConfig {
            stopping: Stopping::KnownK(1),
            ..GpConfig::default()
        };
        let gp_result = GpCpd::new(gp_cfg)
            .expect("gp detector should build")
            .detect(&view, &ctx)
            .expect("gp detection should succeed");
        assert_eq!(gp_result.breakpoints.len(), 2);

        let baseline_cfg = BinSegConfig {
            stopping: Stopping::KnownK(1),
            ..BinSegConfig::default()
        };
        let baseline = BinSeg::new(CostL2Mean::default(), baseline_cfg)
            .expect("binseg should build")
            .detect(&view, &ctx)
            .expect("baseline detection should succeed");

        let gp_cp = gp_result.breakpoints[0];
        let baseline_cp = baseline.breakpoints[0];
        let gp_dist = distance_to_split(gp_cp, split);
        let baseline_dist = distance_to_split(baseline_cp, split);
        assert!(gp_dist <= 15, "GP split too far from target: cp={gp_cp}");
        assert!(
            gp_dist <= baseline_dist + 5,
            "GP should be competitive with baseline: gp_dist={gp_dist}, baseline_dist={baseline_dist}"
        );
    }

    #[test]
    fn argp_detects_ar_regime_change() {
        let n = 200usize;
        let split = 100usize;
        let mut values = vec![0.0; n];
        for t in 1..n {
            let phi = if t < split { 0.9 } else { -0.6 };
            let drift = if t < split { 0.0 } else { 1.2 };
            let eps = 0.12 * (0.17 * t as f64).sin() + 0.05 * (0.03 * t as f64).cos();
            values[t] = drift + phi * (values[t - 1] - drift) + eps;
        }

        let view = make_view(values.as_slice(), n);
        let constraints = Constraints {
            min_segment_len: 12,
            max_change_points: Some(1),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);

        let cfg = ArgpConfig {
            gp: GpConfig {
                stopping: Stopping::KnownK(1),
                ..GpConfig::default()
            },
            ..ArgpConfig::default()
        };
        let result = ArgpCpd::new(cfg)
            .expect("argp detector should build")
            .detect(&view, &ctx)
            .expect("argp detection should succeed");
        let cp = result.breakpoints[0];
        assert!(
            cp.abs_diff(split) <= 14,
            "expected AR regime change near {split}, got {cp}"
        );
    }
}
