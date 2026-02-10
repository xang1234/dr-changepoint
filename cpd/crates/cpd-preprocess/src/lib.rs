// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

#[cfg(feature = "preprocess")]
use cpd_core::{CpdError, DTypeView, MemoryLayout, MissingPolicy, TimeIndex, TimeSeriesView};

#[cfg(feature = "preprocess")]
const DEFAULT_WINSOR_LOWER: f64 = 0.01;
#[cfg(feature = "preprocess")]
const DEFAULT_WINSOR_UPPER: f64 = 0.99;
#[cfg(feature = "preprocess")]
const DEFAULT_MAD_EPSILON: f64 = 1.0e-9;
#[cfg(feature = "preprocess")]
const DEFAULT_NORMAL_CONSISTENCY: f64 = 1.4826;

#[cfg(feature = "preprocess")]
#[derive(Clone, Debug, PartialEq)]
pub enum DetrendMethod {
    Linear,
    Polynomial { degree: usize },
}

#[cfg(feature = "preprocess")]
#[derive(Clone, Debug, PartialEq)]
pub struct DetrendConfig {
    pub method: DetrendMethod,
}

#[cfg(feature = "preprocess")]
#[derive(Clone, Debug, PartialEq)]
pub enum DeseasonalizeMethod {
    Differencing { period: usize },
    StlLike { period: usize },
}

#[cfg(feature = "preprocess")]
#[derive(Clone, Debug, PartialEq)]
pub struct DeseasonalizeConfig {
    pub method: DeseasonalizeMethod,
}

#[cfg(feature = "preprocess")]
#[derive(Clone, Debug, PartialEq)]
pub struct WinsorizeConfig {
    pub lower_quantile: f64,
    pub upper_quantile: f64,
}

#[cfg(feature = "preprocess")]
impl Default for WinsorizeConfig {
    fn default() -> Self {
        Self {
            lower_quantile: DEFAULT_WINSOR_LOWER,
            upper_quantile: DEFAULT_WINSOR_UPPER,
        }
    }
}

#[cfg(feature = "preprocess")]
#[derive(Clone, Debug, PartialEq)]
pub struct RobustScaleConfig {
    pub mad_epsilon: f64,
    pub normal_consistency: f64,
}

#[cfg(feature = "preprocess")]
impl Default for RobustScaleConfig {
    fn default() -> Self {
        Self {
            mad_epsilon: DEFAULT_MAD_EPSILON,
            normal_consistency: DEFAULT_NORMAL_CONSISTENCY,
        }
    }
}

#[cfg(feature = "preprocess")]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PreprocessConfig {
    pub detrend: Option<DetrendConfig>,
    pub deseasonalize: Option<DeseasonalizeConfig>,
    pub winsorize: Option<WinsorizeConfig>,
    pub robust_scale: Option<RobustScaleConfig>,
}

#[cfg(feature = "preprocess")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepReport {
    pub step: String,
    pub notes: Vec<String>,
}

#[cfg(feature = "preprocess")]
#[derive(Clone, Debug, PartialEq)]
enum OwnedTimeIndex {
    None,
    Uniform { t0_ns: i64, dt_ns: i64 },
    Explicit(Vec<i64>),
}

#[cfg(feature = "preprocess")]
impl OwnedTimeIndex {
    fn from_borrowed(time: TimeIndex<'_>) -> Self {
        match time {
            TimeIndex::None => Self::None,
            TimeIndex::Uniform { t0_ns, dt_ns } => Self::Uniform { t0_ns, dt_ns },
            TimeIndex::Explicit(ts) => Self::Explicit(ts.to_vec()),
        }
    }

    fn as_borrowed(&self) -> TimeIndex<'_> {
        match self {
            Self::None => TimeIndex::None,
            Self::Uniform { t0_ns, dt_ns } => TimeIndex::Uniform {
                t0_ns: *t0_ns,
                dt_ns: *dt_ns,
            },
            Self::Explicit(ts) => TimeIndex::Explicit(ts),
        }
    }
}

#[cfg(feature = "preprocess")]
#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessedSeries {
    values: Vec<f64>,
    n: usize,
    d: usize,
    missing_mask: Vec<u8>,
    time: OwnedTimeIndex,
    missing: MissingPolicy,
    reports: Vec<StepReport>,
}

#[cfg(feature = "preprocess")]
impl PreprocessedSeries {
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn d(&self) -> usize {
        self.d
    }

    pub fn reports(&self) -> &[StepReport] {
        &self.reports
    }

    pub fn as_view(&self) -> Result<TimeSeriesView<'_>, CpdError> {
        let missing_mask = if self.missing_mask.iter().any(|&v| v == 1) {
            Some(self.missing_mask.as_slice())
        } else {
            None
        };

        TimeSeriesView::from_f64(
            &self.values,
            self.n,
            self.d,
            MemoryLayout::CContiguous,
            missing_mask,
            self.time.as_borrowed(),
            self.missing,
        )
    }
}

#[cfg(feature = "preprocess")]
#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessPipeline {
    config: PreprocessConfig,
}

#[cfg(feature = "preprocess")]
impl PreprocessPipeline {
    pub fn new(config: PreprocessConfig) -> Result<Self, CpdError> {
        validate_config(&config)?;
        Ok(Self { config })
    }

    pub fn config(&self) -> &PreprocessConfig {
        &self.config
    }

    pub fn apply(&self, x: &TimeSeriesView<'_>) -> Result<PreprocessedSeries, CpdError> {
        let (mut values, missing_mask) = flatten_to_c_f64_with_union_missing(x)?;
        let mut reports = vec![];

        if let Some(cfg) = &self.config.detrend {
            apply_detrend(&mut values, x.n, x.d, cfg)?;
            reports.push(StepReport {
                step: "detrend".to_string(),
                notes: vec![format!("method={:?}", cfg.method)],
            });
        }

        if let Some(cfg) = &self.config.deseasonalize {
            let note = apply_deseasonalize(&mut values, x.n, x.d, cfg)?;
            reports.push(StepReport {
                step: "deseasonalize".to_string(),
                notes: vec![note],
            });
        }

        if let Some(cfg) = &self.config.winsorize {
            apply_winsorize(&mut values, x.n, x.d, cfg)?;
            reports.push(StepReport {
                step: "winsorize".to_string(),
                notes: vec![format!(
                    "lower_quantile={}, upper_quantile={}",
                    cfg.lower_quantile, cfg.upper_quantile
                )],
            });
        }

        if let Some(cfg) = &self.config.robust_scale {
            apply_robust_scale(&mut values, x.n, x.d, cfg)?;
            reports.push(StepReport {
                step: "robust_scale".to_string(),
                notes: vec![format!(
                    "mad_epsilon={}, normal_consistency={}",
                    cfg.mad_epsilon, cfg.normal_consistency
                )],
            });
        }

        Ok(PreprocessedSeries {
            values,
            n: x.n,
            d: x.d,
            missing_mask,
            time: OwnedTimeIndex::from_borrowed(x.time),
            missing: x.missing,
            reports,
        })
    }
}

#[cfg(feature = "preprocess")]
fn validate_config(config: &PreprocessConfig) -> Result<(), CpdError> {
    if let Some(detrend) = &config.detrend {
        match detrend.method {
            DetrendMethod::Linear => {}
            DetrendMethod::Polynomial { degree } => {
                if !(1..=3).contains(&degree) {
                    return Err(CpdError::invalid_input(format!(
                        "DetrendMethod::Polynomial degree must be in 1..=3, got {degree}"
                    )));
                }
            }
        }
    }

    if let Some(deseason) = &config.deseasonalize {
        match deseason.method {
            DeseasonalizeMethod::Differencing { period } => {
                if period == 0 {
                    return Err(CpdError::invalid_input(
                        "Deseasonalize differencing period must be >= 1",
                    ));
                }
            }
            DeseasonalizeMethod::StlLike { period } => {
                if period < 2 {
                    return Err(CpdError::invalid_input(format!(
                        "Deseasonalize stl_like period must be >= 2, got {period}"
                    )));
                }
            }
        }
    }

    if let Some(winsor) = &config.winsorize {
        if !winsor.lower_quantile.is_finite() || !winsor.upper_quantile.is_finite() {
            return Err(CpdError::invalid_input(
                "Winsorize quantiles must be finite values",
            ));
        }
        if winsor.lower_quantile < 0.0 || winsor.upper_quantile > 1.0 {
            return Err(CpdError::invalid_input(format!(
                "Winsorize quantiles must satisfy 0.0 <= lower <= upper <= 1.0, got lower={}, upper={}",
                winsor.lower_quantile, winsor.upper_quantile
            )));
        }
        if winsor.lower_quantile >= winsor.upper_quantile {
            return Err(CpdError::invalid_input(format!(
                "Winsorize quantiles must satisfy lower < upper, got lower={}, upper={}",
                winsor.lower_quantile, winsor.upper_quantile
            )));
        }
    }

    if let Some(scale) = &config.robust_scale {
        if !scale.mad_epsilon.is_finite() || scale.mad_epsilon <= 0.0 {
            return Err(CpdError::invalid_input(format!(
                "Robust scaling mad_epsilon must be finite and > 0, got {}",
                scale.mad_epsilon
            )));
        }
        if !scale.normal_consistency.is_finite() || scale.normal_consistency <= 0.0 {
            return Err(CpdError::invalid_input(format!(
                "Robust scaling normal_consistency must be finite and > 0, got {}",
                scale.normal_consistency
            )));
        }
    }

    Ok(())
}

#[cfg(feature = "preprocess")]
fn flatten_to_c_f64_with_union_missing(
    x: &TimeSeriesView<'_>,
) -> Result<(Vec<f64>, Vec<u8>), CpdError> {
    let len = x
        .n
        .checked_mul(x.d)
        .ok_or_else(|| CpdError::invalid_input("n*d overflow while flattening preprocess input"))?;
    let mut out = vec![f64::NAN; len];
    let mut missing = vec![0u8; len];
    let source_len = match x.values {
        DTypeView::F32(values) => values.len(),
        DTypeView::F64(values) => values.len(),
    };

    for t in 0..x.n {
        for j in 0..x.d {
            let src = source_index(x.layout, x.n, x.d, t, j)?;
            if src >= source_len {
                return Err(CpdError::invalid_input(format!(
                    "source index out of bounds for preprocess input: idx={src}, len={source_len}, t={t}, j={j}, layout={:?}",
                    x.layout
                )));
            }

            let dst = t * x.d + j;
            let value = match x.values {
                DTypeView::F32(values) => f64::from(values[src]),
                DTypeView::F64(values) => values[src],
            };
            let mask_missing = x.missing_mask.map(|mask| mask[src] == 1).unwrap_or(false);
            let is_missing = mask_missing || value.is_nan();

            if is_missing {
                out[dst] = f64::NAN;
                missing[dst] = 1;
            } else {
                out[dst] = value;
            }
        }
    }

    Ok((out, missing))
}

#[cfg(feature = "preprocess")]
fn source_index(
    layout: MemoryLayout,
    n: usize,
    d: usize,
    t: usize,
    j: usize,
) -> Result<usize, CpdError> {
    match layout {
        MemoryLayout::CContiguous => t
            .checked_mul(d)
            .and_then(|base| base.checked_add(j))
            .ok_or_else(|| CpdError::invalid_input("C-layout index overflow in preprocessing")),
        MemoryLayout::FContiguous => j
            .checked_mul(n)
            .and_then(|base| base.checked_add(t))
            .ok_or_else(|| CpdError::invalid_input("F-layout index overflow in preprocessing")),
        MemoryLayout::Strided {
            row_stride,
            col_stride,
        } => {
            let t_isize = isize::try_from(t).map_err(|_| {
                CpdError::invalid_input(format!(
                    "time index {t} does not fit in isize for strided preprocessing"
                ))
            })?;
            let j_isize = isize::try_from(j).map_err(|_| {
                CpdError::invalid_input(format!(
                    "dimension index {j} does not fit in isize for strided preprocessing"
                ))
            })?;
            let idx = t_isize
                .checked_mul(row_stride)
                .and_then(|left| {
                    j_isize
                        .checked_mul(col_stride)
                        .and_then(|right| left.checked_add(right))
                })
                .ok_or_else(|| {
                    CpdError::invalid_input(format!(
                        "strided index overflow in preprocessing at t={t}, j={j}, row_stride={row_stride}, col_stride={col_stride}"
                    ))
                })?;
            usize::try_from(idx).map_err(|_| {
                CpdError::invalid_input(format!(
                    "strided index became negative in preprocessing at t={t}, j={j}: idx={idx}"
                ))
            })
        }
    }
}

#[cfg(feature = "preprocess")]
fn apply_detrend(
    values: &mut [f64],
    n: usize,
    d: usize,
    cfg: &DetrendConfig,
) -> Result<(), CpdError> {
    for j in 0..d {
        let valid: Vec<(f64, f64)> = (0..n)
            .filter_map(|t| {
                let idx = t * d + j;
                let y = values[idx];
                (!y.is_nan()).then_some((t as f64, y))
            })
            .collect();

        if valid.is_empty() {
            continue;
        }

        match cfg.method {
            DetrendMethod::Linear => {
                if valid.len() < 2 {
                    return Err(CpdError::invalid_input(format!(
                        "detrend linear requires at least 2 valid samples for dimension {j}"
                    )));
                }
                let (intercept, slope) = fit_linear(&valid).ok_or_else(|| {
                    CpdError::invalid_input(format!(
                        "detrend linear is ill-conditioned for dimension {j}"
                    ))
                })?;
                for t in 0..n {
                    let idx = t * d + j;
                    if values[idx].is_nan() {
                        continue;
                    }
                    let trend = intercept + slope * t as f64;
                    values[idx] -= trend;
                }
            }
            DetrendMethod::Polynomial { degree } => {
                if valid.len() < degree + 1 {
                    return Err(CpdError::invalid_input(format!(
                        "detrend polynomial degree {degree} requires at least {} valid samples for dimension {j}",
                        degree + 1
                    )));
                }
                let coeffs = fit_polynomial(&valid, degree).ok_or_else(|| {
                    CpdError::invalid_input(format!(
                        "detrend polynomial degree {degree} is ill-conditioned for dimension {j}"
                    ))
                })?;
                for t in 0..n {
                    let idx = t * d + j;
                    if values[idx].is_nan() {
                        continue;
                    }
                    let mut trend = 0.0;
                    let mut basis = 1.0;
                    let t_f = t as f64;
                    for coeff in &coeffs {
                        trend += *coeff * basis;
                        basis *= t_f;
                    }
                    values[idx] -= trend;
                }
            }
        }
    }

    Ok(())
}

#[cfg(feature = "preprocess")]
fn fit_linear(samples: &[(f64, f64)]) -> Option<(f64, f64)> {
    let m = samples.len() as f64;
    let (sum_t, sum_y, sum_tt, sum_ty) = samples
        .iter()
        .fold((0.0, 0.0, 0.0, 0.0), |(st, sy, stt, sty), (t, y)| {
            (st + *t, sy + *y, stt + t * t, sty + t * y)
        });
    let denom = m * sum_tt - sum_t * sum_t;
    if !denom.is_finite() || denom.abs() <= f64::EPSILON {
        return None;
    }
    let slope = (m * sum_ty - sum_t * sum_y) / denom;
    let intercept = (sum_y - slope * sum_t) / m;
    Some((intercept, slope))
}

#[cfg(feature = "preprocess")]
fn fit_polynomial(samples: &[(f64, f64)], degree: usize) -> Option<Vec<f64>> {
    let dim = degree + 1;
    let mut a = vec![vec![0.0; dim]; dim];
    let mut b = vec![0.0; dim];

    for (t, y) in samples {
        let mut powers = vec![1.0; dim * 2];
        for p in 1..powers.len() {
            powers[p] = powers[p - 1] * *t;
        }

        for row in 0..dim {
            b[row] += y * powers[row];
            for col in 0..dim {
                a[row][col] += powers[row + col];
            }
        }
    }

    solve_linear_system(a, b)
}

#[cfg(feature = "preprocess")]
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for pivot in 0..n {
        let mut best_row = pivot;
        let mut best_abs = a[pivot][pivot].abs();
        for row in (pivot + 1)..n {
            let cand = a[row][pivot].abs();
            if cand > best_abs {
                best_abs = cand;
                best_row = row;
            }
        }
        if !best_abs.is_finite() || best_abs <= f64::EPSILON {
            return None;
        }
        if best_row != pivot {
            a.swap(pivot, best_row);
            b.swap(pivot, best_row);
        }

        let pivot_val = a[pivot][pivot];
        for col in pivot..n {
            a[pivot][col] /= pivot_val;
        }
        b[pivot] /= pivot_val;

        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = a[row][pivot];
            if factor == 0.0 {
                continue;
            }
            for col in pivot..n {
                a[row][col] -= factor * a[pivot][col];
            }
            b[row] -= factor * b[pivot];
        }
    }
    Some(b)
}

#[cfg(feature = "preprocess")]
fn apply_deseasonalize(
    values: &mut [f64],
    n: usize,
    d: usize,
    cfg: &DeseasonalizeConfig,
) -> Result<String, CpdError> {
    match cfg.method {
        DeseasonalizeMethod::Differencing { period } => {
            if period >= n {
                return Err(CpdError::invalid_input(format!(
                    "deseasonalize differencing requires period < n, got period={period}, n={n}"
                )));
            }
            let original = values.to_vec();
            for t in period..n {
                for j in 0..d {
                    let idx = t * d + j;
                    let lag_idx = (t - period) * d + j;
                    let current = original[idx];
                    let lagged = original[lag_idx];
                    if current.is_nan() || lagged.is_nan() {
                        values[idx] = f64::NAN;
                    } else {
                        values[idx] = current - lagged;
                    }
                }
            }

            Ok(format!(
                "differencing period={period}; warmup prefix [0, {period}) left unchanged"
            ))
        }
        DeseasonalizeMethod::StlLike { period } => {
            if period >= n {
                return Err(CpdError::invalid_input(format!(
                    "deseasonalize stl_like requires period < n, got period={period}, n={n}"
                )));
            }

            let trend_window = stl_like_trend_window(n, period);
            const STL_LIKE_ITERATIONS: usize = 2;

            for j in 0..d {
                let series: Vec<f64> = (0..n).map(|t| values[t * d + j]).collect();
                let mut trend = centered_moving_average_ignore_nan(&series, trend_window);
                let mut deseasoned = series.clone();

                for _ in 0..STL_LIKE_ITERATIONS {
                    let detrended: Vec<f64> = (0..n)
                        .map(|t| {
                            let observed = series[t];
                            let trend_value = trend[t];
                            if observed.is_nan() || trend_value.is_nan() {
                                f64::NAN
                            } else {
                                observed - trend_value
                            }
                        })
                        .collect();
                    let seasonal = mean_centered_phase_profile(&detrended, period);

                    for t in 0..n {
                        let observed = series[t];
                        if observed.is_nan() {
                            deseasoned[t] = f64::NAN;
                        } else {
                            deseasoned[t] = observed - seasonal[t % period];
                        }
                    }

                    trend = centered_moving_average_ignore_nan(&deseasoned, trend_window);
                }

                for t in 0..n {
                    values[t * d + j] = deseasoned[t];
                }
            }

            Ok(format!(
                "method=stl_like, period={period}, trend_window={trend_window}, iterations={STL_LIKE_ITERATIONS}"
            ))
        }
    }
}

#[cfg(feature = "preprocess")]
fn stl_like_trend_window(n: usize, period: usize) -> usize {
    let max_odd_n = if n % 2 == 0 { n.saturating_sub(1) } else { n };
    let bounded = max_odd_n.min(period.saturating_mul(2).saturating_add(1));
    bounded.max(3)
}

#[cfg(feature = "preprocess")]
fn centered_moving_average_ignore_nan(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }
    let half = window / 2;
    let mut out = vec![f64::NAN; n];
    for t in 0..n {
        let start = t.saturating_sub(half);
        let end = t.saturating_add(half).saturating_add(1).min(n);
        let mut sum = 0.0;
        let mut count = 0usize;
        for value in &values[start..end] {
            if value.is_nan() {
                continue;
            }
            sum += *value;
            count += 1;
        }
        if count > 0 {
            out[t] = sum / count as f64;
        }
    }
    out
}

#[cfg(feature = "preprocess")]
fn mean_centered_phase_profile(values: &[f64], period: usize) -> Vec<f64> {
    let mut sums = vec![0.0; period];
    let mut counts = vec![0usize; period];

    for (t, value) in values.iter().enumerate() {
        if value.is_nan() {
            continue;
        }
        let phase = t % period;
        sums[phase] += *value;
        counts[phase] += 1;
    }

    let mut profile = vec![0.0; period];
    for phase in 0..period {
        if counts[phase] > 0 {
            profile[phase] = sums[phase] / counts[phase] as f64;
        }
    }

    let mean = profile.iter().sum::<f64>() / period as f64;
    for value in &mut profile {
        *value -= mean;
    }

    profile
}

#[cfg(feature = "preprocess")]
fn apply_winsorize(
    values: &mut [f64],
    n: usize,
    d: usize,
    cfg: &WinsorizeConfig,
) -> Result<(), CpdError> {
    for j in 0..d {
        let mut valid = collect_valid(values, n, d, j);
        if valid.is_empty() {
            continue;
        }
        valid.sort_by(|a, b| a.total_cmp(b));
        let lo_idx = nearest_rank_index(valid.len(), cfg.lower_quantile);
        let hi_idx = nearest_rank_index(valid.len(), cfg.upper_quantile);
        let lo = valid[lo_idx];
        let hi = valid[hi_idx];
        for t in 0..n {
            let idx = t * d + j;
            let value = values[idx];
            if value.is_nan() {
                continue;
            }
            values[idx] = value.clamp(lo, hi);
        }
    }
    Ok(())
}

#[cfg(feature = "preprocess")]
fn nearest_rank_index(len: usize, q: f64) -> usize {
    if len <= 1 || q <= 0.0 {
        return 0;
    }
    if q >= 1.0 {
        return len - 1;
    }
    ((q * len as f64).ceil() as usize)
        .saturating_sub(1)
        .min(len - 1)
}

#[cfg(feature = "preprocess")]
fn apply_robust_scale(
    values: &mut [f64],
    n: usize,
    d: usize,
    cfg: &RobustScaleConfig,
) -> Result<(), CpdError> {
    for j in 0..d {
        let valid = collect_valid(values, n, d, j);
        if valid.is_empty() {
            continue;
        }
        let median = median_of_slice(&valid).ok_or_else(|| {
            CpdError::invalid_input(format!(
                "robust scaling could not compute median for dimension {j}"
            ))
        })?;
        let deviations: Vec<f64> = valid.iter().map(|v| (v - median).abs()).collect();
        let mad = median_of_slice(&deviations).ok_or_else(|| {
            CpdError::invalid_input(format!(
                "robust scaling could not compute MAD for dimension {j}"
            ))
        })?;
        let scale = (mad * cfg.normal_consistency).max(cfg.mad_epsilon);
        if !scale.is_finite() || scale <= 0.0 {
            return Err(CpdError::invalid_input(format!(
                "robust scaling produced invalid scale for dimension {j}: {scale}"
            )));
        }
        for t in 0..n {
            let idx = t * d + j;
            let value = values[idx];
            if value.is_nan() {
                continue;
            }
            values[idx] = (value - median) / scale;
        }
    }
    Ok(())
}

#[cfg(feature = "preprocess")]
fn collect_valid(values: &[f64], n: usize, d: usize, j: usize) -> Vec<f64> {
    (0..n)
        .filter_map(|t| {
            let value = values[t * d + j];
            (!value.is_nan()).then_some(value)
        })
        .collect()
}

#[cfg(feature = "preprocess")]
fn median_of_slice(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        Some(sorted[mid])
    } else {
        Some((sorted[mid - 1] + sorted[mid]) * 0.5)
    }
}

/// Optional preprocessing namespace.
pub fn crate_name() -> &'static str {
    let _ = cpd_core::crate_name();
    "cpd-preprocess"
}

#[cfg(test)]
mod tests {
    use super::crate_name;

    #[test]
    fn crate_name_matches_expected() {
        assert_eq!(crate_name(), "cpd-preprocess");
    }
}

#[cfg(all(test, feature = "preprocess"))]
mod preprocess_tests {
    use super::{
        DeseasonalizeConfig, DeseasonalizeMethod, DetrendConfig, DetrendMethod, PreprocessConfig,
        PreprocessPipeline, RobustScaleConfig, WinsorizeConfig, median_of_slice,
    };
    use cpd_core::{MemoryLayout, MissingPolicy, TimeIndex, TimeSeriesView};

    fn make_view<'a>(
        values: &'a [f64],
        n: usize,
        d: usize,
        missing_mask: Option<&'a [u8]>,
        missing: MissingPolicy,
    ) -> TimeSeriesView<'a> {
        TimeSeriesView::from_f64(
            values,
            n,
            d,
            MemoryLayout::CContiguous,
            missing_mask,
            TimeIndex::None,
            missing,
        )
        .expect("test view should be valid")
    }

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "actual={actual}, expected={expected}, tol={tol}"
        );
    }

    fn phase_profile_amplitude(values: &[f64], period: usize) -> f64 {
        let mut sums = vec![0.0; period];
        let mut counts = vec![0usize; period];
        for (t, value) in values.iter().enumerate() {
            if value.is_nan() {
                continue;
            }
            let phase = t % period;
            sums[phase] += *value;
            counts[phase] += 1;
        }
        let mut profile = vec![0.0; period];
        for phase in 0..period {
            if counts[phase] > 0 {
                profile[phase] = sums[phase] / counts[phase] as f64;
            }
        }
        let mean = profile.iter().sum::<f64>() / period as f64;
        profile
            .iter()
            .map(|value| (value - mean).abs())
            .sum::<f64>()
            / period as f64
    }

    fn lag_autocorrelation(values: &[f64], lag: usize) -> f64 {
        if lag >= values.len() {
            return 0.0;
        }
        let pairs: Vec<(f64, f64)> = (lag..values.len())
            .filter_map(|t| {
                let current = values[t];
                let lagged = values[t - lag];
                if current.is_nan() || lagged.is_nan() {
                    None
                } else {
                    Some((current, lagged))
                }
            })
            .collect();
        if pairs.len() < 2 {
            return 0.0;
        }
        let inv_len = 1.0 / pairs.len() as f64;
        let mean_x = pairs.iter().map(|(x, _)| *x).sum::<f64>() * inv_len;
        let mean_y = pairs.iter().map(|(_, y)| *y).sum::<f64>() * inv_len;
        let mut numer = 0.0;
        let mut denom_x = 0.0;
        let mut denom_y = 0.0;
        for (x, y) in pairs {
            let dx = x - mean_x;
            let dy = y - mean_y;
            numer += dx * dy;
            denom_x += dx * dx;
            denom_y += dy * dy;
        }
        let denom = denom_x.sqrt() * denom_y.sqrt();
        if !denom.is_finite() || denom <= 1e-12 {
            0.0
        } else {
            (numer / denom).clamp(-1.0, 1.0)
        }
    }

    fn remove_linear_trend(values: &[f64]) -> Vec<f64> {
        let samples: Vec<(f64, f64)> = values
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| (!value.is_nan()).then_some((idx as f64, *value)))
            .collect();
        if samples.len() < 2 {
            return values.to_vec();
        }

        let m = samples.len() as f64;
        let (sum_t, sum_y, sum_tt, sum_ty) = samples
            .iter()
            .fold((0.0, 0.0, 0.0, 0.0), |(st, sy, stt, sty), (t, y)| {
                (st + *t, sy + *y, stt + t * t, sty + t * y)
            });
        let denom = m * sum_tt - sum_t * sum_t;
        if !denom.is_finite() || denom.abs() <= f64::EPSILON {
            return values.to_vec();
        }
        let slope = (m * sum_ty - sum_t * sum_y) / denom;
        let intercept = (sum_y - slope * sum_t) / m;

        values
            .iter()
            .enumerate()
            .map(|(idx, value)| {
                if value.is_nan() {
                    f64::NAN
                } else {
                    value - (intercept + slope * idx as f64)
                }
            })
            .collect()
    }

    #[test]
    fn detrend_linear_removes_linear_signal() {
        let n = 64usize;
        let values: Vec<f64> = (0..n).map(|t| 2.0 * t as f64 + 5.0).collect();
        let view = make_view(&values, n, 1, None, MissingPolicy::Error);
        let config = PreprocessConfig {
            detrend: Some(DetrendConfig {
                method: DetrendMethod::Linear,
            }),
            ..PreprocessConfig::default()
        };
        let out = PreprocessPipeline::new(config)
            .expect("pipeline should build")
            .apply(&view)
            .expect("linear detrend should succeed");
        for value in out.values() {
            assert_close(*value, 0.0, 1e-9);
        }
    }

    #[test]
    fn detrend_polynomial_degree_two_removes_quadratic_signal() {
        let n = 32usize;
        let values: Vec<f64> = (0..n)
            .map(|t| {
                let tf = t as f64;
                tf * tf + 2.0 * tf + 1.0
            })
            .collect();
        let view = make_view(&values, n, 1, None, MissingPolicy::Error);
        let config = PreprocessConfig {
            detrend: Some(DetrendConfig {
                method: DetrendMethod::Polynomial { degree: 2 },
            }),
            ..PreprocessConfig::default()
        };
        let out = PreprocessPipeline::new(config)
            .expect("pipeline should build")
            .apply(&view)
            .expect("quadratic detrend should succeed");
        for value in out.values() {
            assert_close(*value, 0.0, 1e-8);
        }
    }

    #[test]
    fn deseasonalize_differencing_attentuates_periodic_component() {
        let values = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let view = make_view(&values, values.len(), 1, None, MissingPolicy::Error);
        let config = PreprocessConfig {
            deseasonalize: Some(DeseasonalizeConfig {
                method: DeseasonalizeMethod::Differencing { period: 2 },
            }),
            ..PreprocessConfig::default()
        };
        let out = PreprocessPipeline::new(config)
            .expect("pipeline should build")
            .apply(&view)
            .expect("deseasonalize should succeed");
        assert_close(out.values()[0], 1.0, 1e-12);
        assert_close(out.values()[1], 2.0, 1e-12);
        for idx in 2..values.len() {
            assert_close(out.values()[idx], 0.0, 1e-12);
        }
    }

    #[test]
    fn deseasonalize_stl_like_reduces_periodic_phase_amplitude() {
        let period = 4usize;
        let n = 96usize;
        let pattern = [1.0, 3.0, 1.0, 3.0];
        let values: Vec<f64> = (0..n).map(|t| pattern[t % period]).collect();
        let baseline_amp = phase_profile_amplitude(&values, period);
        let view = make_view(&values, n, 1, None, MissingPolicy::Error);
        let config = PreprocessConfig {
            deseasonalize: Some(DeseasonalizeConfig {
                method: DeseasonalizeMethod::StlLike { period },
            }),
            ..PreprocessConfig::default()
        };
        let out = PreprocessPipeline::new(config)
            .expect("pipeline should build")
            .apply(&view)
            .expect("stl-like deseasonalization should succeed");

        let residual_amp = phase_profile_amplitude(out.values(), period);
        assert!(
            residual_amp < baseline_amp * 0.2,
            "expected strong seasonal attenuation: before={baseline_amp}, after={residual_amp}"
        );
    }

    #[test]
    fn deseasonalize_stl_like_reduces_lag_autocorrelation_and_keeps_values_finite() {
        let period = 4usize;
        let n = 96usize;
        let seasonal = [-1.5, 2.5, -1.5, 2.5];
        let values: Vec<f64> = (0..n)
            .map(|t| 0.2 * t as f64 + seasonal[t % period])
            .collect();
        let baseline_corr =
            lag_autocorrelation(&remove_linear_trend(&values), period).abs();
        let view = make_view(&values, n, 1, None, MissingPolicy::Error);
        let config = PreprocessConfig {
            deseasonalize: Some(DeseasonalizeConfig {
                method: DeseasonalizeMethod::StlLike { period },
            }),
            ..PreprocessConfig::default()
        };
        let out = PreprocessPipeline::new(config)
            .expect("pipeline should build")
            .apply(&view)
            .expect("stl-like deseasonalization should succeed");

        for value in out.values() {
            assert!(value.is_finite(), "value should remain finite: {value}");
        }

        let residual_corr =
            lag_autocorrelation(&remove_linear_trend(out.values()), period).abs();
        assert!(
            residual_corr < baseline_corr,
            "expected seasonal lag autocorrelation reduction: before={baseline_corr}, after={residual_corr}"
        );
    }

    #[test]
    fn deseasonalize_stl_like_preserves_missing_mask_and_nan_positions() {
        let values = vec![1.0, 2.0, f64::NAN, 4.0, 1.0, 2.0, 1.0, 2.0];
        let missing_mask = vec![0_u8, 1, 0, 0, 0, 0, 0, 0];
        let view = make_view(&values, values.len(), 1, Some(&missing_mask), MissingPolicy::Ignore);
        let config = PreprocessConfig {
            deseasonalize: Some(DeseasonalizeConfig {
                method: DeseasonalizeMethod::StlLike { period: 2 },
            }),
            ..PreprocessConfig::default()
        };
        let out = PreprocessPipeline::new(config)
            .expect("pipeline should build")
            .apply(&view)
            .expect("stl-like deseasonalization should succeed");

        assert!(out.values()[1].is_nan());
        assert!(out.values()[2].is_nan());
        for (idx, value) in out.values().iter().enumerate() {
            if idx == 1 || idx == 2 {
                continue;
            }
            assert!(value.is_finite(), "expected finite value at idx={idx}, got {value}");
        }
        let roundtrip = out.as_view().expect("roundtrip view should succeed");
        assert_eq!(roundtrip.n_missing(), 2);
    }

    #[test]
    fn deseasonalize_stl_like_is_independent_per_dimension() {
        let n = 64usize;
        let d = 2usize;
        let period = 4usize;
        let season0 = [1.0, 3.0, 1.0, 3.0];
        let season1 = [12.0, -6.0, 8.0, -10.0];
        let mut values = Vec::with_capacity(n * d);
        let mut col0 = Vec::with_capacity(n);
        let mut col1 = Vec::with_capacity(n);
        for t in 0..n {
            let v0 = 0.1 * t as f64 + season0[t % period];
            let v1 = -0.2 * t as f64 + season1[t % period];
            values.push(v0);
            values.push(v1);
            col0.push(v0);
            col1.push(v1);
        }

        let config = PreprocessConfig {
            deseasonalize: Some(DeseasonalizeConfig {
                method: DeseasonalizeMethod::StlLike { period },
            }),
            ..PreprocessConfig::default()
        };

        let multi = PreprocessPipeline::new(config.clone())
            .expect("pipeline should build")
            .apply(&make_view(&values, n, d, None, MissingPolicy::Error))
            .expect("multivariate run should succeed");
        let uni0 = PreprocessPipeline::new(config.clone())
            .expect("pipeline should build")
            .apply(&make_view(&col0, n, 1, None, MissingPolicy::Error))
            .expect("first univariate run should succeed");
        let uni1 = PreprocessPipeline::new(config)
            .expect("pipeline should build")
            .apply(&make_view(&col1, n, 1, None, MissingPolicy::Error))
            .expect("second univariate run should succeed");

        for t in 0..n {
            assert_close(multi.values()[t * d], uni0.values()[t], 1e-9);
            assert_close(multi.values()[t * d + 1], uni1.values()[t], 1e-9);
        }
    }

    #[test]
    fn deseasonalize_stl_like_reports_expected_metadata() {
        let values: Vec<f64> = (0..24).map(|t| [1.0, 3.0, 1.0, 3.0][t % 4]).collect();
        let view = make_view(&values, values.len(), 1, None, MissingPolicy::Error);
        let config = PreprocessConfig {
            deseasonalize: Some(DeseasonalizeConfig {
                method: DeseasonalizeMethod::StlLike { period: 4 },
            }),
            ..PreprocessConfig::default()
        };
        let out = PreprocessPipeline::new(config)
            .expect("pipeline should build")
            .apply(&view)
            .expect("stl-like deseasonalization should succeed");
        let note = out
            .reports()
            .iter()
            .find(|step| step.step == "deseasonalize")
            .expect("deseasonalize report should be present")
            .notes
            .first()
            .expect("deseasonalize report should include note");

        assert!(note.contains("method=stl_like"));
        assert!(note.contains("period=4"));
        assert!(note.contains("trend_window="));
        assert!(note.contains("iterations=2"));
    }

    #[test]
    fn stl_like_period_below_two_is_rejected_at_pipeline_construction() {
        for period in [0usize, 1usize] {
            let err = PreprocessPipeline::new(PreprocessConfig {
                deseasonalize: Some(DeseasonalizeConfig {
                    method: DeseasonalizeMethod::StlLike { period },
                }),
                ..PreprocessConfig::default()
            })
            .expect_err("period below two should be rejected");
            let msg = err.to_string();
            assert!(
                msg.contains("Deseasonalize stl_like period must be >= 2"),
                "unexpected error: {msg}"
            );
        }
    }

    #[test]
    fn stl_like_requires_period_less_than_series_length_at_apply_time() {
        let values = vec![1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0];
        let view = make_view(&values, values.len(), 1, None, MissingPolicy::Error);
        let config = PreprocessConfig {
            deseasonalize: Some(DeseasonalizeConfig {
                method: DeseasonalizeMethod::StlLike {
                    period: values.len(),
                },
            }),
            ..PreprocessConfig::default()
        };
        let err = PreprocessPipeline::new(config)
            .expect("pipeline should build")
            .apply(&view)
            .expect_err("period >= n should fail at apply time");
        let msg = err.to_string();
        assert!(
            msg.contains("deseasonalize stl_like requires period < n"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn winsorize_clips_extremes_to_percentile_bounds() {
        let values = vec![-100.0, 0.0, 1.0, 2.0, 100.0];
        let view = make_view(&values, values.len(), 1, None, MissingPolicy::Error);
        let config = PreprocessConfig {
            winsorize: Some(WinsorizeConfig {
                lower_quantile: 0.4,
                upper_quantile: 0.8,
            }),
            ..PreprocessConfig::default()
        };
        let out = PreprocessPipeline::new(config)
            .expect("pipeline should build")
            .apply(&view)
            .expect("winsorize should succeed");
        assert_eq!(out.values(), &[0.0, 0.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn robust_scaling_centers_median_and_normalizes_mad() {
        let values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let view = make_view(&values, values.len(), 1, None, MissingPolicy::Error);
        let config = PreprocessConfig {
            robust_scale: Some(RobustScaleConfig {
                mad_epsilon: 1e-9,
                normal_consistency: 1.0,
            }),
            ..PreprocessConfig::default()
        };
        let out = PreprocessPipeline::new(config)
            .expect("pipeline should build")
            .apply(&view)
            .expect("robust scaling should succeed");
        let med = median_of_slice(out.values()).expect("median should exist");
        assert_close(med, 0.0, 1e-12);
        let deviations: Vec<f64> = out.values().iter().map(|v| v.abs()).collect();
        let mad = median_of_slice(&deviations).expect("mad should exist");
        assert_close(mad, 1.0, 1e-9);
    }

    #[test]
    fn pipeline_reports_fixed_step_order() {
        let values: Vec<f64> = (0..16).map(|t| t as f64).collect();
        let view = make_view(&values, values.len(), 1, None, MissingPolicy::Error);
        let config = PreprocessConfig {
            detrend: Some(DetrendConfig {
                method: DetrendMethod::Linear,
            }),
            deseasonalize: Some(DeseasonalizeConfig {
                method: DeseasonalizeMethod::Differencing { period: 2 },
            }),
            winsorize: Some(WinsorizeConfig::default()),
            robust_scale: Some(RobustScaleConfig::default()),
        };
        let out = PreprocessPipeline::new(config)
            .expect("pipeline should build")
            .apply(&view)
            .expect("pipeline should succeed");
        let names: Vec<&str> = out.reports().iter().map(|r| r.step.as_str()).collect();
        assert_eq!(
            names,
            vec!["detrend", "deseasonalize", "winsorize", "robust_scale"]
        );
    }

    #[test]
    fn multivariate_preprocessing_is_independent_per_dimension() {
        let n = 6usize;
        let d = 2usize;
        let values: Vec<f64> = vec![
            0.0, 0.0, 1.0, 100.0, 2.0, 200.0, 3.0, 300.0, 4.0, 400.0, 5.0, 500.0,
        ];
        let view = make_view(&values, n, d, None, MissingPolicy::Error);
        let config = PreprocessConfig {
            robust_scale: Some(RobustScaleConfig {
                mad_epsilon: 1e-9,
                normal_consistency: 1.0,
            }),
            ..PreprocessConfig::default()
        };
        let out = PreprocessPipeline::new(config)
            .expect("pipeline should build")
            .apply(&view)
            .expect("robust scaling should succeed");
        let col0: Vec<f64> = (0..n).map(|t| out.values()[t * d]).collect();
        let col1: Vec<f64> = (0..n).map(|t| out.values()[t * d + 1]).collect();
        for t in 0..n {
            assert_close(col0[t], col1[t], 1e-12);
        }
    }

    #[test]
    fn missing_values_are_preserved_and_excluded_from_fit() {
        let values = vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0];
        let missing_mask = vec![1_u8, 0, 0, 0, 0, 0];
        let view = make_view(&values, 6, 1, Some(&missing_mask), MissingPolicy::Ignore);
        let config = PreprocessConfig {
            detrend: Some(DetrendConfig {
                method: DetrendMethod::Linear,
            }),
            winsorize: Some(WinsorizeConfig::default()),
            ..PreprocessConfig::default()
        };
        let out = PreprocessPipeline::new(config)
            .expect("pipeline should build")
            .apply(&view)
            .expect("pipeline should succeed");
        assert!(out.values()[0].is_nan());
        assert!(out.values()[2].is_nan());

        let view_roundtrip = out.as_view().expect("roundtrip view should succeed");
        assert_eq!(view_roundtrip.n_missing(), 2);
    }
}
