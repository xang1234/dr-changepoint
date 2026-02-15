// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::model::CostModel;
use cpd_core::{
    CachePolicy, CpdError, DTypeView, MemoryLayout, MissingSupport, ReproMode, TimeSeriesView,
    prefix_sum_squares, prefix_sum_squares_kahan, prefix_sums, prefix_sums_kahan,
};

const LOG_2PI: f64 = 1.8378770664093453;
const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEFFICIENTS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_7e-7,
];

/// Prior for the Normal-Inverse-Gamma model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NIGPrior {
    /// Prior mean for the segment mean parameter.
    pub mu0: f64,
    /// Prior precision scaling for the mean.
    pub kappa0: f64,
    /// Inverse-gamma shape parameter for variance.
    pub alpha0: f64,
    /// Inverse-gamma scale parameter for variance.
    pub beta0: f64,
}

impl NIGPrior {
    /// Weakly informative defaults.
    pub const fn weakly_informative() -> Self {
        Self {
            mu0: 0.0,
            kappa0: 0.01,
            alpha0: 1.0,
            beta0: 1.0,
        }
    }

    fn validate(&self) -> Result<(), CpdError> {
        if !self.mu0.is_finite() {
            return Err(CpdError::invalid_input(format!(
                "NIG prior mu0 must be finite; got {}",
                self.mu0
            )));
        }
        if !self.kappa0.is_finite() || self.kappa0 <= 0.0 {
            return Err(CpdError::invalid_input(format!(
                "NIG prior kappa0 must be finite and > 0; got {}",
                self.kappa0
            )));
        }
        if !self.alpha0.is_finite() || self.alpha0 <= 0.0 {
            return Err(CpdError::invalid_input(format!(
                "NIG prior alpha0 must be finite and > 0; got {}",
                self.alpha0
            )));
        }
        if !self.beta0.is_finite() || self.beta0 <= 0.0 {
            return Err(CpdError::invalid_input(format!(
                "NIG prior beta0 must be finite and > 0; got {}",
                self.beta0
            )));
        }
        Ok(())
    }
}

impl Default for NIGPrior {
    fn default() -> Self {
        Self::weakly_informative()
    }
}

/// Normal-Inverse-Gamma marginal likelihood segment cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostNIGMarginal {
    pub prior: NIGPrior,
    pub repro_mode: ReproMode,
}

impl CostNIGMarginal {
    pub const fn new(repro_mode: ReproMode) -> Self {
        Self {
            prior: NIGPrior::weakly_informative(),
            repro_mode,
        }
    }

    pub fn with_prior(prior: NIGPrior, repro_mode: ReproMode) -> Result<Self, CpdError> {
        prior.validate()?;
        Ok(Self { prior, repro_mode })
    }
}

impl Default for CostNIGMarginal {
    fn default() -> Self {
        Self::new(ReproMode::Balanced)
    }
}

/// Prefix-stat cache for O(1) NIG marginal segment-cost queries.
#[derive(Clone, Debug, PartialEq)]
pub struct NIGCache {
    prefix_count: Vec<u64>,
    prefix_sum: Vec<f64>,
    prefix_sum_sq: Vec<f64>,
    prior: NIGPrior,
    n: usize,
    d: usize,
}

impl NIGCache {
    fn prefix_len_per_dim(&self) -> usize {
        self.n + 1
    }

    fn dim_offset(&self, dim: usize) -> usize {
        dim * self.prefix_len_per_dim()
    }
}

fn strided_linear_index(
    t: usize,
    dim: usize,
    row_stride: isize,
    col_stride: isize,
    len: usize,
) -> Result<usize, CpdError> {
    let t_isize = isize::try_from(t).map_err(|_| {
        CpdError::invalid_input(format!(
            "strided index overflow: t={t} does not fit into isize"
        ))
    })?;
    let dim_isize = isize::try_from(dim).map_err(|_| {
        CpdError::invalid_input(format!(
            "strided index overflow: dim={dim} does not fit into isize"
        ))
    })?;

    let index = t_isize
        .checked_mul(row_stride)
        .and_then(|left| {
            dim_isize
                .checked_mul(col_stride)
                .and_then(|right| left.checked_add(right))
        })
        .ok_or_else(|| {
            CpdError::invalid_input(format!(
                "strided index overflow at t={t}, dim={dim}, row_stride={row_stride}, col_stride={col_stride}"
            ))
        })?;

    let index_usize = usize::try_from(index).map_err(|_| {
        CpdError::invalid_input(format!(
            "strided index negative at t={t}, dim={dim}: idx={index}"
        ))
    })?;

    if index_usize >= len {
        return Err(CpdError::invalid_input(format!(
            "strided index out of bounds at t={t}, dim={dim}: idx={index_usize}, len={len}"
        )));
    }

    Ok(index_usize)
}

fn read_value(x: &TimeSeriesView<'_>, t: usize, dim: usize) -> Result<f64, CpdError> {
    match (x.values, x.layout) {
        (DTypeView::F32(values), MemoryLayout::CContiguous) => {
            let idx = t
                .checked_mul(x.d)
                .and_then(|base| base.checked_add(dim))
                .ok_or_else(|| CpdError::invalid_input("C-contiguous index overflow"))?;
            values
                .get(idx)
                .map(|v| f64::from(*v))
                .ok_or_else(|| CpdError::invalid_input("C-contiguous index out of bounds"))
        }
        (DTypeView::F64(values), MemoryLayout::CContiguous) => {
            let idx = t
                .checked_mul(x.d)
                .and_then(|base| base.checked_add(dim))
                .ok_or_else(|| CpdError::invalid_input("C-contiguous index overflow"))?;
            values
                .get(idx)
                .copied()
                .ok_or_else(|| CpdError::invalid_input("C-contiguous index out of bounds"))
        }
        (DTypeView::F32(values), MemoryLayout::FContiguous) => {
            let idx = dim
                .checked_mul(x.n)
                .and_then(|base| base.checked_add(t))
                .ok_or_else(|| CpdError::invalid_input("F-contiguous index overflow"))?;
            values
                .get(idx)
                .map(|v| f64::from(*v))
                .ok_or_else(|| CpdError::invalid_input("F-contiguous index out of bounds"))
        }
        (DTypeView::F64(values), MemoryLayout::FContiguous) => {
            let idx = dim
                .checked_mul(x.n)
                .and_then(|base| base.checked_add(t))
                .ok_or_else(|| CpdError::invalid_input("F-contiguous index overflow"))?;
            values
                .get(idx)
                .copied()
                .ok_or_else(|| CpdError::invalid_input("F-contiguous index out of bounds"))
        }
        (
            DTypeView::F32(values),
            MemoryLayout::Strided {
                row_stride,
                col_stride,
            },
        ) => {
            let idx = strided_linear_index(t, dim, row_stride, col_stride, values.len())?;
            Ok(f64::from(values[idx]))
        }
        (
            DTypeView::F64(values),
            MemoryLayout::Strided {
                row_stride,
                col_stride,
            },
        ) => {
            let idx = strided_linear_index(t, dim, row_stride, col_stride, values.len())?;
            Ok(values[idx])
        }
    }
}

fn cache_overflow_err(n: usize, d: usize) -> CpdError {
    CpdError::resource_limit(format!(
        "cache size overflow while planning NIGCache for n={n}, d={d}"
    ))
}

fn normalize_sse(raw_sse: f64) -> f64 {
    if raw_sse.is_nan() || raw_sse <= 0.0 {
        0.0
    } else if raw_sse == f64::INFINITY {
        f64::MAX
    } else {
        raw_sse
    }
}

fn normalize_beta(raw_beta: f64) -> f64 {
    if raw_beta.is_nan() || raw_beta <= f64::MIN_POSITIVE {
        f64::MIN_POSITIVE
    } else if raw_beta == f64::INFINITY {
        f64::MAX
    } else {
        raw_beta
    }
}

fn ln_gamma(z: f64) -> f64 {
    debug_assert!(
        z.is_finite() && z > 0.0,
        "ln_gamma requires z > 0 and finite"
    );

    // Avoid reflection-instability for tiny positive z where sin(pi*z) can underflow.
    // ln Γ(z) = -ln(z) + O(z) as z -> 0+.
    if z < 1e-8 {
        return -z.ln();
    }

    if z < 0.5 {
        let sin_term = (std::f64::consts::PI * z).sin().abs();
        return std::f64::consts::PI.ln() - sin_term.ln() - ln_gamma(1.0 - z);
    }

    let shifted = z - 1.0;
    let mut x = LANCZOS_COEFFICIENTS[0];
    for (idx, coefficient) in LANCZOS_COEFFICIENTS.iter().copied().enumerate().skip(1) {
        x += coefficient / (shifted + idx as f64);
    }

    let t = shifted + LANCZOS_G + 0.5;
    0.5 * LOG_2PI + (shifted + 0.5) * t.ln() - t + x.ln()
}

fn segment_log_marginal(cache: &NIGCache, start: usize, end: usize, dim: usize) -> f64 {
    let base = cache.dim_offset(dim);
    let m = (cache.prefix_count[base + end] - cache.prefix_count[base + start]) as f64;
    let sum = cache.prefix_sum[base + end] - cache.prefix_sum[base + start];
    let sum_sq = cache.prefix_sum_sq[base + end] - cache.prefix_sum_sq[base + start];
    let mean = sum / m;
    let sse = normalize_sse(sum_sq - (sum * sum) / m);

    let prior = cache.prior;
    let kappa_n = prior.kappa0 + m;
    let alpha_n = prior.alpha0 + 0.5 * m;
    let mean_delta = mean - prior.mu0;
    let shrinkage = (prior.kappa0 * m * mean_delta * mean_delta) / (2.0 * kappa_n);
    let beta_n = normalize_beta(prior.beta0 + 0.5 * sse + shrinkage);

    ln_gamma(alpha_n) - ln_gamma(prior.alpha0) + prior.alpha0 * prior.beta0.ln()
        - alpha_n * beta_n.ln()
        + 0.5 * (prior.kappa0.ln() - kappa_n.ln())
        - 0.5 * m * LOG_2PI
}

impl CostModel for CostNIGMarginal {
    type Cache = NIGCache;

    fn name(&self) -> &'static str {
        "nig_marginal"
    }

    fn validate(&self, x: &TimeSeriesView<'_>) -> Result<(), CpdError> {
        self.prior.validate()?;

        if x.n == 0 {
            return Err(CpdError::invalid_input(
                "CostNIGMarginal requires n >= 1; got n=0",
            ));
        }
        if x.d == 0 {
            return Err(CpdError::invalid_input(
                "CostNIGMarginal requires d >= 1; got d=0",
            ));
        }

        if x.has_missing() {
            return Err(CpdError::invalid_input(format!(
                "CostNIGMarginal does not support missing values: effective_missing_count={}",
                x.n_missing()
            )));
        }

        match x.values {
            DTypeView::F32(_) | DTypeView::F64(_) => Ok(()),
        }
    }

    fn missing_support(&self) -> MissingSupport {
        MissingSupport::Reject
    }

    fn precompute(
        &self,
        x: &TimeSeriesView<'_>,
        policy: &CachePolicy,
    ) -> Result<Self::Cache, CpdError> {
        let required_bytes = self.worst_case_cache_bytes(x);

        if matches!(policy, CachePolicy::Approximate { .. }) {
            return Err(CpdError::not_supported(
                "CostNIGMarginal does not support CachePolicy::Approximate",
            ));
        }

        if required_bytes == usize::MAX {
            return Err(cache_overflow_err(x.n, x.d));
        }

        if let CachePolicy::Budgeted { max_bytes } = policy
            && required_bytes > *max_bytes
        {
            return Err(CpdError::resource_limit(format!(
                "CostNIGMarginal cache requires {} bytes, exceeds budget {} bytes",
                required_bytes, max_bytes
            )));
        }

        let prefix_len_per_dim =
            x.n.checked_add(1)
                .ok_or_else(|| cache_overflow_err(x.n, x.d))?;
        let total_prefix_len = prefix_len_per_dim
            .checked_mul(x.d)
            .ok_or_else(|| cache_overflow_err(x.n, x.d))?;

        let mut prefix_count = Vec::with_capacity(total_prefix_len);

        let mut prefix_sum = Vec::with_capacity(total_prefix_len);
        let mut prefix_sum_sq = Vec::with_capacity(total_prefix_len);

        for dim in 0..x.d {
            for idx in 0..=x.n {
                let idx_u64 = u64::try_from(idx).map_err(|_| cache_overflow_err(x.n, x.d))?;
                prefix_count.push(idx_u64);
            }

            let mut series = Vec::with_capacity(x.n);
            for t in 0..x.n {
                series.push(read_value(x, t, dim)?);
            }

            let dim_prefix_sum = if matches!(self.repro_mode, ReproMode::Strict) {
                prefix_sums_kahan(&series)
            } else {
                prefix_sums(&series)
            };

            let dim_prefix_sum_sq = if matches!(self.repro_mode, ReproMode::Strict) {
                prefix_sum_squares_kahan(&series)
            } else {
                prefix_sum_squares(&series)
            };

            debug_assert_eq!(dim_prefix_sum.len(), prefix_len_per_dim);
            debug_assert_eq!(dim_prefix_sum_sq.len(), prefix_len_per_dim);

            prefix_sum.extend_from_slice(&dim_prefix_sum);
            prefix_sum_sq.extend_from_slice(&dim_prefix_sum_sq);
        }

        Ok(NIGCache {
            prefix_count,
            prefix_sum,
            prefix_sum_sq,
            prior: self.prior,
            n: x.n,
            d: x.d,
        })
    }

    fn worst_case_cache_bytes(&self, x: &TimeSeriesView<'_>) -> usize {
        let prefix_len_per_dim = match x.n.checked_add(1) {
            Some(v) => v,
            None => return usize::MAX,
        };
        let total_prefix_len = match prefix_len_per_dim.checked_mul(x.d) {
            Some(v) => v,
            None => return usize::MAX,
        };

        let count_bytes = match total_prefix_len.checked_mul(std::mem::size_of::<u64>()) {
            Some(v) => v,
            None => return usize::MAX,
        };
        let bytes_per_array = match total_prefix_len.checked_mul(std::mem::size_of::<f64>()) {
            Some(v) => v,
            None => return usize::MAX,
        };
        let stat_bytes = match bytes_per_array.checked_mul(2) {
            Some(v) => v,
            None => return usize::MAX,
        };

        match count_bytes.checked_add(stat_bytes) {
            Some(v) => v,
            None => usize::MAX,
        }
    }

    fn supports_approx_cache(&self) -> bool {
        false
    }

    fn segment_cost(&self, cache: &Self::Cache, start: usize, end: usize) -> f64 {
        assert!(
            start < end,
            "segment_cost requires start < end; got start={start}, end={end}"
        );
        assert!(
            end <= cache.n,
            "segment_cost end out of bounds: end={end}, n={} ",
            cache.n
        );

        let mut total_log_marginal = 0.0;
        for dim in 0..cache.d {
            total_log_marginal += segment_log_marginal(cache, start, end, dim);
        }

        -total_log_marginal
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CostNIGMarginal, NIGCache, NIGPrior, cache_overflow_err, ln_gamma, normalize_beta,
        normalize_sse, read_value, segment_log_marginal, strided_linear_index,
    };
    use crate::model::CostModel;
    use crate::normal::CostNormalMeanVar;
    use cpd_core::{
        CachePolicy, CpdError, DTypeView, MemoryLayout, MissingPolicy, MissingSupport, ReproMode,
        TimeIndex, TimeSeriesView,
    };

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tol,
            "expected {expected}, got {actual}, |diff|={diff}, tol={tol}"
        );
    }

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

    fn make_f32_view<'a>(
        values: &'a [f32],
        n: usize,
        d: usize,
        layout: MemoryLayout,
        missing: MissingPolicy,
    ) -> TimeSeriesView<'a> {
        TimeSeriesView::new(
            DTypeView::F32(values),
            n,
            d,
            layout,
            None,
            TimeIndex::None,
            missing,
        )
        .expect("test view should be valid")
    }

    fn naive_nig(values: &[f64], prior: NIGPrior) -> f64 {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let sse = values
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f64>();
        let kappa_n = prior.kappa0 + n;
        let alpha_n = prior.alpha0 + 0.5 * n;
        let beta_n = prior.beta0
            + 0.5 * sse
            + (prior.kappa0 * n * (mean - prior.mu0) * (mean - prior.mu0)) / (2.0 * kappa_n);
        let log_marginal = ln_gamma(alpha_n) - ln_gamma(prior.alpha0)
            + prior.alpha0 * prior.beta0.ln()
            - alpha_n * beta_n.ln()
            + 0.5 * (prior.kappa0.ln() - kappa_n.ln())
            - 0.5 * n * super::LOG_2PI;
        -log_marginal
    }

    fn synthetic_multivariate_values(n: usize, d: usize) -> Vec<f64> {
        let mut values = Vec::with_capacity(n * d);
        for t in 0..n {
            for dim in 0..d {
                let x = t as f64 + 1.0;
                let y = dim as f64 + 1.0;
                values.push((x * y) + (0.03 * x).sin() + (0.07 * y).cos());
            }
        }
        values
    }

    fn dim_series(values: &[f64], n: usize, d: usize, dim: usize) -> Vec<f64> {
        (0..n).map(|t| values[t * d + dim]).collect()
    }

    #[test]
    fn helper_functions_cover_edge_paths() {
        let err_t = strided_linear_index(usize::MAX, 0, 1, 1, 8).expect_err("t overflow expected");
        assert!(matches!(err_t, CpdError::InvalidInput(_)));
        assert!(err_t.to_string().contains("t="));

        let err_dim =
            strided_linear_index(0, usize::MAX, 1, 1, 8).expect_err("dim overflow expected");
        assert!(matches!(err_dim, CpdError::InvalidInput(_)));
        assert!(err_dim.to_string().contains("dim="));

        let err_neg = strided_linear_index(1, 0, -1, 0, 8).expect_err("negative index expected");
        assert!(matches!(err_neg, CpdError::InvalidInput(_)));
        assert!(err_neg.to_string().contains("negative"));

        assert_eq!(normalize_sse(f64::NAN), 0.0);
        assert_eq!(normalize_sse(-1.0), 0.0);
        assert_eq!(normalize_sse(f64::INFINITY), f64::MAX);
        assert_eq!(normalize_beta(f64::NAN), f64::MIN_POSITIVE);
        assert_eq!(normalize_beta(-1.0), f64::MIN_POSITIVE);
        assert_eq!(normalize_beta(f64::INFINITY), f64::MAX);
    }

    #[test]
    fn ln_gamma_matches_known_values() {
        assert_close(ln_gamma(1.0), 0.0, 1e-14);
        assert_close(ln_gamma(0.5), 0.5 * std::f64::consts::PI.ln(), 1e-12);
        assert_close(ln_gamma(5.0), 24.0_f64.ln(), 1e-12);
        let tiny = 1.0e-320;
        assert!(ln_gamma(tiny).is_finite());
        assert_close(ln_gamma(tiny), -tiny.ln(), 1e-10);
    }

    #[test]
    fn read_value_f32_layout_paths_and_errors() {
        let c_values = [1.5_f32, 10.5, 2.5, 20.5];
        let c_view = make_f32_view(
            &c_values,
            2,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        assert_close(
            read_value(&c_view, 1, 0).expect("C f32 read should succeed"),
            2.5,
            1e-12,
        );
        let c_oob = read_value(&c_view, 1, 2).expect_err("C f32 oob expected");
        assert!(c_oob.to_string().contains("out of bounds"));

        let f_values = [1.5_f32, 2.5, 10.5, 20.5];
        let f_view = make_f32_view(
            &f_values,
            2,
            2,
            MemoryLayout::FContiguous,
            MissingPolicy::Error,
        );
        assert_close(
            read_value(&f_view, 1, 0).expect("F f32 read should succeed"),
            2.5,
            1e-12,
        );
        let f_oob = read_value(&f_view, 1, 2).expect_err("F f32 oob expected");
        assert!(f_oob.to_string().contains("out of bounds"));
    }

    #[test]
    fn cache_overflow_error_message_is_stable() {
        let err = cache_overflow_err(7, 11);
        assert!(matches!(err, CpdError::ResourceLimit(_)));
        assert!(err.to_string().contains("n=7, d=11"));
    }

    #[test]
    fn trait_contract_and_defaults() {
        let model = CostNIGMarginal::default();
        assert_eq!(model.name(), "nig_marginal");
        assert_eq!(model.repro_mode, ReproMode::Balanced);
        assert_eq!(model.missing_support(), MissingSupport::Reject);
        assert!(!model.supports_approx_cache());
        assert_eq!(model.prior, NIGPrior::weakly_informative());
    }

    #[test]
    fn with_prior_validates_parameters() {
        let err = CostNIGMarginal::with_prior(
            NIGPrior {
                mu0: 0.0,
                kappa0: 0.0,
                alpha0: 1.0,
                beta0: 1.0,
            },
            ReproMode::Balanced,
        )
        .expect_err("kappa0 must be > 0");
        assert!(matches!(err, CpdError::InvalidInput(_)));

        let ok = CostNIGMarginal::with_prior(
            NIGPrior {
                mu0: 0.5,
                kappa0: 0.25,
                alpha0: 1.5,
                beta0: 0.75,
            },
            ReproMode::Strict,
        )
        .expect("prior should be valid");
        assert_eq!(ok.repro_mode, ReproMode::Strict);
        assert_eq!(ok.prior.mu0, 0.5);
    }

    #[test]
    fn validate_rejects_missing_effective_values() {
        let values = [1.0, f64::NAN, 3.0];
        let view = make_f64_view(
            &values,
            3,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Ignore,
        );
        let err = CostNIGMarginal::default()
            .validate(&view)
            .expect_err("missing values should be rejected");
        assert!(matches!(err, CpdError::InvalidInput(_)));
        assert!(err.to_string().contains("missing values"));
    }

    #[test]
    fn known_answer_univariate_matches_reference_constant() {
        let prior = NIGPrior::weakly_informative();
        let model = CostNIGMarginal::with_prior(prior, ReproMode::Balanced)
            .expect("default prior should be valid");
        let values = [1.0, 2.0, 3.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let cache = model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed");

        let cost = model.segment_cost(&cache, 0, values.len());
        // Reference generated from scipy.special.gammaln with the same closed-form equation.
        let expected = 7.083_349_404_558_763_5;
        assert_close(cost, expected, 1e-12);
        assert_close(cost, naive_nig(&values, prior), 1e-12);
    }

    #[test]
    fn known_answer_multivariate_matches_dimension_sum() {
        let model = CostNIGMarginal::default();
        let values = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        let view = make_f64_view(
            &values,
            4,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let cache = model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed");

        let fast = model.segment_cost(&cache, 1, 4);
        let dim0 = naive_nig(&[2.0, 3.0, 4.0], model.prior);
        let dim1 = naive_nig(&[20.0, 30.0, 40.0], model.prior);
        assert_close(fast, dim0 + dim1, 1e-12);
    }

    #[test]
    fn multivariate_matches_univariate_sum_for_d1_d4_d16() {
        let model = CostNIGMarginal::default();
        let n = 9;
        let start = 2;
        let end = 8;

        for d in [1_usize, 4, 16] {
            let values = synthetic_multivariate_values(n, d);
            let view = make_f64_view(
                &values,
                n,
                d,
                MemoryLayout::CContiguous,
                MissingPolicy::Error,
            );
            let cache = model
                .precompute(&view, &CachePolicy::Full)
                .expect("precompute should succeed");
            let multivariate = model.segment_cost(&cache, start, end);

            let mut per_dimension_sum = 0.0;
            for dim in 0..d {
                let series = dim_series(&values, n, d, dim);
                let dim_view = make_f64_view(
                    &series,
                    n,
                    1,
                    MemoryLayout::CContiguous,
                    MissingPolicy::Error,
                );
                let dim_cache = model
                    .precompute(&dim_view, &CachePolicy::Full)
                    .expect("univariate precompute should succeed");
                per_dimension_sum += model.segment_cost(&dim_cache, start, end);
            }

            assert_close(multivariate, per_dimension_sum, 1e-10);
        }
    }

    #[test]
    fn constant_small_and_extreme_segments_remain_finite() {
        let model = CostNIGMarginal::default();

        let constant = vec![2.0, 2.0, 2.0, 2.0];
        let constant_view = make_f64_view(
            &constant,
            constant.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constant_cache = model
            .precompute(&constant_view, &CachePolicy::Full)
            .expect("constant precompute should succeed");
        assert!(
            model
                .segment_cost(&constant_cache, 0, constant.len())
                .is_finite()
        );

        let singleton = vec![42.0];
        let singleton_view = make_f64_view(
            &singleton,
            singleton.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let singleton_cache = model
            .precompute(&singleton_view, &CachePolicy::Full)
            .expect("singleton precompute should succeed");
        assert!(model.segment_cost(&singleton_cache, 0, 1).is_finite());

        let extreme = vec![1.0e150, -1.0e150, 2.0e150, -2.0e150];
        let extreme_view = make_f64_view(
            &extreme,
            extreme.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let extreme_cache = model
            .precompute(&extreme_view, &CachePolicy::Full)
            .expect("extreme precompute should succeed");
        assert!(
            model
                .segment_cost(&extreme_cache, 0, extreme.len())
                .is_finite()
        );
    }

    #[test]
    fn cache_policy_behavior_budgeted_and_approximate() {
        let model = CostNIGMarginal::default();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = make_f64_view(
            &values,
            3,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let required = model.worst_case_cache_bytes(&view);
        assert!(required > 0);

        model
            .precompute(
                &view,
                &CachePolicy::Budgeted {
                    max_bytes: required,
                },
            )
            .expect("budget equal to required should succeed");

        let err = model
            .precompute(
                &view,
                &CachePolicy::Budgeted {
                    max_bytes: required - 1,
                },
            )
            .expect_err("budget below required should fail");
        assert!(matches!(err, CpdError::ResourceLimit(_)));

        let err = model
            .precompute(
                &view,
                &CachePolicy::Approximate {
                    max_bytes: required,
                    error_tolerance: 0.1,
                },
            )
            .expect_err("approximate policy should be unsupported");
        assert!(matches!(err, CpdError::NotSupported(_)));
    }

    #[test]
    fn worst_case_cache_bytes_matches_multivariate_formula() {
        let model = CostNIGMarginal::default();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = make_f64_view(
            &values,
            3,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let prefix_len_per_dim = view.n + 1;
        let total_prefix_len = prefix_len_per_dim * view.d;
        let expected_small =
            total_prefix_len * (std::mem::size_of::<u64>() + 2 * std::mem::size_of::<f64>());
        assert_eq!(model.worst_case_cache_bytes(&view), expected_small);

        let n_large = 1_000_000usize;
        let d_large = 16usize;
        let expected_large = n_large
            .checked_add(1)
            .and_then(|v| v.checked_mul(d_large))
            .and_then(|v| {
                v.checked_mul(std::mem::size_of::<u64>() + 2 * std::mem::size_of::<f64>())
            })
            .expect("formula should not overflow");
        assert_eq!(expected_large, 384_000_384);
    }

    #[test]
    fn comparison_with_normal_prefers_same_split_but_differs_in_magnitude() {
        let mut values = Vec::new();
        for idx in 0..30 {
            values.push((idx as f64).sin() * 0.05 - 0.2);
        }
        for idx in 0..30 {
            values.push((idx as f64).cos() * 0.05 + 0.8);
        }
        let split = 30;

        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let nig = CostNIGMarginal::default();
        let nig_cache = nig
            .precompute(&view, &CachePolicy::Full)
            .expect("nig precompute should succeed");
        let normal = CostNormalMeanVar::default();
        let normal_cache = normal
            .precompute(&view, &CachePolicy::Full)
            .expect("normal precompute should succeed");

        let nig_gain = nig.segment_cost(&nig_cache, 0, values.len())
            - (nig.segment_cost(&nig_cache, 0, split)
                + nig.segment_cost(&nig_cache, split, values.len()));
        let normal_gain = normal.segment_cost(&normal_cache, 0, values.len())
            - (normal.segment_cost(&normal_cache, 0, split)
                + normal.segment_cost(&normal_cache, split, values.len()));

        assert!(nig_gain > 0.0, "NIG should improve on true split");
        assert!(normal_gain > 0.0, "Normal should improve on true split");
        assert!(
            (nig_gain - normal_gain).abs() > 1e-9,
            "cost gains should not be identical"
        );
    }

    #[test]
    fn nonzero_prior_mean_changes_shifted_cost() {
        let prior = NIGPrior {
            mu0: 5.0,
            kappa0: 0.01,
            alpha0: 1.0,
            beta0: 1.0,
        };
        let model =
            CostNIGMarginal::with_prior(prior, ReproMode::Balanced).expect("prior should be valid");
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let shifted: Vec<f64> = values.iter().map(|v| v + 1.5).collect();

        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let shifted_view = make_f64_view(
            &shifted,
            shifted.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let cache = model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed");
        let shifted_cache = model
            .precompute(&shifted_view, &CachePolicy::Full)
            .expect("shifted precompute should succeed");

        let base = model.segment_cost(&cache, 0, values.len());
        let shifted_cost = model.segment_cost(&shifted_cache, 0, shifted.len());
        assert!(
            (base - shifted_cost).abs() > 1e-6,
            "expected nonzero prior mean to be shift-sensitive"
        );
    }

    #[test]
    fn tiny_positive_alpha_prior_produces_finite_cost() {
        let prior = NIGPrior {
            mu0: 0.0,
            kappa0: 0.01,
            alpha0: 1.0e-320,
            beta0: 1.0,
        };
        let model = CostNIGMarginal::with_prior(prior, ReproMode::Balanced)
            .expect("tiny positive alpha should validate");
        let values = vec![1.0, 2.0, 3.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let cache = model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed");
        let cost = model.segment_cost(&cache, 0, values.len());
        assert!(cost.is_finite());
    }

    #[test]
    fn strict_mode_uses_compensated_prefixes() {
        let values = vec![1.0e16, 1.0, -1.0e16];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let balanced = CostNIGMarginal::new(ReproMode::Balanced)
            .precompute(&view, &CachePolicy::Full)
            .expect("balanced precompute should succeed");
        let strict = CostNIGMarginal::new(ReproMode::Strict)
            .precompute(&view, &CachePolicy::Full)
            .expect("strict precompute should succeed");

        let balanced_final = balanced.prefix_sum[balanced.prefix_len_per_dim() - 1];
        let strict_final = strict.prefix_sum[strict.prefix_len_per_dim() - 1];
        assert_eq!(balanced_final, 0.0);
        assert_eq!(strict_final, 1.0);
    }

    #[test]
    fn segment_log_marginal_matches_segment_cost_sign() {
        let model = CostNIGMarginal::default();
        let values = [1.0, 2.0, 3.0, 4.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let cache = model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed");
        let logp = segment_log_marginal(&cache, 0, values.len(), 0);
        let cost = model.segment_cost(&cache, 0, values.len());
        assert_close(cost, -logp, 1e-12);
    }

    #[test]
    #[should_panic(expected = "start < end")]
    fn segment_cost_panics_when_start_ge_end() {
        let model = CostNIGMarginal::default();
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let view = make_f64_view(
            &values,
            4,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let cache: NIGCache = model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed");

        let _ = model.segment_cost(&cache, 2, 2);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn segment_cost_panics_when_end_exceeds_n() {
        let model = CostNIGMarginal::default();
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let view = make_f64_view(
            &values,
            4,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let cache = model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed");

        let _ = model.segment_cost(&cache, 0, 5);
    }
}
