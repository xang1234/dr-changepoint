// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::model::CostModel;
use cpd_core::{
    CachePolicy, CpdError, DTypeView, MemoryLayout, MissingSupport, ReproMode, TimeSeriesView,
    prefix_sum_squares, prefix_sum_squares_kahan, prefix_sums, prefix_sums_kahan,
};

const VAR_FLOOR: f64 = f64::EPSILON * 1e6;

/// Autoregressive segment cost model.
///
/// Currently supports AR(1) with intercept and Gaussian residual variance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CostAR {
    pub order: usize,
    pub repro_mode: ReproMode,
}

impl CostAR {
    pub const fn new(order: usize, repro_mode: ReproMode) -> Self {
        Self { order, repro_mode }
    }

    pub const fn ar1(repro_mode: ReproMode) -> Self {
        Self::new(1, repro_mode)
    }
}

impl Default for CostAR {
    fn default() -> Self {
        Self::ar1(ReproMode::Balanced)
    }
}

/// Prefix-stat cache for O(1) AR(1) segment-cost queries.
#[derive(Clone, Debug, PartialEq)]
pub struct ARCache {
    prefix_sum: Vec<f64>,
    prefix_sum_sq: Vec<f64>,
    prefix_lag_prod: Vec<f64>,
    n: usize,
    d: usize,
}

impl ARCache {
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
        "cache size overflow while planning ARCache for n={n}, d={d}"
    ))
}

fn normalize_variance(raw_var: f64) -> f64 {
    if raw_var.is_nan() || raw_var <= VAR_FLOOR {
        VAR_FLOOR
    } else if raw_var == f64::INFINITY {
        f64::MAX
    } else {
        raw_var
    }
}

fn lag_variance_tolerance(sum_x: f64, sum_x_sq: f64, m: f64) -> f64 {
    let cross = if m > 0.0 { (sum_x * sum_x) / m } else { 0.0 };
    let scale = sum_x_sq.abs().max(cross.abs()).max(1.0);
    32.0 * f64::EPSILON * scale
}

fn ar1_residual_sse(
    sum_x: f64,
    sum_x_sq: f64,
    sum_y: f64,
    sum_y_sq: f64,
    sum_xy: f64,
    m: f64,
) -> f64 {
    if m <= 1.0 {
        return 0.0;
    }

    let sxx = sum_x_sq - (sum_x * sum_x) / m;
    let syy = (sum_y_sq - (sum_y * sum_y) / m).max(0.0);
    let tol = lag_variance_tolerance(sum_x, sum_x_sq, m);

    if sxx <= tol {
        return syy;
    }

    let sxy = sum_xy - (sum_x * sum_y) / m;
    let explained = (sxy * sxy) / sxx;
    (syy - explained).max(0.0)
}

impl CostModel for CostAR {
    type Cache = ARCache;

    fn name(&self) -> &'static str {
        "ar"
    }

    fn penalty_params_per_segment(&self) -> usize {
        self.order.saturating_add(2)
    }

    fn validate(&self, x: &TimeSeriesView<'_>) -> Result<(), CpdError> {
        if self.order == 0 {
            return Err(CpdError::invalid_input(
                "CostAR requires order >= 1; got order=0",
            ));
        }
        if self.order != 1 {
            return Err(CpdError::not_supported(format!(
                "CostAR currently supports order=1 only; got order={}",
                self.order
            )));
        }
        if x.n == 0 {
            return Err(CpdError::invalid_input("CostAR requires n >= 1; got n=0"));
        }
        if x.d == 0 {
            return Err(CpdError::invalid_input("CostAR requires d >= 1; got d=0"));
        }
        if x.has_missing() {
            return Err(CpdError::invalid_input(format!(
                "CostAR does not support missing values: effective_missing_count={}",
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
                "CostAR does not support CachePolicy::Approximate",
            ));
        }
        if required_bytes == usize::MAX {
            return Err(cache_overflow_err(x.n, x.d));
        }
        if let CachePolicy::Budgeted { max_bytes } = policy
            && required_bytes > *max_bytes
        {
            return Err(CpdError::resource_limit(format!(
                "CostAR cache requires {} bytes, exceeds budget {} bytes",
                required_bytes, max_bytes
            )));
        }

        let prefix_len_per_dim =
            x.n.checked_add(1)
                .ok_or_else(|| cache_overflow_err(x.n, x.d))?;
        let total_prefix_len = prefix_len_per_dim
            .checked_mul(x.d)
            .ok_or_else(|| cache_overflow_err(x.n, x.d))?;

        let mut prefix_sum = Vec::with_capacity(total_prefix_len);
        let mut prefix_sum_sq = Vec::with_capacity(total_prefix_len);
        let mut prefix_lag_prod = Vec::with_capacity(total_prefix_len);

        for dim in 0..x.d {
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

            let mut lag_products = Vec::with_capacity(x.n);
            lag_products.push(0.0);
            for t in 1..x.n {
                lag_products.push(series[t] * series[t - 1]);
            }
            let dim_prefix_lag_prod = if matches!(self.repro_mode, ReproMode::Strict) {
                prefix_sums_kahan(&lag_products)
            } else {
                prefix_sums(&lag_products)
            };

            debug_assert_eq!(dim_prefix_sum.len(), prefix_len_per_dim);
            debug_assert_eq!(dim_prefix_sum_sq.len(), prefix_len_per_dim);
            debug_assert_eq!(dim_prefix_lag_prod.len(), prefix_len_per_dim);

            prefix_sum.extend_from_slice(&dim_prefix_sum);
            prefix_sum_sq.extend_from_slice(&dim_prefix_sum_sq);
            prefix_lag_prod.extend_from_slice(&dim_prefix_lag_prod);
        }

        Ok(ARCache {
            prefix_sum,
            prefix_sum_sq,
            prefix_lag_prod,
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
        let bytes_per_array = match total_prefix_len.checked_mul(std::mem::size_of::<f64>()) {
            Some(v) => v,
            None => return usize::MAX,
        };

        match bytes_per_array.checked_mul(3) {
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

        let segment_len = end - start;
        if segment_len <= self.order.saturating_add(1) {
            return 0.0;
        }
        debug_assert_eq!(self.order, 1);

        let n_residuals = (segment_len - self.order) as f64;
        let mut total = 0.0;

        for dim in 0..cache.d {
            let base = cache.dim_offset(dim);
            let sum_y = cache.prefix_sum[base + end] - cache.prefix_sum[base + start + self.order];
            let sum_y_sq =
                cache.prefix_sum_sq[base + end] - cache.prefix_sum_sq[base + start + self.order];

            let lag_end = end - self.order;
            let sum_x = cache.prefix_sum[base + lag_end] - cache.prefix_sum[base + start];
            let sum_x_sq = cache.prefix_sum_sq[base + lag_end] - cache.prefix_sum_sq[base + start];

            let sum_xy = cache.prefix_lag_prod[base + end]
                - cache.prefix_lag_prod[base + start + self.order];

            let sse = ar1_residual_sse(sum_x, sum_x_sq, sum_y, sum_y_sq, sum_xy, n_residuals);
            let residual_var = normalize_variance(sse / n_residuals);
            total += n_residuals * residual_var.ln();
        }

        total
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ARCache, CostAR, ar1_residual_sse, cache_overflow_err, normalize_variance, read_value,
        strided_linear_index,
    };
    use crate::model::CostModel;
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

    fn naive_ar1_univariate(values: &[f64], start: usize, end: usize) -> f64 {
        let segment_len = end - start;
        if segment_len <= 2 {
            return 0.0;
        }

        let mut sum_x = 0.0;
        let mut sum_x_sq = 0.0;
        let mut sum_y = 0.0;
        let mut sum_y_sq = 0.0;
        let mut sum_xy = 0.0;

        for t in (start + 1)..end {
            let x = values[t - 1];
            let y = values[t];
            sum_x += x;
            sum_x_sq += x * x;
            sum_y += y;
            sum_y_sq += y * y;
            sum_xy += x * y;
        }

        let m = (segment_len - 1) as f64;
        let sse = ar1_residual_sse(sum_x, sum_x_sq, sum_y, sum_y_sq, sum_xy, m);
        let var = normalize_variance(sse / m);
        m * var.ln()
    }

    fn synthetic_univariate_values(n: usize, breakpoint: usize) -> Vec<f64> {
        let phi = 0.8;
        let mut values = Vec::with_capacity(n);
        values.push(0.0);
        for t in 1..n {
            let mu = if t < breakpoint { 0.0 } else { 4.0 };
            let prev = values[t - 1];
            let eps = 0.15 * (0.17 * t as f64).sin();
            values.push(mu + phi * (prev - mu) + eps);
        }
        values
    }

    fn synthetic_multivariate_values(n: usize, d: usize) -> Vec<f64> {
        let mut prev = vec![0.0; d];
        let mut out = Vec::with_capacity(n * d);
        for t in 0..n {
            for (dim, prev_dim) in prev.iter_mut().enumerate() {
                let phi = 0.6 + 0.02 * dim as f64;
                let mu = if t < n / 2 {
                    dim as f64 * 0.3
                } else {
                    3.0 + dim as f64 * 0.3
                };
                let eps = 0.07 * (0.11 * t as f64 + dim as f64 * 0.3).sin();
                let value = mu + phi * (*prev_dim - mu) + eps;
                *prev_dim = value;
                out.push(value);
            }
        }
        out
    }

    fn dim_series(values: &[f64], n: usize, d: usize, dim: usize) -> Vec<f64> {
        (0..n).map(|t| values[t * d + dim]).collect()
    }

    fn lcg_next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state
    }

    #[test]
    fn strided_linear_index_reports_overflow_and_negative_paths() {
        let err_t = strided_linear_index(usize::MAX, 0, 1, 1, 8).expect_err("t overflow expected");
        assert!(matches!(err_t, CpdError::InvalidInput(_)));
        assert!(err_t.to_string().contains("t="));

        let err_dim =
            strided_linear_index(0, usize::MAX, 1, 1, 8).expect_err("dim overflow expected");
        assert!(matches!(err_dim, CpdError::InvalidInput(_)));
        assert!(err_dim.to_string().contains("dim="));

        let err_mul = strided_linear_index(isize::MAX as usize, 0, 2, 0, 8)
            .expect_err("checked_mul overflow expected");
        assert!(matches!(err_mul, CpdError::InvalidInput(_)));
        assert!(err_mul.to_string().contains("overflow"));

        let err_neg = strided_linear_index(1, 0, -1, 0, 8).expect_err("negative index expected");
        assert!(matches!(err_neg, CpdError::InvalidInput(_)));
        assert!(err_neg.to_string().contains("negative"));
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
        let c_overflow = read_value(&c_view, usize::MAX, 0).expect_err("C f32 overflow expected");
        assert!(c_overflow.to_string().contains("overflow"));

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
        let f_overflow = read_value(&f_view, 0, usize::MAX).expect_err("F f32 overflow expected");
        assert!(f_overflow.to_string().contains("overflow"));

        let s_values = [1.5_f32, 10.5, 2.5, 20.5];
        let s_view = make_f32_view(
            &s_values,
            2,
            2,
            MemoryLayout::Strided {
                row_stride: 2,
                col_stride: 1,
            },
            MissingPolicy::Error,
        );
        assert_close(
            read_value(&s_view, 1, 1).expect("strided f32 read should succeed"),
            20.5,
            1e-12,
        );
    }

    #[test]
    fn cache_overflow_error_message_is_stable() {
        let err = cache_overflow_err(7, 11);
        assert!(matches!(err, CpdError::ResourceLimit(_)));
        assert!(err.to_string().contains("n=7, d=11"));
    }

    #[test]
    fn trait_contract_and_defaults() {
        let model = CostAR::default();
        assert_eq!(model.name(), "ar");
        assert_eq!(model.order, 1);
        assert_eq!(model.repro_mode, ReproMode::Balanced);
        assert_eq!(model.penalty_params_per_segment(), 3);
        assert_eq!(model.missing_support(), MissingSupport::Reject);
        assert!(!model.supports_approx_cache());
    }

    #[test]
    fn validate_rejects_invalid_order_and_missing_values() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let clean = make_f64_view(
            &values,
            4,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let zero_order = CostAR::new(0, ReproMode::Balanced)
            .validate(&clean)
            .expect_err("order=0 should fail");
        assert!(matches!(zero_order, CpdError::InvalidInput(_)));
        assert!(zero_order.to_string().contains("order >= 1"));

        let ar2 = CostAR::new(2, ReproMode::Balanced)
            .validate(&clean)
            .expect_err("order=2 should fail");
        assert!(matches!(ar2, CpdError::NotSupported(_)));
        assert!(ar2.to_string().contains("order=1 only"));

        let with_missing = [1.0, f64::NAN, 3.0, 4.0];
        let missing_view = make_f64_view(
            &with_missing,
            4,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Ignore,
        );
        let err = CostAR::default()
            .validate(&missing_view)
            .expect_err("missing values should fail");
        assert!(matches!(err, CpdError::InvalidInput(_)));
        assert!(err.to_string().contains("missing values"));
    }

    #[test]
    fn known_answer_piecewise_mean_shift_prefers_true_split() {
        let model = CostAR::default();
        let n = 200;
        let breakpoint = 100;
        let values = synthetic_univariate_values(n, breakpoint);
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

        let full = model.segment_cost(&cache, 0, n);
        let split_true =
            model.segment_cost(&cache, 0, breakpoint) + model.segment_cost(&cache, breakpoint, n);
        let split_wrong = model.segment_cost(&cache, 0, breakpoint + 20)
            + model.segment_cost(&cache, breakpoint + 20, n);

        assert!(split_true < full - 1.0);
        assert!(split_true < split_wrong - 0.1);
    }

    #[test]
    fn segment_cost_matches_naive_on_deterministic_queries() {
        let model = CostAR::default();
        let n = 256;
        let values = synthetic_univariate_values(n, 128);
        let view = make_f64_view(
            &values,
            n,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let cache = model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed");

        let mut state = 0x4a32_1120_b0fe_cafe_u64;
        for _ in 0..600 {
            let a = (lcg_next(&mut state) as usize) % n;
            let b = (lcg_next(&mut state) as usize) % n;
            let start = a.min(b);
            let mut end = a.max(b) + 1;
            if start == end {
                end = (start + 1).min(n);
            }

            let fast = model.segment_cost(&cache, start, end);
            let naive = naive_ar1_univariate(&values, start, end);
            assert_close(fast, naive, 1e-6);
        }
    }

    #[test]
    fn multivariate_matches_univariate_sum_for_d1_d4_d8() {
        let model = CostAR::default();
        let n = 80;
        let start = 8;
        let end = 72;

        for d in [1_usize, 4, 8] {
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

            assert_close(multivariate, per_dimension_sum, 1e-9);
        }
    }

    #[test]
    fn layout_coverage_c_f_and_strided() {
        let model = CostAR::default();
        let start = 1;
        let end = 6;

        let c_values = vec![
            1.0, 11.0, 1.5, 11.5, 2.0, 12.0, 2.5, 12.5, 3.0, 13.0, 3.5, 13.5,
        ];
        let c_view = make_f64_view(
            &c_values,
            6,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let c_cache = model
            .precompute(&c_view, &CachePolicy::Full)
            .expect("C precompute should succeed");
        let c_cost = model.segment_cost(&c_cache, start, end);

        let f_values = vec![
            1.0, 1.5, 2.0, 2.5, 3.0, 3.5, // dim 0
            11.0, 11.5, 12.0, 12.5, 13.0, 13.5, // dim 1
        ];
        let f_view = make_f64_view(
            &f_values,
            6,
            2,
            MemoryLayout::FContiguous,
            MissingPolicy::Error,
        );
        let f_cache = model
            .precompute(&f_view, &CachePolicy::Full)
            .expect("F precompute should succeed");
        let f_cost = model.segment_cost(&f_cache, start, end);
        assert_close(f_cost, c_cost, 1e-12);

        let s_values = vec![
            1.0, 1.5, 2.0, 2.5, 3.0, 3.5, // dim 0
            11.0, 11.5, 12.0, 12.5, 13.0, 13.5, // dim 1
        ];
        let s_view = make_f64_view(
            &s_values,
            6,
            2,
            MemoryLayout::Strided {
                row_stride: 1,
                col_stride: 6,
            },
            MissingPolicy::Error,
        );
        let s_cache = model
            .precompute(&s_view, &CachePolicy::Full)
            .expect("strided precompute should succeed");
        let s_cost = model.segment_cost(&s_cache, start, end);
        assert_close(s_cost, c_cost, 1e-12);
    }

    #[test]
    fn precompute_respects_budget() {
        let model = CostAR::default();
        let values = synthetic_multivariate_values(16, 3);
        let view = make_f64_view(
            &values,
            16,
            3,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let required = model.worst_case_cache_bytes(&view);
        let tiny_budget = required.saturating_sub(1);
        let err = model
            .precompute(
                &view,
                &CachePolicy::Budgeted {
                    max_bytes: tiny_budget,
                },
            )
            .expect_err("insufficient budget should fail");
        assert!(matches!(err, CpdError::ResourceLimit(_)));
        assert!(err.to_string().contains("exceeds budget"));
    }

    #[test]
    fn approx_cache_policy_is_rejected() {
        let model = CostAR::default();
        let values = synthetic_multivariate_values(12, 2);
        let view = make_f64_view(
            &values,
            12,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let err = model
            .precompute(
                &view,
                &CachePolicy::Approximate {
                    max_bytes: model.worst_case_cache_bytes(&view),
                    error_tolerance: 0.05,
                },
            )
            .expect_err("approx policy should be rejected");
        assert!(matches!(err, CpdError::NotSupported(_)));
    }

    #[test]
    fn short_segments_return_zero() {
        let model = CostAR::default();
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
        assert_eq!(model.segment_cost(&cache, 0, 1), 0.0);
        assert_eq!(model.segment_cost(&cache, 1, 3), 0.0);
    }

    #[test]
    #[should_panic(expected = "segment_cost requires start < end")]
    fn segment_cost_panics_when_start_ge_end() {
        let model = CostAR::default();
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
        let _ = model.segment_cost(&cache, 2, 2);
    }

    #[test]
    #[should_panic(expected = "segment_cost end out of bounds")]
    fn segment_cost_panics_when_end_exceeds_n() {
        let model = CostAR::default();
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
        let _ = model.segment_cost(&cache, 0, 4);
    }

    #[test]
    fn cache_from_parts_roundtrip_is_stable() {
        let model = CostAR::default();
        let values = synthetic_univariate_values(24, 12);
        let view = make_f64_view(
            &values,
            24,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let cache = model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed");
        let rebuilt = ARCache {
            prefix_sum: cache.prefix_sum.clone(),
            prefix_sum_sq: cache.prefix_sum_sq.clone(),
            prefix_lag_prod: cache.prefix_lag_prod.clone(),
            n: cache.n,
            d: cache.d,
        };
        assert_eq!(cache, rebuilt);
    }
}
