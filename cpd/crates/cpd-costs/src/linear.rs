// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::model::CostModel;
use cpd_core::{
    CachePolicy, CpdError, DTypeView, MemoryLayout, MissingSupport, ReproMode, TimeSeriesView,
    prefix_sum_squares, prefix_sum_squares_kahan, prefix_sums, prefix_sums_kahan,
};

/// Piecewise-linear least-squares segment cost model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CostLinear {
    pub repro_mode: ReproMode,
}

impl CostLinear {
    pub const fn new(repro_mode: ReproMode) -> Self {
        Self { repro_mode }
    }
}

impl Default for CostLinear {
    fn default() -> Self {
        Self::new(ReproMode::Balanced)
    }
}

/// Prefix-stat cache for O(1) linear-trend segment-cost queries.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearCache {
    prefix_t: Vec<f64>,
    prefix_t_sq: Vec<f64>,
    prefix_x: Vec<f64>,
    prefix_x_sq: Vec<f64>,
    prefix_t_x: Vec<f64>,
    n: usize,
    d: usize,
}

impl LinearCache {
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
        "cache size overflow while planning LinearCache for n={n}, d={d}"
    ))
}

fn time_variance_tolerance(sum_t: f64, sum_t_sq: f64, m: f64) -> f64 {
    let cross = if m > 0.0 { (sum_t * sum_t) / m } else { 0.0 };
    let scale = sum_t_sq.abs().max(cross.abs()).max(1.0);
    32.0 * f64::EPSILON * scale
}

fn mean_only_sse(sum_x: f64, sum_x_sq: f64, m: f64) -> f64 {
    let centered = sum_x_sq - (sum_x * sum_x) / m;
    centered.max(0.0)
}

impl CostModel for CostLinear {
    type Cache = LinearCache;

    fn name(&self) -> &'static str {
        "linear"
    }

    fn penalty_params_per_segment(&self) -> usize {
        3
    }

    fn validate(&self, x: &TimeSeriesView<'_>) -> Result<(), CpdError> {
        if x.n == 0 {
            return Err(CpdError::invalid_input(
                "CostLinear requires n >= 1; got n=0",
            ));
        }
        if x.d == 0 {
            return Err(CpdError::invalid_input(
                "CostLinear requires d >= 1; got d=0",
            ));
        }
        if x.has_missing() {
            return Err(CpdError::invalid_input(format!(
                "CostLinear does not support missing values: effective_missing_count={}",
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
                "CostLinear does not support CachePolicy::Approximate",
            ));
        }

        if required_bytes == usize::MAX {
            return Err(cache_overflow_err(x.n, x.d));
        }

        if let CachePolicy::Budgeted { max_bytes } = policy
            && required_bytes > *max_bytes
        {
            return Err(CpdError::resource_limit(format!(
                "CostLinear cache requires {} bytes, exceeds budget {} bytes",
                required_bytes, max_bytes
            )));
        }

        let prefix_len_per_dim =
            x.n.checked_add(1)
                .ok_or_else(|| cache_overflow_err(x.n, x.d))?;
        let total_prefix_len = prefix_len_per_dim
            .checked_mul(x.d)
            .ok_or_else(|| cache_overflow_err(x.n, x.d))?;

        let mut t_values = Vec::with_capacity(x.n);
        for t in 0..x.n {
            t_values.push(t as f64);
        }

        let prefix_t = if matches!(self.repro_mode, ReproMode::Strict) {
            prefix_sums_kahan(&t_values)
        } else {
            prefix_sums(&t_values)
        };

        let t_sq_values: Vec<f64> = t_values.iter().map(|t| t * t).collect();
        let prefix_t_sq = if matches!(self.repro_mode, ReproMode::Strict) {
            prefix_sums_kahan(&t_sq_values)
        } else {
            prefix_sums(&t_sq_values)
        };

        let mut prefix_x = Vec::with_capacity(total_prefix_len);
        let mut prefix_x_sq = Vec::with_capacity(total_prefix_len);
        let mut prefix_t_x = Vec::with_capacity(total_prefix_len);

        for dim in 0..x.d {
            let mut series = Vec::with_capacity(x.n);
            let mut series_t_x = Vec::with_capacity(x.n);

            for (t, t_value) in t_values.iter().copied().enumerate() {
                let value = read_value(x, t, dim)?;
                series.push(value);
                series_t_x.push(t_value * value);
            }

            let dim_prefix_x = if matches!(self.repro_mode, ReproMode::Strict) {
                prefix_sums_kahan(&series)
            } else {
                prefix_sums(&series)
            };

            let dim_prefix_x_sq = if matches!(self.repro_mode, ReproMode::Strict) {
                prefix_sum_squares_kahan(&series)
            } else {
                prefix_sum_squares(&series)
            };

            let dim_prefix_t_x = if matches!(self.repro_mode, ReproMode::Strict) {
                prefix_sums_kahan(&series_t_x)
            } else {
                prefix_sums(&series_t_x)
            };

            debug_assert_eq!(dim_prefix_x.len(), prefix_len_per_dim);
            debug_assert_eq!(dim_prefix_x_sq.len(), prefix_len_per_dim);
            debug_assert_eq!(dim_prefix_t_x.len(), prefix_len_per_dim);

            prefix_x.extend_from_slice(&dim_prefix_x);
            prefix_x_sq.extend_from_slice(&dim_prefix_x_sq);
            prefix_t_x.extend_from_slice(&dim_prefix_t_x);
        }

        Ok(LinearCache {
            prefix_t,
            prefix_t_sq,
            prefix_x,
            prefix_x_sq,
            prefix_t_x,
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
        let bytes_per_array_global =
            match prefix_len_per_dim.checked_mul(std::mem::size_of::<f64>()) {
                Some(v) => v,
                None => return usize::MAX,
            };
        let bytes_per_array_dim = match total_prefix_len.checked_mul(std::mem::size_of::<f64>()) {
            Some(v) => v,
            None => return usize::MAX,
        };
        let global_bytes = match bytes_per_array_global.checked_mul(2) {
            Some(v) => v,
            None => return usize::MAX,
        };
        let dim_bytes = match bytes_per_array_dim.checked_mul(3) {
            Some(v) => v,
            None => return usize::MAX,
        };

        match global_bytes.checked_add(dim_bytes) {
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

        let m = (end - start) as f64;
        if m <= 1.0 {
            return 0.0;
        }

        let sum_t = cache.prefix_t[end] - cache.prefix_t[start];
        let sum_t_sq = cache.prefix_t_sq[end] - cache.prefix_t_sq[start];
        let time_centered_ss = sum_t_sq - (sum_t * sum_t) / m;
        let time_tol = time_variance_tolerance(sum_t, sum_t_sq, m);

        let mut total = 0.0;
        for dim in 0..cache.d {
            let base = cache.dim_offset(dim);
            let sum_x = cache.prefix_x[base + end] - cache.prefix_x[base + start];
            let sum_x_sq = cache.prefix_x_sq[base + end] - cache.prefix_x_sq[base + start];
            let base_sse = mean_only_sse(sum_x, sum_x_sq, m);

            if time_centered_ss <= time_tol {
                total += base_sse;
                continue;
            }

            let sum_t_x = cache.prefix_t_x[base + end] - cache.prefix_t_x[base + start];
            let cov_t_x = sum_t_x - (sum_t * sum_x) / m;
            let explained = (cov_t_x * cov_t_x) / time_centered_ss;
            let sse = base_sse - explained;
            total += sse.max(0.0);
        }

        total.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::{CostLinear, LinearCache, cache_overflow_err, read_value, strided_linear_index};
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

    fn naive_linear_univariate(values: &[f64], start: usize, end: usize) -> f64 {
        let m = end - start;
        if m <= 1 {
            return 0.0;
        }

        let mut sum_t = 0.0;
        let mut sum_t_sq = 0.0;
        let mut sum_x = 0.0;
        let mut sum_t_x = 0.0;
        for (offset, value) in values[start..end].iter().copied().enumerate() {
            let t = (start + offset) as f64;
            sum_t += t;
            sum_t_sq += t * t;
            sum_x += value;
            sum_t_x += t * value;
        }

        let m_f = m as f64;
        let denom = m_f * sum_t_sq - sum_t * sum_t;
        if denom <= 1e-15 {
            let mean = sum_x / m_f;
            return values[start..end]
                .iter()
                .map(|v| {
                    let diff = *v - mean;
                    diff * diff
                })
                .sum();
        }

        let slope = (m_f * sum_t_x - sum_t * sum_x) / denom;
        let intercept = (sum_x - slope * sum_t) / m_f;
        values[start..end]
            .iter()
            .enumerate()
            .map(|(offset, value)| {
                let t = (start + offset) as f64;
                let resid = *value - (intercept + slope * t);
                resid * resid
            })
            .sum()
    }

    fn synthetic_multivariate_values(n: usize, d: usize) -> Vec<f64> {
        let mut values = Vec::with_capacity(n * d);
        for t in 0..n {
            let t_f = t as f64;
            for dim in 0..d {
                let dim_f = dim as f64;
                let y = (0.5 + 0.2 * dim_f) * t_f + (2.0 * dim_f) + (0.13 * t_f).sin();
                values.push(y);
            }
        }
        values
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
        let model = CostLinear::default();
        assert_eq!(model.name(), "linear");
        assert_eq!(model.repro_mode, ReproMode::Balanced);
        assert_eq!(model.missing_support(), MissingSupport::Reject);
        assert_eq!(model.penalty_params_per_segment(), 3);
        assert!(!model.supports_approx_cache());
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
        let err = CostLinear::default()
            .validate(&view)
            .expect_err("missing values should be rejected");
        assert!(matches!(err, CpdError::InvalidInput(_)));
        assert!(err.to_string().contains("missing values"));
    }

    #[test]
    fn known_answer_piecewise_linear_breakpoint_gain() {
        let model = CostLinear::default();
        let n = 100;
        let breakpoint = 50;
        let values: Vec<f64> = (0..n)
            .map(|t| {
                if t < breakpoint {
                    t as f64
                } else {
                    (2 * breakpoint) as f64 - t as f64
                }
            })
            .collect();

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

        let full = model.segment_cost(&cache, 0, values.len());
        let split_true =
            model.segment_cost(&cache, 0, breakpoint) + model.segment_cost(&cache, breakpoint, n);
        let split_wrong = model.segment_cost(&cache, 0, 60) + model.segment_cost(&cache, 60, n);

        assert!(
            full > 1_000.0,
            "full fit should have large residual, got {full}"
        );
        assert_close(split_true, 0.0, 1e-9);
        assert!(split_wrong > split_true + 10.0);
    }

    #[test]
    fn segment_cost_matches_naive_on_deterministic_queries() {
        let model = CostLinear::default();
        let n = 256;
        let values: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64;
                1.5 + 0.7 * x + 0.3 * x.sin() + (i % 7) as f64 * 1e-3
            })
            .collect();
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

        let mut state = 0x7654_3210_1234_5678_u64;
        for _ in 0..600 {
            let a = (lcg_next(&mut state) as usize) % n;
            let b = (lcg_next(&mut state) as usize) % n;
            let start = a.min(b);
            let mut end = a.max(b) + 1;
            if start == end {
                end = (start + 1).min(n);
            }

            let fast = model.segment_cost(&cache, start, end);
            let naive = naive_linear_univariate(&values, start, end);
            assert_close(fast, naive, 1e-8);
        }
    }

    #[test]
    fn multivariate_matches_univariate_sum_for_d1_d4_d16() {
        let model = CostLinear::default();
        let n = 64;
        let start = 7;
        let end = 56;

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

            assert_close(multivariate, per_dimension_sum, 1e-8);
        }
    }

    #[test]
    fn layout_coverage_c_f_and_strided() {
        let model = CostLinear::default();

        let c_values = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        let c_view = make_f64_view(
            &c_values,
            4,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let c_cache = model
            .precompute(&c_view, &CachePolicy::Full)
            .expect("C precompute should succeed");

        let f_values = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let f_view = make_f64_view(
            &f_values,
            4,
            2,
            MemoryLayout::FContiguous,
            MissingPolicy::Error,
        );
        let f_cache = model
            .precompute(&f_view, &CachePolicy::Full)
            .expect("F precompute should succeed");

        let c_cost = model.segment_cost(&c_cache, 0, 4);
        let f_cost = model.segment_cost(&f_cache, 0, 4);
        assert_close(c_cost, f_cost, 1e-12);

        let strided_view = make_f64_view(
            &c_values,
            4,
            2,
            MemoryLayout::Strided {
                row_stride: 2,
                col_stride: 1,
            },
            MissingPolicy::Error,
        );
        let strided_cache = model
            .precompute(&strided_view, &CachePolicy::Full)
            .expect("strided precompute should succeed");
        let strided_cost = model.segment_cost(&strided_cache, 0, 4);
        assert_close(c_cost, strided_cost, 1e-12);
    }

    #[test]
    fn cache_policy_behavior_budgeted_and_approximate() {
        let model = CostLinear::default();
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
        let model = CostLinear::default();
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
            (2 * prefix_len_per_dim + 3 * total_prefix_len) * std::mem::size_of::<f64>();
        assert_eq!(model.worst_case_cache_bytes(&view), expected_small);
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

        let balanced = CostLinear::new(ReproMode::Balanced)
            .precompute(&view, &CachePolicy::Full)
            .expect("balanced precompute should succeed");
        let strict = CostLinear::new(ReproMode::Strict)
            .precompute(&view, &CachePolicy::Full)
            .expect("strict precompute should succeed");

        let balanced_final = balanced.prefix_x[balanced.prefix_len_per_dim() - 1];
        let strict_final = strict.prefix_x[strict.prefix_len_per_dim() - 1];
        assert_eq!(balanced_final, 0.0);
        assert_eq!(strict_final, 1.0);
    }

    #[test]
    #[should_panic(expected = "start < end")]
    fn segment_cost_panics_when_start_ge_end() {
        let model = CostLinear::default();
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let view = make_f64_view(
            &values,
            4,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let cache: LinearCache = model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed");

        let _ = model.segment_cost(&cache, 2, 2);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn segment_cost_panics_when_end_exceeds_n() {
        let model = CostLinear::default();
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
