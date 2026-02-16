// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::model::CostModel;
use cpd_core::{
    CachePolicy, CpdError, DTypeView, MemoryLayout, MissingSupport, ReproMode, TimeSeriesView,
    prefix_sums, prefix_sums_kahan,
};

const INTEGER_TOL: f64 = 1e-9;

/// Poisson segment cost model for count/rate changes.
///
/// Segment conventions use half-open intervals: `[start, end)`.
///
/// The returned value uses the negative log-likelihood with MLE rate and omits
/// the per-observation `log(c_i!)` additive constants, which are independent of
/// the segmentation objective.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CostPoissonRate {
    pub repro_mode: ReproMode,
}

impl CostPoissonRate {
    pub const fn new(repro_mode: ReproMode) -> Self {
        Self { repro_mode }
    }
}

impl Default for CostPoissonRate {
    fn default() -> Self {
        Self::new(ReproMode::Balanced)
    }
}

/// Prefix-stat cache for O(1) Poisson segment-cost queries.
#[derive(Clone, Debug, PartialEq)]
pub struct PoissonCache {
    prefix_sum: Vec<f64>,
    n: usize,
    d: usize,
}

impl PoissonCache {
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

fn normalize_count(value: f64, t: usize, dim: usize) -> Result<f64, CpdError> {
    if !value.is_finite() || value < 0.0 {
        return Err(CpdError::invalid_input(format!(
            "CostPoissonRate requires non-negative finite counts; got value={value} at t={t}, dim={dim}"
        )));
    }

    let rounded = value.round();
    if (rounded - value).abs() > INTEGER_TOL {
        return Err(CpdError::invalid_input(format!(
            "CostPoissonRate requires integer-valued counts; got value={value} at t={t}, dim={dim}"
        )));
    }

    Ok(rounded)
}

fn read_count(x: &TimeSeriesView<'_>, t: usize, dim: usize) -> Result<f64, CpdError> {
    normalize_count(read_value(x, t, dim)?, t, dim)
}

fn cache_overflow_err(n: usize, d: usize) -> CpdError {
    CpdError::resource_limit(format!(
        "cache size overflow while planning PoissonCache for n={n}, d={d}"
    ))
}

impl CostModel for CostPoissonRate {
    type Cache = PoissonCache;

    fn name(&self) -> &'static str {
        "poisson_rate"
    }

    fn validate(&self, x: &TimeSeriesView<'_>) -> Result<(), CpdError> {
        if x.n == 0 {
            return Err(CpdError::invalid_input(
                "CostPoissonRate requires n >= 1; got n=0",
            ));
        }
        if x.d == 0 {
            return Err(CpdError::invalid_input(
                "CostPoissonRate requires d >= 1; got d=0",
            ));
        }

        if x.has_missing() {
            return Err(CpdError::invalid_input(format!(
                "CostPoissonRate does not support missing values: effective_missing_count={}",
                x.n_missing()
            )));
        }

        for dim in 0..x.d {
            for t in 0..x.n {
                let _ = read_count(x, t, dim)?;
            }
        }

        Ok(())
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
                "CostPoissonRate does not support CachePolicy::Approximate",
            ));
        }

        if required_bytes == usize::MAX {
            return Err(cache_overflow_err(x.n, x.d));
        }

        if let CachePolicy::Budgeted { max_bytes } = policy
            && required_bytes > *max_bytes
        {
            return Err(CpdError::resource_limit(format!(
                "CostPoissonRate cache requires {} bytes, exceeds budget {} bytes",
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

        for dim in 0..x.d {
            let mut series = Vec::with_capacity(x.n);
            for t in 0..x.n {
                series.push(read_count(x, t, dim)?);
            }

            let dim_prefix_sum = if matches!(self.repro_mode, ReproMode::Strict) {
                prefix_sums_kahan(&series)
            } else {
                prefix_sums(&series)
            };

            debug_assert_eq!(dim_prefix_sum.len(), prefix_len_per_dim);
            prefix_sum.extend_from_slice(&dim_prefix_sum);
        }

        Ok(PoissonCache {
            prefix_sum,
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

        match total_prefix_len.checked_mul(std::mem::size_of::<f64>()) {
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
        let mut total = 0.0;

        for dim in 0..cache.d {
            let base = cache.dim_offset(dim);
            let sum = cache.prefix_sum[base + end] - cache.prefix_sum[base + start];
            if sum <= 0.0 {
                continue;
            }
            let lambda = sum / m;
            total += sum - sum * lambda.ln();
        }

        total
    }
}

#[cfg(test)]
mod tests {
    use super::{CostPoissonRate, INTEGER_TOL, PoissonCache, cache_overflow_err, read_value};
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

    fn naive_poisson_univariate(values: &[f64], start: usize, end: usize) -> f64 {
        let segment = &values[start..end];
        let sum: f64 = segment.iter().sum();
        if sum <= 0.0 {
            return 0.0;
        }
        let m = segment.len() as f64;
        let lambda = sum / m;
        sum - sum * lambda.ln()
    }

    fn synthetic_multivariate_counts(n: usize, d: usize) -> Vec<f64> {
        let mut values = Vec::with_capacity(n * d);
        for t in 0..n {
            for dim in 0..d {
                values.push(((t * (dim + 1)) % 17) as f64);
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
    fn trait_contract_and_defaults() {
        let model = CostPoissonRate::default();
        assert_eq!(model.name(), "poisson_rate");
        assert_eq!(model.repro_mode, ReproMode::Balanced);
        assert_eq!(model.missing_support(), MissingSupport::Reject);
        assert!(!model.supports_approx_cache());
    }

    #[test]
    fn cache_overflow_error_message_is_stable() {
        let err = cache_overflow_err(7, 11);
        assert!(matches!(err, CpdError::ResourceLimit(_)));
        assert!(err.to_string().contains("n=7, d=11"));
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
        let err = CostPoissonRate::default()
            .validate(&view)
            .expect_err("missing values should be rejected");
        assert!(matches!(err, CpdError::InvalidInput(_)));
        assert!(err.to_string().contains("missing values"));
    }

    #[test]
    fn validate_rejects_negative_and_non_integer_values() {
        let negative = [0.0, -1.0, 2.0, 3.0];
        let negative_view = make_f64_view(
            &negative,
            4,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let negative_err = CostPoissonRate::default()
            .validate(&negative_view)
            .expect_err("negative counts should fail");
        assert!(negative_err.to_string().contains("non-negative"));

        let fractional = [0.0, 1.25, 2.0, 3.0];
        let fractional_view = make_f64_view(
            &fractional,
            4,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let fractional_err = CostPoissonRate::default()
            .validate(&fractional_view)
            .expect_err("fractional counts should fail");
        assert!(fractional_err.to_string().contains("integer-valued"));
    }

    #[test]
    fn validate_accepts_integer_valued_f32_and_f64() {
        let model = CostPoissonRate::default();

        let f32_values = [0.0_f32, 1.0, 2.0, 3.0];
        let f32_view = make_f32_view(
            &f32_values,
            4,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        model.validate(&f32_view).expect("f32 integer counts");

        let f64_values = [0.0_f64, 1.0, 2.0, 3.0];
        let f64_view = make_f64_view(
            &f64_values,
            4,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        model.validate(&f64_view).expect("f64 integer counts");
    }

    #[test]
    fn known_answer_univariate_and_zero_rate_segment() {
        let model = CostPoissonRate::default();
        let values = [0.0, 1.0, 2.0, 3.0];
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

        let full = model.segment_cost(&cache, 0, 4);
        assert_close(full, 6.0 - 6.0 * 1.5_f64.ln(), 1e-12);

        let sub = model.segment_cost(&cache, 1, 3);
        assert_close(sub, 3.0 - 3.0 * 1.5_f64.ln(), 1e-12);

        let zero_values = [0.0, 0.0, 0.0, 0.0];
        let zero_view = make_f64_view(
            &zero_values,
            4,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let zero_cache = model
            .precompute(&zero_view, &CachePolicy::Full)
            .expect("precompute should succeed");
        assert_close(model.segment_cost(&zero_cache, 0, 4), 0.0, 1e-15);
    }

    #[test]
    fn segment_cost_matches_naive_on_deterministic_queries() {
        let model = CostPoissonRate::default();
        let n = 256;
        let values: Vec<f64> = (0..n).map(|i| ((i * 7) % 23) as f64).collect();
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

        let mut state = 0x1020_3040_5060_7080_u64;
        for _ in 0..600 {
            let a = (lcg_next(&mut state) as usize) % n;
            let b = (lcg_next(&mut state) as usize) % n;
            let start = a.min(b);
            let mut end = a.max(b) + 1;
            if start == end {
                end = (start + 1).min(n);
            }

            let fast = model.segment_cost(&cache, start, end);
            let naive = naive_poisson_univariate(&values, start, end);
            assert_close(fast, naive, 1e-9);
        }
    }

    #[test]
    fn multivariate_matches_univariate_sum_for_d1_d4_d16() {
        let model = CostPoissonRate::default();
        let n = 9;
        let start = 2;
        let end = 8;

        for d in [1_usize, 4, 16] {
            let values = synthetic_multivariate_counts(n, d);
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

            assert_close(multivariate, per_dimension_sum, 1e-12);
        }
    }

    #[test]
    fn layout_coverage_c_f_and_strided() {
        let model = CostPoissonRate::default();

        let c_values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let c_view = make_f64_view(
            &c_values,
            3,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let c_cache = model
            .precompute(&c_view, &CachePolicy::Full)
            .expect("C precompute should succeed");

        let f_values = vec![0.0, 2.0, 4.0, 1.0, 3.0, 5.0];
        let f_view = make_f64_view(
            &f_values,
            3,
            2,
            MemoryLayout::FContiguous,
            MissingPolicy::Error,
        );
        let f_cache = model
            .precompute(&f_view, &CachePolicy::Full)
            .expect("F precompute should succeed");

        let s_values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let s_view = make_f64_view(
            &s_values,
            3,
            2,
            MemoryLayout::Strided {
                row_stride: 2,
                col_stride: 1,
            },
            MissingPolicy::Error,
        );
        let s_cache = model
            .precompute(&s_view, &CachePolicy::Full)
            .expect("strided precompute should succeed");

        assert_close(
            model.segment_cost(&c_cache, 0, 3),
            model.segment_cost(&f_cache, 0, 3),
            1e-12,
        );
        assert_close(
            model.segment_cost(&c_cache, 0, 3),
            model.segment_cost(&s_cache, 0, 3),
            1e-12,
        );
    }

    #[test]
    fn precompute_rejects_approximate_and_budget_exceeded() {
        let model = CostPoissonRate::default();
        let values = [0.0, 1.0, 2.0, 3.0, 4.0];
        let view = make_f64_view(
            &values,
            5,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let approx_err = model
            .precompute(&view, &CachePolicy::Approximate { max_bytes: 1024 })
            .expect_err("approximate cache should be rejected");
        assert!(matches!(approx_err, CpdError::NotSupported(_)));

        let required = model.worst_case_cache_bytes(&view);
        let budget_err = model
            .precompute(
                &view,
                &CachePolicy::Budgeted {
                    max_bytes: required.saturating_sub(1),
                },
            )
            .expect_err("insufficient budget should fail");
        assert!(matches!(budget_err, CpdError::ResourceLimit(_)));
    }

    #[test]
    fn worst_case_cache_bytes_matches_formula() {
        let model = CostPoissonRate::default();
        let values = vec![0.0; 16];
        let view = make_f64_view(
            &values,
            8,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let expected = (8 + 1) * 2 * std::mem::size_of::<f64>();
        assert_eq!(model.worst_case_cache_bytes(&view), expected);
    }

    #[test]
    fn read_value_f32_layout_paths_and_errors() {
        let c_values = [1.0_f32, 2.0, 3.0, 4.0];
        let c_view = make_f32_view(
            &c_values,
            2,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        assert_close(
            read_value(&c_view, 1, 0).expect("C f32 read should succeed"),
            3.0,
            1e-12,
        );
        let c_oob = read_value(&c_view, 1, 2).expect_err("C f32 oob expected");
        assert!(c_oob.to_string().contains("out of bounds"));

        let f_values = [1.0_f32, 3.0, 2.0, 4.0];
        let f_view = make_f32_view(
            &f_values,
            2,
            2,
            MemoryLayout::FContiguous,
            MissingPolicy::Error,
        );
        assert_close(
            read_value(&f_view, 1, 0).expect("F f32 read should succeed"),
            3.0,
            1e-12,
        );
    }

    #[test]
    fn segment_cost_panics_when_start_ge_end() {
        let model = CostPoissonRate::default();
        let cache = PoissonCache {
            prefix_sum: vec![0.0, 1.0, 3.0],
            n: 2,
            d: 1,
        };

        let panic = std::panic::catch_unwind(|| model.segment_cost(&cache, 1, 1));
        assert!(panic.is_err(), "expected panic for start >= end");
    }

    #[test]
    fn segment_cost_panics_when_end_exceeds_n() {
        let model = CostPoissonRate::default();
        let cache = PoissonCache {
            prefix_sum: vec![0.0, 1.0, 3.0],
            n: 2,
            d: 1,
        };

        let panic = std::panic::catch_unwind(|| model.segment_cost(&cache, 0, 3));
        assert!(panic.is_err(), "expected panic for end > n");
    }

    #[test]
    fn integer_tolerance_allows_tiny_rounding_noise() {
        let model = CostPoissonRate::default();
        let values = [1.0, 2.0 + 0.5 * INTEGER_TOL, 3.0];
        let view = make_f64_view(
            &values,
            3,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        model
            .validate(&view)
            .expect("tiny rounding noise should be accepted");
    }
}
