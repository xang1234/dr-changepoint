// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::model::CostModel;
use cpd_core::{
    CachePolicy, CpdError, DTypeView, MemoryLayout, MissingSupport, ReproMode, TimeSeriesView,
};

/// Directionality-sensitive segment cost based on cosine similarity.
///
/// For each sample, this model normalizes the row vector to unit length
/// (`u_t`). Segment cost is:
///
/// `cost([s,e)) = m - ||sum_{t=s}^{e-1} u_t||_2`, where `m = e-s`.
///
/// Lower values indicate stronger directional coherence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CostCosine {
    pub repro_mode: ReproMode,
}

impl CostCosine {
    pub const fn new(repro_mode: ReproMode) -> Self {
        Self { repro_mode }
    }
}

impl Default for CostCosine {
    fn default() -> Self {
        Self::new(ReproMode::Balanced)
    }
}

/// Prefix cache for cosine segment queries.
#[derive(Clone, Debug, PartialEq)]
pub struct CosineCache {
    prefix_unit_sum: Vec<f64>,
    n: usize,
    d: usize,
}

impl CosineCache {
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
        "cache size overflow while planning CosineCache for n={n}, d={d}"
    ))
}

fn stable_l2_norm(values: &[f64]) -> f64 {
    let mut scale = 0.0;
    let mut sumsq = 1.0;
    let mut has_nonzero = false;

    for &value in values {
        let x = value.abs();
        if x == 0.0 {
            continue;
        }
        has_nonzero = true;
        if scale < x {
            let ratio = if scale == 0.0 { 0.0 } else { scale / x };
            sumsq = 1.0 + sumsq * ratio * ratio;
            scale = x;
        } else {
            let ratio = x / scale;
            sumsq += ratio * ratio;
        }
    }

    if !has_nonzero {
        0.0
    } else {
        scale * sumsq.sqrt()
    }
}

impl CostModel for CostCosine {
    type Cache = CosineCache;

    fn name(&self) -> &'static str {
        "cosine"
    }

    fn penalty_params_per_segment(&self) -> usize {
        2
    }

    fn penalty_effective_params(&self, d: usize) -> Option<usize> {
        d.checked_add(1)
    }

    fn validate(&self, x: &TimeSeriesView<'_>) -> Result<(), CpdError> {
        if x.n == 0 {
            return Err(CpdError::invalid_input(
                "CostCosine requires n >= 1; got n=0",
            ));
        }
        if x.d == 0 {
            return Err(CpdError::invalid_input(
                "CostCosine requires d >= 1; got d=0",
            ));
        }
        if x.has_missing() {
            return Err(CpdError::invalid_input(format!(
                "CostCosine does not support missing values: effective_missing_count={}",
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
                "CostCosine does not support CachePolicy::Approximate",
            ));
        }

        if required_bytes == usize::MAX {
            return Err(cache_overflow_err(x.n, x.d));
        }

        if let CachePolicy::Budgeted { max_bytes } = policy
            && required_bytes > *max_bytes
        {
            return Err(CpdError::resource_limit(format!(
                "CostCosine cache requires {} bytes, exceeds budget {} bytes",
                required_bytes, max_bytes
            )));
        }

        let prefix_len_per_dim =
            x.n.checked_add(1)
                .ok_or_else(|| cache_overflow_err(x.n, x.d))?;
        let total_prefix_len = prefix_len_per_dim
            .checked_mul(x.d)
            .ok_or_else(|| cache_overflow_err(x.n, x.d))?;

        let mut prefix_unit_sum = vec![0.0; total_prefix_len];
        let mut running_sum = vec![0.0; x.d];
        let mut running_comp = vec![0.0; x.d];
        let mut row = vec![0.0; x.d];

        for t in 0..x.n {
            for (dim, slot) in row.iter_mut().enumerate() {
                *slot = read_value(x, t, dim)?;
            }

            let norm = stable_l2_norm(&row);
            let inv_norm = if norm.is_finite() && norm > 0.0 {
                1.0 / norm
            } else {
                0.0
            };

            for (dim, value) in row.iter().copied().enumerate() {
                let unit = value * inv_norm;
                if matches!(self.repro_mode, ReproMode::Strict) {
                    let y = unit - running_comp[dim];
                    let sum = running_sum[dim] + y;
                    running_comp[dim] = (sum - running_sum[dim]) - y;
                    running_sum[dim] = sum;
                } else {
                    running_sum[dim] += unit;
                }
                let base = dim * prefix_len_per_dim;
                prefix_unit_sum[base + t + 1] = running_sum[dim];
            }
        }

        Ok(CosineCache {
            prefix_unit_sum,
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
        let mut resultant = vec![0.0; cache.d];
        for (dim, slot) in resultant.iter_mut().enumerate() {
            let base = cache.dim_offset(dim);
            *slot = cache.prefix_unit_sum[base + end] - cache.prefix_unit_sum[base + start];
        }

        let resultant_norm = stable_l2_norm(&resultant);
        (m - resultant_norm).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CosineCache, CostCosine, cache_overflow_err, read_value, stable_l2_norm,
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

    #[test]
    fn cosine_defaults_match_contract() {
        let model = CostCosine::default();
        assert_eq!(model.name(), "cosine");
        assert_eq!(model.missing_support(), MissingSupport::Reject);
        assert!(!model.supports_approx_cache());
        assert_eq!(model.penalty_params_per_segment(), 2);
        assert_eq!(model.penalty_effective_params(8), Some(9));
    }

    #[test]
    fn coherent_direction_has_zero_cost() {
        let mut values = Vec::new();
        for _ in 0..32 {
            values.extend_from_slice(&[3.0, 4.0]);
        }
        let view = make_f64_view(
            values.as_slice(),
            32,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let model = CostCosine::default();
        let cache = model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed");
        assert_close(model.segment_cost(&cache, 0, 32), 0.0, 1e-10);
    }

    #[test]
    fn magnitude_scaling_does_not_change_cost() {
        let base = vec![1.0, 2.0, 2.0, 1.0, 1.5, 1.5, 0.5, 3.0, 1.0, 2.0, 2.0, 1.0];
        let scales = [0.1, 5.0, 2.0, 10.0, 0.25, 4.0];
        let mut scaled = Vec::with_capacity(base.len());
        for (row, scale) in scales.into_iter().enumerate() {
            scaled.push(base[2 * row] * scale);
            scaled.push(base[2 * row + 1] * scale);
        }

        let base_view = make_f64_view(
            base.as_slice(),
            6,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let scaled_view = make_f64_view(
            scaled.as_slice(),
            6,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let model = CostCosine::default();
        let base_cache = model
            .precompute(&base_view, &CachePolicy::Full)
            .expect("precompute should succeed");
        let scaled_cache = model
            .precompute(&scaled_view, &CachePolicy::Full)
            .expect("precompute should succeed");

        assert_close(
            model.segment_cost(&base_cache, 0, 6),
            model.segment_cost(&scaled_cache, 0, 6),
            1e-10,
        );
    }

    #[test]
    fn mixed_directions_have_higher_whole_segment_cost() {
        let mut values = Vec::with_capacity(80 * 2);
        for _ in 0..40 {
            values.extend_from_slice(&[1.0, 0.0]);
        }
        for _ in 0..40 {
            values.extend_from_slice(&[0.0, 1.0]);
        }

        let view = make_f64_view(
            values.as_slice(),
            80,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let model = CostCosine::default();
        let cache = model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed");

        let whole = model.segment_cost(&cache, 0, 80);
        let split = model.segment_cost(&cache, 0, 40) + model.segment_cost(&cache, 40, 80);
        assert!(whole > split);
    }

    #[test]
    fn strict_mode_remains_close_to_balanced() {
        let mut values = Vec::with_capacity(512 * 3);
        for idx in 0..512 {
            values.push(1.0 + 1.0e-9 * idx as f64);
            values.push(2.0 + (idx as f64 * 0.013).sin());
            values.push(3.0 + (idx as f64 * 0.007).cos());
        }

        let view = make_f64_view(
            values.as_slice(),
            512,
            3,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let balanced_model = CostCosine::new(ReproMode::Balanced);
        let strict_model = CostCosine::new(ReproMode::Strict);
        let balanced_cache = balanced_model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed");
        let strict_cache = strict_model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed");

        let balanced = balanced_model.segment_cost(&balanced_cache, 64, 448);
        let strict = strict_model.segment_cost(&strict_cache, 64, 448);
        assert_close(balanced, strict, 1e-7);
    }

    #[test]
    fn validate_rejects_missing_values() {
        let view = make_f64_view(
            &[1.0, f64::NAN, 2.0, 3.0],
            2,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Ignore,
        );
        let err = CostCosine::default()
            .validate(&view)
            .expect_err("missing values should fail validation");
        assert!(matches!(err, CpdError::InvalidInput(_)));
    }

    #[test]
    fn budgeted_policy_enforces_bytes() {
        let values = vec![1.0, 0.0, 0.0, 1.0];
        let view = make_f64_view(
            values.as_slice(),
            2,
            2,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let model = CostCosine::default();
        let needed = model.worst_case_cache_bytes(&view);
        let err = model
            .precompute(
                &view,
                &CachePolicy::Budgeted {
                    max_bytes: needed.saturating_sub(1),
                },
            )
            .expect_err("budgeted precompute should fail with undersized budget");
        assert!(matches!(err, CpdError::ResourceLimit(_)));
    }

    #[test]
    fn l2_norm_is_stable_and_zero_safe() {
        assert_close(stable_l2_norm(&[]), 0.0, 0.0);
        assert_close(stable_l2_norm(&[0.0, 0.0, 0.0]), 0.0, 0.0);
        assert_close(stable_l2_norm(&[3.0, 4.0]), 5.0, 1e-12);
    }

    #[test]
    fn strided_helpers_cover_bounds() {
        let values = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = TimeSeriesView::new(
            DTypeView::F64(&values),
            3,
            2,
            MemoryLayout::Strided {
                row_stride: 2,
                col_stride: 1,
            },
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect("view should be valid");

        assert_eq!(
            strided_linear_index(1, 1, 2, 1, values.len()).unwrap_or(0),
            3
        );
        assert_close(read_value(&view, 2, 0).unwrap_or(f64::NAN), 5.0, 0.0);
    }

    #[test]
    fn cache_overflow_message_mentions_cosine_cache() {
        let err = cache_overflow_err(9, 4);
        match err {
            CpdError::ResourceLimit(msg) => assert!(msg.contains("CosineCache")),
            _ => panic!("expected resource-limit error"),
        }
    }

    #[test]
    fn cache_layout_offsets_follow_prefix_length() {
        let cache = CosineCache {
            prefix_unit_sum: vec![0.0; 12],
            n: 5,
            d: 2,
        };
        assert_eq!(cache.prefix_len_per_dim(), 6);
        assert_eq!(cache.dim_offset(0), 0);
        assert_eq!(cache.dim_offset(1), 6);
    }
}
