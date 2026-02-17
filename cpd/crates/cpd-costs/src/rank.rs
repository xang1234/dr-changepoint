// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::model::CostModel;
use cpd_core::{
    CachePolicy, CpdError, DTypeView, MemoryLayout, MissingSupport, ReproMode, TimeSeriesView,
    prefix_sum_squares, prefix_sum_squares_kahan, prefix_sums, prefix_sums_kahan,
};

/// Rank-transform segment cost model (piecewise-constant rank mean).
///
/// Values are transformed to average ranks per dimension, then scored with an
/// L2 SSE objective over segment-local rank means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CostRank {
    pub repro_mode: ReproMode,
}

impl CostRank {
    pub const fn new(repro_mode: ReproMode) -> Self {
        Self { repro_mode }
    }
}

impl Default for CostRank {
    fn default() -> Self {
        Self::new(ReproMode::Balanced)
    }
}

/// Prefix-stat cache for O(1) rank-SSE segment-cost queries.
#[derive(Clone, Debug, PartialEq)]
pub struct RankCache {
    prefix_rank_sum: Vec<f64>,
    prefix_rank_sum_sq: Vec<f64>,
    n: usize,
    d: usize,
}

impl RankCache {
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
        "cache size overflow while planning RankCache for n={n}, d={d}"
    ))
}

fn assign_average_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]).then_with(|| a.cmp(&b)));

    let mut ranks = vec![0.0; n];
    let mut group_start = 0usize;
    while group_start < n {
        let mut group_end = group_start + 1;
        while group_end < n
            && values[order[group_end]]
                .total_cmp(&values[order[group_start]])
                .is_eq()
        {
            group_end += 1;
        }

        let rank_low = group_start as f64 + 1.0;
        let rank_high = group_end as f64;
        let avg_rank = 0.5 * (rank_low + rank_high);

        for &idx in &order[group_start..group_end] {
            ranks[idx] = avg_rank;
        }
        group_start = group_end;
    }

    ranks
}

impl CostModel for CostRank {
    type Cache = RankCache;

    fn name(&self) -> &'static str {
        "rank"
    }

    fn penalty_params_per_segment(&self) -> usize {
        2
    }

    fn validate(&self, x: &TimeSeriesView<'_>) -> Result<(), CpdError> {
        if x.n == 0 {
            return Err(CpdError::invalid_input("CostRank requires n >= 1; got n=0"));
        }
        if x.d == 0 {
            return Err(CpdError::invalid_input("CostRank requires d >= 1; got d=0"));
        }
        if x.has_missing() {
            return Err(CpdError::invalid_input(format!(
                "CostRank does not support missing values: effective_missing_count={}",
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
                "CostRank does not support CachePolicy::Approximate",
            ));
        }

        if required_bytes == usize::MAX {
            return Err(cache_overflow_err(x.n, x.d));
        }

        if let CachePolicy::Budgeted { max_bytes } = policy
            && required_bytes > *max_bytes
        {
            return Err(CpdError::resource_limit(format!(
                "CostRank cache requires {} bytes, exceeds budget {} bytes",
                required_bytes, max_bytes
            )));
        }

        let prefix_len_per_dim =
            x.n.checked_add(1)
                .ok_or_else(|| cache_overflow_err(x.n, x.d))?;
        let total_prefix_len = prefix_len_per_dim
            .checked_mul(x.d)
            .ok_or_else(|| cache_overflow_err(x.n, x.d))?;

        let mut prefix_rank_sum = Vec::with_capacity(total_prefix_len);
        let mut prefix_rank_sum_sq = Vec::with_capacity(total_prefix_len);

        for dim in 0..x.d {
            let mut series = Vec::with_capacity(x.n);
            for t in 0..x.n {
                series.push(read_value(x, t, dim)?);
            }
            let ranks = assign_average_ranks(&series);

            let dim_prefix_rank_sum = if matches!(self.repro_mode, ReproMode::Strict) {
                prefix_sums_kahan(&ranks)
            } else {
                prefix_sums(&ranks)
            };

            let dim_prefix_rank_sum_sq = if matches!(self.repro_mode, ReproMode::Strict) {
                prefix_sum_squares_kahan(&ranks)
            } else {
                prefix_sum_squares(&ranks)
            };

            prefix_rank_sum.extend_from_slice(&dim_prefix_rank_sum);
            prefix_rank_sum_sq.extend_from_slice(&dim_prefix_rank_sum_sq);
        }

        Ok(RankCache {
            prefix_rank_sum,
            prefix_rank_sum_sq,
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
        match bytes_per_array.checked_mul(2) {
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
            let sum = cache.prefix_rank_sum[base + end] - cache.prefix_rank_sum[base + start];
            let sum_sq =
                cache.prefix_rank_sum_sq[base + end] - cache.prefix_rank_sum_sq[base + start];
            let cost_dim = sum_sq - (sum * sum) / m;
            total += cost_dim.max(0.0);
        }

        total.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CostRank, RankCache, assign_average_ranks, cache_overflow_err, read_value,
        strided_linear_index,
    };
    use crate::l2::CostL2Mean;
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
    fn rank_defaults_match_contract() {
        let model = CostRank::default();
        assert_eq!(model.name(), "rank");
        assert_eq!(model.missing_support(), MissingSupport::Reject);
        assert!(!model.supports_approx_cache());
        assert_eq!(model.penalty_params_per_segment(), 2);
    }

    #[test]
    fn tie_handling_uses_average_ranks() {
        let ranks = assign_average_ranks(&[1.0, 1.0, 2.0, 4.0, 4.0, 4.0]);
        let expected = [1.5, 1.5, 3.0, 5.0, 5.0, 5.0];
        for (actual, expected) in ranks.iter().zip(expected) {
            assert_close(*actual, expected, 1e-12);
        }
    }

    #[test]
    fn monotone_transform_preserves_segment_costs() {
        let raw: Vec<f64> = vec![0.2, 1.0, 1.5, 4.0, 7.0, 9.0, 12.0, 14.0];
        let transformed: Vec<f64> = raw.iter().map(|v| (2.0_f64 * *v + 1.0).exp()).collect();

        let raw_view = make_f64_view(
            raw.as_slice(),
            raw.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let transformed_view = make_f64_view(
            transformed.as_slice(),
            transformed.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let model = CostRank::default();
        let raw_cache = model
            .precompute(&raw_view, &CachePolicy::Full)
            .expect("precompute should succeed");
        let transformed_cache = model
            .precompute(&transformed_view, &CachePolicy::Full)
            .expect("precompute should succeed");

        for (start, end) in [(0usize, 4usize), (2, 8), (1, 7)] {
            let raw_cost = model.segment_cost(&raw_cache, start, end);
            let transformed_cost = model.segment_cost(&transformed_cache, start, end);
            assert_close(raw_cost, transformed_cost, 1e-9);
        }
    }

    #[test]
    fn rank_cost_is_less_sensitive_to_outlier_than_l2() {
        let base = vec![-0.3, 0.1, -0.2, 0.05, -0.05, 0.2, 0.0, -0.1];
        let mut outlier = base.clone();
        outlier[2] = 1.0e6;

        let base_view = make_f64_view(
            base.as_slice(),
            base.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let outlier_view = make_f64_view(
            outlier.as_slice(),
            outlier.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let rank_model = CostRank::default();
        let rank_base = rank_model
            .precompute(&base_view, &CachePolicy::Full)
            .expect("precompute should succeed");
        let rank_outlier = rank_model
            .precompute(&outlier_view, &CachePolicy::Full)
            .expect("precompute should succeed");

        let l2_model = CostL2Mean::default();
        let l2_base = l2_model
            .precompute(&base_view, &CachePolicy::Full)
            .expect("precompute should succeed");
        let l2_outlier = l2_model
            .precompute(&outlier_view, &CachePolicy::Full)
            .expect("precompute should succeed");

        let segment = (0usize, base.len());
        let rank_delta = (rank_model.segment_cost(&rank_base, segment.0, segment.1)
            - rank_model.segment_cost(&rank_outlier, segment.0, segment.1))
        .abs();
        let l2_delta = (l2_model.segment_cost(&l2_base, segment.0, segment.1)
            - l2_model.segment_cost(&l2_outlier, segment.0, segment.1))
        .abs();

        assert!(
            rank_delta < l2_delta,
            "rank_delta={rank_delta} should be below l2_delta={l2_delta}"
        );
    }

    #[test]
    fn strict_mode_remains_close_to_balanced() {
        let values: Vec<f64> = (0..512)
            .map(|idx| 1.0e12 + idx as f64 * 0.01 + (idx as f64 * 0.1).sin())
            .collect();
        let view = make_f64_view(
            values.as_slice(),
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let balanced = CostRank::new(ReproMode::Balanced)
            .precompute(&view, &CachePolicy::Full)
            .expect("balanced precompute should succeed");
        let strict = CostRank::new(ReproMode::Strict)
            .precompute(&view, &CachePolicy::Full)
            .expect("strict precompute should succeed");

        let balanced_cost = CostRank::new(ReproMode::Balanced).segment_cost(&balanced, 64, 448);
        let strict_cost = CostRank::new(ReproMode::Strict).segment_cost(&strict, 64, 448);
        assert_close(balanced_cost, strict_cost, 1e-6);
    }

    #[test]
    fn validate_rejects_empty_and_missing() {
        let with_missing = make_f64_view(
            &[1.0, f64::NAN, 3.0],
            3,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Ignore,
        );
        let err = CostRank::default()
            .validate(&with_missing)
            .expect_err("missing values should fail");
        assert!(matches!(err, CpdError::InvalidInput(_)));
    }

    #[test]
    fn budgeted_policy_enforces_bytes() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let view = make_f64_view(
            values.as_slice(),
            4,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let model = CostRank::default();
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
    fn cache_overflow_message_mentions_rank_cache() {
        let err = cache_overflow_err(7, 3);
        match err {
            CpdError::ResourceLimit(msg) => assert!(msg.contains("RankCache")),
            _ => panic!("expected resource-limit error"),
        }
    }

    #[test]
    fn cache_layout_offsets_follow_prefix_length() {
        let cache = RankCache {
            prefix_rank_sum: vec![0.0; 12],
            prefix_rank_sum_sq: vec![0.0; 12],
            n: 5,
            d: 2,
        };
        assert_eq!(cache.prefix_len_per_dim(), 6);
        assert_eq!(cache.dim_offset(0), 0);
        assert_eq!(cache.dim_offset(1), 6);
    }
}
