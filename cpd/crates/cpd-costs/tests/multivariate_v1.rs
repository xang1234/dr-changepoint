// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{CachePolicy, MemoryLayout, MissingPolicy, TimeIndex, TimeSeriesView};
use cpd_costs::{
    CostAR, CostBernoulli, CostL1Median, CostL2Mean, CostLinear, CostModel, CostNIGMarginal,
    CostNormalMeanVar, CostPoissonRate, CostRank,
};

const N: usize = 96;
const SEGMENTS: [(usize, usize); 2] = [(8, 56), (16, 92)];

fn make_view<'a>(values: &'a [f64], n: usize, d: usize) -> TimeSeriesView<'a> {
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

fn continuous_values(n: usize, d: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(n * d);
    for t in 0..n {
        for dim in 0..d {
            let x = t as f64 + 1.0;
            let y = dim as f64 + 1.0;
            values.push((0.03 * x).sin() + (0.07 * y).cos() + 0.05 * x + 0.02 * x * y);
        }
    }
    values
}

fn count_values(n: usize, d: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(n * d);
    for t in 0..n {
        for dim in 0..d {
            let raw = (t
                .wrapping_mul(dim.saturating_add(7))
                .wrapping_add(dim.saturating_mul(13)))
                % 31;
            values.push(raw as f64);
        }
    }
    values
}

fn binary_values(n: usize, d: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(n * d);
    for t in 0..n {
        for dim in 0..d {
            values.push(((t + dim.saturating_mul(3)) % 2) as f64);
        }
    }
    values
}

fn column_major_extract(values: &[f64], n: usize, d: usize, dim: usize) -> Vec<f64> {
    let mut column = Vec::with_capacity(n);
    for t in 0..n {
        column.push(values[t * d + dim]);
    }
    column
}

fn assert_close(
    label: &str,
    d: usize,
    start: usize,
    end: usize,
    actual: f64,
    expected: f64,
    tol: f64,
) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= tol,
        "{label} additive mismatch at d={d}, segment=[{start}, {end}): expected {expected}, got {actual}, |diff|={diff}, tol={tol}"
    );
}

fn assert_additive_multivariate<M: CostModel>(
    label: &str,
    model: &M,
    values: &[f64],
    n: usize,
    d: usize,
    tol: f64,
) {
    let view = make_view(values, n, d);
    let multivariate_cache = model
        .precompute(&view, &CachePolicy::Full)
        .expect("multivariate precompute should succeed");

    let mut univariate_caches = Vec::with_capacity(d);
    for dim in 0..d {
        let column = column_major_extract(values, n, d, dim);
        let column_view = make_view(column.as_slice(), n, 1);
        let cache = model
            .precompute(&column_view, &CachePolicy::Full)
            .expect("univariate precompute should succeed");
        univariate_caches.push(cache);
    }

    for (start, end) in SEGMENTS {
        let multivariate = model.segment_cost(&multivariate_cache, start, end);
        let per_dimension_sum = univariate_caches
            .iter()
            .map(|cache| model.segment_cost(cache, start, end))
            .sum::<f64>();
        assert_close(label, d, start, end, multivariate, per_dimension_sum, tol);
    }
}

#[test]
fn all_v1_costs_match_univariate_sum_for_d8_and_d16() {
    let l2 = CostL2Mean::default();
    let l1 = CostL1Median::default();
    let normal = CostNormalMeanVar::default();
    let nig = CostNIGMarginal::default();
    let poisson = CostPoissonRate::default();
    let bernoulli = CostBernoulli::default();
    let linear = CostLinear::default();
    let ar = CostAR::default();
    let rank = CostRank::default();

    for d in [8_usize, 16] {
        let continuous = continuous_values(N, d);
        assert_additive_multivariate("CostL1Median", &l1, continuous.as_slice(), N, d, 1e-10);
        assert_additive_multivariate("CostL2Mean", &l2, continuous.as_slice(), N, d, 1e-10);
        assert_additive_multivariate(
            "CostNormalMeanVar",
            &normal,
            continuous.as_slice(),
            N,
            d,
            1e-10,
        );
        assert_additive_multivariate("CostNIGMarginal", &nig, continuous.as_slice(), N, d, 1e-10);
        assert_additive_multivariate("CostLinear", &linear, continuous.as_slice(), N, d, 1e-8);
        assert_additive_multivariate("CostAR", &ar, continuous.as_slice(), N, d, 1e-9);
        assert_additive_multivariate("CostRank", &rank, continuous.as_slice(), N, d, 1e-10);

        let counts = count_values(N, d);
        assert_additive_multivariate("CostPoissonRate", &poisson, counts.as_slice(), N, d, 1e-12);

        let binary = binary_values(N, d);
        assert_additive_multivariate("CostBernoulli", &bernoulli, binary.as_slice(), N, d, 1e-12);
    }
}
