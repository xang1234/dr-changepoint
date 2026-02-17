// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{CachePolicy, MemoryLayout, MissingPolicy, TimeIndex, TimeSeriesView};
use cpd_costs::{CostCosine, CostL2Mean, CostModel, CostRank};

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

fn split_gain<M: CostModel>(model: &M, view: &TimeSeriesView<'_>, split: usize) -> f64 {
    let cache = model
        .precompute(view, &CachePolicy::Full)
        .expect("precompute should succeed");
    let whole = model.segment_cost(&cache, 0, view.n);
    let split_cost =
        model.segment_cost(&cache, 0, split) + model.segment_cost(&cache, split, view.n);
    whole - split_cost
}

#[test]
fn rank_is_invariant_to_monotone_distortion() {
    let n = 96usize;
    let mut base = Vec::with_capacity(n);
    for idx in 0..n {
        let x = idx as f64 + 1.0;
        base.push((0.03 * x).sin() + 0.01 * x);
    }
    let transformed: Vec<f64> = base.iter().map(|v| (3.0 * v + 2.0).exp()).collect();

    let base_view = make_view(base.as_slice(), n, 1);
    let transformed_view = make_view(transformed.as_slice(), n, 1);
    let model = CostRank::default();

    for split in [24usize, 48, 72] {
        let base_gain = split_gain(&model, &base_view, split);
        let transformed_gain = split_gain(&model, &transformed_view, split);
        let diff = (base_gain - transformed_gain).abs();
        assert!(
            diff <= 1e-8,
            "rank split gain should be monotone-invariant: diff={diff}, split={split}"
        );
    }
}

#[test]
fn rank_and_cosine_have_positive_split_gain_on_controlled_shift() {
    let n = 120usize;
    let mut values = Vec::with_capacity(n * 2);
    for _ in 0..60 {
        values.extend_from_slice(&[0.5, 1.0]);
    }
    for _ in 0..60 {
        values.extend_from_slice(&[6.0, 8.0]);
    }

    let view = make_view(values.as_slice(), n, 2);
    let split = 60usize;
    let rank_gain = split_gain(&CostRank::default(), &view, split);
    let cosine_gain = split_gain(&CostCosine::default(), &view, split);
    let l2_gain = split_gain(&CostL2Mean::default(), &view, split);

    assert!(l2_gain > 0.0, "baseline L2 should prefer the true split");
    assert!(rank_gain > 0.0, "rank should prefer the true split");
    assert!(
        cosine_gain >= 0.0,
        "cosine should not penalize the true split"
    );
}

#[test]
fn heavy_tail_outlier_changes_rank_less_than_l2() {
    let base = vec![-0.2, 0.0, 0.1, -0.1, 0.2, 0.0, 0.1, -0.2];
    let mut outlier = base.clone();
    outlier[3] = 1.0e7;

    let base_view = make_view(base.as_slice(), base.len(), 1);
    let outlier_view = make_view(outlier.as_slice(), outlier.len(), 1);
    let split = 4usize;

    let rank_gain_base = split_gain(&CostRank::default(), &base_view, split);
    let rank_gain_outlier = split_gain(&CostRank::default(), &outlier_view, split);
    let l2_gain_base = split_gain(&CostL2Mean::default(), &base_view, split);
    let l2_gain_outlier = split_gain(&CostL2Mean::default(), &outlier_view, split);

    let rank_delta = (rank_gain_outlier - rank_gain_base).abs();
    let l2_delta = (l2_gain_outlier - l2_gain_base).abs();
    assert!(
        rank_delta < l2_delta,
        "rank should be more outlier-robust than l2: rank_delta={rank_delta}, l2_delta={l2_delta}"
    );
}

#[test]
fn cosine_prefers_directional_split() {
    let n = 120usize;
    let mut values = Vec::with_capacity(n * 2);
    for _ in 0..60 {
        values.extend_from_slice(&[1.0, 0.0]);
    }
    for _ in 0..60 {
        values.extend_from_slice(&[0.0, 1.0]);
    }

    let view = make_view(values.as_slice(), n, 2);
    let gain = split_gain(&CostCosine::default(), &view, 60);
    assert!(
        gain > 10.0,
        "cosine should strongly prefer directional split"
    );
}
