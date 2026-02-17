// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{
    Constraints, ExecutionContext, MemoryLayout, MissingPolicy, OfflineDetector, ReproMode,
    Stopping, TimeIndex, TimeSeriesView, validate_breakpoints,
};
use cpd_costs::{CostCosine, CostRank};
use cpd_offline::{BinSeg, BinSegConfig, Pelt, PeltConfig};

fn make_view(values: &[f64], n: usize, d: usize) -> TimeSeriesView<'_> {
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

fn assert_single_change_near(
    breakpoints: &[usize],
    n: usize,
    expected: usize,
    tolerance: usize,
    min_segment_len: usize,
) {
    validate_breakpoints(n, breakpoints).expect("breakpoints should satisfy core invariants");
    assert_eq!(
        breakpoints.len(),
        2,
        "KnownK(1) should return exactly one change point plus n"
    );
    let cp = breakpoints[0];
    let lower = expected.saturating_sub(tolerance);
    let upper = expected + tolerance;
    assert!(
        (lower..=upper).contains(&cp),
        "change-point {cp} not in expected window [{lower}, {upper}]"
    );
    assert!(cp >= min_segment_len, "left segment too short");
    assert!(n - cp >= min_segment_len, "right segment too short");
}

#[test]
fn rank_runs_with_pelt_and_binseg_on_monotone_distortion() {
    let n = 140usize;
    let split = 70usize;
    let mut raw = Vec::with_capacity(n);
    for idx in 0..split {
        raw.push(0.5 + 0.01 * idx as f64 + (idx as f64 * 0.05).sin() * 0.02);
    }
    for idx in split..n {
        raw.push(5.5 + 0.01 * idx as f64 + (idx as f64 * 0.05).sin() * 0.02);
    }
    let transformed: Vec<f64> = raw.iter().map(|v| (1.2 * v + 0.7).exp()).collect();

    let constraints = Constraints {
        min_segment_len: 6,
        jump: 1,
        max_change_points: Some(1),
        ..Constraints::default()
    };
    let stopping = Stopping::KnownK(1);
    let ctx = ExecutionContext::new(&constraints).with_repro_mode(ReproMode::Balanced);

    let rank = CostRank::new(ReproMode::Balanced);
    let pelt_cfg = PeltConfig {
        stopping: stopping.clone(),
        params_per_segment: 2,
        cancel_check_every: 128,
    };
    let binseg_cfg = BinSegConfig {
        stopping,
        params_per_segment: 2,
        cancel_check_every: 128,
    };

    let raw_view = make_view(raw.as_slice(), n, 1);
    let transformed_view = make_view(transformed.as_slice(), n, 1);

    let pelt_raw = Pelt::new(rank, pelt_cfg.clone())
        .expect("pelt should build")
        .detect(&raw_view, &ctx)
        .expect("pelt rank detect should succeed");
    let pelt_transformed = Pelt::new(rank, pelt_cfg)
        .expect("pelt should build")
        .detect(&transformed_view, &ctx)
        .expect("pelt rank detect should succeed");

    let binseg_raw = BinSeg::new(rank, binseg_cfg.clone())
        .expect("binseg should build")
        .detect(&raw_view, &ctx)
        .expect("binseg rank detect should succeed");
    let binseg_transformed = BinSeg::new(rank, binseg_cfg)
        .expect("binseg should build")
        .detect(&transformed_view, &ctx)
        .expect("binseg rank detect should succeed");

    for result in [
        &pelt_raw,
        &pelt_transformed,
        &binseg_raw,
        &binseg_transformed,
    ] {
        assert_single_change_near(
            &result.breakpoints,
            n,
            split,
            10,
            constraints.min_segment_len,
        );
    }
}

#[test]
fn cosine_runs_with_pelt_and_binseg_on_direction_shift() {
    let n = 160usize;
    let split = 80usize;
    let mut values = Vec::with_capacity(n * 2);
    for idx in 0..split {
        let jitter = 1.0 + (idx as f64 * 0.03).sin() * 0.02;
        values.extend_from_slice(&[jitter, 0.0]);
    }
    for idx in split..n {
        let jitter = 1.0 + (idx as f64 * 0.03).cos() * 0.02;
        values.extend_from_slice(&[0.0, jitter]);
    }

    let constraints = Constraints {
        min_segment_len: 8,
        jump: 1,
        max_change_points: Some(1),
        ..Constraints::default()
    };
    let stopping = Stopping::KnownK(1);
    let ctx = ExecutionContext::new(&constraints).with_repro_mode(ReproMode::Balanced);
    let cosine = CostCosine::new(ReproMode::Balanced);

    let pelt_cfg = PeltConfig {
        stopping: stopping.clone(),
        params_per_segment: 2,
        cancel_check_every: 128,
    };
    let binseg_cfg = BinSegConfig {
        stopping,
        params_per_segment: 2,
        cancel_check_every: 128,
    };

    let view = make_view(values.as_slice(), n, 2);
    let pelt = Pelt::new(cosine, pelt_cfg)
        .expect("pelt should build")
        .detect(&view, &ctx)
        .expect("pelt cosine detect should succeed");
    let binseg = BinSeg::new(cosine, binseg_cfg)
        .expect("binseg should build")
        .detect(&view, &ctx)
        .expect("binseg cosine detect should succeed");

    assert_single_change_near(&pelt.breakpoints, n, split, 12, constraints.min_segment_len);
    assert_single_change_near(
        &binseg.breakpoints,
        n,
        split,
        14,
        constraints.min_segment_len,
    );
}
