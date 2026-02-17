// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{Constraints, Stopping};
use cpd_doctor::{
    CostConfig, DetectorConfig, EnsembleConfig, Explanation, OfflineDetectorConfig, PipelineSpec,
    Recommendation, ResourceEstimate, execute_ensemble,
};
use cpd_offline::{BinSegConfig, PeltConfig, WbsConfig};

fn stopping_from_detector(detector: &OfflineDetectorConfig) -> Stopping {
    match detector {
        OfflineDetectorConfig::Pelt(config) => config.stopping.clone(),
        OfflineDetectorConfig::BinSeg(config) => config.stopping.clone(),
        OfflineDetectorConfig::Fpop(config) => config.stopping.clone(),
        OfflineDetectorConfig::Wbs(config) => config.stopping.clone(),
        OfflineDetectorConfig::SegNeigh(config) => config.stopping.clone(),
    }
}

fn recommendation_for(
    detector: OfflineDetectorConfig,
    constraints: Constraints,
    seed: Option<u64>,
) -> Recommendation {
    Recommendation {
        pipeline: PipelineSpec {
            detector: DetectorConfig::Offline(detector.clone()),
            cost: CostConfig::L2,
            preprocess: None,
            constraints,
            stopping: Some(stopping_from_detector(&detector)),
            seed,
        },
        resource_estimate: ResourceEstimate {
            time_complexity: "O(n log n)".to_string(),
            memory_complexity: "O(n)".to_string(),
            relative_time_score: 0.5,
            relative_memory_score: 0.4,
        },
        warnings: vec![],
        explanation: Explanation {
            summary: "test recommendation".to_string(),
            drivers: vec![],
            tradeoffs: vec![],
        },
        validation: None,
        confidence: 0.8,
        confidence_interval: (0.72, 0.88),
        abstain_reason: None,
        objective_fit: vec![],
    }
}

fn step_signal(n: usize, split: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(n);
    for t in 0..split {
        values.push(0.4 * (0.07 * t as f64).sin());
    }
    for t in split..n {
        values.push(4.0 + 0.4 * (0.07 * t as f64).sin());
    }
    values
}

fn make_view(values: &[f64], n: usize) -> cpd_core::TimeSeriesView<'_> {
    cpd_core::TimeSeriesView::from_f64(
        values,
        n,
        1,
        cpd_core::MemoryLayout::CContiguous,
        None,
        cpd_core::TimeIndex::None,
        cpd_core::MissingPolicy::Error,
    )
    .expect("test view should be valid")
}

#[test]
fn execute_ensemble_produces_consensus_breakpoint_with_confidence() {
    let n = 200usize;
    let split = 100usize;
    let values = step_signal(n, split);
    let view = make_view(values.as_slice(), n);

    let constraints = Constraints {
        min_segment_len: 10,
        max_change_points: Some(1),
        ..Constraints::default()
    };

    let recommendations = vec![
        recommendation_for(
            OfflineDetectorConfig::Pelt(PeltConfig {
                stopping: Stopping::KnownK(1),
                ..PeltConfig::default()
            }),
            constraints.clone(),
            None,
        ),
        recommendation_for(
            OfflineDetectorConfig::BinSeg(BinSegConfig {
                stopping: Stopping::KnownK(1),
                ..BinSegConfig::default()
            }),
            constraints.clone(),
            None,
        ),
        recommendation_for(
            OfflineDetectorConfig::Wbs(WbsConfig {
                stopping: Stopping::KnownK(1),
                seed: 17,
                ..WbsConfig::default()
            }),
            constraints,
            Some(17),
        ),
    ];

    let config = EnsembleConfig {
        top_k: 3,
        tolerance: 10,
        min_consensus_ratio: 0.5,
        seed: Some(17),
    };
    let report = execute_ensemble(&view, recommendations.as_slice(), &config)
        .expect("ensemble execution should succeed");

    assert!(
        report.pipeline_results.len() >= 2,
        "expected at least two successful pipeline runs"
    );
    assert!(
        !report.breakpoints.is_empty(),
        "expected at least one consensus breakpoint"
    );

    let strongest = report
        .breakpoints
        .iter()
        .max_by(|left, right| left.consensus_score.total_cmp(&right.consensus_score))
        .expect("non-empty consensus");
    assert!(
        strongest.index.abs_diff(split) <= 15,
        "expected strongest consensus near split={split}, got {}",
        strongest.index
    );
    assert!(
        strongest.consensus_score >= (2.0 / 3.0),
        "expected majority confidence, got {}",
        strongest.consensus_score
    );
}

#[test]
fn execute_ensemble_filters_single_vote_breakpoints_with_high_threshold() {
    let n = 220usize;
    let split = 110usize;
    let values = step_signal(n, split);
    let view = make_view(values.as_slice(), n);

    let constraints = Constraints {
        min_segment_len: 12,
        max_change_points: Some(2),
        ..Constraints::default()
    };

    let recommendations = vec![
        recommendation_for(
            OfflineDetectorConfig::Pelt(PeltConfig {
                stopping: Stopping::KnownK(1),
                ..PeltConfig::default()
            }),
            constraints.clone(),
            None,
        ),
        recommendation_for(
            OfflineDetectorConfig::BinSeg(BinSegConfig {
                stopping: Stopping::KnownK(1),
                ..BinSegConfig::default()
            }),
            constraints.clone(),
            None,
        ),
        recommendation_for(
            OfflineDetectorConfig::Pelt(PeltConfig {
                stopping: Stopping::KnownK(2),
                ..PeltConfig::default()
            }),
            constraints,
            None,
        ),
    ];

    let config = EnsembleConfig {
        top_k: 3,
        tolerance: 8,
        min_consensus_ratio: 2.0 / 3.0,
        seed: None,
    };
    let report = execute_ensemble(&view, recommendations.as_slice(), &config)
        .expect("ensemble execution should succeed");

    assert!(
        report
            .breakpoints
            .iter()
            .all(|bp| bp.consensus_score >= (2.0 / 3.0) - 1.0e-12),
        "all returned breakpoints should satisfy consensus threshold"
    );
    assert!(
        report
            .breakpoints
            .iter()
            .any(|bp| bp.index.abs_diff(split) <= 16),
        "expected at least one consensus breakpoint near true split"
    );
}
