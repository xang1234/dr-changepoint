// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{
    Constraints, CpdError, ExecutionContext, MemoryLayout, MissingPolicy, OfflineDetector, Penalty,
    ReproMode, Stopping, TimeIndex, TimeSeriesView, validate_breakpoints,
};
use cpd_costs::{CostL2Mean, CostNormalMeanVar};
use cpd_offline::{BinSeg, BinSegConfig, Pelt, PeltConfig};
use proptest::prelude::*;
use proptest::test_runner::{Config as ProptestConfig, FileFailurePersistence};

const MIN_PROPTEST_CASES: u32 = 1000;

fn proptest_cases() -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|raw| raw.parse::<u32>().ok())
        .map(|parsed| parsed.max(MIN_PROPTEST_CASES))
        .unwrap_or(MIN_PROPTEST_CASES)
}

fn make_view(
    values: &[f64],
    n: usize,
    d: usize,
    missing: MissingPolicy,
) -> Result<TimeSeriesView<'_>, CpdError> {
    TimeSeriesView::from_f64(
        values,
        n,
        d,
        MemoryLayout::CContiguous,
        None,
        TimeIndex::None,
        missing,
    )
}

fn pelt_l2_breakpoints(
    values: &[f64],
    n: usize,
    d: usize,
    constraints: &Constraints,
    stopping: Stopping,
) -> Result<Vec<usize>, CpdError> {
    let view = make_view(values, n, d, MissingPolicy::Error)?;
    let ctx = ExecutionContext::new(constraints).with_repro_mode(ReproMode::Balanced);
    let detector = Pelt::new(
        CostL2Mean::new(ReproMode::Balanced),
        PeltConfig {
            stopping,
            params_per_segment: 2,
            cancel_check_every: 64,
        },
    )?;
    Ok(detector.detect(&view, &ctx)?.breakpoints)
}

fn pelt_normal_breakpoints(
    values: &[f64],
    n: usize,
    d: usize,
    constraints: &Constraints,
    stopping: Stopping,
) -> Result<Vec<usize>, CpdError> {
    let view = make_view(values, n, d, MissingPolicy::Error)?;
    let ctx = ExecutionContext::new(constraints).with_repro_mode(ReproMode::Balanced);
    let detector = Pelt::new(
        CostNormalMeanVar::new(ReproMode::Balanced),
        PeltConfig {
            stopping,
            params_per_segment: 3,
            cancel_check_every: 64,
        },
    )?;
    Ok(detector.detect(&view, &ctx)?.breakpoints)
}

fn binseg_l2_breakpoints(
    values: &[f64],
    n: usize,
    d: usize,
    constraints: &Constraints,
    stopping: Stopping,
) -> Result<Vec<usize>, CpdError> {
    let view = make_view(values, n, d, MissingPolicy::Error)?;
    let ctx = ExecutionContext::new(constraints).with_repro_mode(ReproMode::Balanced);
    let detector = BinSeg::new(
        CostL2Mean::new(ReproMode::Balanced),
        BinSegConfig {
            stopping,
            params_per_segment: 2,
            cancel_check_every: 64,
        },
    )?;
    Ok(detector.detect(&view, &ctx)?.breakpoints)
}

fn binseg_normal_breakpoints(
    values: &[f64],
    n: usize,
    d: usize,
    constraints: &Constraints,
    stopping: Stopping,
) -> Result<Vec<usize>, CpdError> {
    let view = make_view(values, n, d, MissingPolicy::Error)?;
    let ctx = ExecutionContext::new(constraints).with_repro_mode(ReproMode::Balanced);
    let detector = BinSeg::new(
        CostNormalMeanVar::new(ReproMode::Balanced),
        BinSegConfig {
            stopping,
            params_per_segment: 3,
            cancel_check_every: 64,
        },
    )?;
    Ok(detector.detect(&view, &ctx)?.breakpoints)
}

fn assert_breakpoint_invariants(
    breakpoints: &[usize],
    n: usize,
    min_segment_len: usize,
    jump: usize,
    max_change_points: Option<usize>,
) {
    validate_breakpoints(n, breakpoints).expect("breakpoint contract must hold");

    let mut start = 0usize;
    for &end in breakpoints {
        assert!(
            end.saturating_sub(start) >= min_segment_len,
            "segment [{start}, {end}) violates min_segment_len={min_segment_len}"
        );
        start = end;
    }

    for &bp in breakpoints {
        if bp != n {
            assert_eq!(
                bp % jump,
                0,
                "non-terminal breakpoint {bp} must respect jump={jump}"
            );
        }
    }

    if let Some(max_changes) = max_change_points {
        assert!(
            breakpoints.len().saturating_sub(1) <= max_changes,
            "change-point count must not exceed max_change_points"
        );
    }
}

fn three_regime_signal() -> Vec<f64> {
    let mut out = Vec::with_capacity(90);
    out.extend(std::iter::repeat_n(0.0, 30));
    out.extend(std::iter::repeat_n(8.0, 30));
    out.extend(std::iter::repeat_n(-4.0, 30));
    out
}

fn three_regime_multivariate_signal(d: usize) -> (Vec<f64>, usize) {
    let regime_len = 24;
    let n = regime_len * 3;
    let mut out = Vec::with_capacity(n * d);

    for _ in 0..regime_len {
        for dim in 0..d {
            out.push(dim as f64 * 0.5);
        }
    }
    for _ in 0..regime_len {
        for dim in 0..d {
            out.push(20.0 + dim as f64 * 0.5);
        }
    }
    for _ in 0..regime_len {
        for dim in 0..d {
            out.push(-12.0 - dim as f64 * 0.25);
        }
    }

    (out, n)
}

fn permute_columns(values: &[f64], n: usize, d: usize, permutation: &[usize]) -> Vec<f64> {
    assert_eq!(values.len(), n * d);
    assert_eq!(permutation.len(), d);

    let mut out = vec![0.0; values.len()];
    for t in 0..n {
        for (new_dim, &old_dim) in permutation.iter().enumerate() {
            out[t * d + new_dim] = values[t * d + old_dim];
        }
    }
    out
}

fn replicate_univariate(values: &[f64], factor: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len() * factor);
    for &value in values {
        out.extend(std::iter::repeat_n(value, factor));
    }
    out
}

fn replicate_rows(values: &[f64], n: usize, d: usize, factor: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len() * factor);
    for row_idx in 0..n {
        let start = row_idx * d;
        let end = start + d;
        let row = &values[start..end];
        for _ in 0..factor {
            out.extend_from_slice(row);
        }
    }
    out
}

fn internal_breakpoints(breakpoints: &[usize]) -> &[usize] {
    &breakpoints[..breakpoints.len().saturating_sub(1)]
}

fn scale_breakpoints(breakpoints: &[usize], factor: usize) -> Vec<usize> {
    breakpoints
        .iter()
        .map(|bp| bp.saturating_mul(factor))
        .collect()
}

fn stress_signal(case_id: u8, n: usize) -> Vec<f64> {
    match case_id % 4 {
        0 => (0..n)
            .map(|idx| {
                let sign = if idx % 2 == 0 { 1.0 } else { -1.0 };
                sign * 1.0e15 + idx as f64 * 1.0e3
            })
            .collect(),
        1 => {
            let center = (n as f64 - 1.0) * 0.5;
            (0..n)
                .map(|idx| ((idx as f64) - center) * 1.0e-15)
                .collect()
        }
        2 => {
            let denormal = f64::from_bits(1);
            (0..n)
                .map(|idx| {
                    let sign = if idx % 2 == 0 { 1.0 } else { -1.0 };
                    sign * denormal * (idx as f64 + 1.0)
                })
                .collect()
        }
        _ => {
            let base = 7.0;
            (0..n).map(|idx| base + idx as f64 * 1.0e-12).collect()
        }
    }
}

#[test]
fn nan_with_error_policy_is_rejected_before_detector_runs() {
    let values = [1.0_f64, f64::NAN, 3.0, 4.0];
    let err = make_view(&values, 2, 2, MissingPolicy::Error)
        .expect_err("NaN + MissingPolicy::Error should fail at view construction");
    assert!(matches!(err, CpdError::InvalidInput(_)));
}

#[test]
fn infinities_are_rejected_before_detector_runs() {
    let plus_inf = [1.0_f64, f64::INFINITY, 3.0, 4.0];
    let plus_err = make_view(&plus_inf, 2, 2, MissingPolicy::Error)
        .expect_err("+inf should fail at view construction");
    assert!(matches!(plus_err, CpdError::InvalidInput(_)));

    let minus_inf = [1.0_f64, f64::NEG_INFINITY, 3.0, 4.0];
    let minus_err = make_view(&minus_inf, 2, 2, MissingPolicy::Ignore)
        .expect_err("-inf should fail at view construction for all missing policies");
    assert!(matches!(minus_err, CpdError::InvalidInput(_)));
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: proptest_cases(),
        max_shrink_iters: 1024,
        failure_persistence: Some(Box::new(FileFailurePersistence::Direct("proptest-regressions/tests/proptest_invariants.txt"))),
        .. ProptestConfig::default()
    })]

    #[test]
    fn pelt_and_binseg_outputs_respect_breakpoint_constraints(
        values in prop::collection::vec(-50.0f64..50.0, 32..128),
        min_segment_len in 1usize..8,
        jump in 1usize..6,
        max_change_points in 1usize..10,
    ) {
        let n = values.len();
        prop_assume!(min_segment_len.saturating_mul(2) <= n);

        let constraints = Constraints {
            min_segment_len,
            jump,
            max_change_points: Some(max_change_points),
            ..Constraints::default()
        };
        let stopping = Stopping::Penalized(Penalty::Manual(5.0));

        let pelt_first = pelt_l2_breakpoints(&values, n, 1, &constraints, stopping.clone())
            .expect("pelt should succeed for generated input");
        let pelt_second = pelt_l2_breakpoints(&values, n, 1, &constraints, stopping.clone())
            .expect("pelt should be deterministic");
        prop_assert_eq!(&pelt_first, &pelt_second);
        assert_breakpoint_invariants(
            &pelt_first,
            n,
            constraints.min_segment_len,
            constraints.jump,
            constraints.max_change_points,
        );

        let binseg_first = binseg_l2_breakpoints(&values, n, 1, &constraints, stopping.clone())
            .expect("binseg should succeed for generated input");
        let binseg_second = binseg_l2_breakpoints(&values, n, 1, &constraints, stopping)
            .expect("binseg should be deterministic");
        prop_assert_eq!(&binseg_first, &binseg_second);
        assert_breakpoint_invariants(
            &binseg_first,
            n,
            constraints.min_segment_len,
            constraints.jump,
            constraints.max_change_points,
        );
    }

    #[test]
    fn constant_series_with_large_penalty_has_no_spurious_changes(
        value in -20.0f64..20.0,
        n in 16usize..128,
    ) {
        let series = vec![value; n];
        let constraints = Constraints {
            min_segment_len: 2,
            ..Constraints::default()
        };
        let stopping = Stopping::Penalized(Penalty::Manual(1_000_000.0));

        let pelt = pelt_l2_breakpoints(&series, n, 1, &constraints, stopping.clone())
            .expect("pelt should succeed on constant data");
        let binseg = binseg_l2_breakpoints(&series, n, 1, &constraints, stopping)
            .expect("binseg should succeed on constant data");

        prop_assert_eq!(pelt, vec![n]);
        prop_assert_eq!(binseg, vec![n]);
    }

    #[test]
    fn known_k_detection_is_invariant_to_shift_and_scale_for_pelt_and_binseg(
        shift in -100.0f64..100.0,
        scale in 0.2f64..8.0,
    ) {
        let base = three_regime_signal();
        let n = base.len();
        let shifted: Vec<f64> = base.iter().map(|value| value + shift).collect();
        let scaled: Vec<f64> = base.iter().map(|value| value * scale).collect();

        let constraints = Constraints {
            min_segment_len: 2,
            max_change_points: Some(2),
            ..Constraints::default()
        };
        let stopping = Stopping::KnownK(2);

        let pelt_l2_base = pelt_l2_breakpoints(&base, n, 1, &constraints, stopping.clone())
            .expect("pelt l2 base should succeed");
        let pelt_l2_shifted = pelt_l2_breakpoints(&shifted, n, 1, &constraints, stopping.clone())
            .expect("pelt l2 shifted should succeed");
        let pelt_l2_scaled = pelt_l2_breakpoints(&scaled, n, 1, &constraints, stopping.clone())
            .expect("pelt l2 scaled should succeed");
        prop_assert_eq!(&pelt_l2_base, &pelt_l2_shifted);
        prop_assert_eq!(&pelt_l2_base, &pelt_l2_scaled);

        let pelt_normal_base =
            pelt_normal_breakpoints(&base, n, 1, &constraints, stopping.clone())
                .expect("pelt normal base should succeed");
        let pelt_normal_shifted =
            pelt_normal_breakpoints(&shifted, n, 1, &constraints, stopping.clone())
                .expect("pelt normal shifted should succeed");
        let pelt_normal_scaled = pelt_normal_breakpoints(&scaled, n, 1, &constraints, stopping.clone())
            .expect("pelt normal scaled should succeed");
        prop_assert_eq!(&pelt_normal_base, &pelt_normal_shifted);
        prop_assert_eq!(&pelt_normal_base, &pelt_normal_scaled);

        let binseg_l2_base = binseg_l2_breakpoints(&base, n, 1, &constraints, stopping.clone())
            .expect("binseg l2 base should succeed");
        let binseg_l2_shifted =
            binseg_l2_breakpoints(&shifted, n, 1, &constraints, stopping.clone())
                .expect("binseg l2 shifted should succeed");
        let binseg_l2_scaled =
            binseg_l2_breakpoints(&scaled, n, 1, &constraints, stopping.clone())
                .expect("binseg l2 scaled should succeed");
        prop_assert_eq!(&binseg_l2_base, &binseg_l2_shifted);
        prop_assert_eq!(&binseg_l2_base, &binseg_l2_scaled);

        let binseg_normal_base =
            binseg_normal_breakpoints(&base, n, 1, &constraints, stopping.clone())
                .expect("binseg normal base should succeed");
        let binseg_normal_shifted =
            binseg_normal_breakpoints(&shifted, n, 1, &constraints, stopping.clone())
                .expect("binseg normal shifted should succeed");
        let binseg_normal_scaled =
            binseg_normal_breakpoints(&scaled, n, 1, &constraints, stopping)
                .expect("binseg normal scaled should succeed");
        prop_assert_eq!(&binseg_normal_base, &binseg_normal_shifted);
        prop_assert_eq!(&binseg_normal_base, &binseg_normal_scaled);
    }

    #[test]
    fn concatenated_constant_segments_produce_join_breakpoint_for_all_detectors_and_costs(
        left_len in 8usize..64,
        right_len in 8usize..64,
        left_level in -30.0f64..30.0,
        right_level in -30.0f64..30.0,
        shift in -50.0f64..50.0,
        scale in 0.5f64..6.0,
    ) {
        prop_assume!((left_level - right_level).abs() >= 5.0);

        let mut base = Vec::with_capacity(left_len + right_len);
        base.extend(std::iter::repeat_n(left_level, left_len));
        base.extend(std::iter::repeat_n(right_level, right_len));

        let values: Vec<f64> = base.iter().map(|value| value * scale + shift).collect();
        let n = values.len();

        let constraints = Constraints {
            min_segment_len: 2,
            max_change_points: Some(1),
            ..Constraints::default()
        };
        let stopping_known_k = Stopping::KnownK(1);

        let pelt_l2 = pelt_l2_breakpoints(
            &values,
            n,
            1,
            &constraints,
            stopping_known_k.clone(),
        )
            .expect("pelt l2 should succeed");
        let pelt_normal = pelt_normal_breakpoints(
            &values,
            n,
            1,
            &constraints,
            stopping_known_k.clone(),
        )
        .expect("pelt normal should succeed");
        let binseg_l2 = binseg_l2_breakpoints(
            &values,
            n,
            1,
            &constraints,
            stopping_known_k.clone(),
        )
            .expect("binseg l2 should succeed");
        let binseg_normal = binseg_normal_breakpoints(&values, n, 1, &constraints, stopping_known_k)
            .expect("binseg normal should succeed");

        let expected = vec![left_len, n];
        prop_assert_eq!(&pelt_l2, &expected);
        prop_assert_eq!(&pelt_normal, &expected);
        prop_assert_eq!(&binseg_l2, &expected);
        prop_assert_eq!(&binseg_normal, &expected);
    }

    #[test]
    fn replication_scales_internal_breakpoints_for_all_detectors_and_costs(
        replicate_factor in 2usize..5,
        shift in -100.0f64..100.0,
        scale in 0.4f64..4.0,
    ) {
        let base_raw = three_regime_signal();
        let base: Vec<f64> = base_raw
            .iter()
            .map(|value| value * scale + shift)
            .collect();
        let base_n = base.len();
        let replicated = replicate_univariate(&base, replicate_factor);
        let replicated_n = replicated.len();

        let constraints = Constraints {
            min_segment_len: 2,
            max_change_points: Some(2),
            ..Constraints::default()
        };
        let stopping_known_k = Stopping::KnownK(2);

        let pelt_l2_base = pelt_l2_breakpoints(
            &base,
            base_n,
            1,
            &constraints,
            stopping_known_k.clone(),
        )
        .expect("base pelt l2 should succeed");
        let pelt_l2_replicated =
            pelt_l2_breakpoints(
                &replicated,
                replicated_n,
                1,
                &constraints,
                stopping_known_k.clone(),
            )
                .expect("replicated pelt l2 should succeed");

        let pelt_normal_base = pelt_normal_breakpoints(
            &base,
            base_n,
            1,
            &constraints,
            stopping_known_k.clone(),
        )
        .expect("base pelt normal should succeed");
        let pelt_normal_replicated =
            pelt_normal_breakpoints(
                &replicated,
                replicated_n,
                1,
                &constraints,
                stopping_known_k.clone(),
            )
            .expect("replicated pelt normal should succeed");

        let binseg_l2_base =
            binseg_l2_breakpoints(&base, base_n, 1, &constraints, stopping_known_k.clone())
                .expect("base binseg l2 should succeed");
        let binseg_l2_replicated =
            binseg_l2_breakpoints(
                &replicated,
                replicated_n,
                1,
                &constraints,
                stopping_known_k.clone(),
            )
                .expect("replicated binseg l2 should succeed");

        let binseg_normal_base =
            binseg_normal_breakpoints(&base, base_n, 1, &constraints, stopping_known_k.clone())
                .expect("base binseg normal should succeed");
        let binseg_normal_replicated =
            binseg_normal_breakpoints(
                &replicated,
                replicated_n,
                1,
                &constraints,
                stopping_known_k,
            )
                .expect("replicated binseg normal should succeed");

        prop_assert_eq!(
            internal_breakpoints(&pelt_l2_replicated),
            scale_breakpoints(internal_breakpoints(&pelt_l2_base), replicate_factor)
        );
        prop_assert_eq!(
            internal_breakpoints(&pelt_normal_replicated),
            scale_breakpoints(internal_breakpoints(&pelt_normal_base), replicate_factor)
        );
        prop_assert_eq!(
            internal_breakpoints(&binseg_l2_replicated),
            scale_breakpoints(internal_breakpoints(&binseg_l2_base), replicate_factor)
        );
        prop_assert_eq!(
            internal_breakpoints(&binseg_normal_replicated),
            scale_breakpoints(internal_breakpoints(&binseg_normal_base), replicate_factor)
        );
    }

    #[test]
    fn multivariate_column_permutation_preserves_breakpoints_for_symmetric_costs(
        d in 2usize..6,
        left_seed in 0usize..16,
        right_seed in 0usize..16,
        shift in -100.0f64..100.0,
        scale in 0.4f64..4.0,
        replicate_factor in 1usize..3,
    ) {
        let left_col = left_seed % d;
        let right_col = (left_col + 1 + (right_seed % (d - 1))) % d;

        let (base, n) = three_regime_multivariate_signal(d);
        let transformed: Vec<f64> = base.iter().map(|value| value * scale + shift).collect();
        let transformed_n = n;

        let mut permutation: Vec<usize> = (0..d).collect();
        permutation.swap(left_col, right_col);
        let permuted = permute_columns(&transformed, transformed_n, d, &permutation);

        let transformed_rep = replicate_rows(&transformed, transformed_n, d, replicate_factor);
        let permuted_rep = replicate_rows(&permuted, transformed_n, d, replicate_factor);
        let rep_n = transformed_n * replicate_factor;

        let constraints = Constraints {
            min_segment_len: 2,
            max_change_points: Some(2),
            ..Constraints::default()
        };
        let stopping_known_k = Stopping::KnownK(2);

        let pelt_l2_base =
            pelt_l2_breakpoints(
                &transformed_rep,
                rep_n,
                d,
                &constraints,
                stopping_known_k.clone(),
            )
                .expect("pelt l2 base should succeed");
        let pelt_l2_perm =
            pelt_l2_breakpoints(
                &permuted_rep,
                rep_n,
                d,
                &constraints,
                stopping_known_k.clone(),
            )
                .expect("pelt l2 permuted should succeed");

        let pelt_normal_base =
            pelt_normal_breakpoints(
                &transformed_rep,
                rep_n,
                d,
                &constraints,
                stopping_known_k.clone(),
            )
                .expect("pelt normal base should succeed");
        let pelt_normal_perm =
            pelt_normal_breakpoints(
                &permuted_rep,
                rep_n,
                d,
                &constraints,
                stopping_known_k.clone(),
            )
                .expect("pelt normal permuted should succeed");

        let binseg_l2_base =
            binseg_l2_breakpoints(
                &transformed_rep,
                rep_n,
                d,
                &constraints,
                stopping_known_k.clone(),
            )
                .expect("binseg l2 base should succeed");
        let binseg_l2_perm =
            binseg_l2_breakpoints(
                &permuted_rep,
                rep_n,
                d,
                &constraints,
                stopping_known_k.clone(),
            )
                .expect("binseg l2 permuted should succeed");

        let binseg_normal_base =
            binseg_normal_breakpoints(
                &transformed_rep,
                rep_n,
                d,
                &constraints,
                stopping_known_k.clone(),
            )
                .expect("binseg normal base should succeed");
        let binseg_normal_perm =
            binseg_normal_breakpoints(
                &permuted_rep,
                rep_n,
                d,
                &constraints,
                stopping_known_k,
            )
                .expect("binseg normal permuted should succeed");

        prop_assert_eq!(pelt_l2_base, pelt_l2_perm);
        prop_assert_eq!(pelt_normal_base, pelt_normal_perm);
        prop_assert_eq!(binseg_l2_base, binseg_l2_perm);
        prop_assert_eq!(binseg_normal_base, binseg_normal_perm);
    }

    #[test]
    fn detector_numerical_stress_returns_valid_breakpoints_or_explicit_error(
        n in 24usize..96,
        case_id in 0u8..4,
    ) {
        let values = stress_signal(case_id, n);
        let constraints = Constraints {
            min_segment_len: 2,
            max_change_points: Some(4),
            ..Constraints::default()
        };
        let stopping = Stopping::Penalized(Penalty::BIC);

        let runs = [
            pelt_l2_breakpoints(&values, n, 1, &constraints, stopping.clone()),
            pelt_normal_breakpoints(&values, n, 1, &constraints, stopping.clone()),
            binseg_l2_breakpoints(&values, n, 1, &constraints, stopping.clone()),
            binseg_normal_breakpoints(&values, n, 1, &constraints, stopping),
        ];

        for run in runs {
            match run {
                Ok(breakpoints) => {
                    assert_breakpoint_invariants(
                        &breakpoints,
                        n,
                        constraints.min_segment_len,
                        constraints.jump,
                        constraints.max_change_points,
                    );
                }
                Err(err) => match err {
                    CpdError::InvalidInput(_)
                    | CpdError::NumericalIssue(_)
                    | CpdError::NotSupported(_)
                    | CpdError::ResourceLimit(_)
                    | CpdError::Cancelled => {}
                },
            }
        }
    }
}
