// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{
    CachePolicy, CpdError, MemoryLayout, MissingPolicy, ReproMode, TimeIndex, TimeSeriesView,
};
use cpd_costs::{
    CostBernoulli, CostL2Mean, CostModel, CostNIGMarginal, CostNormalMeanVar, CostPoissonRate,
    NIGPrior,
};
use proptest::prelude::*;
use proptest::test_runner::{Config as ProptestConfig, FileFailurePersistence};

const ABS_TOL: f64 = 1e-7;
const REL_TOL: f64 = 1e-6;
const VAR_FLOOR: f64 = f64::EPSILON * 1e6;
const BERNOULLI_PROB_FLOOR: f64 = f64::EPSILON * 1e6;
const MIN_PROPTEST_CASES: u32 = 1000;
const LOG_2PI: f64 = 1.8378770664093453;

fn proptest_cases() -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|raw| raw.parse::<u32>().ok())
        .map(|parsed| parsed.max(MIN_PROPTEST_CASES))
        .unwrap_or(MIN_PROPTEST_CASES)
}

fn make_f64_view(
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

fn make_univariate_view(values: &[f64]) -> TimeSeriesView<'_> {
    make_f64_view(values, values.len(), 1, MissingPolicy::Error)
        .expect("generated test data should always form a valid TimeSeriesView")
}

fn relative_close(actual: f64, expected: f64) -> bool {
    let diff = (actual - expected).abs();
    let scale = 1.0 + expected.abs();
    diff <= ABS_TOL || diff <= REL_TOL * scale
}

fn naive_l2(values: &[f64], start: usize, end: usize) -> f64 {
    let segment = &values[start..end];
    let len = segment.len() as f64;
    let sum: f64 = segment.iter().sum();
    let mean = sum / len;
    segment
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum()
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

fn log_factorial(n: usize) -> f64 {
    (2..=n).map(|v| (v as f64).ln()).sum()
}

fn ln_gamma_for_default_prior_alpha(segment_len: usize) -> f64 {
    if segment_len.is_multiple_of(2) {
        // alpha_n = 1 + n/2 = integer.
        let alpha_integer = 1 + segment_len / 2;
        log_factorial(alpha_integer.saturating_sub(1))
    } else {
        // alpha_n = 1 + n/2 = k + 1/2 with k = (n + 1)/2.
        let k = segment_len.div_ceil(2);
        log_factorial(2 * k) - (k as f64) * 4.0_f64.ln() - log_factorial(k)
            + 0.5 * std::f64::consts::PI.ln()
    }
}

fn naive_normal(values: &[f64], start: usize, end: usize) -> f64 {
    let segment = &values[start..end];
    let len = segment.len() as f64;
    let sum: f64 = segment.iter().sum();
    let sum_sq: f64 = segment.iter().map(|value| value * value).sum();
    let mean = sum / len;
    let raw_var = sum_sq / len - mean * mean;
    let var = normalize_variance(raw_var);
    len * var.ln()
}

fn naive_poisson(values: &[f64], start: usize, end: usize) -> f64 {
    let segment = &values[start..end];
    let sum: f64 = segment.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }
    let len = segment.len() as f64;
    let lambda = sum / len;
    sum - sum * lambda.ln()
}

fn naive_bernoulli(values: &[f64], start: usize, end: usize) -> f64 {
    let segment = &values[start..end];
    let len = segment.len() as f64;
    let ones: f64 = segment.iter().sum();
    let zeros = len - ones;
    let p_hat = ones / len;
    let log_p = p_hat.max(BERNOULLI_PROB_FLOOR).ln();
    let log_one_minus_p = (1.0 - p_hat).max(BERNOULLI_PROB_FLOOR).ln();
    -(ones * log_p + zeros * log_one_minus_p)
}

fn naive_nig_default_prior(values: &[f64], start: usize, end: usize) -> f64 {
    let prior = NIGPrior::weakly_informative();
    let segment = &values[start..end];
    let n = segment.len() as f64;
    let n_usize = segment.len();
    let mean = segment.iter().sum::<f64>() / n;
    let sse = segment
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f64>();
    let kappa_n = prior.kappa0 + n;
    let alpha_n = prior.alpha0 + 0.5 * n;
    let beta_n = prior.beta0
        + 0.5 * sse
        + (prior.kappa0 * n * (mean - prior.mu0) * (mean - prior.mu0)) / (2.0 * kappa_n);
    let log_marginal = ln_gamma_for_default_prior_alpha(n_usize) + prior.alpha0 * prior.beta0.ln()
        - alpha_n * beta_n.ln()
        + 0.5 * (prior.kappa0.ln() - kappa_n.ln())
        - 0.5 * n * LOG_2PI;
    -log_marginal
}

fn l2_segment_cost(values: &[f64], n: usize, d: usize, start: usize, end: usize) -> f64 {
    let view = make_f64_view(values, n, d, MissingPolicy::Error).expect("view should be valid");
    let model = CostL2Mean::new(ReproMode::Balanced);
    let cache = model
        .precompute(&view, &CachePolicy::Full)
        .expect("precompute should succeed for valid data");
    model.segment_cost(&cache, start, end)
}

fn normal_segment_cost(values: &[f64], n: usize, d: usize, start: usize, end: usize) -> f64 {
    let view = make_f64_view(values, n, d, MissingPolicy::Error).expect("view should be valid");
    let model = CostNormalMeanVar::new(ReproMode::Balanced);
    let cache = model
        .precompute(&view, &CachePolicy::Full)
        .expect("precompute should succeed for valid data");
    model.segment_cost(&cache, start, end)
}

fn nig_segment_cost_with_prior(
    values: &[f64],
    n: usize,
    d: usize,
    start: usize,
    end: usize,
    prior: NIGPrior,
) -> f64 {
    let view = make_f64_view(values, n, d, MissingPolicy::Error).expect("view should be valid");
    let model =
        CostNIGMarginal::with_prior(prior, ReproMode::Balanced).expect("prior should be valid");
    let cache = model
        .precompute(&view, &CachePolicy::Full)
        .expect("precompute should succeed for valid data");
    model.segment_cost(&cache, start, end)
}

fn permute_columns(values: &[f64], n: usize, d: usize, permutation: &[usize]) -> Vec<f64> {
    assert_eq!(permutation.len(), d);
    let mut out = vec![0.0; values.len()];
    for t in 0..n {
        for (new_dim, &old_dim) in permutation.iter().enumerate() {
            out[t * d + new_dim] = values[t * d + old_dim];
        }
    }
    out
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
            let base = 42.0;
            (0..n).map(|idx| base + idx as f64 * 1.0e-12).collect()
        }
    }
}

fn univariate_segment_case_strategy(
    min_segment_len: usize,
) -> impl Strategy<Value = (Vec<f64>, usize, usize)> {
    prop::collection::vec(-1_000.0f64..1_000.0, 8..96).prop_flat_map(move |values| {
        let n = values.len();
        (0usize..=(n - min_segment_len)).prop_flat_map(move |start| {
            let values_for_start = values.clone();
            ((start + min_segment_len)..=n)
                .prop_map(move |end| (values_for_start.clone(), start, end))
        })
    })
}

fn univariate_count_segment_case_strategy(
    min_segment_len: usize,
) -> impl Strategy<Value = (Vec<f64>, usize, usize)> {
    prop::collection::vec(0u16..80u16, 8..96).prop_flat_map(move |counts| {
        let values: Vec<f64> = counts.into_iter().map(f64::from).collect();
        let n = values.len();
        (0usize..=(n - min_segment_len)).prop_flat_map(move |start| {
            let values_for_start = values.clone();
            ((start + min_segment_len)..=n)
                .prop_map(move |end| (values_for_start.clone(), start, end))
        })
    })
}

fn univariate_binary_segment_case_strategy(
    min_segment_len: usize,
) -> impl Strategy<Value = (Vec<f64>, usize, usize)> {
    prop::collection::vec(0u8..=1u8, 8..96).prop_flat_map(move |bits| {
        let values: Vec<f64> = bits.into_iter().map(f64::from).collect();
        let n = values.len();
        (0usize..=(n - min_segment_len)).prop_flat_map(move |start| {
            let values_for_start = values.clone();
            ((start + min_segment_len)..=n)
                .prop_map(move |end| (values_for_start.clone(), start, end))
        })
    })
}

fn stress_segment_case_strategy(
    min_segment_len: usize,
) -> impl Strategy<Value = (usize, u8, usize, usize)> {
    (8usize..96, 0u8..4).prop_flat_map(move |(n, case_id)| {
        (0usize..=(n - min_segment_len)).prop_flat_map(move |start| {
            ((start + min_segment_len)..=n).prop_map(move |end| (n, case_id, start, end))
        })
    })
}

fn multivariate_permutation_case_strategy()
-> impl Strategy<Value = (usize, usize, Vec<f64>, usize, usize, usize, usize)> {
    (8usize..64, 2usize..6).prop_flat_map(|(n, d)| {
        (
            Just(n),
            Just(d),
            prop::collection::vec(-500.0f64..500.0, n * d),
            0usize..(n - 1),
            0usize..d,
            0usize..(d - 1),
        )
            .prop_flat_map(|(n, d, values, start, left_col, right_offset)| {
                let right_col = if right_offset >= left_col {
                    right_offset + 1
                } else {
                    right_offset
                };
                ((start + 2)..=n)
                    .prop_map(move |end| (n, d, values.clone(), start, end, left_col, right_col))
            })
    })
}

#[test]
fn nan_with_missing_policy_error_returns_invalid_input() {
    let values = [1.0_f64, f64::NAN, 3.0, 4.0];
    let err = make_f64_view(&values, 2, 2, MissingPolicy::Error)
        .expect_err("NaN + MissingPolicy::Error should fail");
    assert!(matches!(err, CpdError::InvalidInput(_)));
}

#[test]
fn infinities_are_rejected_at_timeseries_boundary() {
    let policies = [
        MissingPolicy::Error,
        MissingPolicy::ImputeZero,
        MissingPolicy::ImputeLast,
        MissingPolicy::Ignore,
    ];

    for policy in policies {
        let plus_inf = [1.0_f64, f64::INFINITY, 3.0, 4.0];
        let plus_err = make_f64_view(&plus_inf, 2, 2, policy)
            .expect_err("+inf should be rejected for every missing policy");
        assert!(matches!(plus_err, CpdError::InvalidInput(_)));

        let minus_inf = [1.0_f64, f64::NEG_INFINITY, 3.0, 4.0];
        let minus_err = make_f64_view(&minus_inf, 2, 2, policy)
            .expect_err("-inf should be rejected for every missing policy");
        assert!(matches!(minus_err, CpdError::InvalidInput(_)));

        let plus_inf_f32 = [1.0_f32, f32::INFINITY, 3.0, 4.0];
        let plus_err_f32 = TimeSeriesView::from_f32(
            &plus_inf_f32,
            2,
            2,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            policy,
        )
        .expect_err("+inf f32 should be rejected for every missing policy");
        assert!(matches!(plus_err_f32, CpdError::InvalidInput(_)));
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: proptest_cases(),
        max_shrink_iters: 1024,
        failure_persistence: Some(Box::new(FileFailurePersistence::Direct("proptest-regressions/tests/proptest_invariants.txt"))),
        .. ProptestConfig::default()
    })]

    #[test]
    fn l2_segment_cost_is_non_negative_deterministic_and_matches_naive(
        (values, start, end) in univariate_segment_case_strategy(1),
    ) {
        let view = make_univariate_view(&values);
        let model = CostL2Mean::new(ReproMode::Balanced);
        let cache = model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed for valid generated data");

        let actual_first = model.segment_cost(&cache, start, end);
        let actual_second = model.segment_cost(&cache, start, end);
        let expected = naive_l2(&values, start, end);

        prop_assert!(actual_first >= -ABS_TOL);
        prop_assert!(relative_close(actual_first, expected));
        prop_assert!(relative_close(actual_first, actual_second));
    }

    #[test]
    fn normal_segment_cost_is_deterministic_and_matches_naive(
        (values, start, end) in univariate_segment_case_strategy(2),
    ) {
        let view = make_univariate_view(&values);
        let model = CostNormalMeanVar::new(ReproMode::Balanced);
        let cache = model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed for valid generated data");

        let actual_first = model.segment_cost(&cache, start, end);
        let actual_second = model.segment_cost(&cache, start, end);
        let expected = naive_normal(&values, start, end);

        prop_assert!(relative_close(actual_first, expected));
        prop_assert!(relative_close(actual_first, actual_second));
    }

    #[test]
    fn nig_segment_cost_is_deterministic_and_matches_naive(
        (values, start, end) in univariate_segment_case_strategy(1),
    ) {
        let view = make_univariate_view(&values);
        let model = CostNIGMarginal::new(ReproMode::Balanced);
        let cache = model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed for valid generated data");

        let actual_first = model.segment_cost(&cache, start, end);
        let actual_second = model.segment_cost(&cache, start, end);
        let expected = naive_nig_default_prior(&values, start, end);

        prop_assert!(relative_close(actual_first, expected));
        prop_assert!(relative_close(actual_first, actual_second));
    }

    #[test]
    fn poisson_segment_cost_is_deterministic_and_matches_naive(
        (values, start, end) in univariate_count_segment_case_strategy(1),
    ) {
        let view = make_univariate_view(&values);
        let model = CostPoissonRate::new(ReproMode::Balanced);
        let cache = model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed for valid generated data");

        let actual_first = model.segment_cost(&cache, start, end);
        let actual_second = model.segment_cost(&cache, start, end);
        let expected = naive_poisson(&values, start, end);

        prop_assert!(relative_close(actual_first, expected));
        prop_assert!(relative_close(actual_first, actual_second));
    }

    #[test]
    fn bernoulli_segment_cost_is_deterministic_and_matches_naive(
        (values, start, end) in univariate_binary_segment_case_strategy(1),
    ) {
        let view = make_univariate_view(&values);
        let model = CostBernoulli::new(ReproMode::Balanced);
        let cache = model
            .precompute(&view, &CachePolicy::Full)
            .expect("precompute should succeed for valid generated data");

        let actual_first = model.segment_cost(&cache, start, end);
        let actual_second = model.segment_cost(&cache, start, end);
        let expected = naive_bernoulli(&values, start, end);

        prop_assert!(relative_close(actual_first, expected));
        prop_assert!(relative_close(actual_first, actual_second));
    }

    #[test]
    fn l2_segment_cost_is_shift_invariant(
        (values, start, end) in univariate_segment_case_strategy(1),
        shift in -500.0f64..500.0,
    ) {
        let shifted: Vec<f64> = values.iter().map(|value| value + shift).collect();

        let base_view = make_univariate_view(&values);
        let shifted_view = make_univariate_view(&shifted);

        let model = CostL2Mean::new(ReproMode::Balanced);
        let base_cache = model
            .precompute(&base_view, &CachePolicy::Full)
            .expect("base precompute should succeed");
        let shifted_cache = model
            .precompute(&shifted_view, &CachePolicy::Full)
            .expect("shifted precompute should succeed");

        let base_cost = model.segment_cost(&base_cache, start, end);
        let shifted_cost = model.segment_cost(&shifted_cache, start, end);

        prop_assert!(relative_close(base_cost, shifted_cost));
    }

    #[test]
    fn l2_segment_cost_obeys_scale_equivariance_under_affine_transform(
        (values, start, end) in univariate_segment_case_strategy(1),
        shift in -500.0f64..500.0,
        scale in prop_oneof![-8.0f64..-0.2, 0.2f64..8.0],
    ) {
        let n = values.len();

        let transformed: Vec<f64> = values
            .iter()
            .map(|value| scale * value + shift)
            .collect();

        let base_cost = l2_segment_cost(&values, n, 1, start, end);
        let transformed_cost = l2_segment_cost(&transformed, n, 1, start, end);
        let expected = base_cost * scale * scale;

        prop_assert!(relative_close(transformed_cost, expected));
    }

    #[test]
    fn normal_segment_cost_is_shift_invariant(
        (values, start, end) in univariate_segment_case_strategy(2),
        shift in -500.0f64..500.0,
    ) {
        let n = values.len();

        let shifted: Vec<f64> = values.iter().map(|value| value + shift).collect();

        let base_cost = normal_segment_cost(&values, n, 1, start, end);
        let shifted_cost = normal_segment_cost(&shifted, n, 1, start, end);

        prop_assert!(relative_close(base_cost, shifted_cost));
    }

    #[test]
    fn normal_segment_cost_tracks_scale_law_under_affine_transform(
        (values, start, end) in univariate_segment_case_strategy(2),
        shift in -500.0f64..500.0,
        scale in prop_oneof![-8.0f64..-0.2, 0.2f64..8.0],
    ) {
        let n = values.len();

        let transformed: Vec<f64> = values
            .iter()
            .map(|value| scale * value + shift)
            .collect();

        let base_cost = normal_segment_cost(&values, n, 1, start, end);
        let transformed_cost = normal_segment_cost(&transformed, n, 1, start, end);
        let segment_len = (end - start) as f64;
        let expected = base_cost + 2.0 * segment_len * scale.abs().ln();

        prop_assert!(relative_close(transformed_cost, expected));
    }

    #[test]
    fn nig_segment_cost_with_nonzero_prior_mean_remains_finite_under_shift(
        (values, start, end) in univariate_segment_case_strategy(1),
        shift in -50.0f64..50.0,
    ) {
        let n = values.len();
        let shifted: Vec<f64> = values.iter().map(|value| value + shift).collect();
        let prior = NIGPrior {
            mu0: 3.0,
            kappa0: 0.01,
            alpha0: 1.0,
            beta0: 1.0,
        };
        let base = nig_segment_cost_with_prior(&values, n, 1, start, end, prior);
        let shifted_cost = nig_segment_cost_with_prior(&shifted, n, 1, start, end, prior);
        prop_assert!(base.is_finite());
        prop_assert!(shifted_cost.is_finite());
    }

    #[test]
    fn multivariate_column_permutation_preserves_l2_and_normal_costs(
        (n, d, values, start, end, left_col, right_col) in multivariate_permutation_case_strategy(),
    ) {
        let mut permutation: Vec<usize> = (0..d).collect();
        permutation.swap(left_col, right_col);
        let permuted = permute_columns(&values, n, d, &permutation);

        let l2_base = l2_segment_cost(&values, n, d, start, end);
        let l2_permuted = l2_segment_cost(&permuted, n, d, start, end);
        let normal_base = normal_segment_cost(&values, n, d, start, end);
        let normal_permuted = normal_segment_cost(&permuted, n, d, start, end);

        prop_assert!(relative_close(l2_base, l2_permuted));
        prop_assert!(relative_close(normal_base, normal_permuted));
    }

    #[test]
    fn numerical_stress_inputs_produce_finite_costs(
        (n, case_id, start, end) in stress_segment_case_strategy(2),
    ) {
        let values = stress_signal(case_id, n);
        let l2_cost = l2_segment_cost(&values, n, 1, start, end);
        let normal_cost = normal_segment_cost(&values, n, 1, start, end);

        prop_assert!(l2_cost.is_finite());
        prop_assert!(l2_cost >= -ABS_TOL);
        prop_assert!(normal_cost.is_finite());
    }
}
