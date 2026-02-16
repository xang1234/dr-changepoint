// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{segments_from_breakpoints, CpdError, OfflineChangePointResult, OnlineStepResult};

/// Precision/recall/F1 summary for tolerance-based matching.
#[derive(Clone, Debug, PartialEq)]
pub struct F1Metrics {
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

/// Aggregated offline metrics.
#[derive(Clone, Debug, PartialEq)]
pub struct OfflineMetrics {
    pub f1: F1Metrics,
    pub hausdorff_distance: f64,
    pub rand_index: f64,
    pub annotation_error: f64,
}

/// Point on an ROC curve for online alerting.
#[derive(Clone, Debug, PartialEq)]
pub struct RocPoint {
    pub threshold: f64,
    pub true_positive_rate: f64,
    pub false_positive_rate: f64,
    pub detected_changes: usize,
    pub false_alerts: usize,
}

/// Aggregated online metrics.
#[derive(Clone, Debug, PartialEq)]
pub struct OnlineMetrics {
    pub mean_detection_delay: Option<f64>,
    pub false_alarm_rate: f64,
    pub arl0: f64,
    pub arl1: Option<f64>,
    pub detected_changes: usize,
    pub missed_changes: usize,
    pub false_alerts: usize,
    pub total_alerts: usize,
    pub roc_curve: Vec<RocPoint>,
}

/// Computes online/streaming evaluation metrics from per-step detector outputs.
///
/// `true_change_points` must be strictly increasing change indices.
pub fn online_metrics(
    steps: &[OnlineStepResult],
    true_change_points: &[usize],
) -> Result<OnlineMetrics, CpdError> {
    validate_online_inputs(steps, true_change_points)?;
    let observed_change_points = observed_change_points_within_horizon(steps, true_change_points);

    let classification =
        classify_alerts(steps, observed_change_points.as_slice(), |step| step.alert);
    let mean_detection_delay = mean_usize(classification.detection_delays.as_slice());
    let false_alarm_rate = rate(classification.false_alerts, steps.len());
    let arl0 =
        mean_run_length_between_false_alerts(classification.false_alert_positions.as_slice());
    // ARL1 uses the same origin as detection delay: steps from true change index
    // to the first alert that detects the change.
    let arl1 = mean_detection_delay;
    let roc_curve = roc_curve_data_with_validation(
        steps,
        observed_change_points.as_slice(),
        default_roc_thresholds(steps).as_slice(),
    )?;

    Ok(OnlineMetrics {
        mean_detection_delay,
        false_alarm_rate,
        arl0,
        arl1,
        detected_changes: classification.detected_changes,
        missed_changes: classification.missed_changes,
        false_alerts: classification.false_alerts,
        total_alerts: classification.total_alerts,
        roc_curve,
    })
}

/// Computes ROC curve points by sweeping explicit alert thresholds over
/// `OnlineStepResult::p_change`.
///
/// `false_positive_rate` is computed as `false_positives / negatives`, where
/// `negatives` is the number of observed steps that are not true change points
/// in the evaluated horizon.
pub fn roc_curve_data(
    steps: &[OnlineStepResult],
    true_change_points: &[usize],
    thresholds: &[f64],
) -> Result<Vec<RocPoint>, CpdError> {
    validate_online_inputs(steps, true_change_points)?;
    let observed_change_points = observed_change_points_within_horizon(steps, true_change_points);
    roc_curve_data_with_validation(steps, observed_change_points.as_slice(), thresholds)
}

/// Computes offline evaluation metrics from detected and true segmentation outputs.
///
/// Returns an error when exactly one side has no change points, because
/// Hausdorff distance and annotation error are undefined for that case.
pub fn offline_metrics(
    detected: &OfflineChangePointResult,
    truth: &OfflineChangePointResult,
    tolerance: usize,
) -> Result<OfflineMetrics, CpdError> {
    validate_pair(detected, truth)?;
    let detected_cp = detected.change_points.as_slice();
    let truth_cp = truth.change_points.as_slice();
    if exactly_one_empty(detected_cp, truth_cp) {
        return Err(CpdError::invalid_input(
            "offline metrics are undefined when exactly one change-point set is empty",
        ));
    }

    Ok(OfflineMetrics {
        f1: f1_with_tolerance(detected, truth, tolerance)?,
        hausdorff_distance: hausdorff_distance(detected, truth)?,
        rand_index: rand_index(detected, truth)?,
        annotation_error: annotation_error(detected, truth)?,
    })
}

/// Computes precision, recall, and F1 using one-to-one tolerance matching.
///
/// A detected change point is considered a true positive when it can be paired
/// to a true change point within `tolerance`.
pub fn f1_with_tolerance(
    detected: &OfflineChangePointResult,
    truth: &OfflineChangePointResult,
    tolerance: usize,
) -> Result<F1Metrics, CpdError> {
    validate_pair(detected, truth)?;
    let detected_cp = detected.change_points.as_slice();
    let truth_cp = truth.change_points.as_slice();

    let true_positives = count_tolerance_matches(detected_cp, truth_cp, tolerance);
    let false_positives = detected_cp.len() - true_positives;
    let false_negatives = truth_cp.len() - true_positives;

    if detected_cp.is_empty() && truth_cp.is_empty() {
        return Ok(F1Metrics {
            true_positives,
            false_positives,
            false_negatives,
            precision: 1.0,
            recall: 1.0,
            f1: 1.0,
        });
    }

    let precision = if detected_cp.is_empty() {
        0.0
    } else {
        true_positives as f64 / detected_cp.len() as f64
    };
    let recall = if truth_cp.is_empty() {
        0.0
    } else {
        true_positives as f64 / truth_cp.len() as f64
    };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    Ok(F1Metrics {
        true_positives,
        false_positives,
        false_negatives,
        precision,
        recall,
        f1,
    })
}

/// Computes the symmetric Hausdorff distance between detected and true
/// change-point sets.
pub fn hausdorff_distance(
    detected: &OfflineChangePointResult,
    truth: &OfflineChangePointResult,
) -> Result<f64, CpdError> {
    validate_pair(detected, truth)?;
    let detected_cp = detected.change_points.as_slice();
    let truth_cp = truth.change_points.as_slice();

    if detected_cp.is_empty() && truth_cp.is_empty() {
        return Ok(0.0);
    }
    if exactly_one_empty(detected_cp, truth_cp) {
        return Err(CpdError::invalid_input(
            "hausdorff distance is undefined when exactly one change-point set is empty",
        ));
    }

    let d_ab = directed_hausdorff(detected_cp, truth_cp);
    let d_ba = directed_hausdorff(truth_cp, detected_cp);
    Ok(d_ab.max(d_ba) as f64)
}

/// Computes the Rand index between detected and true segmentations.
pub fn rand_index(
    detected: &OfflineChangePointResult,
    truth: &OfflineChangePointResult,
) -> Result<f64, CpdError> {
    let n = validate_pair(detected, truth)?;
    let total_pairs = choose2(n);
    if total_pairs == 0 {
        return Ok(1.0);
    }

    let truth_segments = segments_from_breakpoints(n, truth.breakpoints.as_slice());
    let detected_segments = segments_from_breakpoints(n, detected.breakpoints.as_slice());

    let same_true = truth_segments
        .iter()
        .map(|(start, end)| choose2(end - start))
        .sum::<u128>();
    let same_detected = detected_segments
        .iter()
        .map(|(start, end)| choose2(end - start))
        .sum::<u128>();
    let same_both = overlapping_same_pairs(truth_segments.as_slice(), detected_segments.as_slice());

    let disagreements = same_true + same_detected - 2 * same_both;
    Ok(1.0 - disagreements as f64 / total_pairs as f64)
}

/// Computes mean absolute distance from each detected change point to the
/// nearest true change point.
pub fn annotation_error(
    detected: &OfflineChangePointResult,
    truth: &OfflineChangePointResult,
) -> Result<f64, CpdError> {
    validate_pair(detected, truth)?;
    let detected_cp = detected.change_points.as_slice();
    let truth_cp = truth.change_points.as_slice();

    if detected_cp.is_empty() {
        return Ok(0.0);
    }
    if exactly_one_empty(detected_cp, truth_cp) {
        return Err(CpdError::invalid_input(
            "annotation error is undefined when true change-point set is empty and detected set is non-empty",
        ));
    }

    let total_distance = detected_cp
        .iter()
        .map(|&point| nearest_distance(point, truth_cp) as u128)
        .sum::<u128>();
    Ok(total_distance as f64 / detected_cp.len() as f64)
}

fn validate_pair(
    detected: &OfflineChangePointResult,
    truth: &OfflineChangePointResult,
) -> Result<usize, CpdError> {
    let detected_n = validate_result(detected, "detected")?;
    let truth_n = validate_result(truth, "true")?;
    if detected_n != truth_n {
        return Err(CpdError::invalid_input(format!(
            "detected and true results must share n; got detected_n={detected_n}, true_n={truth_n}"
        )));
    }
    Ok(truth_n)
}

fn validate_result(result: &OfflineChangePointResult, label: &str) -> Result<usize, CpdError> {
    let inferred_n = result
        .breakpoints
        .last()
        .copied()
        .unwrap_or(result.diagnostics.n);
    result.validate(inferred_n).map_err(|err| {
        CpdError::invalid_input(format!(
            "{label} OfflineChangePointResult is invalid: {err}"
        ))
    })?;
    Ok(inferred_n)
}

fn exactly_one_empty(a: &[usize], b: &[usize]) -> bool {
    a.is_empty() != b.is_empty()
}

fn count_tolerance_matches(detected: &[usize], truth: &[usize], tolerance: usize) -> usize {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut matches = 0usize;

    while i < detected.len() && j < truth.len() {
        let d = detected[i];
        let t = truth[j];
        if d.abs_diff(t) <= tolerance {
            matches += 1;
            i += 1;
            j += 1;
            continue;
        }
        if d < t {
            i += 1;
        } else {
            j += 1;
        }
    }

    matches
}

fn directed_hausdorff(a: &[usize], b: &[usize]) -> usize {
    a.iter()
        .map(|&point| nearest_distance(point, b))
        .max()
        .unwrap_or(0)
}

fn nearest_distance(point: usize, sorted_points: &[usize]) -> usize {
    let insertion = sorted_points.partition_point(|&candidate| candidate < point);
    let mut best = usize::MAX;
    if insertion < sorted_points.len() {
        best = best.min(point.abs_diff(sorted_points[insertion]));
    }
    if insertion > 0 {
        best = best.min(point.abs_diff(sorted_points[insertion - 1]));
    }
    best
}

fn choose2(value: usize) -> u128 {
    if value < 2 {
        0
    } else {
        let value_u128 = value as u128;
        value_u128 * (value_u128 - 1) / 2
    }
}

fn overlapping_same_pairs(truth: &[(usize, usize)], detected: &[(usize, usize)]) -> u128 {
    let mut truth_idx = 0usize;
    let mut detected_idx = 0usize;
    let mut pairs = 0u128;

    while truth_idx < truth.len() && detected_idx < detected.len() {
        let (truth_start, truth_end) = truth[truth_idx];
        let (detected_start, detected_end) = detected[detected_idx];
        let overlap_start = truth_start.max(detected_start);
        let overlap_end = truth_end.min(detected_end);
        if overlap_end > overlap_start {
            pairs += choose2(overlap_end - overlap_start);
        }

        if truth_end <= detected_end {
            truth_idx += 1;
        }
        if detected_end <= truth_end {
            detected_idx += 1;
        }
    }

    pairs
}

fn roc_curve_data_with_validation(
    steps: &[OnlineStepResult],
    true_change_points: &[usize],
    thresholds: &[f64],
) -> Result<Vec<RocPoint>, CpdError> {
    for (index, threshold) in thresholds.iter().enumerate() {
        if !threshold.is_finite() {
            return Err(CpdError::invalid_input(format!(
                "thresholds must be finite; thresholds[{index}]={threshold}"
            )));
        }
    }

    let true_change_steps = count_true_change_steps(steps, true_change_points);
    let negative_steps = steps.len().saturating_sub(true_change_steps);
    let mut points = Vec::with_capacity(thresholds.len());
    for &threshold in thresholds {
        let classification =
            classify_alerts(steps, true_change_points, |step| step.p_change >= threshold);
        let true_positive_rate = if true_change_points.is_empty() {
            1.0
        } else {
            classification.detected_changes as f64 / true_change_points.len() as f64
        };
        let false_positive_rate = rate(classification.false_alerts, negative_steps);

        points.push(RocPoint {
            threshold,
            true_positive_rate,
            false_positive_rate,
            detected_changes: classification.detected_changes,
            false_alerts: classification.false_alerts,
        });
    }

    Ok(points)
}

fn observed_change_points_within_horizon(
    steps: &[OnlineStepResult],
    true_change_points: &[usize],
) -> Vec<usize> {
    let Some(last_step) = steps.last() else {
        return Vec::new();
    };
    true_change_points
        .iter()
        .copied()
        .take_while(|&change_point| change_point <= last_step.t)
        .collect()
}

fn count_true_change_steps(steps: &[OnlineStepResult], true_change_points: &[usize]) -> usize {
    let mut change_idx = 0usize;
    let mut count = 0usize;

    for step in steps {
        while change_idx < true_change_points.len() && true_change_points[change_idx] < step.t {
            change_idx += 1;
        }
        if change_idx < true_change_points.len() && true_change_points[change_idx] == step.t {
            count += 1;
            change_idx += 1;
        }
    }

    count
}

fn validate_online_inputs(
    steps: &[OnlineStepResult],
    true_change_points: &[usize],
) -> Result<(), CpdError> {
    for (index, step) in steps.iter().enumerate() {
        if !step.p_change.is_finite() {
            return Err(CpdError::invalid_input(format!(
                "OnlineStepResult::p_change must be finite; steps[{index}].p_change={}",
                step.p_change
            )));
        }
    }

    for index in 1..steps.len() {
        if steps[index - 1].t >= steps[index].t {
            return Err(CpdError::invalid_input(format!(
                "OnlineStepResult::t must be strictly increasing; steps[{}].t={} and steps[{index}].t={}",
                index - 1,
                steps[index - 1].t,
                steps[index].t,
            )));
        }
    }

    for index in 1..true_change_points.len() {
        if true_change_points[index - 1] >= true_change_points[index] {
            return Err(CpdError::invalid_input(format!(
                "true_change_points must be strictly increasing; true_change_points[{}]={} and true_change_points[{index}]={}",
                index - 1,
                true_change_points[index - 1],
                true_change_points[index],
            )));
        }
    }

    Ok(())
}

fn default_roc_thresholds(steps: &[OnlineStepResult]) -> Vec<f64> {
    if steps.is_empty() {
        return vec![1.0, 0.0];
    }

    let mut unique = steps.iter().map(|step| step.p_change).collect::<Vec<_>>();
    unique.sort_by(f64::total_cmp);
    unique.dedup_by(|left, right| left.total_cmp(right).is_eq());

    let min = unique[0];
    let max = unique[unique.len() - 1];
    let mut thresholds = Vec::with_capacity(unique.len() + 2);
    thresholds.push(max + 1.0);
    thresholds.extend(unique.iter().rev().copied());
    thresholds.push(min - 1.0);
    thresholds
}

#[derive(Debug)]
struct AlertClassification {
    detection_delays: Vec<usize>,
    false_alert_positions: Vec<usize>,
    detected_changes: usize,
    missed_changes: usize,
    false_alerts: usize,
    total_alerts: usize,
}

fn classify_alerts<F>(
    steps: &[OnlineStepResult],
    true_change_points: &[usize],
    mut is_alert: F,
) -> AlertClassification
where
    F: FnMut(&OnlineStepResult) -> bool,
{
    let mut cp_idx = 0usize;
    let mut current_change_detected = false;
    let mut detection_delays = Vec::new();
    let mut false_alert_positions = Vec::new();
    let mut missed_changes = 0usize;
    let mut total_alerts = 0usize;

    for (position, step) in steps.iter().enumerate() {
        while cp_idx + 1 < true_change_points.len() && step.t >= true_change_points[cp_idx + 1] {
            if !current_change_detected {
                missed_changes += 1;
            }
            cp_idx += 1;
            current_change_detected = false;
        }

        if !is_alert(step) {
            continue;
        }
        total_alerts += 1;

        if cp_idx < true_change_points.len() {
            let current_change = true_change_points[cp_idx];
            let next_change = true_change_points
                .get(cp_idx + 1)
                .copied()
                .unwrap_or(usize::MAX);
            if step.t >= current_change && step.t < next_change && !current_change_detected {
                detection_delays.push(step.t - current_change);
                current_change_detected = true;
            } else {
                false_alert_positions.push(position);
            }
        } else {
            false_alert_positions.push(position);
        }
    }

    if cp_idx < true_change_points.len() {
        if !current_change_detected {
            missed_changes += 1;
        }
        missed_changes += true_change_points.len() - cp_idx - 1;
    }

    let detected_changes = detection_delays.len();
    let false_alerts = false_alert_positions.len();
    AlertClassification {
        detection_delays,
        false_alert_positions,
        detected_changes,
        missed_changes,
        false_alerts,
        total_alerts,
    }
}

fn mean_usize(values: &[usize]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let total = values.iter().map(|&value| value as u128).sum::<u128>();
    Some(total as f64 / values.len() as f64)
}

fn mean_run_length_between_false_alerts(false_alert_positions: &[usize]) -> f64 {
    if false_alert_positions.is_empty() {
        return f64::INFINITY;
    }

    let mut total_run_length = false_alert_positions[0] + 1;
    for pair in false_alert_positions.windows(2) {
        total_run_length += pair[1] - pair[0];
    }
    total_run_length as f64 / false_alert_positions.len() as f64
}

fn rate(count: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        count as f64 / total as f64
    }
}

/// Evaluation utilities crate name helper.
pub fn crate_name() -> &'static str {
    "cpd-eval"
}

#[cfg(test)]
mod tests {
    use super::{
        annotation_error, f1_with_tolerance, hausdorff_distance, offline_metrics, online_metrics,
        rand_index, roc_curve_data,
    };
    use cpd_core::{Diagnostics, OfflineChangePointResult, OnlineStepResult};
    use std::borrow::Cow;

    fn diagnostics_with_n(n: usize) -> Diagnostics {
        Diagnostics {
            n,
            d: 1,
            algorithm: Cow::Borrowed("test"),
            cost_model: Cow::Borrowed("l2"),
            ..Diagnostics::default()
        }
    }

    fn result(n: usize, breakpoints: &[usize]) -> OfflineChangePointResult {
        OfflineChangePointResult::new(n, breakpoints.to_vec(), diagnostics_with_n(n))
            .expect("test result should be valid")
    }

    fn assert_approx_eq(actual: f64, expected: f64) {
        let delta = (actual - expected).abs();
        assert!(
            delta <= 1e-12,
            "expected {expected}, got {actual} (delta={delta})"
        );
    }

    fn step(t: usize, p_change: f64, alert: bool) -> OnlineStepResult {
        OnlineStepResult {
            t,
            p_change,
            alert,
            alert_reason: alert.then(|| "test-threshold".to_string()),
            run_length_mode: t,
            run_length_mean: t as f64,
            processing_latency_us: None,
        }
    }

    #[test]
    fn f1_with_tolerance_uses_one_to_one_matching() {
        let detected = result(100, &[18, 22, 48, 85, 100]);
        let truth = result(100, &[20, 50, 80, 100]);

        let metrics = f1_with_tolerance(&detected, &truth, 3).expect("f1 should compute");

        assert_eq!(metrics.true_positives, 2);
        assert_eq!(metrics.false_positives, 2);
        assert_eq!(metrics.false_negatives, 1);
        assert_approx_eq(metrics.precision, 0.5);
        assert_approx_eq(metrics.recall, 2.0 / 3.0);
        assert_approx_eq(metrics.f1, 4.0 / 7.0);
    }

    #[test]
    fn f1_with_tolerance_returns_perfect_score_for_no_change_case() {
        let detected = result(100, &[100]);
        let truth = result(100, &[100]);

        let metrics = f1_with_tolerance(&detected, &truth, 0).expect("f1 should compute");
        assert_eq!(metrics.true_positives, 0);
        assert_eq!(metrics.false_positives, 0);
        assert_eq!(metrics.false_negatives, 0);
        assert_approx_eq(metrics.precision, 1.0);
        assert_approx_eq(metrics.recall, 1.0);
        assert_approx_eq(metrics.f1, 1.0);
    }

    #[test]
    fn hausdorff_distance_matches_hand_computed_value() {
        let detected = result(100, &[18, 52, 77, 100]);
        let truth = result(100, &[20, 50, 80, 100]);

        let distance = hausdorff_distance(&detected, &truth).expect("hausdorff should compute");
        assert_approx_eq(distance, 3.0);
    }

    #[test]
    fn hausdorff_distance_rejects_one_sided_empty_change_sets() {
        let detected = result(100, &[25, 100]);
        let truth = result(100, &[100]);

        let err = hausdorff_distance(&detected, &truth)
            .expect_err("one-sided empty set should be rejected");
        assert!(err.to_string().contains("hausdorff distance is undefined"));
    }

    #[test]
    fn rand_index_matches_hand_computed_value() {
        let detected = result(8, &[5, 8]);
        let truth = result(8, &[3, 8]);

        let value = rand_index(&detected, &truth).expect("rand index should compute");
        assert_approx_eq(value, 4.0 / 7.0);
    }

    #[test]
    fn rand_index_returns_one_for_single_sample() {
        let detected = result(1, &[1]);
        let truth = result(1, &[1]);

        let value = rand_index(&detected, &truth).expect("rand index should compute");
        assert_approx_eq(value, 1.0);
    }

    #[test]
    fn annotation_error_matches_hand_computed_mean_distance() {
        let detected = result(100, &[18, 52, 77, 95, 100]);
        let truth = result(100, &[20, 50, 80, 100]);

        let value = annotation_error(&detected, &truth).expect("annotation error should compute");
        assert_approx_eq(value, 5.5);
    }

    #[test]
    fn annotation_error_returns_zero_when_detected_has_no_change_points() {
        let detected = result(100, &[100]);
        let truth = result(100, &[20, 50, 80, 100]);

        let value = annotation_error(&detected, &truth).expect("annotation error should compute");
        assert_approx_eq(value, 0.0);
    }

    #[test]
    fn annotation_error_rejects_empty_true_change_set_with_detections() {
        let detected = result(100, &[25, 100]);
        let truth = result(100, &[100]);

        let err = annotation_error(&detected, &truth)
            .expect_err("one-sided empty set should be rejected");
        assert!(err.to_string().contains("annotation error is undefined"));
    }

    #[test]
    fn offline_metrics_returns_all_metric_components() {
        let detected = result(8, &[5, 8]);
        let truth = result(8, &[3, 8]);

        let metrics = offline_metrics(&detected, &truth, 2).expect("metrics should compute");
        assert_eq!(metrics.f1.true_positives, 1);
        assert_approx_eq(metrics.hausdorff_distance, 2.0);
        assert_approx_eq(metrics.rand_index, 4.0 / 7.0);
        assert_approx_eq(metrics.annotation_error, 2.0);
    }

    #[test]
    fn offline_metrics_rejects_one_sided_empty_change_sets() {
        let detected = result(100, &[25, 100]);
        let truth = result(100, &[100]);

        let err = offline_metrics(&detected, &truth, 3)
            .expect_err("one-sided empty set should be rejected");
        assert!(err.to_string().contains("offline metrics are undefined"));
    }

    #[test]
    fn metrics_reject_results_with_mismatched_n() {
        let detected = result(100, &[20, 100]);
        let truth = result(120, &[20, 120]);

        let err = f1_with_tolerance(&detected, &truth, 0).expect_err("mismatched n should fail");
        assert!(err.to_string().contains("must share n"));
    }

    #[test]
    fn online_metrics_compute_delay_false_alarm_rate_and_arls() {
        let steps = vec![
            step(0, 0.1, false),
            step(1, 0.2, false),
            step(2, 0.6, true),
            step(3, 0.2, false),
            step(4, 0.9, true),
            step(5, 0.7, true),
            step(6, 0.1, false),
            step(7, 0.3, false),
            step(8, 0.8, true),
            step(9, 0.1, false),
        ];

        let metrics = online_metrics(&steps, &[3, 7]).expect("online metrics should compute");
        assert_approx_eq(
            metrics
                .mean_detection_delay
                .expect("delay should be defined when changes are detected"),
            1.0,
        );
        assert_approx_eq(metrics.false_alarm_rate, 0.2);
        assert_approx_eq(metrics.arl0, 3.0);
        assert_approx_eq(
            metrics
                .arl1
                .expect("arl1 should be defined when changes are detected"),
            1.0,
        );
        assert_eq!(metrics.detected_changes, 2);
        assert_eq!(metrics.missed_changes, 0);
        assert_eq!(metrics.false_alerts, 2);
        assert_eq!(metrics.total_alerts, 4);
        assert!(!metrics.roc_curve.is_empty());
    }

    #[test]
    fn online_metrics_reports_misses_and_infinite_arl0_without_false_alerts() {
        let steps = vec![
            step(0, 0.1, false),
            step(1, 0.2, false),
            step(2, 0.3, false),
            step(3, 0.1, false),
            step(4, 0.2, false),
        ];

        let metrics = online_metrics(&steps, &[2]).expect("online metrics should compute");
        assert!(metrics.mean_detection_delay.is_none());
        assert!(metrics.arl1.is_none());
        assert_approx_eq(metrics.false_alarm_rate, 0.0);
        assert!(metrics.arl0.is_infinite());
        assert!(metrics.arl0.is_sign_positive());
        assert_eq!(metrics.detected_changes, 0);
        assert_eq!(metrics.missed_changes, 1);
        assert_eq!(metrics.false_alerts, 0);
        assert_eq!(metrics.total_alerts, 0);
    }

    #[test]
    fn roc_curve_data_matches_threshold_sweep() {
        let steps = vec![
            step(0, 0.1, false),
            step(1, 0.2, false),
            step(2, 0.6, true),
            step(3, 0.2, false),
            step(4, 0.9, true),
            step(5, 0.7, true),
            step(6, 0.1, false),
            step(7, 0.3, false),
            step(8, 0.8, true),
            step(9, 0.1, false),
        ];
        let thresholds = [0.95, 0.75, 0.55];

        let roc = roc_curve_data(&steps, &[3, 7], &thresholds).expect("roc should compute");
        assert_eq!(roc.len(), thresholds.len());

        assert_approx_eq(roc[0].true_positive_rate, 0.0);
        assert_approx_eq(roc[0].false_positive_rate, 0.0);
        assert_eq!(roc[0].detected_changes, 0);
        assert_eq!(roc[0].false_alerts, 0);

        assert_approx_eq(roc[1].true_positive_rate, 1.0);
        assert_approx_eq(roc[1].false_positive_rate, 0.0);
        assert_eq!(roc[1].detected_changes, 2);
        assert_eq!(roc[1].false_alerts, 0);

        assert_approx_eq(roc[2].true_positive_rate, 1.0);
        assert_approx_eq(roc[2].false_positive_rate, 0.25);
        assert_eq!(roc[2].detected_changes, 2);
        assert_eq!(roc[2].false_alerts, 2);
    }

    #[test]
    fn online_metrics_ignore_truth_change_points_after_observed_horizon() {
        let steps = vec![
            step(0, 0.1, false),
            step(1, 0.2, false),
            step(2, 0.3, false),
            step(3, 0.4, false),
            step(4, 0.5, false),
        ];

        let metrics = online_metrics(&steps, &[2, 10]).expect("online metrics should compute");
        assert_eq!(metrics.detected_changes, 0);
        assert_eq!(metrics.missed_changes, 1);
        assert_eq!(metrics.false_alerts, 0);
        assert!(metrics.mean_detection_delay.is_none());
        assert!(metrics.arl1.is_none());
    }

    #[test]
    fn online_metrics_reject_unsorted_truth_change_points() {
        let steps = vec![step(0, 0.1, false), step(1, 0.2, true)];

        let err = online_metrics(&steps, &[4, 2]).expect_err("unsorted change points should fail");
        assert!(err
            .to_string()
            .contains("true_change_points must be strictly increasing"));
    }

    #[test]
    fn online_metrics_reject_non_monotonic_step_times() {
        let steps = vec![step(0, 0.1, false), step(0, 0.2, true)];

        let err = online_metrics(&steps, &[0]).expect_err("non-monotonic times should fail");
        assert!(err
            .to_string()
            .contains("OnlineStepResult::t must be strictly increasing"));
    }
}
