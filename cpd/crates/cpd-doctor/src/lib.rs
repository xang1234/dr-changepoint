// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

pub mod diagnostics;
pub mod recommendation;
pub use diagnostics::{
    DiagnosticsReport, DiagnosticsSummary, DimensionDiagnostics, DoctorDiagnosticsConfig,
    DominantPeriodHint, MissingPattern, compute_diagnostics,
};
pub use recommendation::{
    CalibrationFamily, CalibrationMetrics, CalibrationObservation, CostConfig, DetectorConfig,
    Explanation, FamilyCalibrationMetrics, Objective, OfflineCostKind, OfflineDetectorConfig,
    OnlineDetectorConfig, OnlineDetectorKind, OnlineObservationKind, PipelineConfig, PipelineSpec,
    Recommendation, ResourceEstimate, ValidationReport, ValidationSummary, confidence_formula,
    evaluate_calibration, execute_pipeline, recommend, validate_top_k,
};

#[cfg(feature = "preprocess")]
use cpd_core::{CpdError, DTypeView, MemoryLayout, TimeSeriesView};
#[cfg(feature = "preprocess")]
use cpd_preprocess::{
    DeseasonalizeConfig, DeseasonalizeMethod, DetrendConfig, DetrendMethod, PreprocessConfig,
    PreprocessPipeline, RobustScaleConfig, WinsorizeConfig,
};

#[cfg(feature = "preprocess")]
const DEFAULT_TREND_THRESHOLD: f64 = 0.25;
#[cfg(feature = "preprocess")]
const DEFAULT_SEASONALITY_THRESHOLD: f64 = 0.35;
#[cfg(feature = "preprocess")]
const DEFAULT_OUTLIER_RATE_THRESHOLD: f64 = 0.02;
#[cfg(feature = "preprocess")]
const DEFAULT_SCALE_INSTABILITY_THRESHOLD: f64 = 0.35;
#[cfg(feature = "preprocess")]
const DEFAULT_OUTLIER_Z_THRESHOLD: f64 = 3.5;
#[cfg(feature = "preprocess")]
const DEFAULT_MAX_AUTOCORR_LAG: usize = 128;
#[cfg(feature = "preprocess")]
const DEFAULT_MAD_EPSILON: f64 = 1.0e-9;
#[cfg(feature = "preprocess")]
const DEFAULT_NORMAL_CONSISTENCY: f64 = 1.4826;

#[cfg(feature = "preprocess")]
#[derive(Clone, Debug, PartialEq)]
pub struct DoctorPreprocessConfig {
    pub trend_threshold: f64,
    pub seasonality_threshold: f64,
    pub outlier_rate_threshold: f64,
    pub scale_instability_threshold: f64,
    pub outlier_z_threshold: f64,
    pub max_autocorr_lag: usize,
    pub mad_epsilon: f64,
    pub robust_normal_consistency: f64,
}

#[cfg(feature = "preprocess")]
impl Default for DoctorPreprocessConfig {
    fn default() -> Self {
        Self {
            trend_threshold: DEFAULT_TREND_THRESHOLD,
            seasonality_threshold: DEFAULT_SEASONALITY_THRESHOLD,
            outlier_rate_threshold: DEFAULT_OUTLIER_RATE_THRESHOLD,
            scale_instability_threshold: DEFAULT_SCALE_INSTABILITY_THRESHOLD,
            outlier_z_threshold: DEFAULT_OUTLIER_Z_THRESHOLD,
            max_autocorr_lag: DEFAULT_MAX_AUTOCORR_LAG,
            mad_epsilon: DEFAULT_MAD_EPSILON,
            robust_normal_consistency: DEFAULT_NORMAL_CONSISTENCY,
        }
    }
}

#[cfg(feature = "preprocess")]
#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessRecommendation {
    pub pipeline: Option<PreprocessPipeline>,
    pub reasons: Vec<String>,
    pub signals: Vec<(String, f64)>,
    pub warnings: Vec<String>,
}

#[cfg(feature = "preprocess")]
#[derive(Clone, Copy, Debug)]
struct SignalSummary {
    trend_strength: f64,
    seasonality_strength: f64,
    best_lag: usize,
    outlier_rate: f64,
    scale_instability: f64,
}

#[cfg(feature = "preprocess")]
pub fn recommend_preprocessing(
    x: &TimeSeriesView<'_>,
    cfg: &DoctorPreprocessConfig,
) -> Result<PreprocessRecommendation, CpdError> {
    validate_doctor_config(cfg)?;
    let (values, missing_mask) = flatten_to_c_f64_with_union_missing(x)?;
    let mut warnings = vec![];

    if x.n < 16 {
        warnings.push(format!(
            "small sample size (n={}); preprocessing recommendations may be unstable",
            x.n
        ));
    }

    let summary = compute_signals(&values, &missing_mask, x.n, x.d, cfg, &mut warnings)?;
    let mut reasons = vec![];
    let mut preprocess = PreprocessConfig::default();

    if summary.trend_strength >= cfg.trend_threshold {
        preprocess.detrend = Some(DetrendConfig {
            method: DetrendMethod::Linear,
        });
        reasons.push(format!(
            "trend_strength {:.3} >= threshold {:.3}; recommend linear detrend",
            summary.trend_strength, cfg.trend_threshold
        ));
    }

    if summary.seasonality_strength >= cfg.seasonality_threshold && summary.best_lag >= 2 {
        preprocess.deseasonalize = Some(DeseasonalizeConfig {
            method: DeseasonalizeMethod::Differencing {
                period: summary.best_lag,
            },
        });
        reasons.push(format!(
            "seasonality_strength {:.3} >= threshold {:.3} at lag={}; recommend differencing",
            summary.seasonality_strength, cfg.seasonality_threshold, summary.best_lag
        ));
    }

    if summary.outlier_rate >= cfg.outlier_rate_threshold {
        preprocess.winsorize = Some(WinsorizeConfig::default());
        reasons.push(format!(
            "outlier_rate {:.3} >= threshold {:.3}; recommend winsorization",
            summary.outlier_rate, cfg.outlier_rate_threshold
        ));
    }

    if summary.scale_instability >= cfg.scale_instability_threshold {
        preprocess.robust_scale = Some(RobustScaleConfig::default());
        reasons.push(format!(
            "scale_instability {:.3} >= threshold {:.3}; recommend robust scaling",
            summary.scale_instability, cfg.scale_instability_threshold
        ));
    }

    let has_any = preprocess.detrend.is_some()
        || preprocess.deseasonalize.is_some()
        || preprocess.winsorize.is_some()
        || preprocess.robust_scale.is_some();

    if !has_any {
        reasons.push(
            "no preprocessing strongly indicated; recommend running detector on raw signal"
                .to_string(),
        );
    }

    let pipeline = if has_any {
        Some(PreprocessPipeline::new(preprocess)?)
    } else {
        None
    };

    let signals = vec![
        ("trend_strength".to_string(), summary.trend_strength),
        (
            "seasonality_strength".to_string(),
            summary.seasonality_strength,
        ),
        ("seasonality_best_lag".to_string(), summary.best_lag as f64),
        ("outlier_rate".to_string(), summary.outlier_rate),
        ("scale_instability".to_string(), summary.scale_instability),
    ];

    Ok(PreprocessRecommendation {
        pipeline,
        reasons,
        signals,
        warnings,
    })
}

#[cfg(feature = "preprocess")]
fn validate_doctor_config(cfg: &DoctorPreprocessConfig) -> Result<(), CpdError> {
    if cfg.max_autocorr_lag < 2 {
        return Err(CpdError::invalid_input(format!(
            "DoctorPreprocessConfig.max_autocorr_lag must be >= 2, got {}",
            cfg.max_autocorr_lag
        )));
    }
    for (name, value) in [
        ("trend_threshold", cfg.trend_threshold),
        ("seasonality_threshold", cfg.seasonality_threshold),
        ("outlier_rate_threshold", cfg.outlier_rate_threshold),
        (
            "scale_instability_threshold",
            cfg.scale_instability_threshold,
        ),
        ("outlier_z_threshold", cfg.outlier_z_threshold),
        ("mad_epsilon", cfg.mad_epsilon),
        ("robust_normal_consistency", cfg.robust_normal_consistency),
    ] {
        if !value.is_finite() {
            return Err(CpdError::invalid_input(format!(
                "DoctorPreprocessConfig.{name} must be finite, got {value}"
            )));
        }
    }
    if cfg.trend_threshold < 0.0
        || cfg.seasonality_threshold < 0.0
        || cfg.outlier_rate_threshold < 0.0
        || cfg.scale_instability_threshold < 0.0
    {
        return Err(CpdError::invalid_input(
            "DoctorPreprocessConfig thresholds must be >= 0".to_string(),
        ));
    }
    if cfg.mad_epsilon <= 0.0
        || cfg.robust_normal_consistency <= 0.0
        || cfg.outlier_z_threshold <= 0.0
    {
        return Err(CpdError::invalid_input(
            "DoctorPreprocessConfig mad_epsilon, robust_normal_consistency, and outlier_z_threshold must be > 0".to_string(),
        ));
    }
    Ok(())
}

#[cfg(feature = "preprocess")]
fn compute_signals(
    values: &[f64],
    missing_mask: &[u8],
    n: usize,
    d: usize,
    cfg: &DoctorPreprocessConfig,
    warnings: &mut Vec<String>,
) -> Result<SignalSummary, CpdError> {
    let mut trend_scores = vec![];
    let mut seasonality_scores = vec![];
    let mut lag_candidates = vec![];
    let mut outlier_rates = vec![];
    let mut scale_instabilities = vec![];

    for j in 0..d {
        let series = collect_dim(values, missing_mask, n, d, j);
        let valid_count = series.iter().filter(|v| !v.is_nan()).count();
        if valid_count < 3 {
            warnings.push(format!(
                "dimension {j} has only {valid_count} valid samples; skipping for signal scoring"
            ));
            continue;
        }

        trend_scores.push(trend_strength(&series));
        let (seasonality, lag) = seasonality_strength(&series, n, cfg.max_autocorr_lag);
        seasonality_scores.push(seasonality);
        lag_candidates.push(lag);
        outlier_rates.push(outlier_rate(
            &series,
            cfg.outlier_z_threshold,
            cfg.mad_epsilon,
            cfg.robust_normal_consistency,
        ));
        scale_instabilities.push(scale_instability(
            &series,
            cfg.mad_epsilon,
            cfg.robust_normal_consistency,
        ));
    }

    if trend_scores.is_empty() {
        return Err(CpdError::invalid_input(
            "doctor preprocess recommendation requires at least one dimension with >= 3 valid samples"
                .to_string(),
        ));
    }

    let trend_strength = mean(&trend_scores);
    let seasonality_strength = mean(&seasonality_scores);
    let outlier_rate = mean(&outlier_rates);
    let scale_instability = mean(&scale_instabilities);

    let mut best_idx = 0usize;
    for idx in 1..seasonality_scores.len() {
        if seasonality_scores[idx] > seasonality_scores[best_idx] {
            best_idx = idx;
        }
    }
    let best_lag = lag_candidates[best_idx];

    Ok(SignalSummary {
        trend_strength,
        seasonality_strength,
        best_lag,
        outlier_rate,
        scale_instability,
    })
}

#[cfg(feature = "preprocess")]
fn collect_dim(values: &[f64], missing_mask: &[u8], n: usize, d: usize, j: usize) -> Vec<f64> {
    (0..n)
        .map(|t| {
            let idx = t * d + j;
            if missing_mask[idx] == 1 {
                f64::NAN
            } else {
                values[idx]
            }
        })
        .collect()
}

#[cfg(feature = "preprocess")]
fn trend_strength(series: &[f64]) -> f64 {
    let samples: Vec<(f64, f64)> = series
        .iter()
        .enumerate()
        .filter_map(|(t, y)| (!y.is_nan()).then_some((t as f64, *y)))
        .collect();
    if samples.len() < 3 {
        return 0.0;
    }
    let m = samples.len() as f64;
    let mean_t = samples.iter().map(|(t, _)| *t).sum::<f64>() / m;
    let mean_y = samples.iter().map(|(_, y)| *y).sum::<f64>() / m;
    let mut cov = 0.0;
    let mut var_t = 0.0;
    let mut var_y = 0.0;
    for (t, y) in &samples {
        let dt = *t - mean_t;
        let dy = *y - mean_y;
        cov += dt * dy;
        var_t += dt * dt;
        var_y += dy * dy;
    }
    let denom = (var_t * var_y).sqrt();
    if denom <= f64::EPSILON {
        0.0
    } else {
        (cov / denom).abs().min(1.0)
    }
}

#[cfg(feature = "preprocess")]
fn seasonality_strength(series: &[f64], n: usize, max_autocorr_lag: usize) -> (f64, usize) {
    let max_lag = max_autocorr_lag.min(n / 4);
    if max_lag < 2 {
        return (0.0, 0);
    }
    let mut best = 0.0;
    let mut best_lag = 0usize;
    for lag in 2..=max_lag {
        let corr = autocorr_at_lag(series, lag).abs();
        if corr > best {
            best = corr;
            best_lag = lag;
        }
    }
    (best, best_lag)
}

#[cfg(feature = "preprocess")]
fn autocorr_at_lag(series: &[f64], lag: usize) -> f64 {
    let pairs: Vec<(f64, f64)> = (lag..series.len())
        .filter_map(|t| {
            let a = series[t];
            let b = series[t - lag];
            (!a.is_nan() && !b.is_nan()).then_some((a, b))
        })
        .collect();
    if pairs.len() < 3 {
        return 0.0;
    }
    let m = pairs.len() as f64;
    let mean_a = pairs.iter().map(|(a, _)| *a).sum::<f64>() / m;
    let mean_b = pairs.iter().map(|(_, b)| *b).sum::<f64>() / m;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (a, b) in &pairs {
        let da = *a - mean_a;
        let db = *b - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = (var_a * var_b).sqrt();
    if denom <= f64::EPSILON {
        0.0
    } else {
        (cov / denom).clamp(-1.0, 1.0)
    }
}

#[cfg(feature = "preprocess")]
fn outlier_rate(
    series: &[f64],
    z_threshold: f64,
    mad_epsilon: f64,
    normal_consistency: f64,
) -> f64 {
    let valid: Vec<f64> = series.iter().copied().filter(|v| !v.is_nan()).collect();
    if valid.is_empty() {
        return 0.0;
    }
    let med = median(&valid).unwrap_or(0.0);
    let deviations: Vec<f64> = valid.iter().map(|v| (v - med).abs()).collect();
    let mad = median(&deviations).unwrap_or(0.0);
    let scale = (mad * normal_consistency).max(mad_epsilon);
    let outliers = valid
        .iter()
        .filter(|v| ((**v - med).abs() / scale) > z_threshold)
        .count();
    outliers as f64 / valid.len() as f64
}

#[cfg(feature = "preprocess")]
fn scale_instability(series: &[f64], mad_epsilon: f64, normal_consistency: f64) -> f64 {
    let valid: Vec<f64> = series.iter().copied().filter(|v| !v.is_nan()).collect();
    if valid.len() < 2 {
        return 0.0;
    }
    let mean = valid.iter().sum::<f64>() / valid.len() as f64;
    let variance = valid
        .iter()
        .map(|v| {
            let diff = *v - mean;
            diff * diff
        })
        .sum::<f64>()
        / valid.len() as f64;
    let std = variance.sqrt();

    let med = median(&valid).unwrap_or(0.0);
    let deviations: Vec<f64> = valid.iter().map(|v| (v - med).abs()).collect();
    let mad = median(&deviations).unwrap_or(0.0);
    let robust_std = (mad * normal_consistency).max(mad_epsilon);
    let ratio = (std / robust_std).max(1e-12);
    ratio.ln().abs()
}

#[cfg(feature = "preprocess")]
fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        Some(sorted[mid])
    } else {
        Some((sorted[mid - 1] + sorted[mid]) * 0.5)
    }
}

#[cfg(feature = "preprocess")]
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

#[cfg(feature = "preprocess")]
fn flatten_to_c_f64_with_union_missing(
    x: &TimeSeriesView<'_>,
) -> Result<(Vec<f64>, Vec<u8>), CpdError> {
    let len =
        x.n.checked_mul(x.d)
            .ok_or_else(|| CpdError::invalid_input("n*d overflow while flattening doctor input"))?;
    let mut out = vec![f64::NAN; len];
    let mut missing = vec![0u8; len];
    let source_len = match x.values {
        DTypeView::F32(values) => values.len(),
        DTypeView::F64(values) => values.len(),
    };

    for t in 0..x.n {
        for j in 0..x.d {
            let src = source_index(x.layout, x.n, x.d, t, j)?;
            if src >= source_len {
                return Err(CpdError::invalid_input(format!(
                    "source index out of bounds for doctor input: idx={src}, len={source_len}, t={t}, j={j}, layout={:?}",
                    x.layout
                )));
            }
            let dst = t * x.d + j;
            let value = match x.values {
                DTypeView::F32(values) => f64::from(values[src]),
                DTypeView::F64(values) => values[src],
            };
            let mask_missing = x.missing_mask.map(|mask| mask[src] == 1).unwrap_or(false);
            let is_missing = mask_missing || value.is_nan();
            if is_missing {
                out[dst] = f64::NAN;
                missing[dst] = 1;
            } else {
                out[dst] = value;
            }
        }
    }

    Ok((out, missing))
}

#[cfg(feature = "preprocess")]
fn source_index(
    layout: MemoryLayout,
    n: usize,
    d: usize,
    t: usize,
    j: usize,
) -> Result<usize, CpdError> {
    match layout {
        MemoryLayout::CContiguous => t
            .checked_mul(d)
            .and_then(|base| base.checked_add(j))
            .ok_or_else(|| CpdError::invalid_input("C-layout index overflow in doctor")),
        MemoryLayout::FContiguous => j
            .checked_mul(n)
            .and_then(|base| base.checked_add(t))
            .ok_or_else(|| CpdError::invalid_input("F-layout index overflow in doctor")),
        MemoryLayout::Strided {
            row_stride,
            col_stride,
        } => {
            let t_isize = isize::try_from(t).map_err(|_| {
                CpdError::invalid_input(format!(
                    "time index {t} does not fit in isize for doctor preprocessing"
                ))
            })?;
            let j_isize = isize::try_from(j).map_err(|_| {
                CpdError::invalid_input(format!(
                    "dimension index {j} does not fit in isize for doctor preprocessing"
                ))
            })?;
            let idx = t_isize
                .checked_mul(row_stride)
                .and_then(|left| {
                    j_isize
                        .checked_mul(col_stride)
                        .and_then(|right| left.checked_add(right))
                })
                .ok_or_else(|| {
                    CpdError::invalid_input(format!(
                        "strided index overflow in doctor preprocessing at t={t}, j={j}, row_stride={row_stride}, col_stride={col_stride}"
                    ))
                })?;
            usize::try_from(idx).map_err(|_| {
                CpdError::invalid_input(format!(
                    "strided index became negative in doctor preprocessing at t={t}, j={j}: idx={idx}"
                ))
            })
        }
    }
}

/// Diagnostics and recommendation namespace placeholder.
pub fn crate_name() -> &'static str {
    let _ = (
        cpd_core::crate_name(),
        cpd_offline::crate_name(),
        cpd_online::crate_name(),
    );
    "cpd-doctor"
}

#[cfg(test)]
mod tests {
    use super::crate_name;

    #[test]
    fn crate_name_matches_expected() {
        assert_eq!(crate_name(), "cpd-doctor");
    }
}

#[cfg(all(test, feature = "preprocess"))]
mod preprocess_tests {
    use super::{DoctorPreprocessConfig, recommend_preprocessing};
    use cpd_core::{MemoryLayout, MissingPolicy, TimeIndex, TimeSeriesView};
    use cpd_preprocess::DeseasonalizeMethod;

    fn make_view(values: &[f64], n: usize) -> TimeSeriesView<'_> {
        TimeSeriesView::from_f64(
            values,
            n,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect("test view should be valid")
    }

    #[test]
    fn recommends_detrend_for_strong_trend_signal() {
        let values: Vec<f64> = (0..80).map(|t| 3.0 * t as f64 + 2.0).collect();
        let view = make_view(&values, values.len());
        let rec = recommend_preprocessing(&view, &DoctorPreprocessConfig::default())
            .expect("recommendation should succeed");
        let pipeline = rec.pipeline.expect("pipeline should be present");
        assert!(pipeline.config().detrend.is_some());
    }

    #[test]
    fn recommends_deseasonalize_for_periodic_signal() {
        let mut values = Vec::with_capacity(96);
        for _ in 0..24 {
            values.extend([1.0, 3.0, 1.0, 3.0]);
        }
        let view = make_view(&values, values.len());
        let rec = recommend_preprocessing(&view, &DoctorPreprocessConfig::default())
            .expect("recommendation should succeed");
        let pipeline = rec.pipeline.expect("pipeline should be present");
        let deseasonalize = pipeline
            .config()
            .deseasonalize
            .as_ref()
            .expect("deseasonalize should be present");
        assert_eq!(
            deseasonalize.method,
            DeseasonalizeMethod::Differencing { period: 2 }
        );
    }

    #[test]
    fn recommends_winsorize_for_outlier_heavy_signal() {
        let mut values = vec![0.0; 100];
        values[5] = 25.0;
        values[25] = -30.0;
        values[50] = 22.0;
        values[75] = -27.0;
        let view = make_view(&values, values.len());
        let rec = recommend_preprocessing(&view, &DoctorPreprocessConfig::default())
            .expect("recommendation should succeed");
        let pipeline = rec.pipeline.expect("pipeline should be present");
        assert!(pipeline.config().winsorize.is_some());
    }

    #[test]
    fn recommends_robust_scaling_when_scale_is_unstable() {
        let mut values = vec![0.0; 100];
        values[40] = 30.0;
        values[60] = -30.0;
        let view = make_view(&values, values.len());
        let rec = recommend_preprocessing(&view, &DoctorPreprocessConfig::default())
            .expect("recommendation should succeed");
        let pipeline = rec.pipeline.expect("pipeline should be present");
        assert!(pipeline.config().robust_scale.is_some());
    }

    #[test]
    fn returns_no_pipeline_for_well_behaved_signal() {
        let mut state = 7_u64;
        let mut values = Vec::with_capacity(160);
        for _ in 0..160 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let unit = ((state >> 33) as f64) / ((1_u64 << 31) as f64);
            values.push(unit * 2.0 - 1.0);
        }
        let view = make_view(&values, values.len());
        let rec = recommend_preprocessing(&view, &DoctorPreprocessConfig::default())
            .expect("recommendation should succeed");
        assert!(rec.pipeline.is_none());
    }

    #[test]
    fn recommendation_is_deterministic() {
        let values: Vec<f64> = (0..96)
            .map(|i| if i < 48 { i as f64 } else { i as f64 + 12.0 })
            .collect();
        let view = make_view(&values, values.len());
        let cfg = DoctorPreprocessConfig::default();
        let first = recommend_preprocessing(&view, &cfg).expect("first run should succeed");
        let second = recommend_preprocessing(&view, &cfg).expect("second run should succeed");
        assert_eq!(first, second);
    }
}
