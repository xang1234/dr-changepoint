// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::diagnostics::{
    DiagnosticsSummary, DoctorDiagnosticsConfig, MissingPattern, compute_diagnostics,
};
use cpd_core::{
    Constraints, CpdError, DTypeView, MemoryLayout, TimeSeriesView, validate_constraints,
    validate_constraints_config,
};
use cpd_offline::{BinSegConfig, PeltConfig, WbsConfig, WbsIntervalStrategy};
use cpd_online::{
    BernoulliBetaPrior, BocpdConfig, CusumConfig, GaussianNigPrior, ObservationModel,
    PageHinkleyConfig, PoissonGammaPrior,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

const BINARY_TOLERANCE: f64 = 1.0e-9;
const FAMILY_THRESHOLD: f64 = 0.98;
const MAX_CALIBRATION_BINS: usize = 100;
const CONFIDENCE_FLOOR: f64 = 0.01;
const CONFIDENCE_CEILING: f64 = 0.99;
const OOD_GATING_LAMBDA: f64 = 0.90;
const OOD_GATING_MAX_PENALTY: f64 = 0.80;
const DEFAULT_CALIBRATION_BINS: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Objective {
    Balanced,
    Speed,
    Accuracy,
    Robustness,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CalibrationFamily {
    Gaussian,
    HeavyTailed,
    Autocorrelated,
    Seasonal,
    Multivariate,
    Binary,
    Count,
}

impl CalibrationFamily {
    fn as_str(self) -> &'static str {
        match self {
            Self::Gaussian => "gaussian",
            Self::HeavyTailed => "heavy_tailed",
            Self::Autocorrelated => "autocorrelated",
            Self::Seasonal => "seasonal",
            Self::Multivariate => "multivariate",
            Self::Binary => "binary",
            Self::Count => "count",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationObservation {
    pub family: CalibrationFamily,
    pub predicted_confidence: f64,
    pub top1_within_tolerance: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FamilyCalibrationMetrics {
    pub family: CalibrationFamily,
    pub count: usize,
    pub ece: f64,
    pub brier: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationMetrics {
    pub sample_count: usize,
    pub bins: usize,
    pub ece: f64,
    pub brier: f64,
    pub per_family: Vec<FamilyCalibrationMetrics>,
}

/// Documented confidence formula used by `recommend`.
pub fn confidence_formula() -> &'static str {
    static FORMULA: OnceLock<String> = OnceLock::new();
    FORMULA.get_or_init(|| {
        format!(
            "confidence = clamp((intercept_family + slope_family * heuristic_confidence) * (1 - ood_penalty), {CONFIDENCE_FLOOR:.2}, {CONFIDENCE_CEILING:.2}), where ood_penalty = clamp(1 - exp(-{OOD_GATING_LAMBDA:.2} * diagnostic_divergence), 0.0, {OOD_GATING_MAX_PENALTY:.1})"
        )
    })
}

/// Offline calibration pipeline utility.
///
/// Feed held-out corpus outcomes to compute expected calibration error (ECE) and
/// Brier score overall and per family.
pub fn evaluate_calibration(
    observations: &[CalibrationObservation],
    bins: Option<usize>,
) -> Result<CalibrationMetrics, CpdError> {
    if observations.is_empty() {
        return Err(CpdError::invalid_input(
            "calibration requires at least one observation",
        ));
    }

    let bins = bins.unwrap_or(DEFAULT_CALIBRATION_BINS);
    if bins == 0 || bins > MAX_CALIBRATION_BINS {
        return Err(CpdError::invalid_input(format!(
            "calibration bins must be in [1, {MAX_CALIBRATION_BINS}], got {bins}"
        )));
    }

    let mut overall = CalibrationAccumulator::new(bins);
    let mut grouped = BTreeMap::<CalibrationFamily, CalibrationAccumulator>::new();

    for (idx, observation) in observations.iter().enumerate() {
        let predicted = observation.predicted_confidence;
        if !predicted.is_finite() || !(0.0..=1.0).contains(&predicted) {
            return Err(CpdError::invalid_input(format!(
                "observation[{idx}].predicted_confidence must be finite and within [0, 1], got {predicted}"
            )));
        }

        overall.push(
            predicted,
            if observation.top1_within_tolerance {
                1.0
            } else {
                0.0
            },
        );
        grouped
            .entry(observation.family)
            .or_insert_with(|| CalibrationAccumulator::new(bins))
            .push(
                predicted,
                if observation.top1_within_tolerance {
                    1.0
                } else {
                    0.0
                },
            );
    }
    let (ece, brier, _) = overall.metrics();

    let mut per_family = Vec::with_capacity(grouped.len());
    for (family, accumulator) in grouped {
        let (family_ece, family_brier, count) = accumulator.metrics();
        per_family.push(FamilyCalibrationMetrics {
            family,
            count,
            ece: family_ece,
            brier: family_brier,
        });
    }

    Ok(CalibrationMetrics {
        sample_count: observations.len(),
        bins,
        ece,
        brier,
        per_family,
    })
}

#[derive(Debug)]
struct CalibrationAccumulator {
    bucket_count: Vec<usize>,
    bucket_confidence: Vec<f64>,
    bucket_accuracy: Vec<f64>,
    brier_sum: f64,
    count: usize,
}

impl CalibrationAccumulator {
    fn new(bins: usize) -> Self {
        Self {
            bucket_count: vec![0usize; bins],
            bucket_confidence: vec![0.0; bins],
            bucket_accuracy: vec![0.0; bins],
            brier_sum: 0.0,
            count: 0,
        }
    }

    fn push(&mut self, predicted_confidence: f64, observed: f64) {
        let bins = self.bucket_count.len();
        let idx = ((predicted_confidence * bins as f64).floor() as usize).min(bins - 1);
        self.bucket_count[idx] = self.bucket_count[idx].saturating_add(1);
        self.bucket_confidence[idx] += predicted_confidence;
        self.bucket_accuracy[idx] += observed;
        self.brier_sum += (predicted_confidence - observed).powi(2);
        self.count = self.count.saturating_add(1);
    }

    fn metrics(self) -> (f64, f64, usize) {
        if self.count == 0 {
            return (0.0, 0.0, 0);
        }

        let total = self.count as f64;
        let mut ece = 0.0;
        for bucket in 0..self.bucket_count.len() {
            if self.bucket_count[bucket] == 0 {
                continue;
            }
            let inv = 1.0 / self.bucket_count[bucket] as f64;
            let avg_confidence = self.bucket_confidence[bucket] * inv;
            let avg_accuracy = self.bucket_accuracy[bucket] * inv;
            ece +=
                (self.bucket_count[bucket] as f64 / total) * (avg_accuracy - avg_confidence).abs();
        }

        (ece, self.brier_sum / total, self.count)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Recommendation {
    pub pipeline: PipelineConfig,
    pub resource_estimate: ResourceEstimate,
    pub warnings: Vec<String>,
    pub explanation: Explanation,
    pub validation: Option<ValidationSummary>,
    pub confidence: f64,
    pub confidence_interval: (f64, f64),
    pub abstain_reason: Option<String>,
    pub objective_fit: Vec<(String, f64)>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PipelineConfig {
    Offline {
        detector: OfflineDetectorConfig,
        cost: OfflineCostKind,
        constraints: Constraints,
    },
    Online {
        detector: OnlineDetectorConfig,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum OfflineDetectorConfig {
    Pelt(PeltConfig),
    BinSeg(BinSegConfig),
    Wbs(WbsConfig),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OfflineCostKind {
    L2,
    Normal,
    Nig,
}

#[derive(Clone, Debug, PartialEq)]
pub enum OnlineDetectorConfig {
    Bocpd(BocpdConfig),
    Cusum(CusumConfig),
    PageHinkley(PageHinkleyConfig),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResourceEstimate {
    pub time_complexity: String,
    pub memory_complexity: String,
    pub relative_time_score: f64,
    pub relative_memory_score: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Explanation {
    pub summary: String,
    pub drivers: Vec<(String, f64)>,
    pub tradeoffs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ValidationSummary {
    pub method: String,
    pub notes: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CandidateFamily {
    StrongMapped,
    Generic,
    SafeFallback,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SignalKind {
    Autocorrelated,
    HeavyTail,
    ChangeDense,
    FewStrongChanges,
    MaskingRisk,
    LowSignal,
    BinaryLike,
    CountLike,
}

#[derive(Clone, Copy, Debug)]
struct PerformanceProfile {
    speed: f64,
    accuracy: f64,
    robustness: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Axis {
    Speed,
    Accuracy,
    Robustness,
}

impl PerformanceProfile {
    fn best_axis(self) -> Axis {
        if self.speed >= self.accuracy && self.speed >= self.robustness {
            Axis::Speed
        } else if self.accuracy >= self.speed && self.accuracy >= self.robustness {
            Axis::Accuracy
        } else {
            Axis::Robustness
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct DataHints {
    binary_like: bool,
    count_like: bool,
}

#[derive(Clone, Copy, Debug)]
struct SignalFlags {
    huge_n: bool,
    medium_n: bool,
    autocorrelated: bool,
    heavy_tail: bool,
    change_dense: bool,
    few_strong_changes: bool,
    masking_risk: bool,
    low_signal: bool,
    conflicting_signals: bool,
}

#[derive(Clone, Copy, Debug)]
struct CalibrationRange {
    lower: f64,
    upper: f64,
    weight: f64,
}

#[derive(Clone, Copy, Debug)]
struct CalibrationProfile {
    family: CalibrationFamily,
    intercept: f64,
    slope: f64,
    nan_rate: CalibrationRange,
    kurtosis_proxy: CalibrationRange,
    outlier_rate_iqr: CalibrationRange,
    mad_to_std_ratio: CalibrationRange,
    lag1_autocorr: CalibrationRange,
    change_density_score: CalibrationRange,
    regime_change_proxy: CalibrationRange,
    dominant_period_strength: CalibrationRange,
}

#[derive(Clone, Debug)]
struct CalibrationAssessment {
    family: CalibrationFamily,
    raw_confidence: f64,
    calibrated_confidence: f64,
    final_confidence: f64,
    diagnostic_divergence: f64,
    ood_penalty: f64,
    ood_signals: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
struct ConfidenceContext {
    profile: CalibrationProfile,
}

#[derive(Clone, Debug)]
struct Candidate {
    pipeline: PipelineConfig,
    pipeline_id: String,
    warnings: Vec<String>,
    primary_reason: String,
    driver_keys: Vec<&'static str>,
    profile: PerformanceProfile,
    family: CandidateFamily,
    supported_signals: Vec<SignalKind>,
    evidence_support: f64,
    has_approximation_warning: bool,
}

#[derive(Clone, Debug)]
struct ScoredRecommendation {
    recommendation: Recommendation,
    final_score: f64,
    pipeline_id: String,
    confidence: CalibrationAssessment,
}

pub fn recommend(
    x: &TimeSeriesView<'_>,
    objective: Objective,
    online: bool,
    constraints: Option<Constraints>,
    min_confidence: f64,
    allow_abstain: bool,
) -> Result<Vec<Recommendation>, CpdError> {
    if !min_confidence.is_finite() || !(0.0..=1.0).contains(&min_confidence) {
        return Err(CpdError::invalid_input(format!(
            "min_confidence must be finite and within [0, 1]; got {min_confidence}"
        )));
    }

    let base_constraints = constraints.unwrap_or_default();
    validate_constraints_config(&base_constraints)?;
    let _ = validate_constraints(&base_constraints, x.n)?;

    let diagnostics = compute_diagnostics(x, &DoctorDiagnosticsConfig::default())?;
    let data_hints = sample_data_hints(x, diagnostics.subsample_stride.max(1), BINARY_TOLERANCE)?;
    let flags = signal_flags(x, &diagnostics.summary);
    let strongest_signal = strongest_active_signal(&diagnostics.summary, flags, data_hints);
    let confidence_context = ConfidenceContext {
        profile: calibration_profile(select_calibration_family(
            &diagnostics.summary,
            flags,
            data_hints,
            x.d,
        )),
    };

    let candidates = build_candidates(x, objective, online, &base_constraints, flags, data_hints);
    let mut scored = candidates
        .iter()
        .map(|candidate| {
            score_candidate(
                candidate,
                objective,
                &diagnostics.summary,
                diagnostics.used_subsampling,
                flags,
                strongest_signal,
                confidence_context,
                x.d,
            )
        })
        .collect::<Vec<_>>();

    scored.sort_by(|a, b| {
        b.final_score
            .total_cmp(&a.final_score)
            .then_with(|| {
                b.recommendation
                    .confidence
                    .total_cmp(&a.recommendation.confidence)
            })
            .then_with(|| a.pipeline_id.cmp(&b.pipeline_id))
    });

    if scored.is_empty() {
        return Err(CpdError::invalid_input(
            "recommendation engine failed to produce any candidates",
        ));
    }

    let top_confidence = scored[0].recommendation.confidence;
    if allow_abstain && top_confidence < min_confidence {
        let safe_candidate = build_safe_baseline_candidate(x, online, &base_constraints, flags);
        let top_confidence_assessment = scored[0].confidence.clone();
        let mut safe = score_candidate(
            &safe_candidate,
            objective,
            &diagnostics.summary,
            diagnostics.used_subsampling,
            flags,
            strongest_signal,
            confidence_context,
            x.d,
        )
        .recommendation;
        safe.abstain_reason = Some(abstain_reason(
            &top_confidence_assessment,
            &diagnostics.summary,
            top_confidence,
            min_confidence,
        ));
        safe.warnings.push(
            "returned safe baseline because allow_abstain=true and threshold was not met"
                .to_string(),
        );
        return Ok(vec![safe]);
    }

    if !allow_abstain && top_confidence < min_confidence {
        scored[0].recommendation.warnings.push(format!(
            "top confidence {:.3} is below min_confidence {:.3}; abstain disabled",
            top_confidence, min_confidence
        ));
    }

    let recommendations = scored
        .into_iter()
        .take(5)
        .map(|entry| entry.recommendation)
        .collect::<Vec<_>>();

    Ok(recommendations)
}

fn score_candidate(
    candidate: &Candidate,
    objective: Objective,
    summary: &DiagnosticsSummary,
    used_subsampling: bool,
    flags: SignalFlags,
    strongest_signal: Option<SignalKind>,
    confidence_context: ConfidenceContext,
    dimension_count: usize,
) -> ScoredRecommendation {
    let objective_fit = objective_fit(candidate.profile);
    let selected_objective_score = objective_score(candidate.profile, objective);

    let raw_confidence = confidence_score(
        candidate,
        objective,
        used_subsampling,
        flags,
        strongest_signal,
    );
    let confidence_assessment =
        assess_confidence(raw_confidence, summary, confidence_context, dimension_count);
    let confidence = confidence_assessment.final_confidence;
    let confidence_interval = confidence_interval(
        confidence,
        flags.conflicting_signals,
        confidence_assessment.ood_penalty,
    );
    let mut warnings = candidate.warnings.clone();
    if confidence_assessment.ood_penalty > 0.0 {
        warnings.push(format!(
            "confidence reduced by OOD gating (family={}, divergence={:.3}, penalty={:.3}, raw={:.3}, calibrated={:.3})",
            confidence_assessment.family.as_str(),
            confidence_assessment.diagnostic_divergence,
            confidence_assessment.ood_penalty,
            confidence_assessment.raw_confidence,
            confidence_assessment.calibrated_confidence
        ));
    }

    let explanation = build_explanation(candidate, summary);
    let final_score =
        0.55 * selected_objective_score + 0.35 * confidence + 0.10 * candidate.evidence_support;

    ScoredRecommendation {
        pipeline_id: candidate.pipeline_id.clone(),
        final_score,
        confidence: confidence_assessment,
        recommendation: Recommendation {
            pipeline: candidate.pipeline.clone(),
            resource_estimate: resource_estimate(&candidate.pipeline),
            warnings,
            explanation,
            validation: None,
            confidence,
            confidence_interval,
            abstain_reason: None,
            objective_fit,
        },
    }
}

fn build_explanation(candidate: &Candidate, summary: &DiagnosticsSummary) -> Explanation {
    let mut drivers = Vec::with_capacity(3);
    for key in &candidate.driver_keys {
        drivers.push(((*key).to_string(), diagnostic_value(summary, key)));
        if drivers.len() == 3 {
            break;
        }
    }

    let tradeoff = format!(
        "Tradeoff: speed {:.2}, accuracy {:.2}, robustness {:.2}.",
        candidate.profile.speed, candidate.profile.accuracy, candidate.profile.robustness
    );
    let detail = if candidate.has_approximation_warning {
        "Uses nearest implemented alternative for unavailable model/algorithm.".to_string()
    } else {
        "No approximation mapping required for this path.".to_string()
    };

    Explanation {
        summary: format!(
            "{} because {}",
            pipeline_label(&candidate.pipeline),
            candidate.primary_reason
        ),
        drivers,
        tradeoffs: vec![tradeoff, detail],
    }
}

fn confidence_score(
    candidate: &Candidate,
    objective: Objective,
    used_subsampling: bool,
    flags: SignalFlags,
    strongest_signal: Option<SignalKind>,
) -> f64 {
    let mut confidence: f64 = match candidate.family {
        CandidateFamily::StrongMapped => 0.74,
        CandidateFamily::Generic => 0.64,
        CandidateFamily::SafeFallback => 0.56,
    };

    if used_subsampling {
        confidence -= 0.05;
    }
    if flags.masking_risk {
        confidence -= 0.07;
    }
    if flags.conflicting_signals {
        confidence -= 0.15;
    }
    if candidate.has_approximation_warning {
        confidence -= 0.08;
    }

    if let Some(signal) = strongest_signal
        && candidate.supported_signals.contains(&signal)
    {
        confidence += 0.10;
    }

    if objective_matches_best_axis(candidate.profile, objective) {
        confidence += 0.05;
    }

    confidence.clamp(0.05, 0.95)
}

fn select_calibration_family(
    summary: &DiagnosticsSummary,
    flags: SignalFlags,
    hints: DataHints,
    dimension_count: usize,
) -> CalibrationFamily {
    if dimension_count >= 4 {
        return CalibrationFamily::Multivariate;
    }
    if hints.binary_like {
        return CalibrationFamily::Binary;
    }
    if hints.count_like {
        return CalibrationFamily::Count;
    }

    let seasonal_strength = strongest_period_strength(summary);
    let heavy_tail_score = (summary.kurtosis_proxy / 4.0)
        .max(summary.outlier_rate_iqr / 0.02)
        .max(if summary.mad_to_std_ratio <= 0.80 {
            0.80 / summary.mad_to_std_ratio.max(1.0e-12)
        } else {
            0.0
        });
    let autocorr_score = (summary.lag1_autocorr / 0.55)
        .max(summary.lagk_autocorr / 0.45)
        .max(summary.residual_lag1_autocorr / 0.40);

    if seasonal_strength >= 0.40 {
        return CalibrationFamily::Seasonal;
    }
    if flags.autocorrelated {
        if heavy_tail_score > autocorr_score + 0.25 {
            return CalibrationFamily::HeavyTailed;
        }
        return CalibrationFamily::Autocorrelated;
    }
    if flags.heavy_tail {
        return CalibrationFamily::HeavyTailed;
    }
    CalibrationFamily::Gaussian
}

fn strongest_period_strength(summary: &DiagnosticsSummary) -> f64 {
    summary
        .dominant_period_hints
        .iter()
        .map(|hint| hint.strength)
        .fold(0.0, f64::max)
}

fn calibration_profile(family: CalibrationFamily) -> CalibrationProfile {
    let unit_weight = 1.0;
    match family {
        CalibrationFamily::Gaussian => CalibrationProfile {
            family,
            intercept: 0.04,
            slope: 0.90,
            nan_rate: CalibrationRange {
                lower: 0.0,
                upper: 0.08,
                weight: unit_weight,
            },
            kurtosis_proxy: CalibrationRange {
                lower: 1.8,
                upper: 4.8,
                weight: unit_weight,
            },
            outlier_rate_iqr: CalibrationRange {
                lower: 0.0,
                upper: 0.04,
                weight: 0.9,
            },
            mad_to_std_ratio: CalibrationRange {
                lower: 0.85,
                upper: 1.35,
                weight: 0.7,
            },
            lag1_autocorr: CalibrationRange {
                lower: -0.20,
                upper: 0.45,
                weight: 0.8,
            },
            change_density_score: CalibrationRange {
                lower: 0.03,
                upper: 0.38,
                weight: 0.8,
            },
            regime_change_proxy: CalibrationRange {
                lower: 0.05,
                upper: 1.30,
                weight: 0.7,
            },
            dominant_period_strength: CalibrationRange {
                lower: 0.0,
                upper: 0.35,
                weight: 0.5,
            },
        },
        CalibrationFamily::HeavyTailed => CalibrationProfile {
            family,
            intercept: 0.03,
            slope: 0.88,
            nan_rate: CalibrationRange {
                lower: 0.0,
                upper: 0.12,
                weight: unit_weight,
            },
            kurtosis_proxy: CalibrationRange {
                lower: 3.0,
                upper: 12.0,
                weight: 1.1,
            },
            outlier_rate_iqr: CalibrationRange {
                lower: 0.01,
                upper: 0.20,
                weight: unit_weight,
            },
            mad_to_std_ratio: CalibrationRange {
                lower: 0.15,
                upper: 0.95,
                weight: unit_weight,
            },
            lag1_autocorr: CalibrationRange {
                lower: -0.20,
                upper: 0.50,
                weight: 0.6,
            },
            change_density_score: CalibrationRange {
                lower: 0.02,
                upper: 0.45,
                weight: 0.7,
            },
            regime_change_proxy: CalibrationRange {
                lower: 0.05,
                upper: 1.80,
                weight: 0.7,
            },
            dominant_period_strength: CalibrationRange {
                lower: 0.0,
                upper: 0.45,
                weight: 0.5,
            },
        },
        CalibrationFamily::Autocorrelated => CalibrationProfile {
            family,
            intercept: 0.04,
            slope: 0.89,
            nan_rate: CalibrationRange {
                lower: 0.0,
                upper: 0.12,
                weight: unit_weight,
            },
            kurtosis_proxy: CalibrationRange {
                lower: 1.5,
                upper: 6.5,
                weight: 0.7,
            },
            outlier_rate_iqr: CalibrationRange {
                lower: 0.0,
                upper: 0.08,
                weight: 0.7,
            },
            mad_to_std_ratio: CalibrationRange {
                lower: 0.60,
                upper: 1.30,
                weight: 0.8,
            },
            lag1_autocorr: CalibrationRange {
                lower: 0.35,
                upper: 0.98,
                weight: 1.2,
            },
            change_density_score: CalibrationRange {
                lower: 0.01,
                upper: 0.42,
                weight: 0.7,
            },
            regime_change_proxy: CalibrationRange {
                lower: 0.05,
                upper: 1.60,
                weight: 0.7,
            },
            dominant_period_strength: CalibrationRange {
                lower: 0.0,
                upper: 0.65,
                weight: 0.6,
            },
        },
        CalibrationFamily::Seasonal => CalibrationProfile {
            family,
            intercept: 0.05,
            slope: 0.86,
            nan_rate: CalibrationRange {
                lower: 0.0,
                upper: 0.10,
                weight: unit_weight,
            },
            kurtosis_proxy: CalibrationRange {
                lower: 1.5,
                upper: 7.0,
                weight: 0.8,
            },
            outlier_rate_iqr: CalibrationRange {
                lower: 0.0,
                upper: 0.08,
                weight: 0.8,
            },
            mad_to_std_ratio: CalibrationRange {
                lower: 0.55,
                upper: 1.35,
                weight: 0.8,
            },
            lag1_autocorr: CalibrationRange {
                lower: 0.10,
                upper: 0.95,
                weight: 0.9,
            },
            change_density_score: CalibrationRange {
                lower: 0.01,
                upper: 0.55,
                weight: 0.6,
            },
            regime_change_proxy: CalibrationRange {
                lower: 0.05,
                upper: 1.80,
                weight: 0.6,
            },
            dominant_period_strength: CalibrationRange {
                lower: 0.25,
                upper: 1.0,
                weight: 1.4,
            },
        },
        CalibrationFamily::Multivariate => CalibrationProfile {
            family,
            intercept: 0.06,
            slope: 0.84,
            nan_rate: CalibrationRange {
                lower: 0.0,
                upper: 0.15,
                weight: unit_weight,
            },
            kurtosis_proxy: CalibrationRange {
                lower: 1.2,
                upper: 9.0,
                weight: 0.8,
            },
            outlier_rate_iqr: CalibrationRange {
                lower: 0.0,
                upper: 0.12,
                weight: 0.8,
            },
            mad_to_std_ratio: CalibrationRange {
                lower: 0.45,
                upper: 1.45,
                weight: 0.9,
            },
            lag1_autocorr: CalibrationRange {
                lower: -0.25,
                upper: 0.90,
                weight: 0.8,
            },
            change_density_score: CalibrationRange {
                lower: 0.01,
                upper: 0.55,
                weight: 0.8,
            },
            regime_change_proxy: CalibrationRange {
                lower: 0.05,
                upper: 2.00,
                weight: 0.8,
            },
            dominant_period_strength: CalibrationRange {
                lower: 0.0,
                upper: 0.75,
                weight: 0.7,
            },
        },
        CalibrationFamily::Binary => CalibrationProfile {
            family,
            intercept: 0.04,
            slope: 0.88,
            nan_rate: CalibrationRange {
                lower: 0.0,
                upper: 0.12,
                weight: 1.0,
            },
            kurtosis_proxy: CalibrationRange {
                lower: 0.2,
                upper: 30.0,
                weight: 0.3,
            },
            outlier_rate_iqr: CalibrationRange {
                lower: 0.0,
                upper: 0.15,
                weight: 0.4,
            },
            mad_to_std_ratio: CalibrationRange {
                lower: 0.0,
                upper: 1.6,
                weight: 0.3,
            },
            lag1_autocorr: CalibrationRange {
                lower: -1.0,
                upper: 1.0,
                weight: 0.3,
            },
            change_density_score: CalibrationRange {
                lower: 0.0,
                upper: 1.0,
                weight: 0.8,
            },
            regime_change_proxy: CalibrationRange {
                lower: 0.0,
                upper: 3.0,
                weight: 0.8,
            },
            dominant_period_strength: CalibrationRange {
                lower: 0.0,
                upper: 1.0,
                weight: 0.4,
            },
        },
        CalibrationFamily::Count => CalibrationProfile {
            family,
            intercept: 0.04,
            slope: 0.87,
            nan_rate: CalibrationRange {
                lower: 0.0,
                upper: 0.12,
                weight: 1.0,
            },
            kurtosis_proxy: CalibrationRange {
                lower: 0.5,
                upper: 20.0,
                weight: 0.5,
            },
            outlier_rate_iqr: CalibrationRange {
                lower: 0.0,
                upper: 0.25,
                weight: 0.7,
            },
            mad_to_std_ratio: CalibrationRange {
                lower: 0.0,
                upper: 1.8,
                weight: 0.4,
            },
            lag1_autocorr: CalibrationRange {
                lower: -1.0,
                upper: 1.0,
                weight: 0.4,
            },
            change_density_score: CalibrationRange {
                lower: 0.0,
                upper: 1.0,
                weight: 0.8,
            },
            regime_change_proxy: CalibrationRange {
                lower: 0.0,
                upper: 3.0,
                weight: 0.8,
            },
            dominant_period_strength: CalibrationRange {
                lower: 0.0,
                upper: 1.0,
                weight: 0.4,
            },
        },
    }
}

fn assess_confidence(
    raw_confidence: f64,
    summary: &DiagnosticsSummary,
    confidence_context: ConfidenceContext,
    dimension_count: usize,
) -> CalibrationAssessment {
    let profile = confidence_context.profile;
    let calibrated_confidence = (profile.intercept + profile.slope * raw_confidence)
        .clamp(CONFIDENCE_FLOOR, CONFIDENCE_CEILING);
    let (diagnostic_divergence, ood_signals) =
        diagnostic_divergence(summary, profile, dimension_count);
    let ood_penalty = (1.0 - (-OOD_GATING_LAMBDA * diagnostic_divergence).exp())
        .clamp(0.0, OOD_GATING_MAX_PENALTY);
    let final_confidence =
        (calibrated_confidence * (1.0 - ood_penalty)).clamp(CONFIDENCE_FLOOR, CONFIDENCE_CEILING);

    CalibrationAssessment {
        family: profile.family,
        raw_confidence,
        calibrated_confidence,
        final_confidence,
        diagnostic_divergence,
        ood_penalty,
        ood_signals,
    }
}

fn diagnostic_divergence(
    summary: &DiagnosticsSummary,
    profile: CalibrationProfile,
    dimension_count: usize,
) -> (f64, Vec<String>) {
    let dominant_period_strength = strongest_period_strength(summary);
    let features = [
        ("nan_rate", summary.nan_rate, profile.nan_rate),
        (
            "kurtosis_proxy",
            summary.kurtosis_proxy,
            profile.kurtosis_proxy,
        ),
        (
            "outlier_rate_iqr",
            summary.outlier_rate_iqr,
            profile.outlier_rate_iqr,
        ),
        (
            "mad_to_std_ratio",
            summary.mad_to_std_ratio,
            profile.mad_to_std_ratio,
        ),
        (
            "lag1_autocorr",
            summary.lag1_autocorr,
            profile.lag1_autocorr,
        ),
        (
            "change_density_score",
            summary.change_density_score,
            profile.change_density_score,
        ),
        (
            "regime_change_proxy",
            summary.regime_change_proxy,
            profile.regime_change_proxy,
        ),
        (
            "dominant_period_strength",
            dominant_period_strength,
            profile.dominant_period_strength,
        ),
    ];

    let mut weighted_divergence = 0.0;
    let mut total_weight = 0.0;
    let mut ood_signals = Vec::new();

    for (name, value, range) in features {
        let (divergence, reason) = range_divergence(name, value, range);
        weighted_divergence += range.weight * divergence;
        total_weight += range.weight;
        if let Some(reason) = reason {
            ood_signals.push(reason);
        }
    }

    if summary.missing_pattern == MissingPattern::Block {
        weighted_divergence += 0.8;
        total_weight += 1.0;
        ood_signals.push("missing_pattern=block outside calibration support".to_string());
    }

    if dimension_count > 0 {
        let coverage = summary.valid_dimensions as f64 / dimension_count as f64;
        if coverage < 0.70 {
            let divergence = ((0.70 - coverage) / 0.70).clamp(0.0, 3.0);
            weighted_divergence += divergence;
            total_weight += 1.0;
            ood_signals.push(format!(
                "valid_dimension_coverage={coverage:.3} below calibration minimum 0.700"
            ));
        }
    }

    if summary.longest_nan_run > 512 {
        let run_divergence = ((summary.longest_nan_run as f64 - 512.0) / 512.0).clamp(0.0, 3.0);
        weighted_divergence += 0.8 * run_divergence;
        total_weight += 0.8;
        ood_signals.push(format!(
            "longest_nan_run={} exceeds calibration support up to 512",
            summary.longest_nan_run
        ));
    }

    let diagnostic_divergence = if total_weight <= 0.0 {
        0.0
    } else {
        weighted_divergence / total_weight
    };
    (diagnostic_divergence.max(0.0), ood_signals)
}

fn range_divergence(name: &str, value: f64, range: CalibrationRange) -> (f64, Option<String>) {
    let width = (range.upper - range.lower).abs().max(1.0e-9);
    if value < range.lower {
        let divergence = ((range.lower - value) / width).clamp(0.0, 3.0);
        (
            divergence,
            Some(format!(
                "{name}={value:.3} outside [{:.3}, {:.3}]",
                range.lower, range.upper
            )),
        )
    } else if value > range.upper {
        let divergence = ((value - range.upper) / width).clamp(0.0, 3.0);
        (
            divergence,
            Some(format!(
                "{name}={value:.3} outside [{:.3}, {:.3}]",
                range.lower, range.upper
            )),
        )
    } else {
        (0.0, None)
    }
}

fn abstain_reason(
    confidence: &CalibrationAssessment,
    summary: &DiagnosticsSummary,
    top_confidence: f64,
    min_confidence: f64,
) -> String {
    if confidence.ood_signals.is_empty() {
        format!(
            "top confidence {:.3} below min_confidence {:.3}; drivers: {}",
            top_confidence,
            min_confidence,
            top_driver_summary(summary)
        )
    } else {
        format!(
            "diagnostics outside calibration support: {}; adjusted confidence {:.3} below min_confidence {:.3}",
            confidence.ood_signals.join("; "),
            top_confidence,
            min_confidence
        )
    }
}

fn confidence_interval(confidence: f64, conflicting_signals: bool, ood_penalty: f64) -> (f64, f64) {
    let half_width = (0.10
        + 0.25 * (1.0 - confidence)
        + if conflicting_signals { 0.05 } else { 0.0 }
        + 0.20 * ood_penalty)
        .clamp(0.08, 0.45);
    (
        (confidence - half_width).max(0.0),
        (confidence + half_width).min(1.0),
    )
}

fn objective_matches_best_axis(profile: PerformanceProfile, objective: Objective) -> bool {
    match objective {
        Objective::Balanced => false,
        Objective::Speed => profile.best_axis() == Axis::Speed,
        Objective::Accuracy => profile.best_axis() == Axis::Accuracy,
        Objective::Robustness => profile.best_axis() == Axis::Robustness,
    }
}

fn objective_fit(profile: PerformanceProfile) -> Vec<(String, f64)> {
    vec![
        (
            "balanced".to_string(),
            objective_score(profile, Objective::Balanced),
        ),
        (
            "speed".to_string(),
            objective_score(profile, Objective::Speed),
        ),
        (
            "accuracy".to_string(),
            objective_score(profile, Objective::Accuracy),
        ),
        (
            "robustness".to_string(),
            objective_score(profile, Objective::Robustness),
        ),
    ]
}

fn objective_score(profile: PerformanceProfile, objective: Objective) -> f64 {
    let (w_speed, w_accuracy, w_robustness) = match objective {
        Objective::Balanced => (0.34, 0.33, 0.33),
        Objective::Speed => (0.60, 0.20, 0.20),
        Objective::Accuracy => (0.15, 0.65, 0.20),
        Objective::Robustness => (0.15, 0.25, 0.60),
    };
    (w_speed * profile.speed + w_accuracy * profile.accuracy + w_robustness * profile.robustness)
        .clamp(0.0, 1.0)
}

fn signal_flags(x: &TimeSeriesView<'_>, summary: &DiagnosticsSummary) -> SignalFlags {
    let huge_n = x.n >= 100_000;
    let medium_n = (1_000..100_000).contains(&x.n);

    let autocorrelated = summary.lag1_autocorr >= 0.55
        || summary.lagk_autocorr >= 0.45
        || summary.residual_lag1_autocorr >= 0.40;
    let heavy_tail = summary.kurtosis_proxy >= 4.0
        || summary.outlier_rate_iqr >= 0.02
        || summary.mad_to_std_ratio <= 0.80;
    let change_dense = summary.change_density_score >= 0.35 || summary.regime_change_proxy >= 1.20;
    let few_strong_changes =
        summary.change_density_score < 0.12 && summary.regime_change_proxy >= 0.80;
    let masking_risk = summary.missing_pattern == MissingPattern::Block || summary.nan_rate >= 0.10;
    let low_signal = summary.change_density_score < 0.05 && summary.regime_change_proxy < 0.25;

    let conflicting_signals = (autocorrelated && heavy_tail && x.n < 1_000)
        || (low_signal && (autocorrelated || heavy_tail))
        || summary.valid_dimensions.saturating_mul(2) < x.d;

    SignalFlags {
        huge_n,
        medium_n,
        autocorrelated,
        heavy_tail,
        change_dense,
        few_strong_changes,
        masking_risk,
        low_signal,
        conflicting_signals,
    }
}

fn strongest_active_signal(
    summary: &DiagnosticsSummary,
    flags: SignalFlags,
    hints: DataHints,
) -> Option<SignalKind> {
    let mut scored = Vec::<(SignalKind, f64)>::new();

    if hints.binary_like {
        scored.push((SignalKind::BinaryLike, 2.0));
    } else if hints.count_like {
        scored.push((SignalKind::CountLike, 1.8));
    }

    if flags.autocorrelated {
        let score = (summary.lag1_autocorr / 0.55)
            .max(summary.lagk_autocorr / 0.45)
            .max(summary.residual_lag1_autocorr / 0.40);
        scored.push((SignalKind::Autocorrelated, score));
    }
    if flags.heavy_tail {
        let mad_term = if summary.mad_to_std_ratio <= 0.80 {
            0.80 / summary.mad_to_std_ratio.max(1e-12)
        } else {
            0.0
        };
        let score = (summary.kurtosis_proxy / 4.0)
            .max(summary.outlier_rate_iqr / 0.02)
            .max(mad_term);
        scored.push((SignalKind::HeavyTail, score));
    }
    if flags.change_dense {
        let score = (summary.change_density_score / 0.35).max(summary.regime_change_proxy / 1.20);
        scored.push((SignalKind::ChangeDense, score));
    }
    if flags.few_strong_changes {
        let score = (summary.regime_change_proxy / 0.80).max(1.0);
        scored.push((SignalKind::FewStrongChanges, score));
    }
    if flags.masking_risk {
        let score = (summary.nan_rate / 0.10).max(1.0);
        scored.push((SignalKind::MaskingRisk, score));
    }
    if flags.low_signal {
        let score = 1.0
            + ((0.05 - summary.change_density_score).max(0.0) / 0.05)
            + ((0.25 - summary.regime_change_proxy).max(0.0) / 0.25);
        scored.push((SignalKind::LowSignal, score));
    }

    scored
        .into_iter()
        .max_by(|(signal_a, score_a), (signal_b, score_b)| {
            score_a
                .total_cmp(score_b)
                .then_with(|| signal_priority(*signal_a).cmp(&signal_priority(*signal_b)))
        })
        .map(|(signal, _)| signal)
}

fn signal_priority(signal: SignalKind) -> usize {
    match signal {
        SignalKind::BinaryLike => 8,
        SignalKind::CountLike => 7,
        SignalKind::Autocorrelated => 6,
        SignalKind::HeavyTail => 5,
        SignalKind::ChangeDense => 4,
        SignalKind::FewStrongChanges => 3,
        SignalKind::MaskingRisk => 2,
        SignalKind::LowSignal => 1,
    }
}

fn build_candidates(
    x: &TimeSeriesView<'_>,
    objective: Objective,
    online: bool,
    base_constraints: &Constraints,
    flags: SignalFlags,
    hints: DataHints,
) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    let mut seen = BTreeSet::new();

    if online {
        build_online_candidates(x, objective, hints, flags, &mut candidates, &mut seen);
    } else {
        build_offline_candidates(x, base_constraints, flags, &mut candidates, &mut seen);
    }

    if candidates.is_empty() {
        candidates.push(build_safe_baseline_candidate(
            x,
            online,
            base_constraints,
            flags,
        ));
    }

    candidates
}

fn build_offline_candidates(
    x: &TimeSeriesView<'_>,
    base_constraints: &Constraints,
    flags: SignalFlags,
    out: &mut Vec<Candidate>,
    seen: &mut BTreeSet<String>,
) {
    if flags.huge_n {
        if flags.autocorrelated {
            let constraints = apply_jump_thinning(base_constraints, x.n, true);
            let candidate = Candidate {
                pipeline: PipelineConfig::Offline {
                    detector: OfflineDetectorConfig::Pelt(PeltConfig::default()),
                    cost: OfflineCostKind::Normal,
                    constraints,
                },
                pipeline_id: format!(
                    "offline:pelt:normal:jump={}",
                    apply_jump_thinning(base_constraints, x.n, true).jump
                ),
                warnings: vec![
                    "mapped autocorrelation intent (CostAR/CostLinear) to available Normal cost"
                        .to_string(),
                ],
                primary_reason:
                    "large-n offline series with autocorrelation favors robust PELT+Normal"
                        .to_string(),
                driver_keys: vec!["lag1_autocorr", "lagk_autocorr", "residual_lag1_autocorr"],
                profile: PerformanceProfile {
                    speed: 0.80,
                    accuracy: 0.78,
                    robustness: 0.62,
                },
                family: CandidateFamily::StrongMapped,
                supported_signals: vec![SignalKind::Autocorrelated],
                evidence_support: evidence_support(0.78, flags, true),
                has_approximation_warning: true,
            };
            push_candidate(candidate, out, seen);
        }

        if flags.heavy_tail {
            let constraints = apply_jump_thinning(base_constraints, x.n, true);
            let candidate = Candidate {
                pipeline: PipelineConfig::Offline {
                    detector: OfflineDetectorConfig::Pelt(PeltConfig::default()),
                    cost: OfflineCostKind::Nig,
                    constraints,
                },
                pipeline_id: format!(
                    "offline:pelt:nig:jump={}",
                    apply_jump_thinning(base_constraints, x.n, true).jump
                ),
                warnings: vec![],
                primary_reason: "heavy-tail and outlier indicators favor robust NIG marginal cost"
                    .to_string(),
                driver_keys: vec!["kurtosis_proxy", "outlier_rate_iqr", "mad_to_std_ratio"],
                profile: PerformanceProfile {
                    speed: 0.68,
                    accuracy: 0.82,
                    robustness: 0.85,
                },
                family: CandidateFamily::StrongMapped,
                supported_signals: vec![SignalKind::HeavyTail],
                evidence_support: evidence_support(0.82, flags, false),
                has_approximation_warning: false,
            };
            push_candidate(candidate, out, seen);
        }

        let constraints = apply_jump_thinning(base_constraints, x.n, true);
        let candidate = Candidate {
            pipeline: PipelineConfig::Offline {
                detector: OfflineDetectorConfig::Pelt(PeltConfig::default()),
                cost: OfflineCostKind::L2,
                constraints,
            },
            pipeline_id: format!(
                "offline:pelt:l2:jump={}",
                apply_jump_thinning(base_constraints, x.n, true).jump
            ),
            warnings: vec![],
            primary_reason:
                "default large-n offline path uses PELT+L2 with jump thinning for speed".to_string(),
            driver_keys: vec!["change_density_score", "regime_change_proxy", "nan_rate"],
            profile: PerformanceProfile {
                speed: 0.90,
                accuracy: 0.72,
                robustness: 0.55,
            },
            family: CandidateFamily::Generic,
            supported_signals: vec![SignalKind::ChangeDense],
            evidence_support: evidence_support(0.60, flags, false),
            has_approximation_warning: false,
        };
        push_candidate(candidate, out, seen);
    } else if flags.medium_n {
        if flags.few_strong_changes && flags.masking_risk {
            let mut wbs = WbsConfig::default();
            wbs.interval_strategy = WbsIntervalStrategy::Stratified;

            let candidate = Candidate {
                pipeline: PipelineConfig::Offline {
                    detector: OfflineDetectorConfig::Wbs(wbs),
                    cost: OfflineCostKind::Normal,
                    constraints: base_constraints.clone(),
                },
                pipeline_id: format!("offline:wbs:normal:jump={}", base_constraints.jump),
                warnings: vec![],
                primary_reason:
                    "few strong changes with masking risk favors WBS to reduce masking artifacts"
                        .to_string(),
                driver_keys: vec!["nan_rate", "change_density_score", "regime_change_proxy"],
                profile: PerformanceProfile {
                    speed: 0.62,
                    accuracy: 0.84,
                    robustness: 0.73,
                },
                family: CandidateFamily::StrongMapped,
                supported_signals: vec![SignalKind::FewStrongChanges, SignalKind::MaskingRisk],
                evidence_support: evidence_support(0.80, flags, false),
                has_approximation_warning: false,
            };
            push_candidate(candidate, out, seen);
        }

        if flags.few_strong_changes && !flags.masking_risk {
            let candidate = Candidate {
                pipeline: PipelineConfig::Offline {
                    detector: OfflineDetectorConfig::BinSeg(BinSegConfig::default()),
                    cost: OfflineCostKind::Normal,
                    constraints: base_constraints.clone(),
                },
                pipeline_id: format!("offline:binseg:normal:jump={}", base_constraints.jump),
                warnings: vec![],
                primary_reason:
                    "few strong changes in medium-n series favors BinSeg for fast hierarchy"
                        .to_string(),
                driver_keys: vec![
                    "change_density_score",
                    "regime_change_proxy",
                    "lag1_autocorr",
                ],
                profile: PerformanceProfile {
                    speed: 0.86,
                    accuracy: 0.70,
                    robustness: 0.60,
                },
                family: CandidateFamily::StrongMapped,
                supported_signals: vec![SignalKind::FewStrongChanges],
                evidence_support: evidence_support(0.76, flags, false),
                has_approximation_warning: false,
            };
            push_candidate(candidate, out, seen);
        }

        if flags.change_dense {
            let pelt_candidate = Candidate {
                pipeline: PipelineConfig::Offline {
                    detector: OfflineDetectorConfig::Pelt(PeltConfig::default()),
                    cost: OfflineCostKind::L2,
                    constraints: base_constraints.clone(),
                },
                pipeline_id: format!("offline:pelt:l2:jump={}", base_constraints.jump),
                warnings: vec![],
                primary_reason:
                    "change-dense medium-n series benefits from stable global PELT search"
                        .to_string(),
                driver_keys: vec![
                    "change_density_score",
                    "regime_change_proxy",
                    "lagk_autocorr",
                ],
                profile: PerformanceProfile {
                    speed: 0.90,
                    accuracy: 0.72,
                    robustness: 0.55,
                },
                family: CandidateFamily::StrongMapped,
                supported_signals: vec![SignalKind::ChangeDense],
                evidence_support: evidence_support(0.70, flags, false),
                has_approximation_warning: false,
            };
            push_candidate(pelt_candidate, out, seen);

            let mut wbs = WbsConfig::default();
            wbs.interval_strategy = WbsIntervalStrategy::Random;
            let wbs_candidate = Candidate {
                pipeline: PipelineConfig::Offline {
                    detector: OfflineDetectorConfig::Wbs(wbs),
                    cost: OfflineCostKind::L2,
                    constraints: base_constraints.clone(),
                },
                pipeline_id: format!("offline:wbs:l2:jump={}", base_constraints.jump),
                warnings: vec![
                    "BottomUp is unavailable; mapped change-dense branch to WBS".to_string(),
                ],
                primary_reason:
                    "change-dense branch mapped to WBS where BottomUp is not yet available"
                        .to_string(),
                driver_keys: vec!["change_density_score", "regime_change_proxy", "nan_rate"],
                profile: PerformanceProfile {
                    speed: 0.66,
                    accuracy: 0.76,
                    robustness: 0.58,
                },
                family: CandidateFamily::StrongMapped,
                supported_signals: vec![SignalKind::ChangeDense],
                evidence_support: evidence_support(0.66, flags, true),
                has_approximation_warning: true,
            };
            push_candidate(wbs_candidate, out, seen);
        }
    } else {
        let candidate = Candidate {
            pipeline: PipelineConfig::Offline {
                detector: OfflineDetectorConfig::BinSeg(BinSegConfig::default()),
                cost: OfflineCostKind::Normal,
                constraints: base_constraints.clone(),
            },
            pipeline_id: format!("offline:binseg:normal:jump={}", base_constraints.jump),
            warnings: vec![
                "Dynp is unavailable; mapped small-n exact branch to BinSeg".to_string(),
            ],
            primary_reason:
                "small-n branch mapped to BinSeg as nearest available exact-like option".to_string(),
            driver_keys: vec!["regime_change_proxy", "change_density_score", "nan_rate"],
            profile: PerformanceProfile {
                speed: 0.86,
                accuracy: 0.70,
                robustness: 0.60,
            },
            family: CandidateFamily::StrongMapped,
            supported_signals: vec![SignalKind::FewStrongChanges],
            evidence_support: evidence_support(0.62, flags, true),
            has_approximation_warning: true,
        };
        push_candidate(candidate, out, seen);
    }

    let fallback = build_safe_baseline_candidate(x, false, base_constraints, flags);
    push_candidate(fallback, out, seen);
}

fn build_online_candidates(
    x: &TimeSeriesView<'_>,
    objective: Objective,
    hints: DataHints,
    flags: SignalFlags,
    out: &mut Vec<Candidate>,
    seen: &mut BTreeSet<String>,
) {
    let max_run_length = if x.d > 10 { 512 } else { 2_000 };

    if hints.binary_like {
        let mut config = BocpdConfig::default();
        config.max_run_length = max_run_length;
        config.observation = ObservationModel::Bernoulli {
            prior: BernoulliBetaPrior::default(),
        };
        let candidate = Candidate {
            pipeline: PipelineConfig::Online {
                detector: OnlineDetectorConfig::Bocpd(config),
            },
            pipeline_id: format!("online:bocpd:bernoulli:max_run_length={max_run_length}"),
            warnings: vec![],
            primary_reason:
                "binary-like observations map directly to Bernoulli BOCPD observation model"
                    .to_string(),
            driver_keys: vec!["nan_rate", "lag1_autocorr", "change_density_score"],
            profile: PerformanceProfile {
                speed: 0.78,
                accuracy: 0.84,
                robustness: 0.74,
            },
            family: CandidateFamily::StrongMapped,
            supported_signals: vec![SignalKind::BinaryLike],
            evidence_support: evidence_support(0.78, flags, false),
            has_approximation_warning: false,
        };
        push_candidate(candidate, out, seen);
    } else if hints.count_like {
        let mut config = BocpdConfig::default();
        config.max_run_length = max_run_length;
        config.observation = ObservationModel::Poisson {
            prior: PoissonGammaPrior::default(),
        };
        let candidate = Candidate {
            pipeline: PipelineConfig::Online {
                detector: OnlineDetectorConfig::Bocpd(config),
            },
            pipeline_id: format!("online:bocpd:poisson:max_run_length={max_run_length}"),
            warnings: vec![],
            primary_reason:
                "count-like observations map directly to Poisson BOCPD observation model"
                    .to_string(),
            driver_keys: vec!["nan_rate", "lag1_autocorr", "change_density_score"],
            profile: PerformanceProfile {
                speed: 0.76,
                accuracy: 0.84,
                robustness: 0.74,
            },
            family: CandidateFamily::StrongMapped,
            supported_signals: vec![SignalKind::CountLike],
            evidence_support: evidence_support(0.76, flags, false),
            has_approximation_warning: false,
        };
        push_candidate(candidate, out, seen);
    } else {
        let mut config = BocpdConfig::default();
        config.max_run_length = max_run_length;
        config.observation = ObservationModel::Gaussian {
            prior: GaussianNigPrior::default(),
        };
        let candidate = Candidate {
            pipeline: PipelineConfig::Online {
                detector: OnlineDetectorConfig::Bocpd(config),
            },
            pipeline_id: format!("online:bocpd:gaussian:max_run_length={max_run_length}"),
            warnings: vec![],
            primary_reason:
                "default online path uses Gaussian BOCPD when no discrete family dominates"
                    .to_string(),
            driver_keys: vec!["lag1_autocorr", "change_density_score", "nan_rate"],
            profile: PerformanceProfile {
                speed: 0.74,
                accuracy: 0.82,
                robustness: 0.72,
            },
            family: CandidateFamily::Generic,
            supported_signals: vec![SignalKind::Autocorrelated],
            evidence_support: evidence_support(0.62, flags, false),
            has_approximation_warning: false,
        };
        push_candidate(candidate, out, seen);
    }

    if objective == Objective::Speed {
        let candidate = Candidate {
            pipeline: PipelineConfig::Online {
                detector: OnlineDetectorConfig::Cusum(CusumConfig::default()),
            },
            pipeline_id: "online:cusum".to_string(),
            warnings: vec![],
            primary_reason: "speed objective adds lightweight CUSUM baseline".to_string(),
            driver_keys: vec!["change_density_score", "lag1_autocorr", "nan_rate"],
            profile: PerformanceProfile {
                speed: 0.95,
                accuracy: 0.58,
                robustness: 0.48,
            },
            family: CandidateFamily::Generic,
            supported_signals: vec![SignalKind::LowSignal],
            evidence_support: evidence_support(0.55, flags, false),
            has_approximation_warning: false,
        };
        push_candidate(candidate, out, seen);
    }

    if objective == Objective::Robustness {
        let candidate = Candidate {
            pipeline: PipelineConfig::Online {
                detector: OnlineDetectorConfig::PageHinkley(PageHinkleyConfig::default()),
            },
            pipeline_id: "online:page_hinkley".to_string(),
            warnings: vec![],
            primary_reason: "robustness objective adds Page-Hinkley as conservative baseline"
                .to_string(),
            driver_keys: vec!["change_density_score", "lag1_autocorr", "nan_rate"],
            profile: PerformanceProfile {
                speed: 0.88,
                accuracy: 0.64,
                robustness: 0.62,
            },
            family: CandidateFamily::Generic,
            supported_signals: vec![SignalKind::LowSignal],
            evidence_support: evidence_support(0.56, flags, false),
            has_approximation_warning: false,
        };
        push_candidate(candidate, out, seen);
    }
}

fn build_safe_baseline_candidate(
    x: &TimeSeriesView<'_>,
    online: bool,
    base_constraints: &Constraints,
    flags: SignalFlags,
) -> Candidate {
    if online {
        let max_run_length = if x.d > 10 { 512 } else { 2_000 };
        let mut config = BocpdConfig::default();
        config.max_run_length = max_run_length;
        config.observation = ObservationModel::Gaussian {
            prior: GaussianNigPrior::default(),
        };
        Candidate {
            pipeline: PipelineConfig::Online {
                detector: OnlineDetectorConfig::Bocpd(config),
            },
            pipeline_id: format!("online:bocpd:gaussian:max_run_length={max_run_length}"),
            warnings: vec!["safe baseline candidate".to_string()],
            primary_reason: "safe online fallback when recommendation confidence is low"
                .to_string(),
            driver_keys: vec!["nan_rate", "lag1_autocorr", "change_density_score"],
            profile: PerformanceProfile {
                speed: 0.74,
                accuracy: 0.82,
                robustness: 0.72,
            },
            family: CandidateFamily::SafeFallback,
            supported_signals: vec![SignalKind::LowSignal],
            evidence_support: evidence_support(0.45, flags, false),
            has_approximation_warning: false,
        }
    } else {
        let constraints = apply_jump_thinning(base_constraints, x.n, flags.huge_n);
        Candidate {
            pipeline: PipelineConfig::Offline {
                detector: OfflineDetectorConfig::Pelt(PeltConfig::default()),
                cost: OfflineCostKind::L2,
                constraints: constraints.clone(),
            },
            pipeline_id: format!("offline:pelt:l2:jump={}", constraints.jump),
            warnings: vec!["safe baseline candidate".to_string()],
            primary_reason: "safe offline fallback when recommendation confidence is low"
                .to_string(),
            driver_keys: vec!["change_density_score", "regime_change_proxy", "nan_rate"],
            profile: PerformanceProfile {
                speed: 0.90,
                accuracy: 0.72,
                robustness: 0.55,
            },
            family: CandidateFamily::SafeFallback,
            supported_signals: vec![SignalKind::LowSignal],
            evidence_support: evidence_support(0.42, flags, false),
            has_approximation_warning: false,
        }
    }
}

fn evidence_support(base: f64, flags: SignalFlags, has_approximation_warning: bool) -> f64 {
    let mut score = base;
    if flags.conflicting_signals {
        score -= 0.10;
    }
    if flags.low_signal {
        score -= 0.05;
    }
    if has_approximation_warning {
        score -= 0.05;
    }
    score.clamp(0.0, 1.0)
}

fn push_candidate(candidate: Candidate, out: &mut Vec<Candidate>, seen: &mut BTreeSet<String>) {
    if seen.insert(candidate.pipeline_id.clone()) {
        out.push(candidate);
    }
}

fn apply_jump_thinning(base_constraints: &Constraints, n: usize, enabled: bool) -> Constraints {
    let mut constraints = base_constraints.clone();
    if enabled && constraints.jump == 1 {
        constraints.jump = (n / 25_000).clamp(1, 16);
    }
    constraints
}

fn resource_estimate(pipeline: &PipelineConfig) -> ResourceEstimate {
    match pipeline {
        PipelineConfig::Offline { detector, .. } => match detector {
            OfflineDetectorConfig::Pelt(_) => ResourceEstimate {
                time_complexity: "O(n) avg / O(n^2) worst".to_string(),
                memory_complexity: "O(n)".to_string(),
                relative_time_score: 0.45,
                relative_memory_score: 0.40,
            },
            OfflineDetectorConfig::BinSeg(_) => ResourceEstimate {
                time_complexity: "O(n log n)".to_string(),
                memory_complexity: "O(n)".to_string(),
                relative_time_score: 0.35,
                relative_memory_score: 0.35,
            },
            OfflineDetectorConfig::Wbs(_) => ResourceEstimate {
                time_complexity: "O(M*n)".to_string(),
                memory_complexity: "O(n + M)".to_string(),
                relative_time_score: 0.65,
                relative_memory_score: 0.55,
            },
        },
        PipelineConfig::Online { detector } => match detector {
            OnlineDetectorConfig::Bocpd(_) => ResourceEstimate {
                time_complexity: "O(W) per step".to_string(),
                memory_complexity: "O(W*d)".to_string(),
                relative_time_score: 0.55,
                relative_memory_score: 0.60,
            },
            OnlineDetectorConfig::Cusum(_) | OnlineDetectorConfig::PageHinkley(_) => {
                ResourceEstimate {
                    time_complexity: "O(1) per step".to_string(),
                    memory_complexity: "O(1)".to_string(),
                    relative_time_score: 0.05,
                    relative_memory_score: 0.05,
                }
            }
        },
    }
}

fn top_driver_summary(summary: &DiagnosticsSummary) -> String {
    [
        ("lag1_autocorr", summary.lag1_autocorr),
        ("outlier_rate_iqr", summary.outlier_rate_iqr),
        ("change_density_score", summary.change_density_score),
    ]
    .iter()
    .map(|(name, value)| format!("{name}={value:.3}"))
    .collect::<Vec<_>>()
    .join(", ")
}

fn pipeline_label(pipeline: &PipelineConfig) -> &'static str {
    match pipeline {
        PipelineConfig::Offline { detector, cost, .. } => match (detector, cost) {
            (OfflineDetectorConfig::Pelt(_), OfflineCostKind::L2) => "PELT + L2",
            (OfflineDetectorConfig::Pelt(_), OfflineCostKind::Normal) => "PELT + Normal",
            (OfflineDetectorConfig::Pelt(_), OfflineCostKind::Nig) => "PELT + NIG",
            (OfflineDetectorConfig::BinSeg(_), OfflineCostKind::Normal) => "BinSeg + Normal",
            (OfflineDetectorConfig::BinSeg(_), OfflineCostKind::L2) => "BinSeg + L2",
            (OfflineDetectorConfig::BinSeg(_), OfflineCostKind::Nig) => "BinSeg + NIG",
            (OfflineDetectorConfig::Wbs(_), OfflineCostKind::L2) => "WBS + L2",
            (OfflineDetectorConfig::Wbs(_), OfflineCostKind::Normal) => "WBS + Normal",
            (OfflineDetectorConfig::Wbs(_), OfflineCostKind::Nig) => "WBS + NIG",
        },
        PipelineConfig::Online { detector } => match detector {
            OnlineDetectorConfig::Bocpd(config) => match config.observation {
                ObservationModel::Gaussian { .. } => "BOCPD (Gaussian)",
                ObservationModel::Poisson { .. } => "BOCPD (Poisson)",
                ObservationModel::Bernoulli { .. } => "BOCPD (Bernoulli)",
            },
            OnlineDetectorConfig::Cusum(_) => "CUSUM",
            OnlineDetectorConfig::PageHinkley(_) => "Page-Hinkley",
        },
    }
}

fn diagnostic_value(summary: &DiagnosticsSummary, key: &str) -> f64 {
    match key {
        "nan_rate" => summary.nan_rate,
        "kurtosis_proxy" => summary.kurtosis_proxy,
        "outlier_rate_iqr" => summary.outlier_rate_iqr,
        "mad_to_std_ratio" => summary.mad_to_std_ratio,
        "lag1_autocorr" => summary.lag1_autocorr,
        "lagk_autocorr" => summary.lagk_autocorr,
        "residual_lag1_autocorr" => summary.residual_lag1_autocorr,
        "change_density_score" => summary.change_density_score,
        "regime_change_proxy" => summary.regime_change_proxy,
        _ => 0.0,
    }
}

fn sample_data_hints(
    x: &TimeSeriesView<'_>,
    stride: usize,
    tolerance: f64,
) -> Result<DataHints, CpdError> {
    let source_len = match x.values {
        DTypeView::F32(values) => values.len(),
        DTypeView::F64(values) => values.len(),
    };

    let mut valid = 0usize;
    let mut binary = 0usize;
    let mut count = 0usize;

    let stride = stride.max(1);
    let mut t = 0usize;
    let mut sampled_last = false;
    while t < x.n {
        if t + 1 == x.n {
            sampled_last = true;
        }
        sample_row_hints(
            x,
            source_len,
            t,
            tolerance,
            &mut valid,
            &mut binary,
            &mut count,
        )?;
        t = t.saturating_add(stride);
    }

    if x.n > 0 && !sampled_last {
        sample_row_hints(
            x,
            source_len,
            x.n - 1,
            tolerance,
            &mut valid,
            &mut binary,
            &mut count,
        )?;
    }

    if valid == 0 {
        return Ok(DataHints::default());
    }

    let binary_ratio = binary as f64 / valid as f64;
    if binary_ratio >= FAMILY_THRESHOLD {
        return Ok(DataHints {
            binary_like: true,
            count_like: false,
        });
    }

    let count_ratio = count as f64 / valid as f64;
    Ok(DataHints {
        binary_like: false,
        count_like: count_ratio >= FAMILY_THRESHOLD,
    })
}

fn sample_row_hints(
    x: &TimeSeriesView<'_>,
    source_len: usize,
    t: usize,
    tolerance: f64,
    valid: &mut usize,
    binary: &mut usize,
    count: &mut usize,
) -> Result<(), CpdError> {
    for j in 0..x.d {
        let idx = source_index(x.layout, x.n, x.d, t, j)?;
        if idx >= source_len {
            return Err(CpdError::invalid_input(format!(
                "source index out of bounds while sampling recommendation hints: idx={idx}, len={source_len}"
            )));
        }

        let value = match x.values {
            DTypeView::F32(values) => f64::from(values[idx]),
            DTypeView::F64(values) => values[idx],
        };
        let mask_missing = x.missing_mask.map(|mask| mask[idx] == 1).unwrap_or(false);
        if mask_missing || value.is_nan() {
            continue;
        }

        *valid = valid.saturating_add(1);

        if (value - 0.0).abs() <= tolerance || (value - 1.0).abs() <= tolerance {
            *binary = binary.saturating_add(1);
        }

        let rounded = value.round();
        if value >= 0.0 && (value - rounded).abs() <= tolerance {
            *count = count.saturating_add(1);
        }
    }

    Ok(())
}

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
            .ok_or_else(|| CpdError::invalid_input("C-layout index overflow in recommendation")),
        MemoryLayout::FContiguous => j
            .checked_mul(n)
            .and_then(|base| base.checked_add(t))
            .ok_or_else(|| CpdError::invalid_input("F-layout index overflow in recommendation")),
        MemoryLayout::Strided {
            row_stride,
            col_stride,
        } => {
            let t_isize = isize::try_from(t).map_err(|_| {
                CpdError::invalid_input(format!(
                    "time index {t} does not fit in isize for recommendation"
                ))
            })?;
            let j_isize = isize::try_from(j).map_err(|_| {
                CpdError::invalid_input(format!(
                    "dimension index {j} does not fit in isize for recommendation"
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
                        "strided index overflow in recommendation at t={t}, j={j}"
                    ))
                })?;
            usize::try_from(idx).map_err(|_| {
                CpdError::invalid_input(format!(
                    "strided index became negative in recommendation at t={t}, j={j}: idx={idx}"
                ))
            })
        }
    }
}

#[cfg(test)]
fn pipeline_id(pipeline: &PipelineConfig) -> String {
    match pipeline {
        PipelineConfig::Offline {
            detector,
            cost,
            constraints,
        } => {
            let detector_name = match detector {
                OfflineDetectorConfig::Pelt(_) => "pelt",
                OfflineDetectorConfig::BinSeg(_) => "binseg",
                OfflineDetectorConfig::Wbs(_) => "wbs",
            };
            let cost_name = match cost {
                OfflineCostKind::L2 => "l2",
                OfflineCostKind::Normal => "normal",
                OfflineCostKind::Nig => "nig",
            };
            format!(
                "offline:{detector_name}:{cost_name}:jump={}",
                constraints.jump
            )
        }
        PipelineConfig::Online { detector } => match detector {
            OnlineDetectorConfig::Bocpd(config) => {
                let model = match config.observation {
                    ObservationModel::Gaussian { .. } => "gaussian",
                    ObservationModel::Poisson { .. } => "poisson",
                    ObservationModel::Bernoulli { .. } => "bernoulli",
                };
                format!(
                    "online:bocpd:{model}:max_run_length={}",
                    config.max_run_length
                )
            }
            OnlineDetectorConfig::Cusum(_) => "online:cusum".to_string(),
            OnlineDetectorConfig::PageHinkley(_) => "online:page_hinkley".to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CalibrationFamily, CalibrationObservation, Objective, OfflineCostKind,
        OfflineDetectorConfig, OnlineDetectorConfig, PipelineConfig, confidence_formula,
        evaluate_calibration, recommend,
    };
    use cpd_core::{Constraints, MemoryLayout, MissingPolicy, TimeIndex, TimeSeriesView};
    use cpd_online::ObservationModel;
    use std::collections::BTreeSet;

    fn make_univariate_view(values: &[f64]) -> TimeSeriesView<'_> {
        TimeSeriesView::from_f64(
            values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Ignore,
        )
        .expect("test view should be valid")
    }

    fn make_univariate_view_with_mask<'a>(values: &'a [f64], mask: &'a [u8]) -> TimeSeriesView<'a> {
        TimeSeriesView::from_f64(
            values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            Some(mask),
            TimeIndex::None,
            MissingPolicy::Ignore,
        )
        .expect("masked view should be valid")
    }

    fn pseudo_uniform_noise(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let u = ((*state >> 33) as f64) / ((1_u64 << 31) as f64);
        u * 2.0 - 1.0
    }

    fn base_summary() -> super::DiagnosticsSummary {
        super::DiagnosticsSummary {
            valid_dimensions: 1,
            nan_rate: 0.01,
            longest_nan_run: 0,
            missing_pattern: super::MissingPattern::None,
            kurtosis_proxy: 2.5,
            outlier_rate_iqr: 0.01,
            mad_to_std_ratio: 1.0,
            lag1_autocorr: 0.10,
            lagk_autocorr: 0.05,
            pacf_lagk_proxy: 0.0,
            residual_lag1_autocorr: 0.05,
            rolling_mean_drift: 0.0,
            rolling_variance_drift: 0.0,
            regime_change_proxy: 0.20,
            change_density_score: 0.10,
            dominant_period_hints: vec![],
        }
    }

    #[test]
    fn recommend_offline_huge_n_prefers_pelt_l2_with_jump_thinning() {
        let n = 120_000;
        let mut state = 123_u64;
        let values = (0..n)
            .map(|_| pseudo_uniform_noise(&mut state))
            .collect::<Vec<_>>();
        let view = make_univariate_view(&values);

        let recommendations =
            recommend(&view, Objective::Speed, false, None, 0.20, true).expect("recommend");

        assert!(!recommendations.is_empty());
        match &recommendations[0].pipeline {
            PipelineConfig::Offline {
                detector: OfflineDetectorConfig::Pelt(_),
                cost: OfflineCostKind::L2,
                constraints,
            } => {
                assert_eq!(constraints.jump, (n / 25_000).clamp(1, 16));
            }
            other => panic!("unexpected top recommendation: {other:?}"),
        }
    }

    #[test]
    fn recommend_offline_heavy_tail_prefers_nig_for_robustness() {
        let n = 120_000;
        let mut state = 987_u64;
        let mut values = Vec::with_capacity(n);
        for idx in 0..n {
            let mut v = pseudo_uniform_noise(&mut state);
            if idx % 17 == 0 {
                v *= 22.0;
            }
            values.push(v);
        }

        let view = make_univariate_view(&values);
        let recommendations =
            recommend(&view, Objective::Robustness, false, None, 0.20, true).expect("recommend");

        assert!(!recommendations.is_empty());
        match &recommendations[0].pipeline {
            PipelineConfig::Offline {
                detector: OfflineDetectorConfig::Pelt(_),
                cost: OfflineCostKind::Nig,
                ..
            } => {}
            other => panic!("unexpected top recommendation: {other:?}"),
        }
    }

    #[test]
    fn recommend_offline_autocorrelated_emits_normal_with_ar_mapping_warning() {
        let n = 120_000;
        let values = (0..n)
            .map(|t| (2.0 * std::f64::consts::PI * t as f64 / 200.0).sin())
            .collect::<Vec<_>>();

        let view = make_univariate_view(&values);
        let recommendations =
            recommend(&view, Objective::Accuracy, false, None, 0.20, true).expect("recommend");

        assert!(!recommendations.is_empty());
        match &recommendations[0].pipeline {
            PipelineConfig::Offline {
                detector: OfflineDetectorConfig::Pelt(_),
                cost: OfflineCostKind::Normal,
                ..
            } => {
                assert!(
                    recommendations[0]
                        .warnings
                        .iter()
                        .any(|warning| warning.contains("CostAR/CostLinear"))
                );
            }
            other => panic!("unexpected top recommendation: {other:?}"),
        }
    }

    #[test]
    fn recommend_offline_masking_risk_prefers_wbs() {
        let n = 20_000;
        let mut values = vec![0.0; n];
        for item in values.iter_mut().take(n).skip(n / 2) {
            *item = 8.0;
        }

        let mut mask = vec![0_u8; n];
        for item in mask.iter_mut().take(11_500).skip(9_000) {
            *item = 1;
        }

        let view = make_univariate_view_with_mask(&values, &mask);
        let recommendations =
            recommend(&view, Objective::Accuracy, false, None, 0.20, true).expect("recommend");

        assert!(!recommendations.is_empty());
        match &recommendations[0].pipeline {
            PipelineConfig::Offline {
                detector: OfflineDetectorConfig::Wbs(_),
                cost: OfflineCostKind::Normal,
                ..
            } => {}
            other => panic!("unexpected top recommendation: {other:?}"),
        }
    }

    #[test]
    fn recommend_online_binary_prefers_bocpd_bernoulli() {
        let values = (0..1024)
            .map(|idx| if idx % 2 == 0 { 0.0 } else { 1.0 })
            .collect::<Vec<_>>();
        let view = make_univariate_view(&values);

        let recommendations =
            recommend(&view, Objective::Balanced, true, None, 0.20, true).expect("recommend");

        match &recommendations[0].pipeline {
            PipelineConfig::Online {
                detector: OnlineDetectorConfig::Bocpd(cfg),
            } => {
                assert!(matches!(
                    cfg.observation,
                    ObservationModel::Bernoulli { .. }
                ));
                assert!(
                    !recommendations[0]
                        .warnings
                        .iter()
                        .any(|warning| warning.contains("OOD gating"))
                );
            }
            other => panic!("unexpected top recommendation: {other:?}"),
        }
    }

    #[test]
    fn recommend_online_count_prefers_bocpd_poisson() {
        let values = (0..1024).map(|idx| (idx % 7) as f64).collect::<Vec<_>>();
        let view = make_univariate_view(&values);

        let recommendations =
            recommend(&view, Objective::Balanced, true, None, 0.20, true).expect("recommend");

        match &recommendations[0].pipeline {
            PipelineConfig::Online {
                detector: OnlineDetectorConfig::Bocpd(cfg),
            } => {
                assert!(matches!(cfg.observation, ObservationModel::Poisson { .. }));
                assert!(
                    !recommendations[0]
                        .warnings
                        .iter()
                        .any(|warning| warning.contains("OOD gating"))
                );
            }
            other => panic!("unexpected top recommendation: {other:?}"),
        }
    }

    #[test]
    fn select_calibration_family_uses_binary_and_count_families() {
        let summary = base_summary();
        let flags = super::SignalFlags {
            huge_n: false,
            medium_n: true,
            autocorrelated: false,
            heavy_tail: false,
            change_dense: false,
            few_strong_changes: false,
            masking_risk: false,
            low_signal: false,
            conflicting_signals: false,
        };

        let binary_family = super::select_calibration_family(
            &summary,
            flags,
            super::DataHints {
                binary_like: true,
                count_like: false,
            },
            1,
        );
        let count_family = super::select_calibration_family(
            &summary,
            flags,
            super::DataHints {
                binary_like: false,
                count_like: true,
            },
            1,
        );

        assert_eq!(binary_family, CalibrationFamily::Binary);
        assert_eq!(count_family, CalibrationFamily::Count);
    }

    #[test]
    fn recommend_in_distribution_confidence_is_reasonable() {
        let mut state = 314_159_u64;
        let values = (0..4_096)
            .map(|_| pseudo_uniform_noise(&mut state))
            .collect::<Vec<_>>();
        let view = make_univariate_view(&values);

        let recommendations =
            recommend(&view, Objective::Balanced, false, None, 0.35, true).expect("recommend");

        assert!(!recommendations.is_empty());
        assert!(recommendations[0].abstain_reason.is_none());
        assert!(
            recommendations[0].confidence >= 0.45,
            "expected calibrated in-distribution confidence >= 0.45, got {}",
            recommendations[0].confidence
        );
    }

    #[test]
    fn recommend_abstains_with_safe_baseline_when_threshold_not_met() {
        let values = vec![0.0; 256];
        let view = make_univariate_view(&values);

        let recommendations =
            recommend(&view, Objective::Balanced, false, None, 0.95, true).expect("recommend");

        assert_eq!(recommendations.len(), 1);
        assert!(recommendations[0].abstain_reason.is_some());
        match &recommendations[0].pipeline {
            PipelineConfig::Offline {
                detector: OfflineDetectorConfig::Pelt(_),
                cost: OfflineCostKind::L2,
                ..
            } => {}
            other => panic!("unexpected abstain-safe recommendation: {other:?}"),
        }
    }

    #[test]
    fn recommend_ood_gating_reduces_confidence_and_sets_abstain_reason() {
        let n = 4_096;
        let mut values = vec![0.0; n];
        for idx in 0..n {
            if idx % 11 == 0 {
                values[idx] = 40.0;
            } else if idx > n / 2 {
                values[idx] = -10.0;
            }
        }
        let mut mask = vec![0_u8; n];
        for item in mask.iter_mut().take(3_100).skip(700) {
            *item = 1;
        }
        let view = make_univariate_view_with_mask(&values, &mask);

        let recommendations =
            recommend(&view, Objective::Balanced, false, None, 0.55, true).expect("recommend");

        assert_eq!(recommendations.len(), 1);
        let reason = recommendations[0]
            .abstain_reason
            .as_ref()
            .expect("expected abstain reason");
        assert!(reason.contains("diagnostics outside calibration support"));
        assert!(
            recommendations[0]
                .warnings
                .iter()
                .any(|warning| warning.contains("OOD gating"))
        );
    }

    #[test]
    fn recommend_does_not_abstain_when_allow_abstain_false() {
        let values = vec![0.0; 256];
        let view = make_univariate_view(&values);

        let recommendations =
            recommend(&view, Objective::Balanced, false, None, 0.95, false).expect("recommend");

        assert!(!recommendations.is_empty());
        assert!(recommendations[0].abstain_reason.is_none());
        assert!(
            recommendations[0]
                .warnings
                .iter()
                .any(|warning| warning.contains("below min_confidence"))
        );
    }

    #[test]
    fn recommend_is_deterministic_for_same_input() {
        let mut state = 77_u64;
        let values = (0..2_048)
            .map(|_| pseudo_uniform_noise(&mut state))
            .collect::<Vec<_>>();
        let view = make_univariate_view(&values);

        let first = recommend(&view, Objective::Balanced, false, None, 0.25, true)
            .expect("first recommendation");
        let second = recommend(&view, Objective::Balanced, false, None, 0.25, true)
            .expect("second recommendation");

        assert_eq!(first, second);
    }

    #[test]
    fn recommend_rejects_invalid_min_confidence() {
        let values = vec![0.0; 128];
        let view = make_univariate_view(&values);

        let err = recommend(&view, Objective::Balanced, false, None, 1.2, true)
            .expect_err("min_confidence should be validated");
        assert!(err.to_string().contains("min_confidence"));
    }

    #[test]
    fn recommend_rejects_constraints_invalid_for_series_length() {
        let values = vec![0.0; 64];
        let view = make_univariate_view(&values);
        let constraints = Constraints {
            candidate_splits: Some(vec![70]),
            ..Constraints::default()
        };

        let err = recommend(
            &view,
            Objective::Balanced,
            false,
            Some(constraints),
            0.20,
            true,
        )
        .expect_err("constraint validation should use x.n");
        assert!(err.to_string().contains("candidate_splits"));
    }

    #[test]
    fn strongest_active_signal_prefers_higher_priority_on_score_tie() {
        let summary = super::DiagnosticsSummary {
            valid_dimensions: 1,
            nan_rate: 0.0,
            longest_nan_run: 0,
            missing_pattern: super::MissingPattern::None,
            kurtosis_proxy: 4.0,
            outlier_rate_iqr: 0.0,
            mad_to_std_ratio: 1.0,
            lag1_autocorr: 0.55,
            lagk_autocorr: 0.0,
            pacf_lagk_proxy: 0.0,
            residual_lag1_autocorr: 0.0,
            rolling_mean_drift: 0.0,
            rolling_variance_drift: 0.0,
            regime_change_proxy: 0.0,
            change_density_score: 0.0,
            dominant_period_hints: vec![],
        };
        let flags = super::SignalFlags {
            huge_n: false,
            medium_n: false,
            autocorrelated: true,
            heavy_tail: true,
            change_dense: false,
            few_strong_changes: false,
            masking_risk: false,
            low_signal: false,
            conflicting_signals: false,
        };

        let strongest =
            super::strongest_active_signal(&summary, flags, super::DataHints::default());
        assert_eq!(strongest, Some(super::SignalKind::Autocorrelated));
    }

    #[test]
    fn sample_data_hints_does_not_double_count_last_sample() {
        let mut values = vec![2.0; 50];
        values[49] = 2.5;
        let view = make_univariate_view(&values);

        let hints = super::sample_data_hints(&view, 1, super::BINARY_TOLERANCE)
            .expect("hint sampling should succeed");
        assert!(!hints.binary_like);
        assert!(hints.count_like);
    }

    #[test]
    fn sample_data_hints_reports_oob_for_tail_row_sampling() {
        let values = (0..10).map(|i| i as f64).collect::<Vec<_>>();
        let view = TimeSeriesView::from_f64(
            &values,
            5,
            2,
            MemoryLayout::Strided {
                row_stride: 2,
                col_stride: 2,
            },
            None,
            TimeIndex::None,
            MissingPolicy::Ignore,
        )
        .expect("strided view should construct");

        let err = super::sample_data_hints(&view, 3, super::BINARY_TOLERANCE)
            .expect_err("oob tail row should error");
        assert!(err.to_string().contains("out of bounds"));
    }

    #[test]
    fn recommend_returns_objective_fit_in_fixed_order() {
        let values = vec![0.0; 1024];
        let view = make_univariate_view(&values);

        let recommendations =
            recommend(&view, Objective::Balanced, false, None, 0.20, true).expect("recommend");

        let names = recommendations[0]
            .objective_fit
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["balanced", "speed", "accuracy", "robustness"]);
    }

    #[test]
    fn recommend_includes_at_least_five_distinct_paths_across_fixture_set() {
        let mut path_ids = BTreeSet::new();

        let mut state = 101_u64;
        let large_noise = (0..120_000)
            .map(|_| pseudo_uniform_noise(&mut state))
            .collect::<Vec<_>>();
        let large_noise_view = make_univariate_view(&large_noise);
        for recommendation in
            recommend(&large_noise_view, Objective::Speed, false, None, 0.20, true)
                .expect("large noise recommendation")
        {
            path_ids.insert(super::pipeline_id(&recommendation.pipeline));
        }

        let mut ar = vec![0.0; 120_000];
        for t in 1..ar.len() {
            ar[t] = 0.88 * ar[t - 1] + 0.2 * pseudo_uniform_noise(&mut state);
        }
        let ar_view = make_univariate_view(&ar);
        for recommendation in recommend(&ar_view, Objective::Accuracy, false, None, 0.20, true)
            .expect("ar recommendation")
        {
            path_ids.insert(super::pipeline_id(&recommendation.pipeline));
        }

        let mut heavy = Vec::with_capacity(120_000);
        for idx in 0..120_000 {
            let mut v = pseudo_uniform_noise(&mut state);
            if idx % 13 == 0 {
                v *= 18.0;
            }
            heavy.push(v);
        }
        let heavy_view = make_univariate_view(&heavy);
        for recommendation in recommend(&heavy_view, Objective::Robustness, false, None, 0.20, true)
            .expect("heavy recommendation")
        {
            path_ids.insert(super::pipeline_id(&recommendation.pipeline));
        }

        let mut masked_values = vec![0.0; 2_000];
        for item in masked_values.iter_mut().take(2_000).skip(1_000) {
            *item = 9.0;
        }
        let mut mask = vec![0_u8; 2_000];
        for item in mask.iter_mut().take(900).skip(500) {
            *item = 1;
        }
        let masked_view = make_univariate_view_with_mask(&masked_values, &mask);
        for recommendation in recommend(&masked_view, Objective::Accuracy, false, None, 0.20, true)
            .expect("masked recommendation")
        {
            path_ids.insert(super::pipeline_id(&recommendation.pipeline));
        }

        let binary = (0..1024)
            .map(|idx| if idx % 2 == 0 { 0.0 } else { 1.0 })
            .collect::<Vec<_>>();
        let binary_view = make_univariate_view(&binary);
        for recommendation in recommend(&binary_view, Objective::Speed, true, None, 0.20, true)
            .expect("binary recommendation")
        {
            path_ids.insert(super::pipeline_id(&recommendation.pipeline));
        }

        assert!(
            path_ids.len() >= 5,
            "expected at least five distinct recommendation paths, got {}: {:?}",
            path_ids.len(),
            path_ids
        );
    }

    #[test]
    fn evaluate_calibration_reports_global_and_per_family_metrics() {
        let observations = vec![
            CalibrationObservation {
                family: CalibrationFamily::Gaussian,
                predicted_confidence: 0.80,
                top1_within_tolerance: true,
            },
            CalibrationObservation {
                family: CalibrationFamily::Gaussian,
                predicted_confidence: 0.70,
                top1_within_tolerance: true,
            },
            CalibrationObservation {
                family: CalibrationFamily::Gaussian,
                predicted_confidence: 0.40,
                top1_within_tolerance: false,
            },
            CalibrationObservation {
                family: CalibrationFamily::HeavyTailed,
                predicted_confidence: 0.90,
                top1_within_tolerance: false,
            },
            CalibrationObservation {
                family: CalibrationFamily::HeavyTailed,
                predicted_confidence: 0.65,
                top1_within_tolerance: true,
            },
            CalibrationObservation {
                family: CalibrationFamily::HeavyTailed,
                predicted_confidence: 0.55,
                top1_within_tolerance: false,
            },
        ];

        let metrics = evaluate_calibration(&observations, Some(5)).expect("metrics");
        assert_eq!(metrics.sample_count, observations.len());
        assert_eq!(metrics.bins, 5);
        assert!(metrics.ece > 0.0);
        assert!(metrics.brier > 0.0);
        assert_eq!(metrics.per_family.len(), 2);
        assert!(metrics.per_family.iter().all(|family| family.count == 3));
    }

    #[test]
    fn evaluate_calibration_rejects_invalid_probabilities() {
        let observations = vec![CalibrationObservation {
            family: CalibrationFamily::Gaussian,
            predicted_confidence: 1.1,
            top1_within_tolerance: true,
        }];
        let err = evaluate_calibration(&observations, Some(10))
            .expect_err("invalid confidence should fail");
        assert!(err.to_string().contains("predicted_confidence"));
    }

    #[test]
    fn confidence_formula_mentions_ood_penalty() {
        let formula = confidence_formula();
        assert!(formula.contains("ood_penalty"));
        assert!(formula.contains("diagnostic_divergence"));
    }
}
