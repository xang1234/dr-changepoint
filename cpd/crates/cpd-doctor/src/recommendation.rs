// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::diagnostics::{
    DiagnosticsSummary, DoctorDiagnosticsConfig, MissingPattern, compute_diagnostics,
};
use cpd_core::{
    Constraints, CpdError, DTypeView, ExecutionContext, MemoryLayout, MissingPolicy,
    OfflineChangePointResult, OfflineDetector, Penalty, ReproMode, Stopping, TimeIndex,
    TimeSeriesView, penalty_value, validate_breakpoints, validate_constraints,
    validate_constraints_config,
};
use cpd_costs::{
    CostAR, CostCosine, CostL1Median, CostL2Mean, CostModel, CostNIGMarginal, CostNormalFullCov,
    CostNormalMeanVar, CostRank,
};
use cpd_offline::{
    BinSeg, BinSegConfig, Fpop, FpopConfig, Pelt, PeltConfig, Wbs, WbsConfig, WbsIntervalStrategy,
};
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
const DEFAULT_VALIDATION_TOLERANCE: usize = 1;
const PENALTY_SCALE_DOWN: f64 = 0.90;
const PENALTY_SCALE_UP: f64 = 1.10;

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
    pub pipeline: PipelineSpec,
    pub resource_estimate: ResourceEstimate,
    pub warnings: Vec<String>,
    pub explanation: Explanation,
    pub validation: Option<ValidationSummary>,
    pub confidence: f64,
    pub confidence_interval: (f64, f64),
    pub abstain_reason: Option<String>,
    pub objective_fit: Vec<(String, f64)>,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct PipelineSpec {
    pub detector: DetectorConfig,
    pub cost: CostConfig,
    #[cfg(feature = "preprocess")]
    pub preprocess: Option<cpd_preprocess::PreprocessConfig>,
    #[cfg(not(feature = "preprocess"))]
    pub preprocess: Option<()>,
    pub constraints: Constraints,
    pub stopping: Option<Stopping>,
    pub seed: Option<u64>,
}

impl PipelineSpec {
    fn from_pipeline_config(pipeline: &PipelineConfig) -> Self {
        match pipeline {
            PipelineConfig::Offline {
                detector,
                cost,
                constraints,
            } => Self {
                detector: DetectorConfig::Offline(detector.clone()),
                cost: CostConfig::from(*cost),
                preprocess: None,
                constraints: constraints.clone(),
                stopping: Some(extract_stopping(detector)),
                seed: extract_seed(detector),
            },
            PipelineConfig::Online { detector } => Self {
                detector: DetectorConfig::Online(online_kind_from_config(detector)),
                cost: CostConfig::None,
                preprocess: None,
                constraints: Constraints::default(),
                stopping: None,
                seed: None,
            },
        }
    }

    fn to_pipeline_config(&self) -> Result<PipelineConfig, CpdError> {
        match &self.detector {
            DetectorConfig::Offline(detector) => {
                let cost = match self.cost {
                    CostConfig::Ar => OfflineCostKind::Ar,
                    CostConfig::Cosine => OfflineCostKind::Cosine,
                    CostConfig::L1Median => OfflineCostKind::L1Median,
                    CostConfig::L2 => OfflineCostKind::L2,
                    CostConfig::Normal => OfflineCostKind::Normal,
                    CostConfig::NormalFullCov => OfflineCostKind::NormalFullCov,
                    CostConfig::Nig => OfflineCostKind::Nig,
                    CostConfig::Rank => OfflineCostKind::Rank,
                    CostConfig::None => {
                        return Err(CpdError::invalid_input(
                            "offline pipeline requires a concrete offline cost",
                        ));
                    }
                };
                let stopping = self.stopping.clone().ok_or_else(|| {
                    CpdError::invalid_input("offline pipeline requires stopping configuration")
                })?;

                let detector = with_stopping_and_seed(detector, &stopping, self.seed)?;
                Ok(PipelineConfig::Offline {
                    detector,
                    cost,
                    constraints: self.constraints.clone(),
                })
            }
            DetectorConfig::Online(detector) => {
                if !matches!(self.cost, CostConfig::None) {
                    return Err(CpdError::invalid_input(
                        "online pipeline must use CostConfig::None",
                    ));
                }
                if self.stopping.is_some() {
                    return Err(CpdError::invalid_input(
                        "online pipeline must not include stopping configuration",
                    ));
                }
                Ok(PipelineConfig::Online {
                    detector: online_config_from_kind(detector),
                })
            }
        }
    }

    fn is_online(&self) -> bool {
        matches!(self.detector, DetectorConfig::Online(_))
    }
}

fn extract_stopping(detector: &OfflineDetectorConfig) -> Stopping {
    match detector {
        OfflineDetectorConfig::Pelt(config) => config.stopping.clone(),
        OfflineDetectorConfig::BinSeg(config) => config.stopping.clone(),
        OfflineDetectorConfig::Fpop(config) => config.stopping.clone(),
        OfflineDetectorConfig::Wbs(config) => config.stopping.clone(),
    }
}

fn extract_seed(detector: &OfflineDetectorConfig) -> Option<u64> {
    match detector {
        OfflineDetectorConfig::Wbs(config) => Some(config.seed),
        OfflineDetectorConfig::Pelt(_)
        | OfflineDetectorConfig::BinSeg(_)
        | OfflineDetectorConfig::Fpop(_) => None,
    }
}

fn with_stopping_and_seed(
    detector: &OfflineDetectorConfig,
    stopping: &Stopping,
    seed: Option<u64>,
) -> Result<OfflineDetectorConfig, CpdError> {
    match detector {
        OfflineDetectorConfig::Pelt(config) => {
            if config.stopping != *stopping {
                return Err(CpdError::invalid_input(format!(
                    "offline pipeline has inconsistent stopping configuration for detector=pelt: detector.stopping={:?}, pipeline.stopping={:?}",
                    config.stopping, stopping
                )));
            }
            if seed.is_some() {
                return Err(CpdError::invalid_input(
                    "offline pipeline seed is only supported for detector=wbs",
                ));
            }
            Ok(OfflineDetectorConfig::Pelt(config.clone()))
        }
        OfflineDetectorConfig::BinSeg(config) => {
            if config.stopping != *stopping {
                return Err(CpdError::invalid_input(format!(
                    "offline pipeline has inconsistent stopping configuration for detector=binseg: detector.stopping={:?}, pipeline.stopping={:?}",
                    config.stopping, stopping
                )));
            }
            if seed.is_some() {
                return Err(CpdError::invalid_input(
                    "offline pipeline seed is only supported for detector=wbs",
                ));
            }
            Ok(OfflineDetectorConfig::BinSeg(config.clone()))
        }
        OfflineDetectorConfig::Fpop(config) => {
            if config.stopping != *stopping {
                return Err(CpdError::invalid_input(format!(
                    "offline pipeline has inconsistent stopping configuration for detector=fpop: detector.stopping={:?}, pipeline.stopping={:?}",
                    config.stopping, stopping
                )));
            }
            if seed.is_some() {
                return Err(CpdError::invalid_input(
                    "offline pipeline seed is only supported for detector=wbs",
                ));
            }
            Ok(OfflineDetectorConfig::Fpop(config.clone()))
        }
        OfflineDetectorConfig::Wbs(config) => {
            if config.stopping != *stopping {
                return Err(CpdError::invalid_input(format!(
                    "offline pipeline has inconsistent stopping configuration for detector=wbs: detector.stopping={:?}, pipeline.stopping={:?}",
                    config.stopping, stopping
                )));
            }
            let mut updated = config.clone();
            if let Some(seed_value) = seed {
                updated.seed = seed_value;
            }
            Ok(OfflineDetectorConfig::Wbs(updated))
        }
    }
}

fn online_kind_from_config(detector: &OnlineDetectorConfig) -> OnlineDetectorKind {
    match detector {
        OnlineDetectorConfig::Bocpd(config) => {
            let observation = match &config.observation {
                ObservationModel::Gaussian { .. } => OnlineObservationKind::GaussianNig,
                ObservationModel::Bernoulli { .. } => OnlineObservationKind::Bernoulli,
                ObservationModel::Poisson { .. } => OnlineObservationKind::Poisson,
            };
            OnlineDetectorKind::Bocpd {
                observation,
                max_run_length: config.max_run_length,
            }
        }
        OnlineDetectorConfig::Cusum(_) => OnlineDetectorKind::Cusum,
        OnlineDetectorConfig::PageHinkley(_) => OnlineDetectorKind::PageHinkley,
    }
}

fn online_config_from_kind(detector: &OnlineDetectorKind) -> OnlineDetectorConfig {
    match detector {
        OnlineDetectorKind::Bocpd {
            observation,
            max_run_length,
        } => {
            let mut config = BocpdConfig::default();
            config.max_run_length = *max_run_length;
            config.observation = match observation {
                OnlineObservationKind::GaussianNig => ObservationModel::Gaussian {
                    prior: GaussianNigPrior::default(),
                },
                OnlineObservationKind::Bernoulli => ObservationModel::Bernoulli {
                    prior: BernoulliBetaPrior::default(),
                },
                OnlineObservationKind::Poisson => ObservationModel::Poisson {
                    prior: PoissonGammaPrior::default(),
                },
            };
            OnlineDetectorConfig::Bocpd(config)
        }
        OnlineDetectorKind::Cusum => OnlineDetectorConfig::Cusum(CusumConfig::default()),
        OnlineDetectorKind::PageHinkley => {
            OnlineDetectorConfig::PageHinkley(PageHinkleyConfig::default())
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub enum DetectorConfig {
    Offline(OfflineDetectorConfig),
    Online(OnlineDetectorKind),
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnlineObservationKind {
    GaussianNig,
    Bernoulli,
    Poisson,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OnlineDetectorKind {
    Bocpd {
        observation: OnlineObservationKind,
        max_run_length: usize,
    },
    Cusum,
    PageHinkley,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostConfig {
    Ar,
    Cosine,
    L1Median,
    L2,
    Normal,
    NormalFullCov,
    Nig,
    Rank,
    None,
}

impl From<OfflineCostKind> for CostConfig {
    fn from(value: OfflineCostKind) -> Self {
        match value {
            OfflineCostKind::Ar => Self::Ar,
            OfflineCostKind::Cosine => Self::Cosine,
            OfflineCostKind::L1Median => Self::L1Median,
            OfflineCostKind::L2 => Self::L2,
            OfflineCostKind::Normal => Self::Normal,
            OfflineCostKind::NormalFullCov => Self::NormalFullCov,
            OfflineCostKind::Nig => Self::Nig,
            OfflineCostKind::Rank => Self::Rank,
        }
    }
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub enum OfflineDetectorConfig {
    Pelt(PeltConfig),
    BinSeg(BinSegConfig),
    Fpop(FpopConfig),
    Wbs(WbsConfig),
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OfflineCostKind {
    Ar,
    Cosine,
    L1Median,
    L2,
    Normal,
    NormalFullCov,
    Nig,
    Rank,
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

#[derive(Clone, Debug, PartialEq)]
pub struct ValidationReport {
    pub pipeline_results: Vec<(PipelineSpec, OfflineChangePointResult)>,
    pub stability_score: f64,
    pub agreement_score: f64,
    pub calibration_score: Option<f64>,
    pub penalty_sensitivity: Option<f64>,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug)]
struct ValidationDownsampledInput {
    values: Vec<f64>,
    missing_mask: Option<Vec<u8>>,
    sampled_row_indices: Vec<usize>,
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
    if online && x.d > 1 {
        return Err(CpdError::invalid_input(format!(
            "Doctor online recommendations currently support univariate series only (d=1); got d={}. Use offline recommendations for multivariate data.",
            x.d
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

/// Executes an offline pipeline specification directly from Rust.
pub fn execute_pipeline(
    x: &TimeSeriesView<'_>,
    pipeline: &PipelineSpec,
) -> Result<OfflineChangePointResult, CpdError> {
    execute_pipeline_with_repro_mode(x, pipeline, ReproMode::Balanced)
}

/// Executes an offline pipeline specification directly from Rust with a
/// caller-provided reproducibility mode.
pub fn execute_pipeline_with_repro_mode(
    x: &TimeSeriesView<'_>,
    pipeline: &PipelineSpec,
    repro_mode: ReproMode,
) -> Result<OfflineChangePointResult, CpdError> {
    if pipeline.is_online() {
        return Err(CpdError::invalid_input(
            "execute_pipeline only supports offline pipelines; received online pipeline",
        ));
    }
    run_offline_pipeline(x, pipeline, repro_mode)
}

pub fn validate_top_k(
    x: &TimeSeriesView<'_>,
    recommendations: &[Recommendation],
    k: usize,
    downsample: Option<usize>,
    seed: Option<u64>,
) -> Result<ValidationReport, CpdError> {
    if recommendations.is_empty() {
        return Err(CpdError::invalid_input(
            "validate_top_k requires at least one recommendation",
        ));
    }
    if k == 0 {
        return Err(CpdError::invalid_input(
            "validate_top_k requires k >= 1; got 0",
        ));
    }

    let mut notes = Vec::new();
    let selected_len = recommendations.len().min(k);
    if selected_len < k {
        notes.push(format!(
            "requested k={k} but only {} recommendations were provided",
            recommendations.len()
        ));
    }

    let mut validation_view = *x;
    let mut downsampled_input: Option<ValidationDownsampledInput> = None;
    let mut sampled_row_indices: Option<&[usize]> = None;
    let mut tolerance = DEFAULT_VALIDATION_TOLERANCE;

    if let Some(stride) = downsample {
        if stride == 0 {
            return Err(CpdError::invalid_input(
                "downsample must be >= 1 when provided; got 0",
            ));
        }

        if stride > 1 && x.n > 2 {
            downsampled_input = Some(build_validation_downsampled_input(x, stride)?);
            tolerance = stride.max(DEFAULT_VALIDATION_TOLERANCE);
            notes.push(format!(
                "validation downsampled input with stride={stride} (n={} -> {})",
                x.n,
                downsampled_input
                    .as_ref()
                    .map(|holder| holder.sampled_row_indices.len())
                    .unwrap_or(x.n)
            ));
        } else {
            notes.push(format!(
                "downsampling skipped because stride={stride} does not reduce n={}",
                x.n
            ));
        }
    }

    if let Some(holder) = downsampled_input.as_ref() {
        sampled_row_indices = Some(holder.sampled_row_indices.as_slice());
        validation_view = TimeSeriesView::from_f64(
            holder.values.as_slice(),
            holder.sampled_row_indices.len(),
            x.d,
            MemoryLayout::CContiguous,
            holder.missing_mask.as_deref(),
            TimeIndex::None,
            x.missing,
        )?;
    }
    let imputed_values = if validation_view.has_missing() {
        Some(impute_missing_to_c_f64(&validation_view)?)
    } else {
        None
    };
    if let Some(imputed) = imputed_values.as_ref() {
        validation_view = TimeSeriesView::from_f64(
            imputed.as_slice(),
            validation_view.n,
            validation_view.d,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )?;
        notes.push(
            "validation input contained missing values; applied deterministic forward-fill imputation (leading missings -> 0.0) before running offline pipelines"
                .to_string(),
        );
    }

    let mut pipeline_results = Vec::<(PipelineSpec, OfflineChangePointResult)>::new();
    for (rank, recommendation) in recommendations.iter().take(selected_len).enumerate() {
        if recommendation.pipeline.is_online() {
            notes.push(format!(
                "skipped recommendation rank {} because online pipelines do not produce OfflineChangePointResult",
                rank + 1
            ));
            continue;
        }

        let effective_pipeline = pipeline_with_seed_override(&recommendation.pipeline, seed);
        match run_offline_pipeline(&validation_view, &effective_pipeline, ReproMode::Balanced) {
            Ok(result) => {
                let remapped = remap_validation_result(result, sampled_row_indices, x.n)?;
                pipeline_results.push((effective_pipeline, remapped));
            }
            Err(err) => {
                notes.push(format!(
                    "validation run failed for recommendation rank {} (pipeline={}): {err}",
                    rank + 1,
                    pipeline_id_spec(&effective_pipeline)
                ));
            }
        }
    }

    if pipeline_results.is_empty() {
        return Err(CpdError::invalid_input(
            "validate_top_k could not execute any offline pipelines from the selected recommendations",
        ));
    }
    if selected_len > 1 && pipeline_results.len() < 2 {
        return Err(CpdError::invalid_input(format!(
            "validate_top_k requires at least two successfully validated offline pipelines when k>1; got {} successful pipeline(s) from {} selected recommendation(s)",
            pipeline_results.len(),
            selected_len
        )));
    }

    let offline_results = pipeline_results
        .iter()
        .map(|(_, result)| result.clone())
        .collect::<Vec<_>>();
    let stability_score = pairwise_stability_score(offline_results.as_slice(), tolerance);
    let agreement_score = breakpoint_agreement_score(offline_results.as_slice(), tolerance);
    let penalty_sensitivity = compute_penalty_sensitivity(
        &validation_view,
        pipeline_results.as_slice(),
        tolerance,
        seed,
        sampled_row_indices,
        x.n,
        &mut notes,
    )?;

    notes.push(format!(
        "stability/agreement tolerance set to ±{tolerance} samples"
    ));
    notes.push(
        "calibration_score unavailable because validate_top_k does not take held-out labeled outcomes".to_string(),
    );

    Ok(ValidationReport {
        pipeline_results,
        stability_score,
        agreement_score,
        calibration_score: None,
        penalty_sensitivity,
        notes,
    })
}

fn build_validation_downsampled_input(
    x: &TimeSeriesView<'_>,
    stride: usize,
) -> Result<ValidationDownsampledInput, CpdError> {
    if stride == 0 {
        return Err(CpdError::invalid_input(
            "validation downsampling stride must be >= 1; got 0",
        ));
    }

    let mut sampled_row_indices = Vec::with_capacity(x.n.div_ceil(stride) + 1);
    let mut t = 0usize;
    let mut included_last = false;
    while t < x.n {
        if t + 1 == x.n {
            included_last = true;
        }
        sampled_row_indices.push(t);
        t = t.saturating_add(stride);
    }
    if x.n > 0 && !included_last {
        sampled_row_indices.push(x.n - 1);
    }

    let sampled_n = sampled_row_indices.len();
    let sample_len = sampled_n
        .checked_mul(x.d)
        .ok_or_else(|| CpdError::resource_limit("downsampled value length overflow"))?;
    let source_len = match x.values {
        DTypeView::F32(values) => values.len(),
        DTypeView::F64(values) => values.len(),
    };

    let mut values = Vec::<f64>::with_capacity(sample_len);
    let mut missing_mask = x.missing_mask.map(|_| Vec::<u8>::with_capacity(sample_len));
    for &row in &sampled_row_indices {
        for dim in 0..x.d {
            let idx = source_index(x.layout, x.n, x.d, row, dim)?;
            if idx >= source_len {
                return Err(CpdError::invalid_input(format!(
                    "downsample source index out of bounds at row={row}, dim={dim}: idx={idx}, len={source_len}"
                )));
            }

            let value = match x.values {
                DTypeView::F32(raw) => f64::from(raw[idx]),
                DTypeView::F64(raw) => raw[idx],
            };
            values.push(value);

            if let Some(mask) = x.missing_mask
                && let Some(sampled_mask) = missing_mask.as_mut()
            {
                sampled_mask.push(mask[idx]);
            }
        }
    }

    Ok(ValidationDownsampledInput {
        values,
        missing_mask,
        sampled_row_indices,
    })
}

fn impute_missing_to_c_f64(x: &TimeSeriesView<'_>) -> Result<Vec<f64>, CpdError> {
    let total_len =
        x.n.checked_mul(x.d)
            .ok_or_else(|| CpdError::resource_limit("imputed value length overflow"))?;
    let source_len = match x.values {
        DTypeView::F32(values) => values.len(),
        DTypeView::F64(values) => values.len(),
    };

    let mut out = Vec::with_capacity(total_len);
    let mut last_seen = vec![0.0_f64; x.d];
    let mut seen = vec![false; x.d];

    for t in 0..x.n {
        for dim in 0..x.d {
            let idx = source_index(x.layout, x.n, x.d, t, dim)?;
            if idx >= source_len {
                return Err(CpdError::invalid_input(format!(
                    "imputation source index out of bounds at t={t}, dim={dim}: idx={idx}, len={source_len}"
                )));
            }

            let raw = match x.values {
                DTypeView::F32(values) => f64::from(values[idx]),
                DTypeView::F64(values) => values[idx],
            };
            let mask_missing = x.missing_mask.map(|mask| mask[idx] == 1).unwrap_or(false);
            let is_missing = mask_missing || raw.is_nan();
            if is_missing {
                let fill = if seen[dim] { last_seen[dim] } else { 0.0 };
                out.push(fill);
            } else {
                out.push(raw);
                last_seen[dim] = raw;
                seen[dim] = true;
            }
        }
    }

    Ok(out)
}

fn remap_validation_result(
    mut result: OfflineChangePointResult,
    sampled_row_indices: Option<&[usize]>,
    original_n: usize,
) -> Result<OfflineChangePointResult, CpdError> {
    let Some(sampled_row_indices) = sampled_row_indices else {
        result.validate(original_n)?;
        return Ok(result);
    };

    let mapped_breakpoints = remap_breakpoints_to_original(
        result.breakpoints.as_slice(),
        sampled_row_indices,
        original_n,
    )?;
    result.breakpoints = mapped_breakpoints;
    result.change_points = result
        .breakpoints
        .iter()
        .copied()
        .filter(|&breakpoint| breakpoint < original_n)
        .collect::<Vec<_>>();
    result.diagnostics.n = original_n;
    result.segments = None;
    result.validate(original_n)?;
    Ok(result)
}

fn remap_breakpoints_to_original(
    breakpoints: &[usize],
    sampled_row_indices: &[usize],
    original_n: usize,
) -> Result<Vec<usize>, CpdError> {
    if sampled_row_indices.is_empty() {
        return Err(CpdError::invalid_input(
            "cannot remap breakpoints with an empty sampled_row_indices map",
        ));
    }

    let mut mapped = Vec::with_capacity(breakpoints.len());
    for (idx, &breakpoint) in breakpoints.iter().enumerate() {
        if breakpoint == 0 || breakpoint > sampled_row_indices.len() {
            return Err(CpdError::invalid_input(format!(
                "downsampled breakpoint[{idx}]={breakpoint} is out of range for sampled length {}",
                sampled_row_indices.len()
            )));
        }
        let mapped_breakpoint = if breakpoint == sampled_row_indices.len() {
            original_n
        } else {
            let left_row = sampled_row_indices[breakpoint - 1];
            let right_row = sampled_row_indices[breakpoint];
            if right_row <= left_row {
                return Err(CpdError::invalid_input(format!(
                    "sampled_row_indices must be strictly increasing for remapping; got left_row={left_row}, right_row={right_row}, breakpoint_idx={idx}"
                )));
            }
            let gap = right_row - left_row;
            left_row
                .checked_add(gap.div_ceil(2))
                .ok_or_else(|| CpdError::resource_limit("remapped breakpoint overflow"))?
        };
        if mapped_breakpoint > original_n {
            return Err(CpdError::invalid_input(format!(
                "remapped breakpoint[{idx}]={mapped_breakpoint} exceeds original n={original_n}"
            )));
        }
        mapped.push(mapped_breakpoint);
    }

    validate_breakpoints(original_n, mapped.as_slice())?;
    Ok(mapped)
}

fn run_offline_pipeline(
    x: &TimeSeriesView<'_>,
    pipeline: &PipelineSpec,
    repro_mode: ReproMode,
) -> Result<OfflineChangePointResult, CpdError> {
    let config = pipeline.to_pipeline_config()?;
    #[cfg(feature = "preprocess")]
    if let Some(preprocess) = pipeline.preprocess.as_ref() {
        let preprocess = cpd_preprocess::PreprocessPipeline::new(preprocess.clone())?;
        let preprocessed = preprocess.apply(x)?;
        let preprocessed_view = preprocessed.as_view()?;
        return run_offline_pipeline_with_config(&preprocessed_view, &config, repro_mode);
    }
    run_offline_pipeline_with_config(x, &config, repro_mode)
}

fn run_offline_pipeline_with_config(
    x: &TimeSeriesView<'_>,
    pipeline: &PipelineConfig,
    repro_mode: ReproMode,
) -> Result<OfflineChangePointResult, CpdError> {
    let PipelineConfig::Offline {
        detector,
        cost,
        constraints,
    } = pipeline
    else {
        return Err(CpdError::invalid_input(
            "offline pipeline execution requires an offline detector configuration",
        ));
    };

    validate_constraints_config(constraints)?;
    let _ = validate_constraints(constraints, x.n)?;
    let sanitized_view = if matches!(x.missing, MissingPolicy::Ignore) && !x.has_missing() {
        Some(TimeSeriesView::new(
            x.values,
            x.n,
            x.d,
            x.layout,
            x.missing_mask,
            x.time,
            MissingPolicy::Error,
        )?)
    } else {
        None
    };
    let detect_view = sanitized_view.as_ref().unwrap_or(x);
    let ctx = ExecutionContext::new(constraints).with_repro_mode(repro_mode);

    match cost {
        OfflineCostKind::Ar => {
            run_offline_detector_with_cost(detect_view, detector, &ctx, CostAR::new(1, repro_mode))
        }
        OfflineCostKind::Cosine => {
            run_offline_detector_with_cost(detect_view, detector, &ctx, CostCosine::new(repro_mode))
        }
        OfflineCostKind::L1Median => run_offline_detector_with_cost(
            detect_view,
            detector,
            &ctx,
            CostL1Median::new(repro_mode),
        ),
        OfflineCostKind::L2 => {
            run_offline_detector_with_l2_cost(detect_view, detector, &ctx, repro_mode)
        }
        OfflineCostKind::Normal => run_offline_detector_with_cost(
            detect_view,
            detector,
            &ctx,
            CostNormalMeanVar::new(repro_mode),
        ),
        OfflineCostKind::NormalFullCov => run_offline_detector_with_cost(
            detect_view,
            detector,
            &ctx,
            CostNormalFullCov::new(repro_mode),
        ),
        OfflineCostKind::Nig => run_offline_detector_with_cost(
            detect_view,
            detector,
            &ctx,
            CostNIGMarginal::new(repro_mode),
        ),
        OfflineCostKind::Rank => {
            run_offline_detector_with_cost(detect_view, detector, &ctx, CostRank::new(repro_mode))
        }
    }
}

fn run_offline_detector_with_cost<C: CostModel>(
    x: &TimeSeriesView<'_>,
    detector: &OfflineDetectorConfig,
    ctx: &ExecutionContext<'_>,
    cost_model: C,
) -> Result<OfflineChangePointResult, CpdError> {
    match detector {
        OfflineDetectorConfig::Pelt(config) => {
            Pelt::new(cost_model, config.clone())?.detect(x, ctx)
        }
        OfflineDetectorConfig::BinSeg(config) => {
            BinSeg::new(cost_model, config.clone())?.detect(x, ctx)
        }
        OfflineDetectorConfig::Fpop(_) => {
            Err(CpdError::invalid_input("detector=fpop requires cost=l2"))
        }
        OfflineDetectorConfig::Wbs(config) => Wbs::new(cost_model, config.clone())?.detect(x, ctx),
    }
}

fn run_offline_detector_with_l2_cost(
    x: &TimeSeriesView<'_>,
    detector: &OfflineDetectorConfig,
    ctx: &ExecutionContext<'_>,
    repro_mode: ReproMode,
) -> Result<OfflineChangePointResult, CpdError> {
    match detector {
        OfflineDetectorConfig::Pelt(config) => {
            Pelt::new(CostL2Mean::new(repro_mode), config.clone())?.detect(x, ctx)
        }
        OfflineDetectorConfig::BinSeg(config) => {
            BinSeg::new(CostL2Mean::new(repro_mode), config.clone())?.detect(x, ctx)
        }
        OfflineDetectorConfig::Fpop(config) => {
            Fpop::new(CostL2Mean::new(repro_mode), config.clone())?.detect(x, ctx)
        }
        OfflineDetectorConfig::Wbs(config) => {
            Wbs::new(CostL2Mean::new(repro_mode), config.clone())?.detect(x, ctx)
        }
    }
}

fn pipeline_with_seed_override(pipeline: &PipelineSpec, seed: Option<u64>) -> PipelineSpec {
    let Some(seed_value) = seed else {
        return pipeline.clone();
    };
    if !matches!(
        pipeline.detector,
        DetectorConfig::Offline(OfflineDetectorConfig::Wbs(_))
    ) {
        return pipeline.clone();
    }
    let mut updated = pipeline.clone();
    updated.seed = Some(seed_value);
    if let DetectorConfig::Offline(OfflineDetectorConfig::Wbs(config)) = &mut updated.detector {
        config.seed = seed_value;
    }
    updated
}

fn pipeline_with_scaled_penalty(
    pipeline: &PipelineSpec,
    scale: f64,
    n: usize,
    d: usize,
    seed: Option<u64>,
) -> Result<Option<PipelineSpec>, CpdError> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(CpdError::invalid_input(format!(
            "penalty scale must be finite and > 0; got {scale}"
        )));
    }

    let PipelineConfig::Offline {
        detector,
        cost,
        constraints,
    } = pipeline.to_pipeline_config()?
    else {
        return Ok(None);
    };

    match detector {
        OfflineDetectorConfig::Pelt(config) => {
            let Some(stopping) =
                scaled_stopping(&config.stopping, scale, n, d, config.params_per_segment)?
            else {
                return Ok(None);
            };
            let mut scaled = config.clone();
            scaled.stopping = stopping;
            Ok(Some(PipelineSpec::from_pipeline_config(
                &PipelineConfig::Offline {
                    detector: OfflineDetectorConfig::Pelt(scaled),
                    cost,
                    constraints: constraints.clone(),
                },
            )))
        }
        OfflineDetectorConfig::BinSeg(config) => {
            let Some(stopping) =
                scaled_stopping(&config.stopping, scale, n, d, config.params_per_segment)?
            else {
                return Ok(None);
            };
            let mut scaled = config.clone();
            scaled.stopping = stopping;
            Ok(Some(PipelineSpec::from_pipeline_config(
                &PipelineConfig::Offline {
                    detector: OfflineDetectorConfig::BinSeg(scaled),
                    cost,
                    constraints: constraints.clone(),
                },
            )))
        }
        OfflineDetectorConfig::Fpop(config) => {
            let Some(stopping) =
                scaled_stopping(&config.stopping, scale, n, d, config.params_per_segment)?
            else {
                return Ok(None);
            };
            let mut scaled = config.clone();
            scaled.stopping = stopping;
            Ok(Some(PipelineSpec::from_pipeline_config(
                &PipelineConfig::Offline {
                    detector: OfflineDetectorConfig::Fpop(scaled),
                    cost,
                    constraints: constraints.clone(),
                },
            )))
        }
        OfflineDetectorConfig::Wbs(config) => {
            let Some(stopping) =
                scaled_stopping(&config.stopping, scale, n, d, config.params_per_segment)?
            else {
                return Ok(None);
            };
            let mut scaled = config.clone();
            scaled.stopping = stopping;
            if let Some(seed_value) = seed {
                scaled.seed = seed_value;
            }
            Ok(Some(PipelineSpec::from_pipeline_config(
                &PipelineConfig::Offline {
                    detector: OfflineDetectorConfig::Wbs(scaled),
                    cost,
                    constraints: constraints.clone(),
                },
            )))
        }
    }
}

fn scaled_stopping(
    stopping: &Stopping,
    scale: f64,
    n: usize,
    d: usize,
    params_per_segment: usize,
) -> Result<Option<Stopping>, CpdError> {
    match stopping {
        Stopping::KnownK(_) => Ok(None),
        Stopping::Penalized(penalty) => Ok(Some(Stopping::Penalized(scaled_penalty(
            penalty,
            scale,
            n,
            d,
            params_per_segment,
        )?))),
        Stopping::PenaltyPath(path) => {
            let mut scaled_path = Vec::with_capacity(path.len());
            for penalty in path {
                scaled_path.push(scaled_penalty(penalty, scale, n, d, params_per_segment)?);
            }
            Ok(Some(Stopping::PenaltyPath(scaled_path)))
        }
    }
}

fn scaled_penalty(
    penalty: &Penalty,
    scale: f64,
    n: usize,
    d: usize,
    params_per_segment: usize,
) -> Result<Penalty, CpdError> {
    let base = penalty_value(penalty, n, d, params_per_segment)?;
    let scaled = base * scale;
    if !scaled.is_finite() || scaled <= 0.0 {
        return Err(CpdError::invalid_input(format!(
            "scaled penalty must be finite and > 0; base={base}, scale={scale}, scaled={scaled}"
        )));
    }
    Ok(Penalty::Manual(scaled))
}

fn pairwise_stability_score(results: &[OfflineChangePointResult], tolerance: usize) -> f64 {
    if results.len() < 2 {
        return 1.0;
    }

    let mut pair_count = 0usize;
    let mut total = 0.0;
    for left in 0..results.len() - 1 {
        for right in left + 1..results.len() {
            total += jaccard_with_tolerance(
                results[left].change_points.as_slice(),
                results[right].change_points.as_slice(),
                tolerance,
            );
            pair_count += 1;
        }
    }

    if pair_count == 0 {
        1.0
    } else {
        total / pair_count as f64
    }
}

#[derive(Clone, Debug)]
struct BreakpointCluster {
    anchor: usize,
    pipelines: Vec<usize>,
}

fn breakpoint_agreement_score(results: &[OfflineChangePointResult], tolerance: usize) -> f64 {
    if results.is_empty() {
        return 0.0;
    }

    let mut clusters = Vec::<BreakpointCluster>::new();
    for (pipeline_idx, result) in results.iter().enumerate() {
        for &change_point in result.change_points.as_slice() {
            let mut best_cluster = None;
            let mut best_distance = usize::MAX;
            for (cluster_idx, cluster) in clusters.iter().enumerate() {
                let distance = cluster.anchor.abs_diff(change_point);
                if distance <= tolerance && distance < best_distance {
                    best_distance = distance;
                    best_cluster = Some(cluster_idx);
                }
            }

            if let Some(cluster_idx) = best_cluster {
                let cluster = &mut clusters[cluster_idx];
                if !cluster.pipelines.contains(&pipeline_idx) {
                    cluster.pipelines.push(pipeline_idx);
                }
            } else {
                clusters.push(BreakpointCluster {
                    anchor: change_point,
                    pipelines: vec![pipeline_idx],
                });
            }
        }
    }

    if clusters.is_empty() {
        return 1.0;
    }

    let pipeline_count = results.len() as f64;
    let total_support = clusters
        .iter()
        .map(|cluster| cluster.pipelines.len() as f64 / pipeline_count)
        .sum::<f64>();
    total_support / clusters.len() as f64
}

fn jaccard_with_tolerance(left: &[usize], right: &[usize], tolerance: usize) -> f64 {
    if left.is_empty() && right.is_empty() {
        return 1.0;
    }
    let matches = count_tolerance_matches(left, right, tolerance);
    let union = left.len() + right.len() - matches;
    if union == 0 {
        1.0
    } else {
        matches as f64 / union as f64
    }
}

fn count_tolerance_matches(left: &[usize], right: &[usize], tolerance: usize) -> usize {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut matches = 0usize;

    while i < left.len() && j < right.len() {
        let left_cp = left[i];
        let right_cp = right[j];
        if left_cp.abs_diff(right_cp) <= tolerance {
            matches += 1;
            i += 1;
            j += 1;
            continue;
        }
        if left_cp < right_cp {
            i += 1;
        } else {
            j += 1;
        }
    }

    matches
}

fn compute_penalty_sensitivity(
    validation_view: &TimeSeriesView<'_>,
    pipeline_results: &[(PipelineSpec, OfflineChangePointResult)],
    tolerance: usize,
    seed: Option<u64>,
    sampled_row_indices: Option<&[usize]>,
    original_n: usize,
    notes: &mut Vec<String>,
) -> Result<Option<f64>, CpdError> {
    let mut sensitivities = Vec::<f64>::new();
    let mut penalty_based_pipeline_count = 0usize;
    let mut penalty_run_failure_count = 0usize;

    for (pipeline, base_result) in pipeline_results {
        let Some(scaled_low) = pipeline_with_scaled_penalty(
            pipeline,
            PENALTY_SCALE_DOWN,
            validation_view.n,
            validation_view.d,
            seed,
        )?
        else {
            continue;
        };
        penalty_based_pipeline_count = penalty_based_pipeline_count.saturating_add(1);
        let Some(scaled_high) = pipeline_with_scaled_penalty(
            pipeline,
            PENALTY_SCALE_UP,
            validation_view.n,
            validation_view.d,
            seed,
        )?
        else {
            continue;
        };

        let low_result =
            match run_offline_pipeline(validation_view, &scaled_low, ReproMode::Balanced) {
                Ok(result) => remap_validation_result(result, sampled_row_indices, original_n)?,
                Err(err) => {
                    penalty_run_failure_count = penalty_run_failure_count.saturating_add(1);
                    notes.push(format!(
                    "penalty sensitivity skipped for pipeline={} because 0.9x run failed: {err}",
                    pipeline_id_spec(pipeline)
                ));
                    continue;
                }
            };

        let high_result =
            match run_offline_pipeline(validation_view, &scaled_high, ReproMode::Balanced) {
                Ok(result) => remap_validation_result(result, sampled_row_indices, original_n)?,
                Err(err) => {
                    penalty_run_failure_count = penalty_run_failure_count.saturating_add(1);
                    notes.push(format!(
                    "penalty sensitivity skipped for pipeline={} because 1.1x run failed: {err}",
                    pipeline_id_spec(pipeline)
                ));
                    continue;
                }
            };

        let low_overlap = jaccard_with_tolerance(
            base_result.change_points.as_slice(),
            low_result.change_points.as_slice(),
            tolerance,
        );
        let high_overlap = jaccard_with_tolerance(
            base_result.change_points.as_slice(),
            high_result.change_points.as_slice(),
            tolerance,
        );
        sensitivities.push((1.0 - 0.5 * (low_overlap + high_overlap)).clamp(0.0, 1.0));
    }

    if sensitivities.is_empty() {
        if penalty_based_pipeline_count == 0 {
            notes.push(
                "penalty_sensitivity unavailable because no validated pipelines used penalty-based stopping"
                    .to_string(),
            );
        } else {
            notes.push(format!(
                "penalty_sensitivity unavailable because all ±10% penalty reruns failed for {penalty_based_pipeline_count} penalty-based pipeline(s) (failures={penalty_run_failure_count})"
            ));
        }
        Ok(None)
    } else {
        Ok(Some(mean_f64(sensitivities.as_slice())))
    }
}

fn mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
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
    warnings.extend(multivariate_pipeline_warnings(
        &candidate.pipeline,
        dimension_count,
    ));
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
            pipeline: PipelineSpec::from_pipeline_config(&candidate.pipeline),
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

fn multivariate_pipeline_warnings(
    pipeline: &PipelineConfig,
    dimension_count: usize,
) -> Vec<String> {
    if dimension_count <= 1 {
        return Vec::new();
    }

    match pipeline {
        PipelineConfig::Offline { cost, .. } => {
            let message = match cost {
                OfflineCostKind::Normal => Some(
                    "multivariate semantics: CostNormalMeanVar uses diagonal covariance (independent per-dimension Gaussian terms summed across dimensions); prefer this when d is large, compute budget is tight, or cross-dimension correlation is weak",
                ),
                OfflineCostKind::NormalFullCov => Some(
                    "multivariate semantics: CostNormalFullCov estimates a regularized full covariance per segment (captures cross-dimension correlation); prefer this when d is moderate and cross-dimension covariance carries change signal",
                ),
                OfflineCostKind::Nig => Some(
                    "multivariate semantics: CostNIGMarginal uses diagonal covariance assumptions (independent per-dimension NIG marginals summed across dimensions)",
                ),
                OfflineCostKind::Ar => Some(
                    "multivariate semantics: CostAR fits autoregressive residual models independently per dimension and sums segment costs",
                ),
                OfflineCostKind::Cosine => Some(
                    "multivariate semantics: CostCosine scores directional coherence from row-wise unit-vector resultants (captures direction shifts while being scale-invariant)",
                ),
                OfflineCostKind::L2 => Some(
                    "multivariate semantics: CostL2Mean computes additive per-dimension SSE and sums across dimensions",
                ),
                OfflineCostKind::L1Median => Some(
                    "multivariate semantics: CostL1Median computes per-dimension median absolute deviation costs and sums across dimensions",
                ),
                OfflineCostKind::Rank => Some(
                    "multivariate semantics: CostRank applies per-dimension rank transforms, then sums additive per-dimension rank-SSE costs",
                ),
            };
            message
                .map(|entry| vec![entry.to_string()])
                .unwrap_or_default()
        }
        PipelineConfig::Online { .. } => vec![
            "multivariate limitation: online detectors (BOCPD/CUSUM/Page-Hinkley) currently require univariate updates (d=1)".to_string(),
        ],
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

fn normal_family_offline_cost(x: &TimeSeriesView<'_>, flags: SignalFlags) -> OfflineCostKind {
    if x.d > 1 && x.d <= 16 && !flags.huge_n {
        OfflineCostKind::NormalFullCov
    } else {
        OfflineCostKind::Normal
    }
}

fn normal_family_cost_name(cost: OfflineCostKind) -> &'static str {
    match cost {
        OfflineCostKind::Normal => "normal",
        OfflineCostKind::NormalFullCov => "normal_full_cov",
        _ => unreachable!("normal-family helper expects Normal or NormalFullCov"),
    }
}

fn normal_family_cost_warnings(cost: OfflineCostKind) -> Vec<String> {
    if matches!(cost, OfflineCostKind::NormalFullCov) {
        vec![
            "CostNormalFullCov is heavier than diagonal Normal (cache O(n*d^2), segment solve O(d^3)); use when cross-dimension covariance carries signal"
                .to_string(),
        ]
    } else {
        Vec::new()
    }
}

fn build_offline_candidates(
    x: &TimeSeriesView<'_>,
    base_constraints: &Constraints,
    flags: SignalFlags,
    out: &mut Vec<Candidate>,
    seen: &mut BTreeSet<String>,
) {
    let normal_cost = normal_family_offline_cost(x, flags);
    let normal_cost_name = normal_family_cost_name(normal_cost);
    let normal_cost_warnings = normal_family_cost_warnings(normal_cost);

    if flags.huge_n {
        if flags.autocorrelated {
            let constraints = apply_jump_thinning(base_constraints, x.n, true);
            let candidate = Candidate {
                pipeline: PipelineConfig::Offline {
                    detector: OfflineDetectorConfig::Pelt(PeltConfig::default()),
                    cost: OfflineCostKind::Ar,
                    constraints,
                },
                pipeline_id: format!(
                    "offline:pelt:ar:jump={}",
                    apply_jump_thinning(base_constraints, x.n, true).jump
                ),
                warnings: vec![],
                primary_reason: "large-n offline series with autocorrelation favors PELT+AR"
                    .to_string(),
                driver_keys: vec!["lag1_autocorr", "lagk_autocorr", "residual_lag1_autocorr"],
                profile: PerformanceProfile {
                    speed: 0.78,
                    accuracy: 0.82,
                    robustness: 0.70,
                },
                family: CandidateFamily::StrongMapped,
                supported_signals: vec![SignalKind::Autocorrelated],
                evidence_support: evidence_support(0.82, flags, false),
                has_approximation_warning: false,
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
                detector: OfflineDetectorConfig::Fpop(FpopConfig::default()),
                cost: OfflineCostKind::L2,
                constraints,
            },
            pipeline_id: format!(
                "offline:fpop:l2:jump={}",
                apply_jump_thinning(base_constraints, x.n, true).jump
            ),
            warnings: vec![],
            primary_reason:
                "default large-n offline path uses FPOP+L2 with jump thinning for speed".to_string(),
            driver_keys: vec!["change_density_score", "regime_change_proxy", "nan_rate"],
            profile: PerformanceProfile {
                speed: 0.92,
                accuracy: 0.73,
                robustness: 0.55,
            },
            family: CandidateFamily::Generic,
            supported_signals: vec![SignalKind::ChangeDense],
            evidence_support: evidence_support(0.60, flags, false),
            has_approximation_warning: false,
        };
        push_candidate(candidate, out, seen);
    } else if flags.medium_n {
        if flags.heavy_tail {
            let candidate = Candidate {
                pipeline: PipelineConfig::Offline {
                    detector: OfflineDetectorConfig::BinSeg(BinSegConfig::default()),
                    cost: OfflineCostKind::L1Median,
                    constraints: base_constraints.clone(),
                },
                pipeline_id: format!("offline:binseg:l1_median:jump={}", base_constraints.jump),
                warnings: vec![
                    "CostL1Median is a slow path: O(m) median recomputation per segment (not O(1)); use for moderate n with heavy outlier contamination"
                        .to_string(),
                ],
                primary_reason:
                    "medium-n heavy-tail signal favors robust L1 median segment cost".to_string(),
                driver_keys: vec!["kurtosis_proxy", "outlier_rate_iqr", "mad_to_std_ratio"],
                profile: PerformanceProfile {
                    speed: 0.38,
                    accuracy: 0.74,
                    robustness: 0.93,
                },
                family: CandidateFamily::StrongMapped,
                supported_signals: vec![SignalKind::HeavyTail],
                evidence_support: evidence_support(0.86, flags, false),
                has_approximation_warning: false,
            };
            push_candidate(candidate, out, seen);
        }

        if flags.few_strong_changes && flags.masking_risk {
            let mut wbs = WbsConfig::default();
            wbs.interval_strategy = WbsIntervalStrategy::Stratified;

            let candidate = Candidate {
                pipeline: PipelineConfig::Offline {
                    detector: OfflineDetectorConfig::Wbs(wbs),
                    cost: normal_cost,
                    constraints: base_constraints.clone(),
                },
                pipeline_id: format!(
                    "offline:wbs:{normal_cost_name}:jump={}",
                    base_constraints.jump
                ),
                warnings: normal_cost_warnings.clone(),
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
                    cost: normal_cost,
                    constraints: base_constraints.clone(),
                },
                pipeline_id: format!(
                    "offline:binseg:{normal_cost_name}:jump={}",
                    base_constraints.jump
                ),
                warnings: normal_cost_warnings.clone(),
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
            let fpop_candidate = Candidate {
                pipeline: PipelineConfig::Offline {
                    detector: OfflineDetectorConfig::Fpop(FpopConfig::default()),
                    cost: OfflineCostKind::L2,
                    constraints: base_constraints.clone(),
                },
                pipeline_id: format!("offline:fpop:l2:jump={}", base_constraints.jump),
                warnings: vec![],
                primary_reason: "change-dense medium-n series benefits from global FPOP search"
                    .to_string(),
                driver_keys: vec![
                    "change_density_score",
                    "regime_change_proxy",
                    "lagk_autocorr",
                ],
                profile: PerformanceProfile {
                    speed: 0.92,
                    accuracy: 0.73,
                    robustness: 0.55,
                },
                family: CandidateFamily::StrongMapped,
                supported_signals: vec![SignalKind::ChangeDense],
                evidence_support: evidence_support(0.70, flags, false),
                has_approximation_warning: false,
            };
            push_candidate(fpop_candidate, out, seen);

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
                cost: normal_cost,
                constraints: base_constraints.clone(),
            },
            pipeline_id: format!(
                "offline:binseg:{normal_cost_name}:jump={}",
                base_constraints.jump
            ),
            warnings: {
                let mut warnings =
                    vec!["Dynp is unavailable; mapped small-n exact branch to BinSeg".to_string()];
                warnings.extend(normal_cost_warnings.clone());
                warnings
            },
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
        PipelineConfig::Offline { detector, cost, .. } => {
            if matches!(cost, OfflineCostKind::L1Median) {
                return ResourceEstimate {
                    time_complexity:
                        "slow path: O(m) median per segment query (not O(1)); detector overhead applies"
                            .to_string(),
                    memory_complexity: "O(n)".to_string(),
                    relative_time_score: 0.88,
                    relative_memory_score: 0.45,
                };
            }

            match detector {
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
                OfflineDetectorConfig::Fpop(_) => ResourceEstimate {
                    time_complexity: "O(n) avg / O(n^2) worst".to_string(),
                    memory_complexity: "O(n)".to_string(),
                    relative_time_score: 0.38,
                    relative_memory_score: 0.40,
                },
                OfflineDetectorConfig::Wbs(_) => ResourceEstimate {
                    time_complexity: "O(M*n)".to_string(),
                    memory_complexity: "O(n + M)".to_string(),
                    relative_time_score: 0.65,
                    relative_memory_score: 0.55,
                },
            }
        }
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
            (OfflineDetectorConfig::Pelt(_), OfflineCostKind::Ar) => "PELT + AR",
            (OfflineDetectorConfig::Pelt(_), OfflineCostKind::Cosine) => "PELT + Cosine",
            (OfflineDetectorConfig::Pelt(_), OfflineCostKind::L1Median) => "PELT + L1 median",
            (OfflineDetectorConfig::Pelt(_), OfflineCostKind::L2) => "PELT + L2",
            (OfflineDetectorConfig::Pelt(_), OfflineCostKind::Normal) => "PELT + Normal",
            (OfflineDetectorConfig::Pelt(_), OfflineCostKind::NormalFullCov) => {
                "PELT + Normal (full covariance)"
            }
            (OfflineDetectorConfig::Pelt(_), OfflineCostKind::Nig) => "PELT + NIG",
            (OfflineDetectorConfig::Pelt(_), OfflineCostKind::Rank) => "PELT + Rank",
            (OfflineDetectorConfig::BinSeg(_), OfflineCostKind::Ar) => "BinSeg + AR",
            (OfflineDetectorConfig::BinSeg(_), OfflineCostKind::Cosine) => "BinSeg + Cosine",
            (OfflineDetectorConfig::BinSeg(_), OfflineCostKind::L1Median) => "BinSeg + L1 median",
            (OfflineDetectorConfig::BinSeg(_), OfflineCostKind::Normal) => "BinSeg + Normal",
            (OfflineDetectorConfig::BinSeg(_), OfflineCostKind::NormalFullCov) => {
                "BinSeg + Normal (full covariance)"
            }
            (OfflineDetectorConfig::BinSeg(_), OfflineCostKind::L2) => "BinSeg + L2",
            (OfflineDetectorConfig::BinSeg(_), OfflineCostKind::Nig) => "BinSeg + NIG",
            (OfflineDetectorConfig::BinSeg(_), OfflineCostKind::Rank) => "BinSeg + Rank",
            (OfflineDetectorConfig::Fpop(_), OfflineCostKind::L2) => "FPOP + L2",
            (OfflineDetectorConfig::Fpop(_), OfflineCostKind::Ar)
            | (OfflineDetectorConfig::Fpop(_), OfflineCostKind::Cosine)
            | (OfflineDetectorConfig::Fpop(_), OfflineCostKind::L1Median)
            | (OfflineDetectorConfig::Fpop(_), OfflineCostKind::Normal)
            | (OfflineDetectorConfig::Fpop(_), OfflineCostKind::NormalFullCov)
            | (OfflineDetectorConfig::Fpop(_), OfflineCostKind::Nig)
            | (OfflineDetectorConfig::Fpop(_), OfflineCostKind::Rank) => {
                "FPOP + non-L2 (unsupported)"
            }
            (OfflineDetectorConfig::Wbs(_), OfflineCostKind::Ar) => "WBS + AR",
            (OfflineDetectorConfig::Wbs(_), OfflineCostKind::Cosine) => "WBS + Cosine",
            (OfflineDetectorConfig::Wbs(_), OfflineCostKind::L1Median) => "WBS + L1 median",
            (OfflineDetectorConfig::Wbs(_), OfflineCostKind::L2) => "WBS + L2",
            (OfflineDetectorConfig::Wbs(_), OfflineCostKind::Normal) => "WBS + Normal",
            (OfflineDetectorConfig::Wbs(_), OfflineCostKind::NormalFullCov) => {
                "WBS + Normal (full covariance)"
            }
            (OfflineDetectorConfig::Wbs(_), OfflineCostKind::Nig) => "WBS + NIG",
            (OfflineDetectorConfig::Wbs(_), OfflineCostKind::Rank) => "WBS + Rank",
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
                OfflineDetectorConfig::Fpop(_) => "fpop",
                OfflineDetectorConfig::Wbs(_) => "wbs",
            };
            let cost_name = match cost {
                OfflineCostKind::Ar => "ar",
                OfflineCostKind::Cosine => "cosine",
                OfflineCostKind::L1Median => "l1_median",
                OfflineCostKind::L2 => "l2",
                OfflineCostKind::Normal => "normal",
                OfflineCostKind::NormalFullCov => "normal_full_cov",
                OfflineCostKind::Nig => "nig",
                OfflineCostKind::Rank => "rank",
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

fn pipeline_id_spec(pipeline: &PipelineSpec) -> String {
    match pipeline.to_pipeline_config() {
        Ok(config) => pipeline_id(&config),
        Err(_) => "invalid_pipeline_spec".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CalibrationFamily, CalibrationObservation, CostConfig, DetectorConfig, Objective,
        OfflineCostKind, OfflineDetectorConfig, OnlineDetectorConfig, OnlineDetectorKind,
        PipelineConfig, PipelineSpec, confidence_formula, evaluate_calibration, recommend,
        validate_top_k,
    };
    use cpd_core::{
        Constraints, MemoryLayout, MissingPolicy, Penalty, ReproMode, Stopping, TimeIndex,
        TimeSeriesView,
    };
    use cpd_online::{CusumConfig, ObservationModel};
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

    fn make_multivariate_view(values: &[f64], n: usize, d: usize) -> TimeSeriesView<'_> {
        TimeSeriesView::from_f64(
            values,
            n,
            d,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Ignore,
        )
        .expect("multivariate view should be valid")
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

    fn make_recommendation(pipeline: PipelineConfig) -> super::Recommendation {
        super::Recommendation {
            pipeline: PipelineSpec::from_pipeline_config(&pipeline),
            resource_estimate: super::ResourceEstimate {
                time_complexity: "test".to_string(),
                memory_complexity: "test".to_string(),
                relative_time_score: 0.5,
                relative_memory_score: 0.5,
            },
            warnings: vec![],
            explanation: super::Explanation {
                summary: "test recommendation".to_string(),
                drivers: vec![],
                tradeoffs: vec![],
            },
            validation: None,
            confidence: 0.8,
            confidence_interval: (0.7, 0.9),
            abstain_reason: None,
            objective_fit: vec![],
        }
    }

    fn pseudo_uniform_noise(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let u = ((*state >> 33) as f64) / ((1_u64 << 31) as f64);
        u * 2.0 - 1.0
    }

    fn collect_offline_coverage(
        recommendations: &[super::Recommendation],
        detectors: &mut BTreeSet<&'static str>,
        costs: &mut BTreeSet<&'static str>,
    ) {
        for recommendation in recommendations {
            let pipeline = recommendation
                .pipeline
                .to_pipeline_config()
                .expect("pipeline should convert");
            if let PipelineConfig::Offline { detector, cost, .. } = pipeline {
                detectors.insert(match detector {
                    OfflineDetectorConfig::Pelt(_) => "pelt",
                    OfflineDetectorConfig::BinSeg(_) => "binseg",
                    OfflineDetectorConfig::Fpop(_) => "fpop",
                    OfflineDetectorConfig::Wbs(_) => "wbs",
                });
                costs.insert(match cost {
                    OfflineCostKind::Ar => "ar",
                    OfflineCostKind::Cosine => "cosine",
                    OfflineCostKind::L1Median => "l1_median",
                    OfflineCostKind::L2 => "l2",
                    OfflineCostKind::Normal => "normal",
                    OfflineCostKind::NormalFullCov => "normal_full_cov",
                    OfflineCostKind::Nig => "nig",
                    OfflineCostKind::Rank => "rank",
                });
            }
        }
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
    fn recommend_offline_huge_n_prefers_fpop_l2_with_jump_thinning() {
        let n = 120_000;
        let mut state = 123_u64;
        let values = (0..n)
            .map(|_| pseudo_uniform_noise(&mut state))
            .collect::<Vec<_>>();
        let view = make_univariate_view(&values);

        let recommendations =
            recommend(&view, Objective::Speed, false, None, 0.20, true).expect("recommend");

        assert!(!recommendations.is_empty());
        let pipeline = recommendations[0]
            .pipeline
            .to_pipeline_config()
            .expect("pipeline should convert");
        match &pipeline {
            PipelineConfig::Offline {
                detector: OfflineDetectorConfig::Fpop(_),
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
        let pipeline = recommendations[0]
            .pipeline
            .to_pipeline_config()
            .expect("pipeline should convert");
        match &pipeline {
            PipelineConfig::Offline {
                detector: OfflineDetectorConfig::Pelt(_),
                cost: OfflineCostKind::Nig,
                ..
            } => {}
            other => panic!("unexpected top recommendation: {other:?}"),
        }
    }

    #[test]
    fn recommend_offline_medium_n_heavy_tail_prefers_l1_median_slow_path() {
        let n = 20_000;
        let mut state = 11_u64;
        let mut values = Vec::with_capacity(n);
        for idx in 0..n {
            let mut v = pseudo_uniform_noise(&mut state);
            if idx % 29 == 0 {
                v *= 30.0;
            }
            values.push(v);
        }

        let view = make_univariate_view(&values);
        let recommendations =
            recommend(&view, Objective::Robustness, false, None, 0.20, true).expect("recommend");

        assert!(!recommendations.is_empty());
        let pipeline = recommendations[0]
            .pipeline
            .to_pipeline_config()
            .expect("pipeline should convert");
        match &pipeline {
            PipelineConfig::Offline {
                detector: OfflineDetectorConfig::BinSeg(_),
                cost: OfflineCostKind::L1Median,
                ..
            } => {}
            other => panic!("unexpected top recommendation: {other:?}"),
        }
        assert!(
            recommendations[0]
                .warnings
                .iter()
                .any(|warning| warning.contains("slow path")),
            "expected slow-path warning, got {:?}",
            recommendations[0].warnings
        );
    }

    #[test]
    fn l1_median_detects_true_shift_when_l2_is_outlier_dominated() {
        let n = 2_048;
        let true_cp = n / 2;
        let mut values = vec![0.0; n];
        for item in values.iter_mut().take(n).skip(true_cp) {
            *item = 3.0;
        }
        for item in values.iter_mut().take(325).skip(300) {
            *item = 60.0;
        }
        let view = make_univariate_view(&values);

        let detector = OfflineDetectorConfig::BinSeg(cpd_offline::BinSegConfig {
            stopping: Stopping::KnownK(1),
            params_per_segment: 2,
            cancel_check_every: 1_000,
        });
        let constraints = Constraints {
            min_segment_len: 16,
            max_change_points: Some(1),
            ..Constraints::default()
        };

        let l2_pipeline = PipelineSpec::from_pipeline_config(&PipelineConfig::Offline {
            detector: detector.clone(),
            cost: OfflineCostKind::L2,
            constraints: constraints.clone(),
        });
        let l1_pipeline = PipelineSpec::from_pipeline_config(&PipelineConfig::Offline {
            detector,
            cost: OfflineCostKind::L1Median,
            constraints,
        });

        let l2 = super::execute_pipeline_with_repro_mode(&view, &l2_pipeline, ReproMode::Balanced)
            .expect("L2 pipeline should execute");
        let l1 = super::execute_pipeline_with_repro_mode(&view, &l1_pipeline, ReproMode::Balanced)
            .expect("L1 pipeline should execute");

        let l2_cp = l2.breakpoints.first().copied().expect("L2 cp should exist");
        let l1_cp = l1.breakpoints.first().copied().expect("L1 cp should exist");

        assert!(
            l1_cp.abs_diff(true_cp) <= 12,
            "expected L1 cp near {true_cp}, got {l1_cp}"
        );
        assert!(
            l2_cp.abs_diff(true_cp) >= 128,
            "expected L2 cp to be materially displaced by outliers, got {l2_cp}"
        );
    }

    #[test]
    fn recommend_offline_autocorrelated_prefers_ar_cost() {
        let n = 120_000;
        let values = (0..n)
            .map(|t| (2.0 * std::f64::consts::PI * t as f64 / 200.0).sin())
            .collect::<Vec<_>>();

        let view = make_univariate_view(&values);
        let recommendations =
            recommend(&view, Objective::Accuracy, false, None, 0.20, true).expect("recommend");

        assert!(!recommendations.is_empty());
        let pipeline = recommendations[0]
            .pipeline
            .to_pipeline_config()
            .expect("pipeline should convert");
        match &pipeline {
            PipelineConfig::Offline {
                detector: OfflineDetectorConfig::Pelt(_),
                cost: OfflineCostKind::Ar,
                ..
            } => {
                assert!(
                    recommendations[0]
                        .warnings
                        .iter()
                        .all(|warning| !warning.contains("CostAR/CostLinear"))
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
        let pipeline = recommendations[0]
            .pipeline
            .to_pipeline_config()
            .expect("pipeline should convert");
        match &pipeline {
            PipelineConfig::Offline {
                detector: OfflineDetectorConfig::Wbs(_),
                cost: OfflineCostKind::Normal,
                ..
            } => {}
            other => panic!("unexpected top recommendation: {other:?}"),
        }
    }

    #[test]
    fn recommend_offline_multivariate_adds_semantics_warning() {
        let n = 300;
        let d = 4;
        let mut values = vec![0.0; n * d];
        for t in 0..n {
            for dim in 0..d {
                let base = if t < n / 2 { 0.0 } else { 6.0 };
                values[t * d + dim] = base + 0.25 * dim as f64 + 0.001 * t as f64;
            }
        }

        let view = make_multivariate_view(values.as_slice(), n, d);
        let recommendations =
            recommend(&view, Objective::Balanced, false, None, 0.20, true).expect("recommend");

        assert!(!recommendations.is_empty());
        assert!(
            recommendations[0]
                .warnings
                .iter()
                .any(|warning| warning.contains("multivariate semantics")),
            "expected multivariate semantics warning, got {:?}",
            recommendations[0].warnings
        );
    }

    #[test]
    fn recommend_offline_multivariate_surfaces_full_covariance_normal_cost() {
        let n = 512;
        let d = 4;
        let mut values = vec![0.0; n * d];
        for t in 0..n {
            let regime_shift = if t < n / 2 { 0.0 } else { 5.0 };
            let base = regime_shift + 0.01 * t as f64;
            for dim in 0..d {
                values[t * d + dim] = base * (1.0 + 0.1 * dim as f64);
            }
        }

        let view = make_multivariate_view(values.as_slice(), n, d);
        let recommendations =
            recommend(&view, Objective::Balanced, false, None, 0.20, true).expect("recommend");
        assert!(!recommendations.is_empty());

        let has_full_cov = recommendations.iter().any(|recommendation| {
            matches!(
                recommendation
                    .pipeline
                    .to_pipeline_config()
                    .expect("pipeline should convert"),
                PipelineConfig::Offline {
                    cost: OfflineCostKind::NormalFullCov,
                    ..
                }
            )
        });
        assert!(
            has_full_cov,
            "expected at least one multivariate recommendation to use normal_full_cov, got {:?}",
            recommendations
                .iter()
                .map(|entry| super::pipeline_id_spec(&entry.pipeline))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn recommend_online_binary_prefers_bocpd_bernoulli() {
        let values = (0..1024)
            .map(|idx| if idx % 2 == 0 { 0.0 } else { 1.0 })
            .collect::<Vec<_>>();
        let view = make_univariate_view(&values);

        let recommendations =
            recommend(&view, Objective::Balanced, true, None, 0.20, true).expect("recommend");

        let pipeline = recommendations[0]
            .pipeline
            .to_pipeline_config()
            .expect("pipeline should convert");
        match &pipeline {
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
    fn recommend_online_multivariate_is_rejected_with_clear_error() {
        let n = 256;
        let d = 3;
        let mut values = vec![0.0; n * d];
        for t in 0..n {
            for dim in 0..d {
                values[t * d + dim] = (0.01 * t as f64).sin() + dim as f64 * 0.1;
            }
        }
        let view = make_multivariate_view(values.as_slice(), n, d);

        let err = recommend(&view, Objective::Balanced, true, None, 0.20, true)
            .expect_err("online multivariate recommend should fail");
        let msg = err.to_string();
        assert!(msg.contains("support univariate series only"));
        assert!(msg.contains("Use offline recommendations for multivariate data"));
    }

    #[test]
    fn recommend_online_count_prefers_bocpd_poisson() {
        let values = (0..1024).map(|idx| (idx % 7) as f64).collect::<Vec<_>>();
        let view = make_univariate_view(&values);

        let recommendations =
            recommend(&view, Objective::Balanced, true, None, 0.20, true).expect("recommend");

        let pipeline = recommendations[0]
            .pipeline
            .to_pipeline_config()
            .expect("pipeline should convert");
        match &pipeline {
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
        let pipeline = recommendations[0]
            .pipeline
            .to_pipeline_config()
            .expect("pipeline should convert");
        match &pipeline {
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
    fn validate_top_k_reports_high_scores_for_agreeing_pipelines() {
        let mut values = vec![0.0; 180];
        for item in values.iter_mut().take(120).skip(60) {
            *item = 7.0;
        }
        for item in values.iter_mut().skip(120) {
            *item = -4.0;
        }
        let view = make_univariate_view(&values);

        let base_pipeline = PipelineConfig::Offline {
            detector: OfflineDetectorConfig::Pelt(super::PeltConfig {
                stopping: Stopping::KnownK(2),
                params_per_segment: 2,
                cancel_check_every: 16,
            }),
            cost: OfflineCostKind::L2,
            constraints: Constraints {
                min_segment_len: 8,
                ..Constraints::default()
            },
        };
        let recommendations = vec![
            make_recommendation(base_pipeline.clone()),
            make_recommendation(base_pipeline.clone()),
            make_recommendation(base_pipeline),
        ];

        let report = validate_top_k(&view, &recommendations, 3, None, Some(17))
            .expect("top-k validation should succeed");
        assert_eq!(report.pipeline_results.len(), 3);
        assert!(
            report.stability_score > 0.95,
            "expected high stability for agreeing pipelines, got {}",
            report.stability_score
        );
        assert!(
            report.agreement_score > 0.95,
            "expected high agreement for agreeing pipelines, got {}",
            report.agreement_score
        );
    }

    #[test]
    fn validate_top_k_reports_low_scores_for_disagreeing_pipelines() {
        let mut state = 202_u64;
        let values = (0..600)
            .map(|_| pseudo_uniform_noise(&mut state))
            .collect::<Vec<_>>();
        let view = make_univariate_view(&values);

        let constraints = Constraints {
            min_segment_len: 12,
            ..Constraints::default()
        };
        let recommendations = vec![
            make_recommendation(PipelineConfig::Offline {
                detector: OfflineDetectorConfig::Pelt(super::PeltConfig {
                    stopping: Stopping::Penalized(Penalty::Manual(250.0)),
                    params_per_segment: 2,
                    cancel_check_every: 16,
                }),
                cost: OfflineCostKind::L2,
                constraints: constraints.clone(),
            }),
            make_recommendation(PipelineConfig::Offline {
                detector: OfflineDetectorConfig::Pelt(super::PeltConfig {
                    stopping: Stopping::Penalized(Penalty::Manual(5.0)),
                    params_per_segment: 2,
                    cancel_check_every: 16,
                }),
                cost: OfflineCostKind::L2,
                constraints: constraints.clone(),
            }),
            make_recommendation(PipelineConfig::Offline {
                detector: OfflineDetectorConfig::Pelt(super::PeltConfig {
                    stopping: Stopping::Penalized(Penalty::Manual(0.2)),
                    params_per_segment: 2,
                    cancel_check_every: 16,
                }),
                cost: OfflineCostKind::L2,
                constraints,
            }),
        ];

        let report = validate_top_k(&view, &recommendations, 3, None, Some(17))
            .expect("top-k validation should succeed");
        let cps = report
            .pipeline_results
            .iter()
            .map(|(_, result)| result.change_points.clone())
            .collect::<Vec<_>>();
        assert!(
            report.stability_score < 0.70,
            "expected low stability for disagreeing pipelines, got {}, cps={cps:?}",
            report.stability_score,
        );
        assert!(
            report.agreement_score < 0.80,
            "expected low agreement for disagreeing pipelines, got {}, cps={cps:?}",
            report.agreement_score,
        );
    }

    #[test]
    fn validate_top_k_computes_penalty_sensitivity_with_downsampling() {
        let mut values = vec![0.0; 200];
        for item in values.iter_mut().take(100).skip(50) {
            *item = 10.0;
        }
        for item in values.iter_mut().take(150).skip(100) {
            *item = -8.0;
        }
        for item in values.iter_mut().skip(150) {
            *item = 4.0;
        }
        let view = make_univariate_view(&values);

        let penalized_pipeline = PipelineConfig::Offline {
            detector: OfflineDetectorConfig::Pelt(super::PeltConfig {
                stopping: Stopping::Penalized(Penalty::Manual(6.0)),
                params_per_segment: 2,
                cancel_check_every: 16,
            }),
            cost: OfflineCostKind::L2,
            constraints: Constraints {
                min_segment_len: 10,
                ..Constraints::default()
            },
        };
        let recommendations = vec![
            make_recommendation(penalized_pipeline.clone()),
            make_recommendation(penalized_pipeline),
        ];

        let report = validate_top_k(&view, &recommendations, 2, Some(4), Some(99))
            .expect("top-k validation should succeed");
        assert!(report.penalty_sensitivity.is_some());
        assert_eq!(
            report.pipeline_results[0].1.breakpoints.last().copied(),
            Some(values.len())
        );
        assert!(report.notes.iter().any(|note| note.contains("downsampled")));
    }

    #[test]
    fn downsample_breakpoint_remap_uses_midpoint_between_sampled_rows() {
        let mapped =
            super::remap_breakpoints_to_original(&[2, 5], &[0, 4, 8, 12, 15], 16).expect("remap");
        assert_eq!(mapped, vec![6, 16]);
    }

    #[test]
    fn validate_top_k_imputes_missing_values_for_offline_validation() {
        let values = vec![
            f64::NAN,
            f64::NAN,
            0.0,
            0.0,
            0.0,
            0.0,
            5.0,
            5.0,
            f64::NAN,
            5.0,
            5.0,
            5.0,
        ];
        let view = TimeSeriesView::from_f64(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Ignore,
        )
        .expect("view with missing should be valid");

        let recommendation = make_recommendation(PipelineConfig::Offline {
            detector: OfflineDetectorConfig::Pelt(super::PeltConfig {
                stopping: Stopping::KnownK(1),
                params_per_segment: 2,
                cancel_check_every: 16,
            }),
            cost: OfflineCostKind::L2,
            constraints: Constraints {
                min_segment_len: 2,
                ..Constraints::default()
            },
        });

        let report =
            validate_top_k(&view, &[recommendation], 1, None, Some(123)).expect("validation");
        assert_eq!(report.pipeline_results.len(), 1);
        assert!(
            report
                .notes
                .iter()
                .any(|note| note.contains("forward-fill imputation"))
        );
    }

    #[test]
    fn validate_top_k_requires_two_successful_offline_runs_when_k_greater_than_one() {
        let values = vec![0.0; 64];
        let view = make_univariate_view(&values);

        let recommendations = vec![
            make_recommendation(PipelineConfig::Offline {
                detector: OfflineDetectorConfig::Pelt(super::PeltConfig {
                    stopping: Stopping::Penalized(Penalty::Manual(10.0)),
                    params_per_segment: 2,
                    cancel_check_every: 16,
                }),
                cost: OfflineCostKind::L2,
                constraints: Constraints {
                    min_segment_len: 2,
                    ..Constraints::default()
                },
            }),
            make_recommendation(PipelineConfig::Online {
                detector: OnlineDetectorConfig::Cusum(CusumConfig::default()),
            }),
        ];

        let err = validate_top_k(&view, &recommendations, 2, None, None)
            .expect_err("single successful offline validation should fail for k>1");
        assert!(
            err.to_string()
                .contains("at least two successfully validated offline pipelines")
        );
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
            path_ids.insert(super::pipeline_id_spec(&recommendation.pipeline));
        }

        let mut ar = vec![0.0; 120_000];
        for t in 1..ar.len() {
            ar[t] = 0.88 * ar[t - 1] + 0.2 * pseudo_uniform_noise(&mut state);
        }
        let ar_view = make_univariate_view(&ar);
        for recommendation in recommend(&ar_view, Objective::Accuracy, false, None, 0.20, true)
            .expect("ar recommendation")
        {
            path_ids.insert(super::pipeline_id_spec(&recommendation.pipeline));
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
            path_ids.insert(super::pipeline_id_spec(&recommendation.pipeline));
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
            path_ids.insert(super::pipeline_id_spec(&recommendation.pipeline));
        }

        let binary = (0..1024)
            .map(|idx| if idx % 2 == 0 { 0.0 } else { 1.0 })
            .collect::<Vec<_>>();
        let binary_view = make_univariate_view(&binary);
        for recommendation in recommend(&binary_view, Objective::Speed, true, None, 0.20, true)
            .expect("binary recommendation")
        {
            path_ids.insert(super::pipeline_id_spec(&recommendation.pipeline));
        }

        assert!(
            path_ids.len() >= 5,
            "expected at least five distinct recommendation paths, got {}: {:?}",
            path_ids.len(),
            path_ids
        );
    }

    #[test]
    fn recommend_fixture_suite_covers_all_current_offline_detectors_and_costs() {
        let mut detectors = BTreeSet::new();
        let mut costs = BTreeSet::new();

        let mut state = 101_u64;
        let large_noise = (0..120_000)
            .map(|_| pseudo_uniform_noise(&mut state))
            .collect::<Vec<_>>();
        let large_noise_view = make_univariate_view(&large_noise);
        let large_noise_recommendations =
            recommend(&large_noise_view, Objective::Speed, false, None, 0.20, true)
                .expect("large noise recommendation");
        collect_offline_coverage(&large_noise_recommendations, &mut detectors, &mut costs);

        let ar = (0..120_000)
            .map(|t| (2.0 * std::f64::consts::PI * t as f64 / 200.0).sin())
            .collect::<Vec<_>>();
        let ar_view = make_univariate_view(&ar);
        let ar_recommendations = recommend(&ar_view, Objective::Accuracy, false, None, 0.20, true)
            .expect("ar recommendation");
        collect_offline_coverage(&ar_recommendations, &mut detectors, &mut costs);

        let mut heavy = Vec::with_capacity(120_000);
        for idx in 0..120_000 {
            let mut v = pseudo_uniform_noise(&mut state);
            if idx % 13 == 0 {
                v *= 18.0;
            }
            heavy.push(v);
        }
        let heavy_view = make_univariate_view(&heavy);
        let heavy_recommendations =
            recommend(&heavy_view, Objective::Robustness, false, None, 0.20, true)
                .expect("heavy recommendation");
        collect_offline_coverage(&heavy_recommendations, &mut detectors, &mut costs);

        let mut medium_heavy = Vec::with_capacity(20_000);
        for idx in 0..20_000 {
            let mut v = pseudo_uniform_noise(&mut state);
            if idx % 29 == 0 {
                v *= 30.0;
            }
            medium_heavy.push(v);
        }
        let medium_heavy_view = make_univariate_view(&medium_heavy);
        let medium_heavy_recommendations = recommend(
            &medium_heavy_view,
            Objective::Robustness,
            false,
            None,
            0.20,
            true,
        )
        .expect("medium heavy recommendation");
        collect_offline_coverage(&medium_heavy_recommendations, &mut detectors, &mut costs);

        let mut masked_values = vec![0.0; 2_000];
        for item in masked_values.iter_mut().take(2_000).skip(1_000) {
            *item = 9.0;
        }
        let mut mask = vec![0_u8; 2_000];
        for item in mask.iter_mut().take(900).skip(500) {
            *item = 1;
        }
        let masked_view = make_univariate_view_with_mask(&masked_values, &mask);
        let masked_recommendations =
            recommend(&masked_view, Objective::Accuracy, false, None, 0.20, true)
                .expect("masked recommendation");
        collect_offline_coverage(&masked_recommendations, &mut detectors, &mut costs);

        let mut short_values = vec![0.0; 256];
        for item in short_values.iter_mut().skip(128) {
            *item = 6.0;
        }
        let short_view = make_univariate_view(&short_values);
        let short_recommendations =
            recommend(&short_view, Objective::Balanced, false, None, 0.20, true)
                .expect("short recommendation");
        collect_offline_coverage(&short_recommendations, &mut detectors, &mut costs);

        let expected_detectors = ["binseg", "fpop", "pelt", "wbs"]
            .into_iter()
            .collect::<BTreeSet<_>>();
        let expected_costs = ["ar", "l1_median", "l2", "nig", "normal"]
            .into_iter()
            .collect::<BTreeSet<_>>();

        assert_eq!(detectors, expected_detectors);
        assert_eq!(costs, expected_costs);
    }

    #[test]
    fn recommend_top_pipeline_executes_end_to_end() {
        let mut values = vec![0.0; 180];
        for item in values.iter_mut().take(120).skip(60) {
            *item = 7.0;
        }
        for item in values.iter_mut().skip(120) {
            *item = -4.0;
        }
        let view = make_univariate_view(&values);

        let recommendations =
            recommend(&view, Objective::Balanced, false, None, 0.20, true).expect("recommend");
        assert!(!recommendations.is_empty());

        let result = super::execute_pipeline(&view, &recommendations[0].pipeline)
            .expect("top recommendation should execute");
        assert_eq!(result.breakpoints.last().copied(), Some(values.len()));
        assert!(!result.diagnostics.algorithm.is_empty());
        assert!(!result.diagnostics.cost_model.is_empty());
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

    #[cfg(feature = "serde")]
    #[test]
    fn pipeline_spec_roundtrip_serde_executes_consistently() {
        let values = vec![0.0, 0.0, 0.0, 5.0, 5.0, 5.0, -3.0, -3.0, -3.0];
        let view = make_univariate_view(&values);
        let stopping = Stopping::KnownK(2);
        let pipeline = PipelineSpec {
            detector: DetectorConfig::Offline(OfflineDetectorConfig::Pelt(
                cpd_offline::PeltConfig {
                    stopping: stopping.clone(),
                    ..cpd_offline::PeltConfig::default()
                },
            )),
            cost: CostConfig::L2,
            preprocess: None,
            constraints: Constraints {
                min_segment_len: 2,
                ..Constraints::default()
            },
            stopping: Some(stopping),
            seed: None,
        };

        let first = super::execute_pipeline(&view, &pipeline).expect("first run should succeed");
        let encoded = serde_json::to_vec(&pipeline).expect("pipeline should serialize");
        let decoded: PipelineSpec =
            serde_json::from_slice(&encoded).expect("pipeline should decode");
        let second = super::execute_pipeline(&view, &decoded).expect("second run should succeed");

        assert_eq!(first.breakpoints, second.breakpoints);
        assert_eq!(first.diagnostics.algorithm, second.diagnostics.algorithm);
        assert_eq!(first.diagnostics.cost_model, second.diagnostics.cost_model);
    }

    #[test]
    fn execute_pipeline_rejects_online_spec_with_clear_error() {
        let values = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let view = make_univariate_view(&values);
        let pipeline = PipelineSpec {
            detector: DetectorConfig::Online(OnlineDetectorKind::Cusum),
            cost: CostConfig::None,
            preprocess: None,
            constraints: Constraints::default(),
            stopping: None,
            seed: None,
        };

        let err = super::execute_pipeline(&view, &pipeline)
            .expect_err("online pipeline should be rejected");
        assert!(err.to_string().contains("only supports offline pipelines"));
    }

    #[test]
    fn execute_pipeline_rejects_inconsistent_stopping_configuration() {
        let values = vec![0.0, 0.0, 0.0, 5.0, 5.0, 5.0, -3.0, -3.0, -3.0];
        let view = make_univariate_view(&values);
        let pipeline = PipelineSpec {
            detector: DetectorConfig::Offline(OfflineDetectorConfig::Pelt(
                cpd_offline::PeltConfig::default(),
            )),
            cost: CostConfig::L2,
            preprocess: None,
            constraints: Constraints {
                min_segment_len: 2,
                ..Constraints::default()
            },
            stopping: Some(Stopping::KnownK(2)),
            seed: None,
        };

        let err = super::execute_pipeline(&view, &pipeline)
            .expect_err("mismatched stopping configuration should be rejected");
        assert!(
            err.to_string()
                .contains("inconsistent stopping configuration")
        );
    }

    #[test]
    fn execute_pipeline_with_repro_mode_runs_offline_spec() {
        let values = vec![0.0, 0.0, 0.0, 5.0, 5.0, 5.0, -3.0, -3.0, -3.0];
        let view = make_univariate_view(&values);
        let stopping = Stopping::KnownK(2);
        let pipeline = PipelineSpec {
            detector: DetectorConfig::Offline(OfflineDetectorConfig::Pelt(
                cpd_offline::PeltConfig {
                    stopping: stopping.clone(),
                    ..cpd_offline::PeltConfig::default()
                },
            )),
            cost: CostConfig::L2,
            preprocess: None,
            constraints: Constraints {
                min_segment_len: 2,
                ..Constraints::default()
            },
            stopping: Some(stopping),
            seed: None,
        };

        let result = super::execute_pipeline_with_repro_mode(&view, &pipeline, ReproMode::Fast)
            .expect("offline pipeline should execute under fast repro mode");
        assert_eq!(result.breakpoints, vec![3, 6, 9]);
    }

    #[test]
    fn execute_pipeline_with_repro_mode_runs_normal_full_cov_spec() {
        let n = 12;
        let d = 2;
        let mut values = vec![0.0; n * d];
        for t in 0..n {
            let base = if t < 6 {
                0.1 * t as f64
            } else {
                5.0 + 0.1 * t as f64
            };
            values[t * d] = base;
            values[t * d + 1] = 1.1 * base + 0.01 * (t as f64).cos();
        }
        let view = make_multivariate_view(&values, n, d);
        let stopping = Stopping::KnownK(1);
        let pipeline = PipelineSpec {
            detector: DetectorConfig::Offline(OfflineDetectorConfig::Pelt(
                cpd_offline::PeltConfig {
                    stopping: stopping.clone(),
                    ..cpd_offline::PeltConfig::default()
                },
            )),
            cost: CostConfig::NormalFullCov,
            preprocess: None,
            constraints: Constraints {
                min_segment_len: 2,
                ..Constraints::default()
            },
            stopping: Some(stopping),
            seed: None,
        };

        let result = super::execute_pipeline_with_repro_mode(&view, &pipeline, ReproMode::Balanced)
            .expect("normal_full_cov offline pipeline should execute");
        assert_eq!(result.diagnostics.cost_model, "normal_full_cov");
        assert_eq!(result.breakpoints.last().copied(), Some(n));
        let first = result
            .change_points
            .first()
            .copied()
            .expect("KnownK(1) should produce one change point");
        assert!(
            first.abs_diff(6) <= 1,
            "expected a changepoint near 6, got {first}"
        );
    }

    #[test]
    fn confidence_formula_mentions_ood_penalty() {
        let formula = confidence_formula();
        assert!(formula.contains("ood_penalty"));
        assert!(formula.contains("diagnostic_divergence"));
    }
}
