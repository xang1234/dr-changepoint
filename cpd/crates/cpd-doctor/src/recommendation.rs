// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::diagnostics::{
    DiagnosticsSummary, DoctorDiagnosticsConfig, MissingPattern, compute_diagnostics,
};
use cpd_core::{
    Constraints, CpdError, DTypeView, MemoryLayout, TimeSeriesView, validate_constraints_config,
};
use cpd_offline::{BinSegConfig, PeltConfig, WbsConfig, WbsIntervalStrategy};
use cpd_online::{
    BernoulliBetaPrior, BocpdConfig, CusumConfig, GaussianNigPrior, ObservationModel,
    PageHinkleyConfig, PoissonGammaPrior,
};
use std::collections::BTreeSet;

const BINARY_TOLERANCE: f64 = 1.0e-9;
const FAMILY_THRESHOLD: f64 = 0.98;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Objective {
    Balanced,
    Speed,
    Accuracy,
    Robustness,
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

    let diagnostics = compute_diagnostics(x, &DoctorDiagnosticsConfig::default())?;
    let data_hints = sample_data_hints(x, diagnostics.subsample_stride.max(1), BINARY_TOLERANCE)?;
    let flags = signal_flags(x, &diagnostics.summary);
    let strongest_signal = strongest_active_signal(&diagnostics.summary, flags, data_hints);

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

    let mut recommendations = scored
        .into_iter()
        .take(5)
        .map(|entry| entry.recommendation)
        .collect::<Vec<_>>();

    if recommendations.is_empty() {
        return Err(CpdError::invalid_input(
            "recommendation engine failed to produce any candidates",
        ));
    }

    let top_confidence = recommendations[0].confidence;
    if allow_abstain && top_confidence < min_confidence {
        let safe_candidate = build_safe_baseline_candidate(x, online, &base_constraints, flags);
        let mut safe = score_candidate(
            &safe_candidate,
            objective,
            &diagnostics.summary,
            diagnostics.used_subsampling,
            flags,
            strongest_signal,
        )
        .recommendation;
        safe.abstain_reason = Some(format!(
            "top confidence {:.3} below min_confidence {:.3}; drivers: {}",
            top_confidence,
            min_confidence,
            top_driver_summary(&diagnostics.summary)
        ));
        safe.warnings.push(
            "returned safe baseline because allow_abstain=true and threshold was not met"
                .to_string(),
        );
        return Ok(vec![safe]);
    }

    if !allow_abstain && top_confidence < min_confidence {
        recommendations[0].warnings.push(format!(
            "top confidence {:.3} is below min_confidence {:.3}; abstain disabled",
            top_confidence, min_confidence
        ));
    }

    Ok(recommendations)
}

fn score_candidate(
    candidate: &Candidate,
    objective: Objective,
    summary: &DiagnosticsSummary,
    used_subsampling: bool,
    flags: SignalFlags,
    strongest_signal: Option<SignalKind>,
) -> ScoredRecommendation {
    let objective_fit = objective_fit(candidate.profile);
    let selected_objective_score = objective_score(candidate.profile, objective);

    let confidence = confidence_score(
        candidate,
        objective,
        used_subsampling,
        flags,
        strongest_signal,
    );
    let confidence_interval = confidence_interval(confidence, flags.conflicting_signals);

    let explanation = build_explanation(candidate, summary);
    let final_score =
        0.55 * selected_objective_score + 0.35 * confidence + 0.10 * candidate.evidence_support;

    ScoredRecommendation {
        pipeline_id: candidate.pipeline_id.clone(),
        final_score,
        recommendation: Recommendation {
            pipeline: candidate.pipeline.clone(),
            resource_estimate: resource_estimate(&candidate.pipeline),
            warnings: candidate.warnings.clone(),
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

fn confidence_interval(confidence: f64, conflicting_signals: bool) -> (f64, f64) {
    let half_width =
        (0.10 + 0.25 * (1.0 - confidence) + if conflicting_signals { 0.05 } else { 0.0 })
            .clamp(0.08, 0.40);
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
                .then_with(|| signal_priority(*signal_b).cmp(&signal_priority(*signal_a)))
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

    let mut t = 0usize;
    while t < x.n {
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

            valid = valid.saturating_add(1);

            if (value - 0.0).abs() <= tolerance || (value - 1.0).abs() <= tolerance {
                binary = binary.saturating_add(1);
            }

            let rounded = value.round();
            if value >= 0.0 && (value - rounded).abs() <= tolerance {
                count = count.saturating_add(1);
            }
        }

        t = t.saturating_add(stride.max(1));
    }

    if x.n > 0 {
        let last_t = x.n - 1;
        for j in 0..x.d {
            let idx = source_index(x.layout, x.n, x.d, last_t, j)?;
            if idx >= source_len {
                continue;
            }
            let value = match x.values {
                DTypeView::F32(values) => f64::from(values[idx]),
                DTypeView::F64(values) => values[idx],
            };
            let mask_missing = x.missing_mask.map(|mask| mask[idx] == 1).unwrap_or(false);
            if mask_missing || value.is_nan() {
                continue;
            }

            valid = valid.saturating_add(1);
            if (value - 0.0).abs() <= tolerance || (value - 1.0).abs() <= tolerance {
                binary = binary.saturating_add(1);
            }
            let rounded = value.round();
            if value >= 0.0 && (value - rounded).abs() <= tolerance {
                count = count.saturating_add(1);
            }
        }
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
        Objective, OfflineCostKind, OfflineDetectorConfig, OnlineDetectorConfig, PipelineConfig,
        recommend,
    };
    use cpd_core::{MemoryLayout, MissingPolicy, TimeIndex, TimeSeriesView};
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
            }
            other => panic!("unexpected top recommendation: {other:?}"),
        }
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
}
