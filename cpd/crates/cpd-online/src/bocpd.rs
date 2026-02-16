// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::event_time::{
    LateDataCounters, LateDataPolicy, OverflowPolicy, compare_event_time_then_arrival,
};
use cpd_core::{
    CpdError, ExecutionContext, OnlineDetector, OnlineStepResult, log_add_exp, log_sum_exp,
};
use std::f64::consts::PI;
use std::time::Instant;

/// Stable detector identifier used in checkpoint envelopes.
pub const BOCPD_DETECTOR_ID: &str = "bocpd";
/// BOCPD checkpoint state schema version.
pub const BOCPD_STATE_SCHEMA_VERSION: u32 = 1;

/// Hazard-function contract used by BOCPD run-length transitions.
pub trait HazardFunction: Send + Sync {
    fn log_hazard(&self, r: usize) -> f64;
    fn log_survival(&self, r: usize) -> f64;
}

/// Constant hazard: `h(r) = p_change`.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstantHazard {
    p_change: f64,
}

impl ConstantHazard {
    pub fn new(p_change: f64) -> Result<Self, CpdError> {
        if !(p_change.is_finite() && 0.0 < p_change && p_change < 1.0) {
            return Err(CpdError::invalid_input(format!(
                "constant hazard p_change must be finite and in (0,1); got {p_change}"
            )));
        }
        Ok(Self { p_change })
    }
}

impl HazardFunction for ConstantHazard {
    fn log_hazard(&self, _r: usize) -> f64 {
        self.p_change.ln()
    }

    fn log_survival(&self, _r: usize) -> f64 {
        (1.0 - self.p_change).ln()
    }
}

/// Geometric hazard parameterized by mean run length.
#[derive(Clone, Debug, PartialEq)]
pub struct GeometricHazard {
    mean_run_length: f64,
    p_change: f64,
}

impl GeometricHazard {
    pub fn new(mean_run_length: f64) -> Result<Self, CpdError> {
        if !mean_run_length.is_finite() || mean_run_length <= 1.0 {
            return Err(CpdError::invalid_input(format!(
                "geometric hazard mean_run_length must be finite and > 1; got {mean_run_length}"
            )));
        }

        let p_change = 1.0 / mean_run_length;
        Ok(Self {
            mean_run_length,
            p_change,
        })
    }

    pub fn mean_run_length(&self) -> f64 {
        self.mean_run_length
    }
}

impl HazardFunction for GeometricHazard {
    fn log_hazard(&self, _r: usize) -> f64 {
        self.p_change.ln()
    }

    fn log_survival(&self, _r: usize) -> f64 {
        (1.0 - self.p_change).ln()
    }
}

/// Built-in hazard variants for BOCPD.
#[derive(Clone, Debug, PartialEq)]
pub enum HazardSpec {
    Constant(ConstantHazard),
    Geometric(GeometricHazard),
}

impl Default for HazardSpec {
    fn default() -> Self {
        Self::Constant(ConstantHazard::new(1.0 / 200.0).expect("default hazard must be valid"))
    }
}

impl HazardFunction for HazardSpec {
    fn log_hazard(&self, r: usize) -> f64 {
        match self {
            Self::Constant(h) => h.log_hazard(r),
            Self::Geometric(h) => h.log_hazard(r),
        }
    }

    fn log_survival(&self, r: usize) -> f64 {
        match self {
            Self::Constant(h) => h.log_survival(r),
            Self::Geometric(h) => h.log_survival(r),
        }
    }
}

/// Normal-Inverse-Gamma prior for Gaussian observation model.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianNigPrior {
    pub mu0: f64,
    pub kappa0: f64,
    pub alpha0: f64,
    pub beta0: f64,
}

impl Default for GaussianNigPrior {
    fn default() -> Self {
        Self {
            mu0: 0.0,
            kappa0: 1.0,
            alpha0: 1.0,
            beta0: 1.0,
        }
    }
}

impl GaussianNigPrior {
    fn validate(&self) -> Result<(), CpdError> {
        if !self.mu0.is_finite() {
            return Err(CpdError::invalid_input("gaussian prior mu0 must be finite"));
        }
        if !self.kappa0.is_finite() || self.kappa0 <= 0.0 {
            return Err(CpdError::invalid_input(
                "gaussian prior kappa0 must be finite and > 0",
            ));
        }
        if !self.alpha0.is_finite() || self.alpha0 <= 0.0 {
            return Err(CpdError::invalid_input(
                "gaussian prior alpha0 must be finite and > 0",
            ));
        }
        if !self.beta0.is_finite() || self.beta0 <= 0.0 {
            return Err(CpdError::invalid_input(
                "gaussian prior beta0 must be finite and > 0",
            ));
        }
        Ok(())
    }
}

/// Gamma prior for Poisson-rate observation model.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct PoissonGammaPrior {
    pub alpha: f64,
    pub beta: f64,
}

impl Default for PoissonGammaPrior {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }
}

impl PoissonGammaPrior {
    fn validate(&self) -> Result<(), CpdError> {
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            return Err(CpdError::invalid_input(
                "poisson prior alpha must be finite and > 0",
            ));
        }
        if !self.beta.is_finite() || self.beta <= 0.0 {
            return Err(CpdError::invalid_input(
                "poisson prior beta must be finite and > 0",
            ));
        }
        Ok(())
    }
}

/// Beta prior for Bernoulli observation model.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct BernoulliBetaPrior {
    pub alpha: f64,
    pub beta: f64,
}

impl Default for BernoulliBetaPrior {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }
}

impl BernoulliBetaPrior {
    fn validate(&self) -> Result<(), CpdError> {
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            return Err(CpdError::invalid_input(
                "bernoulli prior alpha must be finite and > 0",
            ));
        }
        if !self.beta.is_finite() || self.beta <= 0.0 {
            return Err(CpdError::invalid_input(
                "bernoulli prior beta must be finite and > 0",
            ));
        }
        Ok(())
    }
}

/// Observation model variants for BOCPD.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub enum ObservationModel {
    Gaussian { prior: GaussianNigPrior },
    Poisson { prior: PoissonGammaPrior },
    Bernoulli { prior: BernoulliBetaPrior },
}

impl Default for ObservationModel {
    fn default() -> Self {
        Self::Gaussian {
            prior: GaussianNigPrior::default(),
        }
    }
}

impl ObservationModel {
    fn validate(&self) -> Result<(), CpdError> {
        match self {
            Self::Gaussian { prior } => prior.validate(),
            Self::Poisson { prior } => prior.validate(),
            Self::Bernoulli { prior } => prior.validate(),
        }
    }

    fn prior_stats(&self) -> ObservationStats {
        match self {
            Self::Gaussian { .. } => ObservationStats::Gaussian {
                n: 0,
                sum: 0.0,
                sum_sq: 0.0,
            },
            Self::Poisson { .. } => ObservationStats::Poisson { n: 0, sum: 0 },
            Self::Bernoulli { .. } => ObservationStats::Bernoulli { n: 0, ones: 0 },
        }
    }

    fn update_stats(
        &self,
        current: &ObservationStats,
        x: f64,
    ) -> Result<ObservationStats, CpdError> {
        match (self, current) {
            (Self::Gaussian { .. }, ObservationStats::Gaussian { n, sum, sum_sq }) => {
                if !x.is_finite() {
                    return Err(CpdError::invalid_input(
                        "gaussian observation must be finite",
                    ));
                }
                Ok(ObservationStats::Gaussian {
                    n: n.saturating_add(1),
                    sum: *sum + x,
                    sum_sq: *sum_sq + x * x,
                })
            }
            (Self::Poisson { .. }, ObservationStats::Poisson { n, sum }) => {
                let value = parse_non_negative_count(x)?;
                Ok(ObservationStats::Poisson {
                    n: n.saturating_add(1),
                    sum: sum.saturating_add(value),
                })
            }
            (Self::Bernoulli { .. }, ObservationStats::Bernoulli { n, ones }) => {
                let value = parse_bernoulli(x)?;
                Ok(ObservationStats::Bernoulli {
                    n: n.saturating_add(1),
                    ones: ones.saturating_add(u64::from(value)),
                })
            }
            _ => Err(CpdError::numerical_issue(
                "observation stats variant mismatch with configured model",
            )),
        }
    }

    fn log_predictive(&self, current: &ObservationStats, x: f64) -> Result<f64, CpdError> {
        match (self, current) {
            (Self::Gaussian { prior }, ObservationStats::Gaussian { n, sum, sum_sq }) => {
                gaussian_log_predictive(prior, *n, *sum, *sum_sq, x)
            }
            (Self::Poisson { prior }, ObservationStats::Poisson { n, sum }) => {
                poisson_log_predictive(prior, *n, *sum, x)
            }
            (Self::Bernoulli { prior }, ObservationStats::Bernoulli { n, ones }) => {
                bernoulli_log_predictive(prior, *n, *ones, x)
            }
            _ => Err(CpdError::numerical_issue(
                "observation stats variant mismatch with configured model",
            )),
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub enum ObservationStats {
    Gaussian { n: usize, sum: f64, sum_sq: f64 },
    Poisson { n: usize, sum: u64 },
    Bernoulli { n: usize, ones: u64 },
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
struct PendingEvent {
    x: f64,
    t_ns: i64,
    arrival_seq: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct StepSummary {
    t: usize,
    p_change: f64,
    alert: bool,
    run_length_mode: usize,
    run_length_mean: f64,
}

/// Serializable BOCPD state for checkpoint/restore.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct BocpdState {
    pub t: usize,
    pub watermark_ns: Option<i64>,
    /// Event-time late-data counters that survive checkpoint/restore.
    #[cfg_attr(feature = "serde", serde(default))]
    pub late_data: LateDataCounters,
    pub log_run_probs: Vec<f64>,
    pub run_stats: Vec<ObservationStats>,
    /// Pending events awaiting flush according to `late_data_policy`.
    #[cfg_attr(feature = "serde", serde(default))]
    pending_events: Vec<PendingEvent>,
    #[cfg_attr(feature = "serde", serde(default))]
    next_arrival_seq: u64,
}

impl BocpdState {
    fn new(observation: &ObservationModel) -> Self {
        Self {
            t: 0,
            watermark_ns: None,
            late_data: LateDataCounters::default(),
            log_run_probs: vec![0.0],
            run_stats: vec![observation.prior_stats()],
            pending_events: vec![],
            next_arrival_seq: 0,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), CpdError> {
        if self.log_run_probs.is_empty() {
            return Err(CpdError::invalid_input(
                "bocpd state requires at least one run-length probability",
            ));
        }
        if self.log_run_probs.len() != self.run_stats.len() {
            return Err(CpdError::invalid_input(format!(
                "bocpd state length mismatch: log_run_probs={}, run_stats={}",
                self.log_run_probs.len(),
                self.run_stats.len()
            )));
        }
        for event in &self.pending_events {
            if !event.x.is_finite() {
                return Err(CpdError::invalid_input(
                    "bocpd pending event contains non-finite observation",
                ));
            }
        }
        Ok(())
    }
}

/// BOCPD configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct BocpdConfig {
    pub hazard: HazardSpec,
    pub observation: ObservationModel,
    pub max_run_length: usize,
    /// Relative log-probability threshold (must be <= 0). Entries below `max + threshold` are pruned.
    pub log_prob_threshold: Option<f64>,
    pub alert_threshold: f64,
    /// Event-time late-data policy used when `update(..., t_ns=Some(...), ...)`.
    pub late_data_policy: LateDataPolicy,
}

impl Default for BocpdConfig {
    fn default() -> Self {
        Self {
            hazard: HazardSpec::default(),
            observation: ObservationModel::default(),
            max_run_length: 2_000,
            log_prob_threshold: Some(-35.0),
            alert_threshold: 0.5,
            late_data_policy: LateDataPolicy::Reject,
        }
    }
}

impl BocpdConfig {
    fn validate(&self) -> Result<(), CpdError> {
        self.observation.validate()?;

        if self.max_run_length < 1 {
            return Err(CpdError::invalid_input(
                "max_run_length must be >= 1 for bounded BOCPD state",
            ));
        }

        if let Some(threshold) = self.log_prob_threshold
            && (!threshold.is_finite() || threshold > 0.0)
        {
            return Err(CpdError::invalid_input(
                "log_prob_threshold must be finite and <= 0",
            ));
        }

        if !self.alert_threshold.is_finite() || !(0.0..=1.0).contains(&self.alert_threshold) {
            return Err(CpdError::invalid_input(
                "alert_threshold must be finite and in [0,1]",
            ));
        }

        self.late_data_policy.validate()?;

        Ok(())
    }
}

/// Bayesian Online Change Point Detection implementation.
#[derive(Clone, Debug)]
pub struct BocpdDetector {
    config: BocpdConfig,
    state: BocpdState,
}

impl BocpdDetector {
    pub fn new(config: BocpdConfig) -> Result<Self, CpdError> {
        config.validate()?;
        let state = BocpdState::new(&config.observation);
        Ok(Self { config, state })
    }

    pub fn config(&self) -> &BocpdConfig {
        &self.config
    }

    pub fn state(&self) -> &BocpdState {
        &self.state
    }

    fn step_summary_from_log_probs(
        &self,
        log_probs: &[f64],
        t: usize,
    ) -> Result<StepSummary, CpdError> {
        let first = *log_probs.first().ok_or_else(|| {
            CpdError::numerical_issue("BOCPD step summary requires at least one run-length state")
        })?;

        let mut p_change = first.exp();
        if !p_change.is_finite() {
            return Err(CpdError::numerical_issue(
                "BOCPD p_change became non-finite after normalization",
            ));
        }
        p_change = p_change.clamp(0.0, 1.0);

        Ok(StepSummary {
            t,
            p_change,
            alert: p_change >= self.config.alert_threshold,
            run_length_mode: argmax_index(log_probs),
            run_length_mean: run_length_expectation(log_probs),
        })
    }

    fn current_step_summary(&self) -> Result<StepSummary, CpdError> {
        self.step_summary_from_log_probs(&self.state.log_run_probs, self.state.t.saturating_sub(1))
    }

    fn materialize_step_result(
        &self,
        summary: StepSummary,
        started_at: Instant,
        alert_reason: Option<String>,
    ) -> OnlineStepResult {
        OnlineStepResult {
            t: summary.t,
            p_change: summary.p_change,
            alert: summary.alert,
            alert_reason: alert_reason.or_else(|| {
                summary.alert.then(|| {
                    format!(
                        "bocpd p_change {:.6} >= threshold {:.6}",
                        summary.p_change, self.config.alert_threshold
                    )
                })
            }),
            run_length_mode: summary.run_length_mode,
            run_length_mean: summary.run_length_mean,
            processing_latency_us: Some(started_at.elapsed().as_micros() as u64),
        }
    }

    fn apply_observation(
        &mut self,
        x: f64,
        t_ns: Option<i64>,
        ctx: &ExecutionContext<'_>,
    ) -> Result<StepSummary, CpdError> {
        ctx.check_cancelled()?;
        let _ = ctx.check_cost_eval_budget(self.state.t.saturating_add(1))?;

        if !x.is_finite() {
            return Err(CpdError::invalid_input(
                "BOCPD observation must be finite for update",
            ));
        }

        self.state.validate()?;

        let prior_stats = self.config.observation.prior_stats();
        let prev_len = self.state.log_run_probs.len();
        let hard_cap = self.config.max_run_length.saturating_add(1);
        let candidate_len = prev_len.saturating_add(1);
        let keep_len = candidate_len.min(hard_cap).max(1);

        let mut next_log_probs = vec![f64::NEG_INFINITY; keep_len];
        let mut next_stats = vec![prior_stats.clone(); keep_len];

        let mut cp_mass = f64::NEG_INFINITY;
        let log_pred_reset = self.config.observation.log_predictive(&prior_stats, x)?;

        for run_length in 0..prev_len {
            let log_prev = self.state.log_run_probs[run_length];
            let log_pred_growth = self
                .config
                .observation
                .log_predictive(&self.state.run_stats[run_length], x)?;

            let cp_term = log_prev + self.config.hazard.log_hazard(run_length) + log_pred_reset;
            cp_mass = log_add_exp(cp_mass, cp_term);

            let next_run_length = run_length + 1;
            let growth_term =
                log_prev + self.config.hazard.log_survival(run_length) + log_pred_growth;

            if next_run_length < keep_len {
                next_log_probs[next_run_length] = growth_term;
                next_stats[next_run_length] = self
                    .config
                    .observation
                    .update_stats(&self.state.run_stats[run_length], x)?;
            } else {
                // Truncation redistributes overflow mass to run_length=0.
                cp_mass = log_add_exp(cp_mass, growth_term);
            }
        }

        next_log_probs[0] = cp_mass;
        next_stats[0] = self.config.observation.update_stats(&prior_stats, x)?;

        if let Some(threshold) = self.config.log_prob_threshold {
            let max_log_prob = next_log_probs
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let cutoff = max_log_prob + threshold;

            let mut keep = next_log_probs.len();
            while keep > 1 && next_log_probs[keep - 1] < cutoff {
                keep -= 1;
            }
            next_log_probs.truncate(keep);
            next_stats.truncate(keep);
        }

        normalize_log_probs(&mut next_log_probs)?;

        self.state.t = self.state.t.saturating_add(1);
        if let Some(ts) = t_ns {
            self.state.watermark_ns = Some(self.state.watermark_ns.map_or(ts, |w| w.max(ts)));
        }
        self.state.log_run_probs = next_log_probs;
        self.state.run_stats = next_stats;

        self.step_summary_from_log_probs(&self.state.log_run_probs, self.state.t.saturating_sub(1))
    }

    fn next_arrival_seq(&mut self) -> u64 {
        let seq = self.state.next_arrival_seq;
        self.state.next_arrival_seq = self.state.next_arrival_seq.saturating_add(1);
        seq
    }

    fn enqueue_pending_event(&mut self, x: f64, t_ns: i64, count_as_late_buffered: bool) -> u64 {
        let arrival_seq = self.next_arrival_seq();
        self.state.pending_events.push(PendingEvent {
            x,
            t_ns,
            arrival_seq,
        });
        if count_as_late_buffered {
            self.state.late_data.buffered_events =
                self.state.late_data.buffered_events.saturating_add(1);
        }
        arrival_seq
    }

    fn oldest_pending_index(&self, reorder_by_timestamp: bool) -> Option<usize> {
        if self.state.pending_events.is_empty() {
            return None;
        }

        if !reorder_by_timestamp {
            return Some(0);
        }

        self.state
            .pending_events
            .iter()
            .enumerate()
            .min_by(|(_, lhs), (_, rhs)| {
                compare_event_time_then_arrival(
                    lhs.t_ns,
                    lhs.arrival_seq,
                    rhs.t_ns,
                    rhs.arrival_seq,
                )
            })
            .map(|(idx, _)| idx)
    }

    fn pop_next_pending_event(
        &mut self,
        reorder_by_timestamp: bool,
    ) -> Option<(usize, PendingEvent)> {
        if self.state.pending_events.is_empty() {
            return None;
        }
        let idx = self.oldest_pending_index(reorder_by_timestamp)?;
        Some((idx, self.state.pending_events.remove(idx)))
    }

    fn drain_pending(
        &mut self,
        reorder_by_timestamp: bool,
        ctx: &ExecutionContext<'_>,
    ) -> Result<(), CpdError> {
        while let Some((idx, event)) = self.pop_next_pending_event(reorder_by_timestamp) {
            if reorder_by_timestamp && idx != 0 {
                self.state.late_data.reordered_events =
                    self.state.late_data.reordered_events.saturating_add(1);
            }

            match self.apply_observation(event.x, Some(event.t_ns), ctx) {
                Ok(_) => {}
                Err(err) => {
                    self.state
                        .pending_events
                        .insert(idx.min(self.state.pending_events.len()), event);
                    return Err(err);
                }
            }
        }

        Ok(())
    }
}

impl OnlineDetector for BocpdDetector {
    type State = BocpdState;

    fn reset(&mut self) {
        self.state = BocpdState::new(&self.config.observation);
    }

    fn update(
        &mut self,
        x_t: &[f64],
        t_ns: Option<i64>,
        ctx: &ExecutionContext<'_>,
    ) -> Result<OnlineStepResult, CpdError> {
        ctx.check_cancelled()?;

        if x_t.len() != 1 {
            return Err(CpdError::invalid_input(format!(
                "BOCPD currently supports univariate updates only; got d={} (expected 1)",
                x_t.len()
            )));
        }

        let x = x_t[0];
        if !x.is_finite() {
            return Err(CpdError::invalid_input(
                "BOCPD observation must be finite for update",
            ));
        }

        self.state.validate()?;
        let started_at = Instant::now();
        if t_ns.is_none() {
            if !self.state.pending_events.is_empty() {
                return Err(CpdError::invalid_input(
                    "BOCPD update received t_ns=None while pending late events exist; provide timestamps until the pending buffer drains",
                ));
            }
            let summary = self.apply_observation(x, None, ctx)?;
            return Ok(self.materialize_step_result(summary, started_at, None));
        }

        let ts = t_ns.expect("checked Some above");
        match self.config.late_data_policy.clone() {
            LateDataPolicy::Reject => {
                if self
                    .state
                    .watermark_ns
                    .is_some_and(|watermark| ts < watermark)
                {
                    self.state.late_data.late_events =
                        self.state.late_data.late_events.saturating_add(1);
                    return Err(CpdError::invalid_input(format!(
                        "late event rejected by policy=Reject: t_ns={ts}, watermark_ns={}",
                        self.state
                            .watermark_ns
                            .expect("watermark must exist when event is late"),
                    )));
                }

                let summary = self.apply_observation(x, Some(ts), ctx)?;
                Ok(self.materialize_step_result(summary, started_at, None))
            }
            LateDataPolicy::BufferWithinWindow {
                max_delay_ns,
                max_buffer_items,
                on_overflow,
            }
            | LateDataPolicy::ReorderByTimestamp {
                max_delay_ns,
                max_buffer_items,
                on_overflow,
            } => {
                let reorder_by_timestamp = matches!(
                    self.config.late_data_policy,
                    LateDataPolicy::ReorderByTimestamp { .. }
                );
                let watermark = self.state.watermark_ns;
                let is_late = watermark.is_some_and(|w| ts < w);

                if is_late {
                    self.state.late_data.late_events =
                        self.state.late_data.late_events.saturating_add(1);
                    let watermark_ns = watermark.expect("watermark must exist for late event");
                    let delay_ns = watermark_ns.saturating_sub(ts);
                    let within_window = delay_ns <= max_delay_ns;
                    let buffer_has_capacity = self.state.pending_events.len() < max_buffer_items;

                    if within_window && buffer_has_capacity {
                        self.enqueue_pending_event(x, ts, true);
                        let summary = self.current_step_summary()?;
                        return Ok(self.materialize_step_result(
                            summary,
                            started_at,
                            Some("late data buffered; detector state unchanged".to_string()),
                        ));
                    }

                    let cause = if within_window {
                        format!("buffer capacity exceeded (max_buffer_items={max_buffer_items})")
                    } else {
                        format!(
                            "delay exceeded window (delay_ns={delay_ns}, max_delay_ns={max_delay_ns})"
                        )
                    };

                    match on_overflow {
                        OverflowPolicy::DropNewest => {
                            self.state.late_data.dropped_newest =
                                self.state.late_data.dropped_newest.saturating_add(1);
                            let summary = self.current_step_summary()?;
                            Ok(self.materialize_step_result(
                                summary,
                                started_at,
                                Some(format!(
                                    "late data dropped ({cause}, overflow={})",
                                    on_overflow.as_str()
                                )),
                            ))
                        }
                        OverflowPolicy::Error => {
                            self.state.late_data.overflow_errors =
                                self.state.late_data.overflow_errors.saturating_add(1);
                            Err(CpdError::invalid_input(format!(
                                "late-data overflow (policy={}, overflow={}): {}",
                                self.config.late_data_policy.as_str(),
                                on_overflow.as_str(),
                                cause
                            )))
                        }
                        OverflowPolicy::DropOldest => {
                            if let Some(drop_idx) = self.oldest_pending_index(reorder_by_timestamp)
                            {
                                self.state.pending_events.remove(drop_idx);
                                self.state.late_data.dropped_oldest =
                                    self.state.late_data.dropped_oldest.saturating_add(1);
                            }

                            if self.state.pending_events.len() >= max_buffer_items {
                                self.state.late_data.dropped_newest =
                                    self.state.late_data.dropped_newest.saturating_add(1);
                                let summary = self.current_step_summary()?;
                                return Ok(self.materialize_step_result(
                                    summary,
                                    started_at,
                                    Some(format!(
                                        "late data dropped ({cause}, overflow={})",
                                        on_overflow.as_str()
                                    )),
                                ));
                            }

                            self.enqueue_pending_event(x, ts, true);
                            let summary = self.current_step_summary()?;
                            Ok(self.materialize_step_result(
                                summary,
                                started_at,
                                Some(format!(
                                    "late data buffered after dropping oldest ({cause}, overflow={})",
                                    on_overflow.as_str()
                                )),
                            ))
                        }
                    }
                } else {
                    self.drain_pending(reorder_by_timestamp, ctx)?;
                    let summary = self.apply_observation(x, Some(ts), ctx)?;
                    Ok(self.materialize_step_result(summary, started_at, None))
                }
            }
        }
    }

    fn save_state(&self) -> Self::State {
        self.state.clone()
    }

    fn load_state(&mut self, state: &Self::State) {
        self.state = state.clone();
    }
}

fn normalize_log_probs(log_probs: &mut [f64]) -> Result<(), CpdError> {
    let normalizer = log_sum_exp(log_probs);
    if !normalizer.is_finite() {
        return Err(CpdError::numerical_issue(
            "BOCPD normalization failed (non-finite log_sum_exp)",
        ));
    }

    for value in log_probs {
        *value -= normalizer;
        if value.is_nan() {
            return Err(CpdError::numerical_issue(
                "BOCPD normalization produced NaN run-length log probability",
            ));
        }
    }

    Ok(())
}

fn argmax_index(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn run_length_expectation(log_probs: &[f64]) -> f64 {
    log_probs
        .iter()
        .enumerate()
        .map(|(idx, log_prob)| idx as f64 * log_prob.exp())
        .sum()
}

fn gaussian_log_predictive(
    prior: &GaussianNigPrior,
    n: usize,
    sum: f64,
    sum_sq: f64,
    x: f64,
) -> Result<f64, CpdError> {
    if !x.is_finite() {
        return Err(CpdError::invalid_input(
            "gaussian observation must be finite",
        ));
    }

    let n_f64 = n as f64;
    let kappa_n = prior.kappa0 + n_f64;
    if !(kappa_n.is_finite() && kappa_n > 0.0) {
        return Err(CpdError::numerical_issue(
            "gaussian posterior kappa became non-finite or non-positive",
        ));
    }

    let mu_n = (prior.kappa0 * prior.mu0 + sum) / kappa_n;

    let centered_sse = if n == 0 {
        0.0
    } else {
        let mean = sum / n_f64;
        let raw = sum_sq - n_f64 * mean * mean;
        if raw <= 0.0 { 0.0 } else { raw }
    };

    let alpha_n = prior.alpha0 + 0.5 * n_f64;
    let shrinkage = if n == 0 {
        0.0
    } else {
        let mean = sum / n_f64;
        prior.kappa0 * n_f64 * (mean - prior.mu0).powi(2) / (2.0 * kappa_n)
    };
    let beta_n = prior.beta0 + 0.5 * centered_sse + shrinkage;

    if !(alpha_n.is_finite() && alpha_n > 0.0 && beta_n.is_finite() && beta_n > 0.0) {
        return Err(CpdError::numerical_issue(
            "gaussian posterior alpha/beta became non-finite or non-positive",
        ));
    }

    let nu = 2.0 * alpha_n;
    let scale_sq = beta_n * (kappa_n + 1.0) / (alpha_n * kappa_n);
    if !(nu.is_finite() && nu > 0.0 && scale_sq.is_finite() && scale_sq > 0.0) {
        return Err(CpdError::numerical_issue(
            "gaussian predictive scale became non-finite or non-positive",
        ));
    }

    let z = (x - mu_n).powi(2) / (nu * scale_sq);
    let log_norm =
        ln_gamma(0.5 * (nu + 1.0)) - ln_gamma(0.5 * nu) - 0.5 * (nu.ln() + PI.ln() + scale_sq.ln());
    let log_tail = -0.5 * (nu + 1.0) * (1.0 + z).ln();
    let out = log_norm + log_tail;

    if !out.is_finite() {
        return Err(CpdError::numerical_issue(
            "gaussian predictive log-likelihood became non-finite",
        ));
    }

    Ok(out)
}

fn poisson_log_predictive(
    prior: &PoissonGammaPrior,
    n: usize,
    sum: u64,
    x: f64,
) -> Result<f64, CpdError> {
    let count = parse_non_negative_count(x)?;
    let posterior_alpha = prior.alpha + (sum as f64);
    let posterior_beta = prior.beta + (n as f64);

    if !(posterior_alpha.is_finite() && posterior_alpha > 0.0) {
        return Err(CpdError::numerical_issue(
            "poisson posterior alpha became non-finite or non-positive",
        ));
    }
    if !(posterior_beta.is_finite() && posterior_beta > 0.0) {
        return Err(CpdError::numerical_issue(
            "poisson posterior beta became non-finite or non-positive",
        ));
    }

    let count_f64 = count as f64;
    let out =
        ln_gamma(posterior_alpha + count_f64) - ln_gamma(posterior_alpha) - ln_factorial(count)
            + posterior_alpha * (posterior_beta / (posterior_beta + 1.0)).ln()
            + count_f64 * (1.0 / (posterior_beta + 1.0)).ln();

    if !out.is_finite() {
        return Err(CpdError::numerical_issue(
            "poisson predictive log-likelihood became non-finite",
        ));
    }

    Ok(out)
}

fn bernoulli_log_predictive(
    prior: &BernoulliBetaPrior,
    n: usize,
    ones: u64,
    x: f64,
) -> Result<f64, CpdError> {
    let bit = parse_bernoulli(x)?;
    let alpha_n = prior.alpha + (ones as f64);
    let beta_n = prior.beta + ((n as u64).saturating_sub(ones) as f64);

    if !(alpha_n.is_finite() && alpha_n > 0.0 && beta_n.is_finite() && beta_n > 0.0) {
        return Err(CpdError::numerical_issue(
            "bernoulli posterior alpha/beta became non-finite or non-positive",
        ));
    }

    let denom = alpha_n + beta_n;
    let prob = if bit { alpha_n / denom } else { beta_n / denom };

    if !(prob.is_finite() && prob > 0.0) {
        return Err(CpdError::numerical_issue(
            "bernoulli predictive probability became non-finite or non-positive",
        ));
    }

    Ok(prob.ln())
}

fn parse_non_negative_count(x: f64) -> Result<u64, CpdError> {
    if !x.is_finite() || x < 0.0 {
        return Err(CpdError::invalid_input(format!(
            "poisson observation must be finite and >= 0; got {x}"
        )));
    }

    let rounded = x.round();
    if (rounded - x).abs() > 1e-9 {
        return Err(CpdError::invalid_input(format!(
            "poisson observation must be an integer-valued count; got {x}"
        )));
    }

    if rounded > u64::MAX as f64 {
        return Err(CpdError::invalid_input(format!(
            "poisson observation exceeds u64 range; got {x}"
        )));
    }

    Ok(rounded as u64)
}

fn parse_bernoulli(x: f64) -> Result<bool, CpdError> {
    if (x - 0.0).abs() <= 1e-12 {
        return Ok(false);
    }
    if (x - 1.0).abs() <= 1e-12 {
        return Ok(true);
    }
    Err(CpdError::invalid_input(format!(
        "bernoulli observation must be exactly 0 or 1; got {x}"
    )))
}

fn ln_factorial(n: u64) -> f64 {
    ln_gamma((n + 1) as f64)
}

fn ln_gamma(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if z <= 0.0 || !z.is_finite() {
        return f64::NAN;
    }

    if z < 0.5 {
        let sin_term = (PI * z).sin();
        if sin_term == 0.0 {
            return f64::INFINITY;
        }
        return PI.ln() - sin_term.ln() - ln_gamma(1.0 - z);
    }

    let x = z - 1.0;
    let mut acc = COEFFS[0];
    for (idx, coeff) in COEFFS.iter().enumerate().skip(1) {
        acc += coeff / (x + idx as f64);
    }

    let t = x + 7.5;
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + acc.ln()
}

#[cfg(test)]
mod tests {
    use super::{
        BocpdConfig, BocpdDetector, BocpdState, ConstantHazard, GeometricHazard, HazardSpec,
        LateDataPolicy, ObservationModel, OverflowPolicy,
    };
    use cpd_core::{Constraints, ExecutionContext, OnlineDetector};
    use std::sync::OnceLock;

    fn ctx() -> ExecutionContext<'static> {
        static CONSTRAINTS: OnceLock<Constraints> = OnceLock::new();
        let constraints = CONSTRAINTS.get_or_init(Constraints::default);
        ExecutionContext::new(constraints)
    }

    fn probs_from_log_probs(log_probs: &[f64]) -> Vec<f64> {
        log_probs.iter().map(|value| value.exp()).collect()
    }

    fn make_event_time_detector(policy: LateDataPolicy) -> BocpdDetector {
        BocpdDetector::new(BocpdConfig {
            hazard: HazardSpec::Constant(ConstantHazard::new(0.2).expect("valid hazard")),
            max_run_length: 32,
            log_prob_threshold: None,
            late_data_policy: policy,
            ..BocpdConfig::default()
        })
        .expect("config should be valid")
    }

    #[test]
    fn known_posterior_first_step_matches_closed_form() {
        let hazard_p = 0.2;
        let mut detector = BocpdDetector::new(BocpdConfig {
            hazard: HazardSpec::Constant(ConstantHazard::new(hazard_p).expect("valid hazard")),
            observation: ObservationModel::Bernoulli {
                prior: super::BernoulliBetaPrior {
                    alpha: 1.0,
                    beta: 1.0,
                },
            },
            max_run_length: 32,
            log_prob_threshold: None,
            ..BocpdConfig::default()
        })
        .expect("config should be valid");

        detector
            .update(&[1.0], None, &ctx())
            .expect("update should succeed");

        let probs = probs_from_log_probs(&detector.state().log_run_probs);
        let expected = [hazard_p, 1.0 - hazard_p];

        assert_eq!(probs.len(), expected.len());
        for (idx, (observed, expected)) in probs.iter().zip(expected).enumerate() {
            assert!(
                (observed - expected).abs() < 1e-12,
                "posterior mismatch at run_length={idx}: observed={observed}, expected={expected}",
            );
        }
    }

    #[test]
    fn known_posterior_two_step_bernoulli_matches_closed_form() {
        let hazard_p = 0.2;
        let mut detector = BocpdDetector::new(BocpdConfig {
            hazard: HazardSpec::Constant(ConstantHazard::new(hazard_p).expect("valid hazard")),
            observation: ObservationModel::Bernoulli {
                prior: super::BernoulliBetaPrior {
                    alpha: 1.0,
                    beta: 1.0,
                },
            },
            max_run_length: 32,
            log_prob_threshold: None,
            ..BocpdConfig::default()
        })
        .expect("config should be valid");

        detector
            .update(&[1.0], None, &ctx())
            .expect("first update should succeed");
        let step = detector
            .update(&[1.0], None, &ctx())
            .expect("second update should succeed");

        let normalizer = 4.0 - hazard_p;
        let expected = [
            (3.0 * hazard_p) / normalizer,
            (4.0 * hazard_p * (1.0 - hazard_p)) / normalizer,
            (4.0 * (1.0 - hazard_p) * (1.0 - hazard_p)) / normalizer,
        ];

        let probs = probs_from_log_probs(&detector.state().log_run_probs);
        assert_eq!(probs.len(), expected.len());
        for (idx, (observed, expected)) in probs.iter().zip(expected).enumerate() {
            assert!(
                (observed - expected).abs() < 1e-12,
                "posterior mismatch at run_length={idx}: observed={observed}, expected={expected}",
            );
        }

        assert!(
            (step.p_change - expected[0]).abs() < 1e-12,
            "step p_change mismatch: observed={}, expected={}",
            step.p_change,
            expected[0]
        );
    }

    #[test]
    fn constant_series_keeps_change_probability_low() {
        let mut detector = BocpdDetector::new(BocpdConfig {
            max_run_length: 256,
            alert_threshold: 0.7,
            ..BocpdConfig::default()
        })
        .expect("config should be valid");

        let mut tail_sum = 0.0;
        let mut tail_n = 0usize;
        for step in 0..300 {
            let result = detector
                .update(&[0.0], None, &ctx())
                .expect("update should succeed");
            if step >= 80 {
                tail_sum += result.p_change;
                tail_n += 1;
            }
        }

        let tail_mean = tail_sum / tail_n as f64;
        assert!(tail_mean < 0.2, "tail mean p_change too high: {tail_mean}");
    }

    #[test]
    fn step_shift_produces_change_probability_spike() {
        let mut detector = BocpdDetector::new(BocpdConfig {
            hazard: HazardSpec::Constant(ConstantHazard::new(1.0 / 80.0).expect("valid hazard")),
            max_run_length: 256,
            ..BocpdConfig::default()
        })
        .expect("config should be valid");

        let mut best_idx = 0usize;
        let mut best_prob = 0.0;

        for step in 0..240 {
            let x = if step < 120 { 0.0 } else { 6.0 };
            let result = detector
                .update(&[x], None, &ctx())
                .expect("update should succeed");
            if result.p_change > best_prob {
                best_prob = result.p_change;
                best_idx = step;
            }
        }

        assert!(
            best_prob > 0.25,
            "expected spike; observed p_change={best_prob}"
        );
        assert!(
            (105..=135).contains(&best_idx),
            "expected spike near changepoint; best_idx={best_idx}"
        );
    }

    #[test]
    fn checkpoint_restore_roundtrip_is_equivalent() {
        let mut baseline = BocpdDetector::new(BocpdConfig::default()).expect("valid config");
        let mut first = BocpdDetector::new(BocpdConfig::default()).expect("valid config");

        for i in 0..120 {
            let x = ((i as f64) * 0.07).sin();
            let lhs = baseline
                .update(&[x], None, &ctx())
                .expect("baseline update should succeed");
            let rhs = first
                .update(&[x], None, &ctx())
                .expect("first update should succeed");
            assert!((lhs.p_change - rhs.p_change).abs() < 1e-12);
        }

        let saved: BocpdState = first.save_state();
        let mut restored = BocpdDetector::new(BocpdConfig::default()).expect("valid config");
        restored.load_state(&saved);

        for i in 120..260 {
            let x = if i % 53 < 11 {
                4.0
            } else {
                ((i as f64) * 0.03).cos()
            };
            let lhs = baseline
                .update(&[x], None, &ctx())
                .expect("baseline update should succeed");
            let rhs = restored
                .update(&[x], None, &ctx())
                .expect("restored update should succeed");

            assert!((lhs.p_change - rhs.p_change).abs() < 1e-12);
            assert_eq!(lhs.alert, rhs.alert);
            assert_eq!(lhs.run_length_mode, rhs.run_length_mode);
        }
    }

    #[test]
    fn max_run_length_bounds_state_size() {
        let mut detector = BocpdDetector::new(BocpdConfig {
            max_run_length: 16,
            log_prob_threshold: None,
            ..BocpdConfig::default()
        })
        .expect("valid config");

        for i in 0..512 {
            detector
                .update(&[(i as f64) * 0.001], None, &ctx())
                .expect("update should succeed");
            assert!(detector.state().log_run_probs.len() <= 17);
        }
    }

    #[test]
    fn poisson_and_bernoulli_models_update_successfully() {
        let mut poisson = BocpdDetector::new(BocpdConfig {
            observation: ObservationModel::Poisson {
                prior: super::PoissonGammaPrior {
                    alpha: 1.0,
                    beta: 1.0,
                },
            },
            ..BocpdConfig::default()
        })
        .expect("valid config");

        for x in [0.0, 1.0, 2.0, 3.0, 1.0, 0.0] {
            poisson
                .update(&[x], None, &ctx())
                .expect("poisson update should succeed");
        }

        let mut bernoulli = BocpdDetector::new(BocpdConfig {
            observation: ObservationModel::Bernoulli {
                prior: super::BernoulliBetaPrior {
                    alpha: 1.0,
                    beta: 1.0,
                },
            },
            ..BocpdConfig::default()
        })
        .expect("valid config");

        for x in [0.0, 1.0, 1.0, 0.0, 1.0] {
            bernoulli
                .update(&[x], None, &ctx())
                .expect("bernoulli update should succeed");
        }

        let err = bernoulli
            .update(&[0.2], None, &ctx())
            .expect_err("non-binary observation should fail");
        assert!(
            err.to_string().contains("bernoulli observation"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn late_data_policy_validation_is_enforced_by_config() {
        let invalid = BocpdConfig {
            late_data_policy: LateDataPolicy::BufferWithinWindow {
                max_delay_ns: 0,
                max_buffer_items: 4,
                on_overflow: OverflowPolicy::Error,
            },
            ..BocpdConfig::default()
        };
        let err = BocpdDetector::new(invalid).expect_err("invalid max_delay_ns must fail");
        assert!(err.to_string().contains("max_delay_ns"));
    }

    #[test]
    fn in_order_event_time_updates_watermark() {
        let mut detector = make_event_time_detector(LateDataPolicy::Reject);
        detector
            .update(&[0.0], Some(100), &ctx())
            .expect("first update should succeed");
        detector
            .update(&[0.0], Some(110), &ctx())
            .expect("second update should succeed");
        assert_eq!(detector.state().watermark_ns, Some(110));
    }

    #[test]
    fn reject_policy_errors_on_late_event() {
        let mut detector = make_event_time_detector(LateDataPolicy::Reject);
        detector
            .update(&[0.0], Some(100), &ctx())
            .expect("first update should succeed");

        let err = detector
            .update(&[0.0], Some(90), &ctx())
            .expect_err("late update should fail for Reject policy");
        assert!(err.to_string().contains("late event rejected"));
        assert_eq!(detector.state().late_data.late_events, 1);
    }

    #[test]
    fn buffer_within_window_buffers_and_flushes_on_on_time_event() {
        let mut detector = make_event_time_detector(LateDataPolicy::BufferWithinWindow {
            max_delay_ns: 10,
            max_buffer_items: 8,
            on_overflow: OverflowPolicy::Error,
        });

        detector
            .update(&[0.0], Some(100), &ctx())
            .expect("first update should succeed");

        let buffered = detector
            .update(&[1.0], Some(95), &ctx())
            .expect("late update should be buffered");
        assert_eq!(detector.state().t, 1);
        assert_eq!(buffered.t, 0);
        assert!(
            buffered
                .alert_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("buffered"))
        );

        let flushed = detector
            .update(&[2.0], Some(101), &ctx())
            .expect("on-time update should flush pending events");
        assert_eq!(flushed.t, 2);
        assert_eq!(detector.state().t, 3);
        assert_eq!(detector.state().watermark_ns, Some(101));
        assert_eq!(detector.state().late_data.late_events, 1);
    }

    #[test]
    fn buffered_events_counter_tracks_only_late_buffered_events() {
        let mut detector = make_event_time_detector(LateDataPolicy::BufferWithinWindow {
            max_delay_ns: 10,
            max_buffer_items: 8,
            on_overflow: OverflowPolicy::Error,
        });

        detector
            .update(&[0.0], Some(100), &ctx())
            .expect("first on-time update should succeed");
        assert_eq!(detector.state().late_data.buffered_events, 0);

        detector
            .update(&[1.0], Some(95), &ctx())
            .expect("late update should be buffered");
        assert_eq!(detector.state().late_data.buffered_events, 1);

        detector
            .update(&[2.0], Some(101), &ctx())
            .expect("on-time update should flush pending events");
        detector
            .update(&[3.0], Some(102), &ctx())
            .expect("another on-time update should succeed");
        assert_eq!(detector.state().late_data.buffered_events, 1);
    }

    #[test]
    fn reorder_policy_reorders_by_timestamp_and_tracks_counter() {
        let mut detector = make_event_time_detector(LateDataPolicy::ReorderByTimestamp {
            max_delay_ns: 10,
            max_buffer_items: 8,
            on_overflow: OverflowPolicy::Error,
        });

        detector
            .update(&[0.0], Some(100), &ctx())
            .expect("first update should succeed");
        detector
            .update(&[1.0], Some(99), &ctx())
            .expect("late update should be buffered");
        detector
            .update(&[2.0], Some(98), &ctx())
            .expect("late update should be buffered");
        detector
            .update(&[3.0], Some(101), &ctx())
            .expect("on-time update should flush pending events");

        assert_eq!(detector.state().t, 4);
        assert!(detector.state().late_data.reordered_events > 0);
    }

    #[test]
    fn overflow_drop_newest_returns_noop_and_counts_drop() {
        let mut detector = make_event_time_detector(LateDataPolicy::BufferWithinWindow {
            max_delay_ns: 10,
            max_buffer_items: 1,
            on_overflow: OverflowPolicy::DropNewest,
        });

        detector
            .update(&[0.0], Some(100), &ctx())
            .expect("first update should succeed");
        detector
            .update(&[1.0], Some(99), &ctx())
            .expect("late update should be buffered");
        let dropped = detector
            .update(&[2.0], Some(98), &ctx())
            .expect("overflow drop-newest should return no-op");
        assert_eq!(detector.state().t, 1);
        assert!(
            dropped
                .alert_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("dropped"))
        );
        assert_eq!(detector.state().late_data.dropped_newest, 1);
    }

    #[test]
    fn overflow_drop_oldest_evicts_oldest_buffered_event() {
        let mut detector = make_event_time_detector(LateDataPolicy::BufferWithinWindow {
            max_delay_ns: 10,
            max_buffer_items: 1,
            on_overflow: OverflowPolicy::DropOldest,
        });

        detector
            .update(&[0.0], Some(100), &ctx())
            .expect("first update should succeed");
        detector
            .update(&[1.0], Some(99), &ctx())
            .expect("first late update should be buffered");
        detector
            .update(&[2.0], Some(98), &ctx())
            .expect("second late update should drop oldest and buffer incoming");

        let flushed = detector
            .update(&[3.0], Some(101), &ctx())
            .expect("on-time update should flush pending events");
        assert_eq!(flushed.t, 2);
        assert_eq!(detector.state().t, 3);
        assert_eq!(detector.state().late_data.dropped_oldest, 1);
    }

    #[test]
    fn on_time_update_flushes_full_buffer_without_dropping_current_event() {
        let mut detector = make_event_time_detector(LateDataPolicy::BufferWithinWindow {
            max_delay_ns: 10,
            max_buffer_items: 1,
            on_overflow: OverflowPolicy::DropNewest,
        });

        detector
            .update(&[0.0], Some(100), &ctx())
            .expect("first on-time update should succeed");
        detector
            .update(&[1.0], Some(99), &ctx())
            .expect("late update should be buffered");

        let flushed = detector
            .update(&[2.0], Some(101), &ctx())
            .expect("on-time update should flush buffer and process current event");
        assert_eq!(flushed.t, 2);
        assert_eq!(detector.state().t, 3);
        assert_eq!(detector.state().pending_events.len(), 0);
        assert_eq!(detector.state().late_data.dropped_newest, 0);
    }

    #[test]
    fn overflow_error_returns_invalid_input_and_counts() {
        let mut detector = make_event_time_detector(LateDataPolicy::BufferWithinWindow {
            max_delay_ns: 10,
            max_buffer_items: 1,
            on_overflow: OverflowPolicy::Error,
        });

        detector
            .update(&[0.0], Some(100), &ctx())
            .expect("first update should succeed");
        detector
            .update(&[1.0], Some(99), &ctx())
            .expect("late update should be buffered");
        let err = detector
            .update(&[2.0], Some(98), &ctx())
            .expect_err("overflow policy=Error should fail");
        assert!(err.to_string().contains("late-data overflow"));
        assert_eq!(detector.state().late_data.overflow_errors, 1);
        assert_eq!(detector.state().t, 1);
    }

    #[test]
    fn drain_error_does_not_drop_unprocessed_pending_events() {
        let constraints = Constraints {
            max_cost_evals: Some(1),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let mut detector = make_event_time_detector(LateDataPolicy::BufferWithinWindow {
            max_delay_ns: 10,
            max_buffer_items: 8,
            on_overflow: OverflowPolicy::Error,
        });

        detector
            .update(&[0.0], Some(100), &ctx)
            .expect("first on-time update should consume the only budget slot");
        detector
            .update(&[1.0], Some(95), &ctx)
            .expect("late update should be buffered without consuming budget");
        assert_eq!(detector.state().pending_events.len(), 1);

        let err = detector
            .update(&[2.0], Some(101), &ctx)
            .expect_err("flush should fail due to budget");
        assert!(
            err.to_string()
                .contains("constraints.max_cost_evals exceeded")
        );

        assert_eq!(detector.state().t, 1);
        assert_eq!(detector.state().pending_events.len(), 1);
        assert_eq!(detector.state().pending_events[0].t_ns, 95);
        assert_eq!(detector.state().pending_events[0].x, 1.0);
    }

    #[test]
    fn deterministic_replay_with_reorder_policy_is_stable() {
        let policy = LateDataPolicy::ReorderByTimestamp {
            max_delay_ns: 20,
            max_buffer_items: 16,
            on_overflow: OverflowPolicy::Error,
        };
        let mut lhs = make_event_time_detector(policy.clone());
        let mut rhs = make_event_time_detector(policy);
        let events = [
            (0.0, 100_i64),
            (0.1, 105),
            (2.0, 103),
            (1.5, 104),
            (3.0, 106),
            (0.3, 107),
            (4.0, 105),
            (0.2, 108),
        ];

        for (x, ts) in events {
            let left = lhs
                .update(&[x], Some(ts), &ctx())
                .expect("lhs update should succeed");
            let right = rhs
                .update(&[x], Some(ts), &ctx())
                .expect("rhs update should succeed");
            assert!((left.p_change - right.p_change).abs() < 1e-12);
            assert_eq!(left.alert, right.alert);
            assert_eq!(left.t, right.t);
            assert_eq!(left.run_length_mode, right.run_length_mode);
            assert_eq!(left.alert_reason, right.alert_reason);
        }

        assert_eq!(lhs.save_state(), rhs.save_state());
    }

    #[test]
    fn checkpoint_roundtrip_with_pending_buffer_is_equivalent() {
        let policy = LateDataPolicy::BufferWithinWindow {
            max_delay_ns: 20,
            max_buffer_items: 8,
            on_overflow: OverflowPolicy::Error,
        };
        let mut baseline = make_event_time_detector(policy.clone());
        let mut first = make_event_time_detector(policy.clone());

        baseline
            .update(&[0.0], Some(100), &ctx())
            .expect("baseline first update should succeed");
        first
            .update(&[0.0], Some(100), &ctx())
            .expect("first detector update should succeed");

        let baseline_noop = baseline
            .update(&[1.0], Some(90), &ctx())
            .expect("baseline late event should be buffered");
        let first_noop = first
            .update(&[1.0], Some(90), &ctx())
            .expect("first late event should be buffered");
        assert_eq!(baseline_noop.t, first_noop.t);
        assert_eq!(baseline_noop.alert_reason, first_noop.alert_reason);

        let saved: BocpdState = first.save_state();
        let mut restored = make_event_time_detector(policy);
        restored.load_state(&saved);

        for (x, ts) in [
            (2.0, 101_i64),
            (3.0, 102_i64),
            (4.0, 99_i64),
            (5.0, 103_i64),
        ] {
            let left = baseline
                .update(&[x], Some(ts), &ctx())
                .expect("baseline update should succeed");
            let right = restored
                .update(&[x], Some(ts), &ctx())
                .expect("restored update should succeed");
            assert!((left.p_change - right.p_change).abs() < 1e-12);
            assert_eq!(left.alert, right.alert);
            assert_eq!(left.run_length_mode, right.run_length_mode);
            assert_eq!(left.alert_reason, right.alert_reason);
        }

        assert_eq!(baseline.save_state(), restored.save_state());
    }

    #[test]
    fn geometric_hazard_and_threshold_validation() {
        assert!(GeometricHazard::new(200.0).is_ok());
        assert!(GeometricHazard::new(1.0).is_err());

        let config = BocpdConfig {
            hazard: HazardSpec::Geometric(
                GeometricHazard::new(120.0).expect("mean run length should be valid"),
            ),
            log_prob_threshold: Some(0.1),
            ..BocpdConfig::default()
        };

        let err = BocpdDetector::new(config).expect_err("positive threshold should fail");
        assert!(err.to_string().contains("log_prob_threshold"));
    }
}
