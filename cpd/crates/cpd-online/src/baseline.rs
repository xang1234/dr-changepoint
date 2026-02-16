// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::event_time::{
    LateDataCounters, LateDataPolicy, OverflowPolicy, compare_event_time_then_arrival,
};
use cpd_core::{CpdError, ExecutionContext, OnlineDetector, OnlineStepResult};
use std::time::Instant;

/// Stable detector identifier used in checkpoint envelopes.
pub const CUSUM_DETECTOR_ID: &str = "cusum";
/// CUSUM checkpoint state schema version.
pub const CUSUM_STATE_SCHEMA_VERSION: u32 = 1;
/// Stable detector identifier used in checkpoint envelopes.
pub const PAGE_HINKLEY_DETECTOR_ID: &str = "page_hinkley";
/// Page-Hinkley checkpoint state schema version.
pub const PAGE_HINKLEY_STATE_SCHEMA_VERSION: u32 = 1;

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
    score: f64,
    p_change: f64,
    alert: bool,
    run_length: usize,
}

#[derive(Clone, Debug)]
struct LatencyEstimator {
    window_started_at: Instant,
    samples_in_window: u32,
    last_estimate_us: u64,
}

impl LatencyEstimator {
    const WINDOW: u32 = 256;

    fn new() -> Self {
        Self {
            window_started_at: Instant::now(),
            samples_in_window: 0,
            last_estimate_us: 1,
        }
    }

    fn observe(&mut self) -> u64 {
        self.samples_in_window = self.samples_in_window.saturating_add(1);
        if self.samples_in_window >= Self::WINDOW {
            let elapsed_ns = self.window_started_at.elapsed().as_nanos();
            let avg_ns = elapsed_ns / u128::from(Self::WINDOW);
            // Ceil to 1us minimum so values are non-zero and easy to interpret.
            let avg_us = ((avg_ns.saturating_add(999)) / 1_000) as u64;
            self.last_estimate_us = avg_us.max(1);
            self.window_started_at = Instant::now();
            self.samples_in_window = 0;
        }
        self.last_estimate_us
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

fn normalize_threshold_ratio(score: f64, threshold: f64) -> f64 {
    if !score.is_finite() || !threshold.is_finite() || threshold <= 0.0 {
        return 0.0;
    }
    (score / threshold).clamp(0.0, 1.0)
}

/// CUSUM detector configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct CusumConfig {
    pub drift: f64,
    pub threshold: f64,
    pub target_mean: f64,
    pub late_data_policy: LateDataPolicy,
}

impl Default for CusumConfig {
    fn default() -> Self {
        Self {
            drift: 0.0,
            threshold: 8.0,
            target_mean: 0.0,
            late_data_policy: LateDataPolicy::Reject,
        }
    }
}

impl CusumConfig {
    fn validate(&self) -> Result<(), CpdError> {
        if !self.drift.is_finite() || self.drift < 0.0 {
            return Err(CpdError::invalid_input(format!(
                "CUSUM drift must be finite and >= 0; got {}",
                self.drift
            )));
        }
        if !self.threshold.is_finite() || self.threshold <= 0.0 {
            return Err(CpdError::invalid_input(format!(
                "CUSUM threshold must be finite and > 0; got {}",
                self.threshold
            )));
        }
        if !self.target_mean.is_finite() {
            return Err(CpdError::invalid_input(format!(
                "CUSUM target_mean must be finite; got {}",
                self.target_mean
            )));
        }
        self.late_data_policy.validate()?;
        Ok(())
    }
}

/// Serializable CUSUM state for checkpoint/restore.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct CusumState {
    pub t: usize,
    pub watermark_ns: Option<i64>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub late_data: LateDataCounters,
    pub score: f64,
    pub steps_since_alert: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    pending_events: Vec<PendingEvent>,
    #[cfg_attr(feature = "serde", serde(default))]
    next_arrival_seq: u64,
}

impl CusumState {
    fn new() -> Self {
        Self {
            t: 0,
            watermark_ns: None,
            late_data: LateDataCounters::default(),
            score: 0.0,
            steps_since_alert: 0,
            pending_events: vec![],
            next_arrival_seq: 0,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), CpdError> {
        if !self.score.is_finite() || self.score < 0.0 {
            return Err(CpdError::invalid_input(format!(
                "CUSUM state.score must be finite and >= 0; got {}",
                self.score
            )));
        }

        if self.steps_since_alert > self.t {
            return Err(CpdError::invalid_input(format!(
                "CUSUM state.steps_since_alert={} cannot exceed t={}",
                self.steps_since_alert, self.t
            )));
        }

        if !self.pending_events.is_empty() && self.watermark_ns.is_none() {
            return Err(CpdError::invalid_input(
                "CUSUM pending_events requires watermark_ns",
            ));
        }

        for event in &self.pending_events {
            if !event.x.is_finite() {
                return Err(CpdError::invalid_input(
                    "CUSUM pending event contains non-finite observation",
                ));
            }
        }

        Ok(())
    }
}

/// One-sided CUSUM detector for upward mean shifts.
#[derive(Clone, Debug)]
pub struct CusumDetector {
    config: CusumConfig,
    state: CusumState,
    latency_estimator: LatencyEstimator,
}

impl CusumDetector {
    pub fn new(config: CusumConfig) -> Result<Self, CpdError> {
        config.validate()?;
        Ok(Self {
            config,
            state: CusumState::new(),
            latency_estimator: LatencyEstimator::new(),
        })
    }

    pub fn config(&self) -> &CusumConfig {
        &self.config
    }

    pub fn state(&self) -> &CusumState {
        &self.state
    }

    fn step_summary(&self, t: usize, score: f64, run_length: usize) -> StepSummary {
        let p_change = normalize_threshold_ratio(score, self.config.threshold);
        StepSummary {
            t,
            score,
            p_change,
            alert: score >= self.config.threshold,
            run_length,
        }
    }

    fn current_step_summary(&self) -> StepSummary {
        self.step_summary(
            self.state.t.saturating_sub(1),
            self.state.score,
            self.state.steps_since_alert,
        )
    }

    fn materialize_step_result(
        &mut self,
        summary: StepSummary,
        alert_reason: Option<String>,
    ) -> OnlineStepResult {
        OnlineStepResult {
            t: summary.t,
            p_change: summary.p_change,
            alert: summary.alert,
            alert_reason: alert_reason.or_else(|| {
                summary.alert.then(|| {
                    format!(
                        "cusum score {:.6} > threshold {:.6}",
                        summary.score, self.config.threshold
                    )
                })
            }),
            run_length_mode: summary.run_length,
            run_length_mean: summary.run_length as f64,
            processing_latency_us: Some(self.latency_estimator.observe()),
        }
    }

    fn apply_observation(&mut self, x: f64, t_ns: Option<i64>) -> Result<StepSummary, CpdError> {
        if !x.is_finite() {
            return Err(CpdError::invalid_input(
                "CUSUM observation must be finite for update",
            ));
        }

        let increment = (x - self.config.target_mean) - self.config.drift;
        let next_score = (self.state.score + increment).max(0.0);

        if !next_score.is_finite() {
            return Err(CpdError::numerical_issue(
                "CUSUM score became non-finite during update",
            ));
        }

        let alert = next_score > self.config.threshold;
        self.state.steps_since_alert = if alert {
            0
        } else {
            self.state.steps_since_alert.saturating_add(1)
        };
        self.state.score = next_score;
        self.state.t = self.state.t.saturating_add(1);

        if let Some(ts) = t_ns {
            self.state.watermark_ns = Some(self.state.watermark_ns.map_or(ts, |w| w.max(ts)));
        }

        Ok(self.step_summary(
            self.state.t.saturating_sub(1),
            self.state.score,
            self.state.steps_since_alert,
        ))
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

            ctx.check_cancelled()?;
            let _ = ctx.check_cost_eval_budget(self.state.t.saturating_add(1))?;
            match self.apply_observation(event.x, Some(event.t_ns)) {
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

impl OnlineDetector for CusumDetector {
    type State = CusumState;

    fn reset(&mut self) {
        self.state = CusumState::new();
        self.latency_estimator.reset();
    }

    fn update(
        &mut self,
        x_t: &[f64],
        t_ns: Option<i64>,
        ctx: &ExecutionContext<'_>,
    ) -> Result<OnlineStepResult, CpdError> {
        if x_t.len() != 1 {
            return Err(CpdError::invalid_input(format!(
                "CUSUM currently supports univariate updates only; got d={} (expected 1)",
                x_t.len()
            )));
        }

        let x = x_t[0];
        if !x.is_finite() {
            return Err(CpdError::invalid_input(
                "CUSUM observation must be finite for update",
            ));
        }

        ctx.check_cancelled()?;
        self.state.validate()?;
        if t_ns.is_none() {
            if !self.state.pending_events.is_empty() {
                return Err(CpdError::invalid_input(
                    "CUSUM update received t_ns=None while pending late events exist; provide timestamps until the pending buffer drains",
                ));
            }
            let _ = ctx.check_cost_eval_budget(self.state.t.saturating_add(1))?;
            let summary = self.apply_observation(x, None)?;
            return Ok(self.materialize_step_result(summary, None));
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

                let _ = ctx.check_cost_eval_budget(self.state.t.saturating_add(1))?;
                let summary = self.apply_observation(x, Some(ts))?;
                Ok(self.materialize_step_result(summary, None))
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
                        let summary = self.current_step_summary();
                        return Ok(self.materialize_step_result(
                            summary,
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
                            let summary = self.current_step_summary();
                            Ok(self.materialize_step_result(
                                summary,
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
                                let summary = self.current_step_summary();
                                return Ok(self.materialize_step_result(
                                    summary,
                                    Some(format!(
                                        "late data dropped ({cause}, overflow={})",
                                        on_overflow.as_str()
                                    )),
                                ));
                            }

                            self.enqueue_pending_event(x, ts, true);
                            let summary = self.current_step_summary();
                            Ok(self.materialize_step_result(
                                summary,
                                Some(format!(
                                    "late data buffered after dropping oldest ({cause}, overflow={})",
                                    on_overflow.as_str()
                                )),
                            ))
                        }
                    }
                } else {
                    self.drain_pending(reorder_by_timestamp, ctx)?;
                    let _ = ctx.check_cost_eval_budget(self.state.t.saturating_add(1))?;
                    let summary = self.apply_observation(x, Some(ts))?;
                    Ok(self.materialize_step_result(summary, None))
                }
            }
        }
    }

    fn save_state(&self) -> Self::State {
        self.state.clone()
    }

    fn load_state(&mut self, state: &Self::State) {
        self.state = state.clone();
        self.latency_estimator.reset();
    }
}

/// Page-Hinkley detector configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct PageHinkleyConfig {
    pub delta: f64,
    pub threshold: f64,
    pub initial_mean: f64,
    pub late_data_policy: LateDataPolicy,
}

impl Default for PageHinkleyConfig {
    fn default() -> Self {
        Self {
            delta: 0.01,
            threshold: 8.0,
            initial_mean: 0.0,
            late_data_policy: LateDataPolicy::Reject,
        }
    }
}

impl PageHinkleyConfig {
    fn validate(&self) -> Result<(), CpdError> {
        if !self.delta.is_finite() || self.delta < 0.0 {
            return Err(CpdError::invalid_input(format!(
                "Page-Hinkley delta must be finite and >= 0; got {}",
                self.delta
            )));
        }
        if !self.threshold.is_finite() || self.threshold <= 0.0 {
            return Err(CpdError::invalid_input(format!(
                "Page-Hinkley threshold must be finite and > 0; got {}",
                self.threshold
            )));
        }
        if !self.initial_mean.is_finite() {
            return Err(CpdError::invalid_input(format!(
                "Page-Hinkley initial_mean must be finite; got {}",
                self.initial_mean
            )));
        }
        self.late_data_policy.validate()?;
        Ok(())
    }
}

/// Serializable Page-Hinkley state for checkpoint/restore.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct PageHinkleyState {
    pub t: usize,
    pub watermark_ns: Option<i64>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub late_data: LateDataCounters,
    pub n: usize,
    pub running_mean: f64,
    pub cumulative_sum: f64,
    pub cumulative_min: f64,
    pub steps_since_alert: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    pending_events: Vec<PendingEvent>,
    #[cfg_attr(feature = "serde", serde(default))]
    next_arrival_seq: u64,
}

impl PageHinkleyState {
    fn new(initial_mean: f64) -> Self {
        Self {
            t: 0,
            watermark_ns: None,
            late_data: LateDataCounters::default(),
            n: 0,
            running_mean: initial_mean,
            cumulative_sum: 0.0,
            cumulative_min: 0.0,
            steps_since_alert: 0,
            pending_events: vec![],
            next_arrival_seq: 0,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), CpdError> {
        if self.n != self.t {
            return Err(CpdError::invalid_input(format!(
                "Page-Hinkley state.n={} must equal t={} for incremental updates",
                self.n, self.t
            )));
        }

        if !self.running_mean.is_finite() {
            return Err(CpdError::invalid_input(
                "Page-Hinkley state.running_mean must be finite",
            ));
        }

        if !self.cumulative_sum.is_finite() {
            return Err(CpdError::invalid_input(
                "Page-Hinkley state.cumulative_sum must be finite",
            ));
        }

        if !self.cumulative_min.is_finite() {
            return Err(CpdError::invalid_input(
                "Page-Hinkley state.cumulative_min must be finite",
            ));
        }

        if self.cumulative_min > self.cumulative_sum {
            return Err(CpdError::invalid_input(format!(
                "Page-Hinkley state.cumulative_min={} cannot exceed cumulative_sum={}",
                self.cumulative_min, self.cumulative_sum
            )));
        }

        if self.steps_since_alert > self.t {
            return Err(CpdError::invalid_input(format!(
                "Page-Hinkley state.steps_since_alert={} cannot exceed t={}",
                self.steps_since_alert, self.t
            )));
        }

        if !self.pending_events.is_empty() && self.watermark_ns.is_none() {
            return Err(CpdError::invalid_input(
                "Page-Hinkley pending_events requires watermark_ns",
            ));
        }

        for event in &self.pending_events {
            if !event.x.is_finite() {
                return Err(CpdError::invalid_input(
                    "Page-Hinkley pending event contains non-finite observation",
                ));
            }
        }

        Ok(())
    }
}

/// Page-Hinkley detector for mean-shift detection with drift robustness.
#[derive(Clone, Debug)]
pub struct PageHinkleyDetector {
    config: PageHinkleyConfig,
    state: PageHinkleyState,
    latency_estimator: LatencyEstimator,
}

impl PageHinkleyDetector {
    pub fn new(config: PageHinkleyConfig) -> Result<Self, CpdError> {
        config.validate()?;
        Ok(Self {
            state: PageHinkleyState::new(config.initial_mean),
            config,
            latency_estimator: LatencyEstimator::new(),
        })
    }

    pub fn config(&self) -> &PageHinkleyConfig {
        &self.config
    }

    pub fn state(&self) -> &PageHinkleyState {
        &self.state
    }

    fn score(&self) -> f64 {
        (self.state.cumulative_sum - self.state.cumulative_min).max(0.0)
    }

    fn step_summary(&self, t: usize, score: f64, run_length: usize) -> StepSummary {
        let p_change = normalize_threshold_ratio(score, self.config.threshold);
        StepSummary {
            t,
            score,
            p_change,
            alert: score >= self.config.threshold,
            run_length,
        }
    }

    fn current_step_summary(&self) -> StepSummary {
        self.step_summary(
            self.state.t.saturating_sub(1),
            self.score(),
            self.state.steps_since_alert,
        )
    }

    fn materialize_step_result(
        &mut self,
        summary: StepSummary,
        alert_reason: Option<String>,
    ) -> OnlineStepResult {
        OnlineStepResult {
            t: summary.t,
            p_change: summary.p_change,
            alert: summary.alert,
            alert_reason: alert_reason.or_else(|| {
                summary.alert.then(|| {
                    format!(
                        "page_hinkley score {:.6} > threshold {:.6}",
                        summary.score, self.config.threshold
                    )
                })
            }),
            run_length_mode: summary.run_length,
            run_length_mean: summary.run_length as f64,
            processing_latency_us: Some(self.latency_estimator.observe()),
        }
    }

    fn apply_observation(&mut self, x: f64, t_ns: Option<i64>) -> Result<StepSummary, CpdError> {
        if !x.is_finite() {
            return Err(CpdError::invalid_input(
                "Page-Hinkley observation must be finite for update",
            ));
        }

        let n_next = self.state.n.saturating_add(1);
        if n_next == 0 {
            return Err(CpdError::numerical_issue(
                "Page-Hinkley sample count overflow during update",
            ));
        }

        let prev_mean = self.state.running_mean;
        let new_mean = prev_mean + (x - prev_mean) / (n_next as f64);
        if !new_mean.is_finite() {
            return Err(CpdError::numerical_issue(
                "Page-Hinkley running mean became non-finite during update",
            ));
        }

        let new_cumulative_sum = self.state.cumulative_sum + (x - new_mean) - self.config.delta;
        if !new_cumulative_sum.is_finite() {
            return Err(CpdError::numerical_issue(
                "Page-Hinkley cumulative sum became non-finite during update",
            ));
        }

        let new_cumulative_min = self.state.cumulative_min.min(new_cumulative_sum);
        if !new_cumulative_min.is_finite() {
            return Err(CpdError::numerical_issue(
                "Page-Hinkley cumulative minimum became non-finite during update",
            ));
        }

        self.state.n = n_next;
        self.state.running_mean = new_mean;
        self.state.cumulative_sum = new_cumulative_sum;
        self.state.cumulative_min = new_cumulative_min;
        self.state.t = self.state.t.saturating_add(1);

        if let Some(ts) = t_ns {
            self.state.watermark_ns = Some(self.state.watermark_ns.map_or(ts, |w| w.max(ts)));
        }

        let score = self.score();
        let alert = score > self.config.threshold;
        self.state.steps_since_alert = if alert {
            0
        } else {
            self.state.steps_since_alert.saturating_add(1)
        };

        Ok(self.step_summary(
            self.state.t.saturating_sub(1),
            score,
            self.state.steps_since_alert,
        ))
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

            ctx.check_cancelled()?;
            let _ = ctx.check_cost_eval_budget(self.state.t.saturating_add(1))?;
            match self.apply_observation(event.x, Some(event.t_ns)) {
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

impl OnlineDetector for PageHinkleyDetector {
    type State = PageHinkleyState;

    fn reset(&mut self) {
        self.state = PageHinkleyState::new(self.config.initial_mean);
        self.latency_estimator.reset();
    }

    fn update(
        &mut self,
        x_t: &[f64],
        t_ns: Option<i64>,
        ctx: &ExecutionContext<'_>,
    ) -> Result<OnlineStepResult, CpdError> {
        if x_t.len() != 1 {
            return Err(CpdError::invalid_input(format!(
                "Page-Hinkley currently supports univariate updates only; got d={} (expected 1)",
                x_t.len()
            )));
        }

        let x = x_t[0];
        if !x.is_finite() {
            return Err(CpdError::invalid_input(
                "Page-Hinkley observation must be finite for update",
            ));
        }

        ctx.check_cancelled()?;
        self.state.validate()?;
        if t_ns.is_none() {
            if !self.state.pending_events.is_empty() {
                return Err(CpdError::invalid_input(
                    "Page-Hinkley update received t_ns=None while pending late events exist; provide timestamps until the pending buffer drains",
                ));
            }
            let _ = ctx.check_cost_eval_budget(self.state.t.saturating_add(1))?;
            let summary = self.apply_observation(x, None)?;
            return Ok(self.materialize_step_result(summary, None));
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

                let _ = ctx.check_cost_eval_budget(self.state.t.saturating_add(1))?;
                let summary = self.apply_observation(x, Some(ts))?;
                Ok(self.materialize_step_result(summary, None))
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
                        let summary = self.current_step_summary();
                        return Ok(self.materialize_step_result(
                            summary,
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
                            let summary = self.current_step_summary();
                            Ok(self.materialize_step_result(
                                summary,
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
                                let summary = self.current_step_summary();
                                return Ok(self.materialize_step_result(
                                    summary,
                                    Some(format!(
                                        "late data dropped ({cause}, overflow={})",
                                        on_overflow.as_str()
                                    )),
                                ));
                            }

                            self.enqueue_pending_event(x, ts, true);
                            let summary = self.current_step_summary();
                            Ok(self.materialize_step_result(
                                summary,
                                Some(format!(
                                    "late data buffered after dropping oldest ({cause}, overflow={})",
                                    on_overflow.as_str()
                                )),
                            ))
                        }
                    }
                } else {
                    self.drain_pending(reorder_by_timestamp, ctx)?;
                    let _ = ctx.check_cost_eval_budget(self.state.t.saturating_add(1))?;
                    let summary = self.apply_observation(x, Some(ts))?;
                    Ok(self.materialize_step_result(summary, None))
                }
            }
        }
    }

    fn save_state(&self) -> Self::State {
        self.state.clone()
    }

    fn load_state(&mut self, state: &Self::State) {
        self.state = state.clone();
        self.latency_estimator.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CusumConfig, CusumDetector, CusumState, PageHinkleyConfig, PageHinkleyDetector,
        PageHinkleyState,
    };
    use crate::event_time::{LateDataPolicy, OverflowPolicy};
    use cpd_core::{Constraints, ExecutionContext, OnlineDetector};
    use std::sync::OnceLock;

    fn ctx() -> ExecutionContext<'static> {
        static CONSTRAINTS: OnceLock<Constraints> = OnceLock::new();
        let constraints = CONSTRAINTS.get_or_init(Constraints::default);
        ExecutionContext::new(constraints)
    }

    fn make_cusum_event_time_detector(policy: LateDataPolicy) -> CusumDetector {
        CusumDetector::new(CusumConfig {
            drift: 0.05,
            threshold: 1.0,
            target_mean: 0.0,
            late_data_policy: policy,
        })
        .expect("CUSUM config should be valid")
    }

    fn make_page_hinkley_event_time_detector(policy: LateDataPolicy) -> PageHinkleyDetector {
        PageHinkleyDetector::new(PageHinkleyConfig {
            delta: 0.05,
            threshold: 1.0,
            initial_mean: 0.0,
            late_data_policy: policy,
        })
        .expect("Page-Hinkley config should be valid")
    }

    #[test]
    fn cusum_config_validation_rejects_invalid_values() {
        let bad_drift = CusumConfig {
            drift: -1.0,
            ..CusumConfig::default()
        };
        let err = CusumDetector::new(bad_drift).expect_err("negative drift must fail");
        assert!(err.to_string().contains("drift"));

        let bad_threshold = CusumConfig {
            threshold: 0.0,
            ..CusumConfig::default()
        };
        let err = CusumDetector::new(bad_threshold).expect_err("zero threshold must fail");
        assert!(err.to_string().contains("threshold"));

        let bad_target = CusumConfig {
            target_mean: f64::NAN,
            ..CusumConfig::default()
        };
        let err = CusumDetector::new(bad_target).expect_err("NaN target_mean must fail");
        assert!(err.to_string().contains("target_mean"));
    }

    #[test]
    fn page_hinkley_config_validation_rejects_invalid_values() {
        let bad_delta = PageHinkleyConfig {
            delta: -0.1,
            ..PageHinkleyConfig::default()
        };
        let err = PageHinkleyDetector::new(bad_delta).expect_err("negative delta must fail");
        assert!(err.to_string().contains("delta"));

        let bad_threshold = PageHinkleyConfig {
            threshold: 0.0,
            ..PageHinkleyConfig::default()
        };
        let err = PageHinkleyDetector::new(bad_threshold).expect_err("zero threshold must fail");
        assert!(err.to_string().contains("threshold"));

        let bad_mean = PageHinkleyConfig {
            initial_mean: f64::NAN,
            ..PageHinkleyConfig::default()
        };
        let err = PageHinkleyDetector::new(bad_mean).expect_err("NaN initial_mean must fail");
        assert!(err.to_string().contains("initial_mean"));
    }

    #[test]
    fn cusum_constant_series_keeps_alerts_low() {
        let mut detector = CusumDetector::new(CusumConfig {
            drift: 0.01,
            threshold: 5.0,
            target_mean: 0.0,
            late_data_policy: LateDataPolicy::Reject,
        })
        .expect("valid config");

        for _ in 0..300 {
            let result = detector
                .update(&[0.0], None, &ctx())
                .expect("update should succeed");
            assert!(!result.alert);
            assert!(result.p_change <= 0.05);
        }
    }

    #[test]
    fn page_hinkley_constant_series_keeps_alerts_low() {
        let mut detector = PageHinkleyDetector::new(PageHinkleyConfig {
            delta: 0.01,
            threshold: 5.0,
            initial_mean: 0.0,
            late_data_policy: LateDataPolicy::Reject,
        })
        .expect("valid config");

        for _ in 0..300 {
            let result = detector
                .update(&[0.0], None, &ctx())
                .expect("update should succeed");
            assert!(!result.alert);
            assert!(result.p_change <= 0.05);
        }
    }

    #[test]
    fn cusum_step_shift_triggers_alert_after_change() {
        let mut detector = CusumDetector::new(CusumConfig {
            drift: 0.02,
            threshold: 5.0,
            target_mean: 0.0,
            late_data_policy: LateDataPolicy::Reject,
        })
        .expect("valid config");

        let mut first_alert = None;
        for step in 0..240 {
            let x = if step < 120 { 0.0 } else { 1.5 };
            let result = detector
                .update(&[x], None, &ctx())
                .expect("update should succeed");
            if result.alert && first_alert.is_none() {
                first_alert = Some(step);
            }
        }

        let first_alert = first_alert.expect("expected CUSUM alert after step shift");
        assert!(
            (120..=150).contains(&first_alert),
            "expected first alert near changepoint; got {first_alert}"
        );
    }

    #[test]
    fn page_hinkley_step_shift_triggers_alert_after_change() {
        let mut detector = PageHinkleyDetector::new(PageHinkleyConfig {
            delta: 0.01,
            threshold: 2.0,
            initial_mean: 0.0,
            late_data_policy: LateDataPolicy::Reject,
        })
        .expect("valid config");

        let mut first_alert = None;
        for step in 0..260 {
            let x = if step < 120 { 0.0 } else { 3.0 };
            let result = detector
                .update(&[x], None, &ctx())
                .expect("update should succeed");
            if result.alert && first_alert.is_none() {
                first_alert = Some(step);
            }
        }

        let first_alert = first_alert.expect("expected Page-Hinkley alert after step shift");
        assert!(
            (120..=160).contains(&first_alert),
            "expected first alert near changepoint; got {first_alert}"
        );
    }

    #[test]
    fn non_univariate_updates_are_rejected() {
        let mut cusum = CusumDetector::new(CusumConfig::default()).expect("valid config");
        let err = cusum
            .update(&[0.0, 1.0], None, &ctx())
            .expect_err("multivariate CUSUM update must fail");
        assert!(err.to_string().contains("univariate"));

        let mut ph = PageHinkleyDetector::new(PageHinkleyConfig::default()).expect("valid config");
        let err = ph
            .update(&[0.0, 1.0], None, &ctx())
            .expect_err("multivariate Page-Hinkley update must fail");
        assert!(err.to_string().contains("univariate"));
    }

    #[test]
    fn non_finite_updates_are_rejected() {
        let mut cusum = CusumDetector::new(CusumConfig::default()).expect("valid config");
        let err = cusum
            .update(&[f64::NAN], None, &ctx())
            .expect_err("NaN CUSUM update must fail");
        assert!(err.to_string().contains("finite"));

        let mut ph = PageHinkleyDetector::new(PageHinkleyConfig::default()).expect("valid config");
        let err = ph
            .update(&[f64::INFINITY], None, &ctx())
            .expect_err("non-finite Page-Hinkley update must fail");
        assert!(err.to_string().contains("finite"));
    }

    #[test]
    fn reset_clears_cusum_state() {
        let mut detector = CusumDetector::new(CusumConfig::default()).expect("valid config");
        detector
            .update(&[2.0], None, &ctx())
            .expect("update should succeed");
        assert!(detector.state().t > 0);

        detector.reset();
        assert_eq!(detector.state().t, 0);
        assert_eq!(detector.state().score, 0.0);
        assert_eq!(detector.state().steps_since_alert, 0);
        assert!(detector.state().watermark_ns.is_none());
    }

    #[test]
    fn reset_clears_page_hinkley_state() {
        let mut detector =
            PageHinkleyDetector::new(PageHinkleyConfig::default()).expect("valid config");
        detector
            .update(&[2.0], None, &ctx())
            .expect("update should succeed");
        assert!(detector.state().t > 0);

        detector.reset();
        assert_eq!(detector.state().t, 0);
        assert_eq!(detector.state().n, 0);
        assert_eq!(
            detector.state().running_mean,
            detector.config().initial_mean
        );
        assert_eq!(detector.state().cumulative_sum, 0.0);
        assert_eq!(detector.state().cumulative_min, 0.0);
        assert_eq!(detector.state().steps_since_alert, 0);
        assert!(detector.state().watermark_ns.is_none());
    }

    #[test]
    fn cusum_checkpoint_restore_roundtrip_is_equivalent() {
        let mut baseline = CusumDetector::new(CusumConfig::default()).expect("valid config");
        let mut first = CusumDetector::new(CusumConfig::default()).expect("valid config");

        for i in 0..120 {
            let x = if i < 50 { 0.0 } else { 1.2 };
            let lhs = baseline
                .update(&[x], None, &ctx())
                .expect("baseline update should succeed");
            let rhs = first
                .update(&[x], None, &ctx())
                .expect("first update should succeed");
            assert!((lhs.p_change - rhs.p_change).abs() < 1e-12);
            assert_eq!(lhs.alert, rhs.alert);
            assert_eq!(lhs.run_length_mode, rhs.run_length_mode);
        }

        let saved: CusumState = first.save_state();
        let mut restored = CusumDetector::new(CusumConfig::default()).expect("valid config");
        restored.load_state(&saved);

        for i in 120..260 {
            let x = if i % 37 < 9 { 2.0 } else { 0.1 };
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
    fn page_hinkley_checkpoint_restore_roundtrip_is_equivalent() {
        let mut baseline =
            PageHinkleyDetector::new(PageHinkleyConfig::default()).expect("valid config");
        let mut first =
            PageHinkleyDetector::new(PageHinkleyConfig::default()).expect("valid config");

        for i in 0..120 {
            let x = if i < 50 { 0.0 } else { 1.5 };
            let lhs = baseline
                .update(&[x], None, &ctx())
                .expect("baseline update should succeed");
            let rhs = first
                .update(&[x], None, &ctx())
                .expect("first update should succeed");
            assert!((lhs.p_change - rhs.p_change).abs() < 1e-12);
            assert_eq!(lhs.alert, rhs.alert);
            assert_eq!(lhs.run_length_mode, rhs.run_length_mode);
        }

        let saved: PageHinkleyState = first.save_state();
        let mut restored =
            PageHinkleyDetector::new(PageHinkleyConfig::default()).expect("valid config");
        restored.load_state(&saved);

        for i in 120..260 {
            let x = if i % 31 < 10 { 2.5 } else { 0.2 };
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
    fn cusum_reject_policy_errors_on_late_event() {
        let mut detector = make_cusum_event_time_detector(LateDataPolicy::Reject);
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
    fn page_hinkley_reject_policy_errors_on_late_event() {
        let mut detector = make_page_hinkley_event_time_detector(LateDataPolicy::Reject);
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
    fn cusum_buffer_within_window_buffers_and_flushes_on_on_time_event() {
        let mut detector = make_cusum_event_time_detector(LateDataPolicy::BufferWithinWindow {
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
    fn page_hinkley_buffer_within_window_buffers_and_flushes_on_on_time_event() {
        let mut detector =
            make_page_hinkley_event_time_detector(LateDataPolicy::BufferWithinWindow {
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
    fn cusum_reorder_policy_reorders_by_timestamp_and_tracks_counter() {
        let mut detector = make_cusum_event_time_detector(LateDataPolicy::ReorderByTimestamp {
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
    fn page_hinkley_reorder_policy_reorders_by_timestamp_and_tracks_counter() {
        let mut detector =
            make_page_hinkley_event_time_detector(LateDataPolicy::ReorderByTimestamp {
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
    fn cusum_overflow_drop_newest_returns_noop_and_counts_drop() {
        let mut detector = make_cusum_event_time_detector(LateDataPolicy::BufferWithinWindow {
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
        assert_eq!(detector.state().late_data.dropped_newest, 1);
        assert!(
            dropped
                .alert_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("dropped"))
        );
    }

    #[test]
    fn page_hinkley_overflow_drop_newest_returns_noop_and_counts_drop() {
        let mut detector =
            make_page_hinkley_event_time_detector(LateDataPolicy::BufferWithinWindow {
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
        assert_eq!(detector.state().late_data.dropped_newest, 1);
        assert!(
            dropped
                .alert_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("dropped"))
        );
    }

    #[test]
    fn cusum_overflow_drop_oldest_evicts_oldest_buffered_event() {
        let mut detector = make_cusum_event_time_detector(LateDataPolicy::BufferWithinWindow {
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
    fn page_hinkley_overflow_drop_oldest_evicts_oldest_buffered_event() {
        let mut detector =
            make_page_hinkley_event_time_detector(LateDataPolicy::BufferWithinWindow {
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
    fn cusum_overflow_error_returns_invalid_input_and_counts() {
        let mut detector = make_cusum_event_time_detector(LateDataPolicy::BufferWithinWindow {
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
    }

    #[test]
    fn page_hinkley_overflow_error_returns_invalid_input_and_counts() {
        let mut detector =
            make_page_hinkley_event_time_detector(LateDataPolicy::BufferWithinWindow {
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
    }

    #[test]
    fn cusum_deterministic_replay_with_reorder_policy_is_stable() {
        let policy = LateDataPolicy::ReorderByTimestamp {
            max_delay_ns: 20,
            max_buffer_items: 16,
            on_overflow: OverflowPolicy::Error,
        };
        let mut lhs = make_cusum_event_time_detector(policy.clone());
        let mut rhs = make_cusum_event_time_detector(policy);
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
    fn page_hinkley_deterministic_replay_with_reorder_policy_is_stable() {
        let policy = LateDataPolicy::ReorderByTimestamp {
            max_delay_ns: 20,
            max_buffer_items: 16,
            on_overflow: OverflowPolicy::Error,
        };
        let mut lhs = make_page_hinkley_event_time_detector(policy.clone());
        let mut rhs = make_page_hinkley_event_time_detector(policy);
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
    fn cusum_checkpoint_roundtrip_with_pending_buffer_is_equivalent() {
        let policy = LateDataPolicy::BufferWithinWindow {
            max_delay_ns: 20,
            max_buffer_items: 8,
            on_overflow: OverflowPolicy::Error,
        };
        let mut baseline = make_cusum_event_time_detector(policy.clone());
        let mut first = make_cusum_event_time_detector(policy.clone());

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

        let saved: CusumState = first.save_state();
        let mut restored = make_cusum_event_time_detector(policy);
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
    fn page_hinkley_checkpoint_roundtrip_with_pending_buffer_is_equivalent() {
        let policy = LateDataPolicy::BufferWithinWindow {
            max_delay_ns: 20,
            max_buffer_items: 8,
            on_overflow: OverflowPolicy::Error,
        };
        let mut baseline = make_page_hinkley_event_time_detector(policy.clone());
        let mut first = make_page_hinkley_event_time_detector(policy.clone());

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

        let saved: PageHinkleyState = first.save_state();
        let mut restored = make_page_hinkley_event_time_detector(policy);
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
}
