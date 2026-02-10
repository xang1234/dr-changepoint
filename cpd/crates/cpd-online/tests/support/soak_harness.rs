// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![allow(dead_code)]

use cpd_core::{CpdError, ExecutionContext, OnlineDetector, OnlineStepResult};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoakProfile {
    PrSmoke,
    Nightly1h,
    Weekly24h,
}

impl SoakProfile {
    pub fn target_runtime_seconds(self) -> u64 {
        match self {
            Self::PrSmoke => 120,
            Self::Nightly1h => 3_600,
            Self::Weekly24h => 86_400,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HarnessConfig {
    pub steps: usize,
    pub checkpoint_every: usize,
    pub sleep_per_step_ms: u64,
}

impl HarnessConfig {
    pub fn for_profile(profile: SoakProfile) -> Self {
        match profile {
            SoakProfile::PrSmoke => Self {
                steps: 2_000,
                checkpoint_every: 100,
                sleep_per_step_ms: 0,
            },
            SoakProfile::Nightly1h => Self {
                steps: 10_000,
                checkpoint_every: 200,
                sleep_per_step_ms: 0,
            },
            SoakProfile::Weekly24h => Self {
                steps: 50_000,
                checkpoint_every: 500,
                sleep_per_step_ms: 0,
            },
        }
    }
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self::for_profile(SoakProfile::PrSmoke)
    }
}

pub fn profile_from_env() -> SoakProfile {
    match std::env::var("CPD_SOAK_PROFILE")
        .ok()
        .as_deref()
        .unwrap_or("pr_smoke")
    {
        "nightly_1h" => SoakProfile::Nightly1h,
        "weekly_24h" => SoakProfile::Weekly24h,
        _ => SoakProfile::PrSmoke,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct MockOnlineState {
    pub updates_seen: usize,
}

#[derive(Clone, Debug)]
pub struct MockOnlineDetector {
    state: MockOnlineState,
    budget_checks: usize,
    alert_period: usize,
}

impl MockOnlineDetector {
    pub fn new(alert_period: usize) -> Self {
        Self {
            state: MockOnlineState::default(),
            budget_checks: 0,
            alert_period,
        }
    }
}

impl OnlineDetector for MockOnlineDetector {
    type State = MockOnlineState;

    fn reset(&mut self) {
        self.state = MockOnlineState::default();
        self.budget_checks = 0;
    }

    fn update(
        &mut self,
        x_t: &[f64],
        _t_ns: Option<i64>,
        ctx: &ExecutionContext<'_>,
    ) -> Result<OnlineStepResult, CpdError> {
        ctx.check_cancelled()?;
        self.budget_checks += 1;
        let _ = ctx.check_cost_eval_budget(self.budget_checks)?;

        let t = self.state.updates_seen;
        self.state.updates_seen += 1;

        if self.alert_period == 0 {
            return Err(CpdError::invalid_input(
                "mock detector requires alert_period >= 1",
            ));
        }

        if x_t.is_empty() {
            return Err(CpdError::invalid_input(
                "mock detector expects at least one feature value",
            ));
        }

        let value = x_t[0].abs();
        let p_change = value / (1.0 + value);
        let alert = self.state.updates_seen.is_multiple_of(self.alert_period);

        Ok(OnlineStepResult {
            t,
            p_change,
            alert,
            alert_reason: alert.then(|| "mock-periodic-threshold".to_string()),
            run_length_mode: t,
            run_length_mean: t as f64,
            processing_latency_us: None,
        })
    }

    fn save_state(&self) -> Self::State {
        self.state.clone()
    }

    fn load_state(&mut self, state: &Self::State) {
        self.state = state.clone();
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct HarnessMetrics {
    pub elapsed_ms: u128,
    pub updates_per_sec: f64,
    pub cancellation_latency_ms: Option<u128>,
    pub checkpoint_roundtrip_count: usize,
    pub max_rss_kib: Option<u64>,
    pub rss_slope_kib_per_hr: Option<f64>,
    pub alert_flip_count: usize,
}

pub fn run_soak(
    detector: &mut MockOnlineDetector,
    config: &HarnessConfig,
    ctx: &ExecutionContext<'_>,
) -> Result<HarnessMetrics, CpdError> {
    if config.steps == 0 {
        return Err(CpdError::invalid_input(
            "soak harness requires config.steps >= 1",
        ));
    }

    let started_at = Instant::now();
    let mut checkpoint_roundtrip_count = 0usize;
    let mut alert_flip_count = 0usize;
    let mut last_alert: Option<bool> = None;
    let mut rss_samples: Vec<(f64, u64)> = vec![];

    for step in 0..config.steps {
        let result = detector.update(&[step as f64], None, ctx)?;

        if let Some(previous) = last_alert
            && previous != result.alert
        {
            alert_flip_count += 1;
        }
        last_alert = Some(result.alert);

        if config.checkpoint_every > 0 && (step + 1).is_multiple_of(config.checkpoint_every) {
            let saved = detector.save_state();
            detector.load_state(&saved);
            checkpoint_roundtrip_count += 1;

            if let Some(rss_kib) = current_rss_kib() {
                let elapsed_hours = started_at.elapsed().as_secs_f64() / 3_600.0;
                rss_samples.push((elapsed_hours, rss_kib));
            }
        }

        if config.sleep_per_step_ms > 0 {
            std::thread::sleep(Duration::from_millis(config.sleep_per_step_ms));
        }
    }

    let elapsed = started_at.elapsed();
    let updates_per_sec = config.steps as f64 / elapsed.as_secs_f64().max(1e-9);
    let max_rss_kib = rss_samples.iter().map(|(_, rss)| *rss).max();
    let rss_slope_kib_per_hr = estimate_rss_slope_kib_per_hr(&rss_samples);

    Ok(HarnessMetrics {
        elapsed_ms: elapsed.as_millis(),
        updates_per_sec,
        cancellation_latency_ms: None,
        checkpoint_roundtrip_count,
        max_rss_kib,
        rss_slope_kib_per_hr,
        alert_flip_count,
    })
}

pub fn estimate_rss_slope_kib_per_hr(samples: &[(f64, u64)]) -> Option<f64> {
    if samples.len() < 2 {
        return None;
    }

    let (start_t, start_rss) = samples.first().copied()?;
    let (end_t, end_rss) = samples.last().copied()?;
    let delta_t = end_t - start_t;
    if delta_t <= 0.0 {
        return None;
    }

    Some((end_rss as f64 - start_rss as f64) / delta_t)
}

pub fn emit_metrics_json_if_requested(
    scenario: &str,
    profile: SoakProfile,
    metrics: &HarnessMetrics,
) -> Result<(), CpdError> {
    let Some(path) = std::env::var("CPD_SOAK_METRICS_OUT").ok() else {
        return Ok(());
    };

    let max_rss_kib = metrics
        .max_rss_kib
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string());
    let rss_slope_kib_per_hr = metrics
        .rss_slope_kib_per_hr
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string());
    let cancellation_latency_ms = metrics
        .cancellation_latency_ms
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string());

    let payload = format!(
        "{{\n  \"scenario\": \"{scenario}\",\n  \"profile\": \"{}\",\n  \"target_runtime_seconds\": {},\n  \"updates_per_sec\": {},\n  \"cancellation_latency_ms\": {},\n  \"checkpoint_roundtrip_count\": {},\n  \"max_rss_kib\": {},\n  \"rss_slope_kib_per_hr\": {},\n  \"alert_flip_count\": {}\n}}\n",
        match profile {
            SoakProfile::PrSmoke => "pr_smoke",
            SoakProfile::Nightly1h => "nightly_1h",
            SoakProfile::Weekly24h => "weekly_24h",
        },
        profile.target_runtime_seconds(),
        metrics.updates_per_sec,
        cancellation_latency_ms,
        metrics.checkpoint_roundtrip_count,
        max_rss_kib,
        rss_slope_kib_per_hr,
        metrics.alert_flip_count,
    );

    std::fs::write(path, payload).map_err(|err| {
        CpdError::resource_limit(format!("failed writing soak metrics artifact: {err}"))
    })
}

#[cfg(target_os = "linux")]
fn current_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let value = rest
                .split_whitespace()
                .next()
                .and_then(|raw| raw.parse::<u64>().ok());
            if value.is_some() {
                return value;
            }
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn current_rss_kib() -> Option<u64> {
    None
}
