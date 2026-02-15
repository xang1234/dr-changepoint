// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{Constraints, CpdError, ExecutionContext, OnlineDetector};
use cpd_online::{
    BocpdConfig, BocpdDetector, ConstantHazard, HazardSpec, LateDataPolicy, ObservationModel,
};
use std::sync::OnceLock;
use std::time::Instant;

const MAX_RUN_LENGTH: usize = 2_000;
const LOG_PROB_THRESHOLD: f64 = -20.0;
const WARMUP_STEPS: usize = 2_500;
const MEASURE_STEPS: usize = 12_000;
const SLO_P99_UPDATE_US: u64 = 75;
const SLO_UPDATES_PER_SEC: f64 = 150_000.0;
const MIN_P95_RUN_LENGTH_MODE: usize = 64;
const SIGNAL_REGIME_LEN: usize = 96;

fn ctx() -> ExecutionContext<'static> {
    static CONSTRAINTS: OnceLock<Constraints> = OnceLock::new();
    let constraints = CONSTRAINTS.get_or_init(Constraints::default);
    ExecutionContext::new(constraints)
}

fn parse_env_bool(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn percentile(sorted_values: &[u64], q: f64) -> u64 {
    if sorted_values.is_empty() {
        return 0;
    }

    let q = q.clamp(0.0, 1.0);
    let max_index = sorted_values.len() - 1;
    let rank = (q * (max_index as f64)).ceil() as usize;
    sorted_values[rank.min(max_index)]
}

fn percentile_usize(sorted_values: &[usize], q: f64) -> usize {
    if sorted_values.is_empty() {
        return 0;
    }

    let q = q.clamp(0.0, 1.0);
    let max_index = sorted_values.len() - 1;
    let rank = (q * (max_index as f64)).ceil() as usize;
    sorted_values[rank.min(max_index)]
}

fn perf_signal(step: usize) -> f64 {
    let regime = (step / SIGNAL_REGIME_LEN) % 3;
    let baseline = match regime {
        0 => 0.0,
        1 => 3.5,
        _ => -1.8,
    };
    let wobble = ((step as f64) * 0.03).sin() * 0.2;
    baseline + wobble
}

fn make_signal(total_steps: usize) -> Vec<f64> {
    (0..total_steps).map(perf_signal).collect()
}

fn emit_metrics_if_requested(
    updates_per_sec: f64,
    p99_update_us: u64,
    p95_run_length_mode: usize,
    enforce: bool,
) -> Result<(), CpdError> {
    let Some(path) = std::env::var("CPD_ONLINE_PERF_METRICS_OUT").ok() else {
        return Ok(());
    };

    let payload = format!(
        "{{\n  \"scenario\": \"bocpd_gaussian_d1_max_run_length_2000\",\n  \"max_run_length\": {MAX_RUN_LENGTH},\n  \"warmup_steps\": {WARMUP_STEPS},\n  \"measure_steps\": {MEASURE_STEPS},\n  \"updates_per_sec\": {updates_per_sec},\n  \"p99_update_us\": {p99_update_us},\n  \"p95_run_length_mode\": {p95_run_length_mode},\n  \"min_p95_run_length_mode\": {MIN_P95_RUN_LENGTH_MODE},\n  \"slo_p99_update_us\": {SLO_P99_UPDATE_US},\n  \"slo_updates_per_sec\": {SLO_UPDATES_PER_SEC},\n  \"enforce\": {enforce}\n}}\n",
    );

    std::fs::write(path, payload).map_err(|err| {
        CpdError::resource_limit(format!("failed writing perf metrics artifact: {err}"))
    })
}

#[test]
fn bocpd_gaussian_perf_contract() {
    let enforce = parse_env_bool("CPD_ONLINE_PERF_ENFORCE");
    let signal = make_signal(WARMUP_STEPS + MEASURE_STEPS);
    let exec_ctx = ctx();
    let mut detector = BocpdDetector::new(BocpdConfig {
        hazard: HazardSpec::Constant(ConstantHazard::new(1.0 / 200.0).expect("valid hazard")),
        observation: ObservationModel::default(),
        max_run_length: MAX_RUN_LENGTH,
        log_prob_threshold: Some(LOG_PROB_THRESHOLD),
        alert_threshold: 0.5,
        late_data_policy: LateDataPolicy::Reject,
    })
    .expect("BOCPD config should be valid");

    for step in 0..WARMUP_STEPS {
        detector
            .update(&[signal[step]], None, &exec_ctx)
            .expect("warmup update should succeed");
    }

    let mut latencies_us: Vec<u64> = Vec::with_capacity(MEASURE_STEPS);
    let mut run_length_modes: Vec<usize> = Vec::with_capacity(MEASURE_STEPS);
    let started_at = Instant::now();
    for step in 0..MEASURE_STEPS {
        let result = detector
            .update(&[signal[WARMUP_STEPS + step]], None, &exec_ctx)
            .expect("measurement update should succeed");
        let latency_us = result
            .processing_latency_us
            .expect("BOCPD should report processing latency");
        latencies_us.push(latency_us);
        run_length_modes.push(result.run_length_mode);
    }
    let elapsed = started_at.elapsed();

    latencies_us.sort_unstable();
    run_length_modes.sort_unstable();
    let p99_update_us = percentile(&latencies_us, 0.99);
    let p95_run_length_mode = percentile_usize(&run_length_modes, 0.95);
    let updates_per_sec = (MEASURE_STEPS as f64) / elapsed.as_secs_f64().max(1e-9);

    println!(
        "BOCPD perf: steps={} elapsed_s={:.6} updates_per_sec={:.2} p99_update_us={} p95_run_length_mode={} max_run_length={} enforce={}",
        MEASURE_STEPS,
        elapsed.as_secs_f64(),
        updates_per_sec,
        p99_update_us,
        p95_run_length_mode,
        MAX_RUN_LENGTH,
        enforce
    );

    emit_metrics_if_requested(updates_per_sec, p99_update_us, p95_run_length_mode, enforce)
        .expect("metrics artifact emission should succeed");

    if enforce {
        assert!(
            p99_update_us <= SLO_P99_UPDATE_US,
            "p99 update latency SLO failed: observed={}us, threshold={}us",
            p99_update_us,
            SLO_P99_UPDATE_US
        );
        assert!(
            updates_per_sec >= SLO_UPDATES_PER_SEC,
            "throughput SLO failed: observed={updates_per_sec:.2} updates/sec, threshold={} updates/sec",
            SLO_UPDATES_PER_SEC
        );
        assert!(
            p95_run_length_mode >= MIN_P95_RUN_LENGTH_MODE,
            "run-length occupancy guard failed: observed p95_run_length_mode={}, minimum={}",
            p95_run_length_mode,
            MIN_P95_RUN_LENGTH_MODE
        );
    } else {
        assert!(updates_per_sec.is_finite() && updates_per_sec > 0.0);
        assert!(!latencies_us.is_empty());
    }
}
