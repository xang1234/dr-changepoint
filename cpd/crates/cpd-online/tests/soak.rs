// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

#[path = "support/soak_harness.rs"]
mod soak_harness;

use cpd_core::{Constraints, ExecutionContext, OnlineDetector};
use soak_harness::{
    HarnessConfig, MockOnlineDetector, SoakProfile, emit_metrics_json_if_requested,
    profile_from_env, run_soak,
};

#[test]
fn soak_checkpoint_restore_roundtrip_metrics_are_reported() {
    let constraints = Constraints::default();
    let ctx = ExecutionContext::new(&constraints);
    let mut detector = MockOnlineDetector::new(25);

    let metrics = run_soak(
        &mut detector,
        &HarnessConfig {
            steps: 1_000,
            checkpoint_every: 50,
            sleep_per_step_ms: 0,
        },
        &ctx,
    )
    .expect("soak run should succeed");

    assert_eq!(detector.save_state().updates_seen, 1_000);
    assert_eq!(metrics.checkpoint_roundtrip_count, 20);
    assert!(metrics.updates_per_sec > 0.0);

    emit_metrics_json_if_requested(
        "soak_checkpoint_restore_roundtrip_metrics_are_reported",
        SoakProfile::PrSmoke,
        &metrics,
    )
    .expect("metrics artifact emission should not fail when configured");
}

#[test]
fn soak_reports_alert_stability_markers() {
    let constraints = Constraints::default();
    let ctx = ExecutionContext::new(&constraints);
    let mut detector = MockOnlineDetector::new(40);

    let metrics = run_soak(
        &mut detector,
        &HarnessConfig {
            steps: 400,
            checkpoint_every: 20,
            sleep_per_step_ms: 0,
        },
        &ctx,
    )
    .expect("soak run should succeed");

    assert!(metrics.alert_flip_count > 0);
    assert!(metrics.alert_flip_count < 400);
}

#[test]
fn soak_rss_metrics_are_well_formed_when_available() {
    let constraints = Constraints::default();
    let ctx = ExecutionContext::new(&constraints);
    let mut detector = MockOnlineDetector::new(15);

    let metrics = run_soak(
        &mut detector,
        &HarnessConfig {
            steps: 600,
            checkpoint_every: 30,
            sleep_per_step_ms: 0,
        },
        &ctx,
    )
    .expect("soak run should succeed");

    if let Some(max_rss_kib) = metrics.max_rss_kib {
        assert!(max_rss_kib > 0);
    }
    if let Some(rss_slope_kib_per_hr) = metrics.rss_slope_kib_per_hr {
        assert!(rss_slope_kib_per_hr.is_finite());
    }

    emit_metrics_json_if_requested(
        "soak_rss_metrics_are_well_formed_when_available",
        profile_from_env(),
        &metrics,
    )
    .expect("metrics artifact emission should not fail when configured");
}

#[test]
fn soak_profile_runtime_contract_is_stable() {
    assert_eq!(SoakProfile::PrSmoke.target_runtime_seconds(), 120);
    assert_eq!(SoakProfile::Nightly1h.target_runtime_seconds(), 3_600);
    assert_eq!(SoakProfile::Weekly24h.target_runtime_seconds(), 86_400);

    let profile = profile_from_env();
    let config = HarnessConfig::for_profile(profile);
    assert!(config.steps > 0);
    assert!(config.checkpoint_every > 0);
}
