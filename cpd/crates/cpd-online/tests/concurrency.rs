// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

#[path = "support/soak_harness.rs"]
mod soak_harness;

use cpd_core::{BudgetMode, CancelToken, Constraints, ExecutionContext, OnlineDetector};
use soak_harness::{HarnessConfig, MockOnlineDetector, run_soak};
use std::thread;
use std::time::{Duration, Instant};

#[test]
fn multi_instance_threaded_runs_keep_state_isolated() {
    const THREADS: usize = 4;
    const STEPS: usize = 600;
    const CHECKPOINT_EVERY: usize = 60;

    let mut workers = Vec::with_capacity(THREADS);
    for _ in 0..THREADS {
        workers.push(thread::spawn(move || {
            let constraints = Constraints::default();
            let ctx = ExecutionContext::new(&constraints);
            let mut detector = MockOnlineDetector::new(40);
            let metrics = run_soak(
                &mut detector,
                &HarnessConfig {
                    steps: STEPS,
                    checkpoint_every: CHECKPOINT_EVERY,
                    sleep_per_step_ms: 0,
                },
                &ctx,
            )
            .expect("threaded soak run should succeed");

            (detector.save_state().updates_seen, metrics)
        }));
    }

    for worker in workers {
        let (updates_seen, metrics) = worker.join().expect("thread should join cleanly");
        assert_eq!(updates_seen, STEPS);
        assert_eq!(metrics.checkpoint_roundtrip_count, STEPS / CHECKPOINT_EVERY);
        assert!(metrics.updates_per_sec > 0.0);
    }
}

#[test]
fn concurrent_cancellation_stops_in_flight_updates() {
    let cancel = CancelToken::new();
    let worker_cancel = cancel.clone();
    let worker = thread::spawn(move || {
        let constraints = Constraints::default();
        let ctx = ExecutionContext::new(&constraints).with_cancel(&worker_cancel);
        let mut detector = MockOnlineDetector::new(16);

        let mut updates = 0usize;
        loop {
            match detector.update(&[updates as f64], None, &ctx) {
                Ok(_) => {
                    updates += 1;
                    thread::sleep(Duration::from_millis(1));
                }
                Err(err) => return (updates, err.to_string()),
            }
        }
    });

    thread::sleep(Duration::from_millis(30));
    let cancelled_at = Instant::now();
    cancel.cancel();

    let (updates_before_cancel, err_msg) = worker.join().expect("worker should join");
    let cancellation_latency_ms = cancelled_at.elapsed().as_millis();

    assert!(updates_before_cancel > 0);
    assert_eq!(err_msg, "cancelled");
    assert!(
        cancellation_latency_ms <= 2_000,
        "expected prompt cancellation, got {cancellation_latency_ms} ms"
    );
}

#[test]
fn concurrent_budget_enforcement_is_deterministic() {
    const THREADS: usize = 4;
    const BUDGET_LIMIT: usize = 50;

    let mut workers = Vec::with_capacity(THREADS);
    for _ in 0..THREADS {
        workers.push(thread::spawn(move || {
            let constraints = Constraints {
                max_cost_evals: Some(BUDGET_LIMIT),
                ..Constraints::default()
            };
            let ctx = ExecutionContext::new(&constraints).with_budget_mode(BudgetMode::HardFail);
            let mut detector = MockOnlineDetector::new(32);

            let mut ok_updates = 0usize;
            loop {
                match detector.update(&[ok_updates as f64], None, &ctx) {
                    Ok(_) => ok_updates += 1,
                    Err(err) => return (ok_updates, err.to_string()),
                }
            }
        }));
    }

    for worker in workers {
        let (ok_updates, err_msg) = worker.join().expect("worker should join cleanly");
        assert_eq!(ok_updates, BUDGET_LIMIT);
        assert!(
            err_msg.contains("constraints.max_cost_evals exceeded"),
            "unexpected budget error: {err_msg}"
        );
    }
}
