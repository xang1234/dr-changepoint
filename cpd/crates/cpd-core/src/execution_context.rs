// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::CpdError;
use crate::constraints::Constraints;
use crate::control::{BudgetMode, BudgetStatus, CancelToken};
use crate::observability::{ProgressSink, TelemetrySink};
use crate::repro::ReproMode;
use std::time::Instant;

/// Unified execution context passed through detector calls.
pub struct ExecutionContext<'a> {
    pub constraints: &'a Constraints,
    pub cancel: Option<&'a CancelToken>,
    pub budget_mode: BudgetMode,
    pub repro_mode: ReproMode,
    pub progress: Option<&'a dyn ProgressSink>,
    pub telemetry: Option<&'a dyn TelemetrySink>,
}

impl<'a> ExecutionContext<'a> {
    /// Creates a context with safe defaults and no optional hooks.
    pub fn new(constraints: &'a Constraints) -> Self {
        Self {
            constraints,
            cancel: None,
            budget_mode: BudgetMode::HardFail,
            repro_mode: ReproMode::Balanced,
            progress: None,
            telemetry: None,
        }
    }

    /// Sets the optional cancellation token.
    pub fn with_cancel(mut self, cancel: &'a CancelToken) -> Self {
        self.cancel = Some(cancel);
        self
    }

    /// Sets the budget mode.
    pub fn with_budget_mode(mut self, budget_mode: BudgetMode) -> Self {
        self.budget_mode = budget_mode;
        self
    }

    /// Sets the reproducibility mode.
    pub fn with_repro_mode(mut self, repro_mode: ReproMode) -> Self {
        self.repro_mode = repro_mode;
        self
    }

    /// Sets an optional progress sink.
    pub fn with_progress_sink(mut self, progress: &'a dyn ProgressSink) -> Self {
        self.progress = Some(progress);
        self
    }

    /// Sets an optional telemetry sink.
    pub fn with_telemetry_sink(mut self, telemetry: &'a dyn TelemetrySink) -> Self {
        self.telemetry = Some(telemetry);
        self
    }

    /// Returns true when cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_some_and(CancelToken::is_cancelled)
    }

    /// Returns a cancelled error when cancellation has been requested.
    pub fn check_cancelled(&self) -> Result<(), CpdError> {
        if self.is_cancelled() {
            return Err(CpdError::cancelled());
        }
        Ok(())
    }

    /// Checks cancellation every `every` iterations.
    ///
    /// When `every` is zero, it is treated as one (always poll).
    pub fn check_cancelled_every(&self, iteration: usize, every: usize) -> Result<(), CpdError> {
        let every = every.max(1);
        if iteration % every != 0 {
            return Ok(());
        }
        self.check_cancelled()
    }

    /// Backwards-compatible wrapper for cost-evaluation budget checks.
    pub fn check_budget(&self, cost_evals: usize) -> Result<BudgetStatus, CpdError> {
        self.check_cost_eval_budget(cost_evals)
    }

    /// Checks cost-evaluation budget and reports status based on configured mode.
    pub fn check_cost_eval_budget(&self, cost_evals: usize) -> Result<BudgetStatus, CpdError> {
        let Some(limit) = self.constraints.max_cost_evals else {
            return Ok(BudgetStatus::WithinBudget);
        };

        if cost_evals <= limit {
            return Ok(BudgetStatus::WithinBudget);
        }

        match self.budget_mode {
            BudgetMode::HardFail => Err(CpdError::resource_limit(format!(
                "constraints.max_cost_evals exceeded: used={cost_evals}, limit={limit}, budget_mode=HardFail"
            ))),
            BudgetMode::SoftDegrade => Ok(BudgetStatus::ExceededSoftDegrade),
        }
    }

    /// Checks elapsed time budget and reports status based on configured mode.
    pub fn check_time_budget(&self, started_at: Instant) -> Result<BudgetStatus, CpdError> {
        let Some(limit_ms) = self.constraints.time_budget_ms else {
            return Ok(BudgetStatus::WithinBudget);
        };

        let elapsed_ms = started_at.elapsed().as_millis();
        if elapsed_ms <= u128::from(limit_ms) {
            return Ok(BudgetStatus::WithinBudget);
        }

        match self.budget_mode {
            BudgetMode::HardFail => Err(CpdError::resource_limit(format!(
                "constraints.time_budget_ms exceeded: elapsed_ms={elapsed_ms}, limit_ms={limit_ms}, budget_mode=HardFail"
            ))),
            BudgetMode::SoftDegrade => Ok(BudgetStatus::ExceededSoftDegrade),
        }
    }

    /// Emits clamped progress to the sink, if configured.
    pub fn report_progress(&self, fraction: f32) {
        if !fraction.is_finite() {
            return;
        }

        if let Some(sink) = self.progress {
            sink.on_progress(fraction.clamp(0.0, 1.0));
        }
    }

    /// Emits a scalar telemetry value to the sink, if configured.
    pub fn record_scalar(&self, key: &'static str, value: f64) {
        if let Some(sink) = self.telemetry {
            sink.record_scalar(key, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ExecutionContext;
    use crate::constraints::Constraints;
    use crate::control::{BudgetMode, BudgetStatus, CancelToken};
    use crate::observability::{ProgressSink, TelemetrySink};
    use crate::repro::ReproMode;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    #[derive(Default)]
    struct MockProgressSink {
        values: Mutex<Vec<f32>>,
    }

    impl ProgressSink for MockProgressSink {
        fn on_progress(&self, fraction: f32) {
            self.values
                .lock()
                .expect("progress mutex should lock")
                .push(fraction);
        }
    }

    #[derive(Default)]
    struct MockTelemetrySink {
        values: Mutex<Vec<(&'static str, f64)>>,
    }

    impl TelemetrySink for MockTelemetrySink {
        fn record_scalar(&self, key: &'static str, value: f64) {
            self.values
                .lock()
                .expect("telemetry mutex should lock")
                .push((key, value));
        }
    }

    #[test]
    fn execution_context_new_sets_expected_defaults() {
        let constraints = Constraints::default();
        let ctx = ExecutionContext::new(&constraints);

        assert!(std::ptr::eq(ctx.constraints, &constraints));
        assert!(ctx.cancel.is_none());
        assert_eq!(ctx.budget_mode, BudgetMode::HardFail);
        assert_eq!(ctx.repro_mode, ReproMode::Balanced);
        assert!(ctx.progress.is_none());
        assert!(ctx.telemetry.is_none());
    }

    #[test]
    fn builder_methods_set_requested_fields() {
        let constraints = Constraints::default();
        let cancel = CancelToken::new();
        let progress = MockProgressSink::default();
        let telemetry = MockTelemetrySink::default();

        let ctx = ExecutionContext::new(&constraints)
            .with_cancel(&cancel)
            .with_budget_mode(BudgetMode::SoftDegrade)
            .with_repro_mode(ReproMode::Fast)
            .with_progress_sink(&progress)
            .with_telemetry_sink(&telemetry);

        assert!(ctx.cancel.is_some_and(|token| std::ptr::eq(token, &cancel)));
        assert_eq!(ctx.budget_mode, BudgetMode::SoftDegrade);
        assert_eq!(ctx.repro_mode, ReproMode::Fast);
        assert!(ctx.progress.is_some());
        assert!(ctx.telemetry.is_some());
    }

    #[test]
    fn is_cancelled_returns_false_without_token_and_true_when_cancelled() {
        let constraints = Constraints::default();
        let no_cancel_ctx = ExecutionContext::new(&constraints);
        assert!(!no_cancel_ctx.is_cancelled());

        let cancel = CancelToken::new();
        let with_cancel_ctx = ExecutionContext::new(&constraints).with_cancel(&cancel);
        assert!(!with_cancel_ctx.is_cancelled());
        cancel.cancel();
        assert!(with_cancel_ctx.is_cancelled());
    }

    #[test]
    fn check_cancelled_returns_cancelled_error_when_requested() {
        let constraints = Constraints::default();
        let cancel = CancelToken::new();
        let ctx = ExecutionContext::new(&constraints).with_cancel(&cancel);

        assert!(ctx.check_cancelled().is_ok());
        cancel.cancel();

        let err = ctx
            .check_cancelled()
            .expect_err("cancelled token should return an error");
        assert_eq!(err.to_string(), "cancelled");
    }

    #[test]
    fn check_cancelled_every_polls_on_cadence_and_stops_after_cancel() {
        let constraints = Constraints::default();
        let cancel = CancelToken::new();
        let ctx = ExecutionContext::new(&constraints).with_cancel(&cancel);

        for iteration in 0..10 {
            if iteration == 5 {
                cancel.cancel();
            }

            if iteration < 6 {
                assert!(
                    ctx.check_cancelled_every(iteration, 2).is_ok(),
                    "iteration {iteration} should not yet fail"
                );
            } else if iteration == 6 {
                let err = ctx
                    .check_cancelled_every(iteration, 2)
                    .expect_err("cancelled state should be observed on cadence");
                assert_eq!(err.to_string(), "cancelled");
                break;
            }
        }
    }

    #[test]
    fn check_cancelled_every_zero_interval_is_treated_as_always_poll() {
        let constraints = Constraints::default();
        let cancel = CancelToken::new();
        let ctx = ExecutionContext::new(&constraints).with_cancel(&cancel);

        cancel.cancel();
        let err = ctx
            .check_cancelled_every(3, 0)
            .expect_err("every=0 should behave like every=1");
        assert_eq!(err.to_string(), "cancelled");
    }

    #[test]
    fn check_cost_eval_budget_with_no_limit_or_within_limit_is_within_budget() {
        let no_limit = Constraints::default();
        let no_limit_ctx = ExecutionContext::new(&no_limit);
        assert_eq!(
            no_limit_ctx
                .check_cost_eval_budget(1)
                .expect("no limit must pass"),
            BudgetStatus::WithinBudget
        );

        let with_limit = Constraints {
            max_cost_evals: Some(10),
            ..Constraints::default()
        };
        let with_limit_ctx = ExecutionContext::new(&with_limit);
        assert_eq!(
            with_limit_ctx
                .check_cost_eval_budget(10)
                .expect("at limit should pass"),
            BudgetStatus::WithinBudget
        );
    }

    #[test]
    fn check_cost_eval_budget_over_limit_hard_fail_returns_resource_limit_error() {
        let constraints = Constraints {
            max_cost_evals: Some(10),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints).with_budget_mode(BudgetMode::HardFail);

        let err = ctx
            .check_cost_eval_budget(11)
            .expect_err("hard fail should error on budget exceed");
        assert_eq!(
            err.to_string(),
            "resource limit exceeded: constraints.max_cost_evals exceeded: used=11, limit=10, budget_mode=HardFail"
        );
    }

    #[test]
    fn check_cost_eval_budget_over_limit_soft_degrade_returns_exceeded_status() {
        let constraints = Constraints {
            max_cost_evals: Some(10),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints).with_budget_mode(BudgetMode::SoftDegrade);
        assert_eq!(
            ctx.check_cost_eval_budget(11)
                .expect("soft mode should not error"),
            BudgetStatus::ExceededSoftDegrade
        );
    }

    #[test]
    fn check_budget_wrapper_delegates_to_cost_eval_budget() {
        let constraints = Constraints {
            max_cost_evals: Some(10),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints).with_budget_mode(BudgetMode::SoftDegrade);
        assert_eq!(
            ctx.check_budget(11).expect("wrapper should preserve behavior"),
            BudgetStatus::ExceededSoftDegrade
        );
    }

    #[test]
    fn check_time_budget_with_no_limit_is_within_budget() {
        let constraints = Constraints::default();
        let ctx = ExecutionContext::new(&constraints);
        assert_eq!(
            ctx.check_time_budget(Instant::now())
                .expect("no limit must pass"),
            BudgetStatus::WithinBudget
        );
    }

    #[test]
    fn check_time_budget_over_limit_hard_fail_returns_resource_limit_error() {
        let constraints = Constraints {
            time_budget_ms: Some(1),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints).with_budget_mode(BudgetMode::HardFail);
        let started_at = Instant::now()
            .checked_sub(Duration::from_millis(20))
            .expect("checked_sub should produce a valid earlier instant");

        let err = ctx
            .check_time_budget(started_at)
            .expect_err("hard fail should error on time budget exceed");
        let msg = err.to_string();
        assert!(msg.contains("constraints.time_budget_ms exceeded"));
        assert!(msg.contains("limit_ms=1"));
        assert!(msg.contains("budget_mode=HardFail"));
    }

    #[test]
    fn check_time_budget_over_limit_soft_degrade_returns_exceeded_status() {
        let constraints = Constraints {
            time_budget_ms: Some(1),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints).with_budget_mode(BudgetMode::SoftDegrade);
        let started_at = Instant::now()
            .checked_sub(Duration::from_millis(20))
            .expect("checked_sub should produce a valid earlier instant");

        assert_eq!(
            ctx.check_time_budget(started_at)
                .expect("soft mode should not error"),
            BudgetStatus::ExceededSoftDegrade
        );
    }

    #[test]
    fn report_progress_is_noop_without_sink() {
        let constraints = Constraints::default();
        let ctx = ExecutionContext::new(&constraints);
        ctx.report_progress(0.5);
        ctx.report_progress(-1.0);
        ctx.report_progress(f32::NAN);
    }

    #[test]
    fn report_progress_clamps_and_ignores_non_finite_values() {
        let constraints = Constraints::default();
        let progress = MockProgressSink::default();
        let ctx = ExecutionContext::new(&constraints).with_progress_sink(&progress);

        ctx.report_progress(-0.2);
        ctx.report_progress(0.25);
        ctx.report_progress(1.2);
        ctx.report_progress(f32::NAN);
        ctx.report_progress(f32::INFINITY);
        ctx.report_progress(f32::NEG_INFINITY);

        let got = progress
            .values
            .lock()
            .expect("progress values should lock")
            .clone();
        assert_eq!(got, vec![0.0, 0.25, 1.0]);
    }

    #[test]
    fn record_scalar_is_noop_without_sink() {
        let constraints = Constraints::default();
        let ctx = ExecutionContext::new(&constraints);
        ctx.record_scalar("loss", 1.0);
    }

    #[test]
    fn record_scalar_writes_to_telemetry_sink_when_present() {
        let constraints = Constraints::default();
        let telemetry = MockTelemetrySink::default();
        let ctx = ExecutionContext::new(&constraints).with_telemetry_sink(&telemetry);

        ctx.record_scalar("loss", 1.5);
        ctx.record_scalar("gain", 2.25);

        let got = telemetry
            .values
            .lock()
            .expect("telemetry values should lock")
            .clone();
        assert_eq!(got, vec![("loss", 1.5), ("gain", 2.25)]);
    }
}
