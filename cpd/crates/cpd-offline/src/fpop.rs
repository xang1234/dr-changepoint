// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::{Pelt, PeltConfig};
use cpd_core::{
    CpdError, ExecutionContext, OfflineChangePointResult, OfflineDetector, Penalty, Stopping,
    TimeSeriesView, validate_stopping,
};
use cpd_costs::CostModel;
use std::borrow::Cow;

const DEFAULT_CANCEL_CHECK_EVERY: usize = 1000;
const DEFAULT_PARAMS_PER_SEGMENT: usize = 2;

/// Configuration for [`Fpop`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct FpopConfig {
    pub stopping: Stopping,
    pub params_per_segment: usize,
    pub cancel_check_every: usize,
}

impl Default for FpopConfig {
    fn default() -> Self {
        Self {
            stopping: Stopping::Penalized(Penalty::BIC),
            params_per_segment: DEFAULT_PARAMS_PER_SEGMENT,
            cancel_check_every: DEFAULT_CANCEL_CHECK_EVERY,
        }
    }
}

impl FpopConfig {
    fn validate(&self) -> Result<(), CpdError> {
        validate_stopping(&self.stopping)?;

        if self.params_per_segment == 0 {
            return Err(CpdError::invalid_input(
                "FpopConfig.params_per_segment must be >= 1; got 0",
            ));
        }

        Ok(())
    }
}

/// Functional Pruning Optimal Partitioning detector for L2 mean costs.
///
/// This detector is currently restricted to cost models whose `name()` is
/// `"l2_mean"` and reuses the exact optimal-partitioning kernel used by PELT.
#[derive(Debug)]
pub struct Fpop<C: CostModel + Clone> {
    cost_model: C,
    config: FpopConfig,
}

impl<C: CostModel + Clone> Fpop<C> {
    pub fn new(cost_model: C, config: FpopConfig) -> Result<Self, CpdError> {
        config.validate()?;
        Ok(Self { cost_model, config })
    }

    pub fn cost_model(&self) -> &C {
        &self.cost_model
    }

    pub fn config(&self) -> &FpopConfig {
        &self.config
    }
}

impl<C: CostModel + Clone> OfflineDetector for Fpop<C> {
    fn detect(
        &self,
        x: &TimeSeriesView<'_>,
        ctx: &ExecutionContext<'_>,
    ) -> Result<OfflineChangePointResult, CpdError> {
        self.config.validate()?;

        if self.cost_model.name() != "l2_mean" {
            return Err(CpdError::not_supported(format!(
                "FPOP currently supports only CostL2Mean; got cost_model='{}'",
                self.cost_model.name()
            )));
        }

        let pelt = Pelt::new(
            self.cost_model.clone(),
            PeltConfig {
                stopping: self.config.stopping.clone(),
                params_per_segment: self.config.params_per_segment,
                cancel_check_every: self.config.cancel_check_every,
            },
        )?;

        let mut result = pelt.detect(x, ctx)?;
        result.diagnostics.algorithm = Cow::Borrowed("fpop");
        result.diagnostics.warnings = result
            .diagnostics
            .warnings
            .into_iter()
            .map(|warning| warning.replace("PELT", "FPOP"))
            .collect();
        result
            .diagnostics
            .notes
            .push("kernel=exact_optimal_partitioning (L2)".to_string());

        if let Some(runtime_ms) = result.diagnostics.runtime_ms {
            ctx.record_scalar("offline.fpop.runtime_ms", runtime_ms as f64);
        }
        if let Some(pruning_stats) = &result.diagnostics.pruning_stats {
            ctx.record_scalar(
                "offline.fpop.candidates_considered",
                pruning_stats.candidates_considered as f64,
            );
            ctx.record_scalar(
                "offline.fpop.candidates_pruned",
                pruning_stats.candidates_pruned as f64,
            );
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::{Fpop, FpopConfig};
    use crate::{Pelt, PeltConfig};
    use cpd_core::{
        Constraints, DTypeView, ExecutionContext, MemoryLayout, MissingPolicy, OfflineDetector,
        Penalty, Stopping, TimeIndex, TimeSeriesView,
    };
    use cpd_costs::{CostL2Mean, CostNormalMeanVar};

    fn make_f64_view<'a>(
        values: &'a [f64],
        n: usize,
        d: usize,
        layout: MemoryLayout,
        missing: MissingPolicy,
    ) -> TimeSeriesView<'a> {
        TimeSeriesView::new(
            DTypeView::F64(values),
            n,
            d,
            layout,
            None,
            TimeIndex::None,
            missing,
        )
        .expect("test view should be valid")
    }

    fn constraints_with_min_segment_len(min_segment_len: usize) -> Constraints {
        Constraints {
            min_segment_len,
            ..Constraints::default()
        }
    }

    #[test]
    fn config_defaults_and_validation() {
        let default_cfg = FpopConfig::default();
        assert_eq!(default_cfg.stopping, Stopping::Penalized(Penalty::BIC));
        assert_eq!(default_cfg.params_per_segment, 2);
        assert_eq!(default_cfg.cancel_check_every, 1000);

        let ok =
            Fpop::new(CostL2Mean::default(), default_cfg.clone()).expect("default should validate");
        assert_eq!(ok.config(), &default_cfg);

        let err = Fpop::new(
            CostL2Mean::default(),
            FpopConfig {
                stopping: Stopping::Penalized(Penalty::Manual(1.0)),
                params_per_segment: 0,
                cancel_check_every: 1000,
            },
        )
        .expect_err("params_per_segment=0 must fail");
        assert!(err.to_string().contains("params_per_segment"));
    }

    #[test]
    fn rejects_non_l2_cost_models() {
        let detector = Fpop::new(
            CostNormalMeanVar::default(),
            FpopConfig {
                stopping: Stopping::Penalized(Penalty::Manual(1.0)),
                params_per_segment: 3,
                cancel_check_every: 1000,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 0.0, 10.0, 10.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = constraints_with_min_segment_len(1);
        let ctx = ExecutionContext::new(&constraints);
        let err = detector
            .detect(&view, &ctx)
            .expect_err("non-L2 model must be rejected");
        assert!(matches!(err, cpd_core::CpdError::NotSupported(_)));
    }

    #[test]
    fn penalized_matches_pelt_on_piecewise_constant_signal() {
        let values = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, -5.0, -5.0, -5.0,
            -5.0, -5.0, -5.0,
        ];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = constraints_with_min_segment_len(2);
        let ctx = ExecutionContext::new(&constraints);

        let pelt = Pelt::new(
            CostL2Mean::default(),
            PeltConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.5)),
                params_per_segment: 2,
                cancel_check_every: 64,
            },
        )
        .expect("pelt config should be valid");
        let pelt_result = pelt.detect(&view, &ctx).expect("pelt should detect");

        let fpop = Fpop::new(
            CostL2Mean::default(),
            FpopConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.5)),
                params_per_segment: 2,
                cancel_check_every: 64,
            },
        )
        .expect("fpop config should be valid");
        let fpop_result = fpop.detect(&view, &ctx).expect("fpop should detect");

        assert_eq!(fpop_result.breakpoints, pelt_result.breakpoints);
        assert_eq!(fpop_result.change_points, pelt_result.change_points);
        assert_eq!(fpop_result.diagnostics.algorithm, "fpop");
    }

    #[test]
    fn known_k_matches_pelt() {
        let values = vec![
            0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0, -4.0, -4.0, -4.0, -4.0,
        ];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = constraints_with_min_segment_len(2);
        let ctx = ExecutionContext::new(&constraints);

        let pelt = Pelt::new(
            CostL2Mean::default(),
            PeltConfig {
                stopping: Stopping::KnownK(2),
                params_per_segment: 2,
                cancel_check_every: 32,
            },
        )
        .expect("pelt config should be valid");
        let pelt_result = pelt.detect(&view, &ctx).expect("pelt should detect");

        let fpop = Fpop::new(
            CostL2Mean::default(),
            FpopConfig {
                stopping: Stopping::KnownK(2),
                params_per_segment: 2,
                cancel_check_every: 32,
            },
        )
        .expect("fpop config should be valid");
        let fpop_result = fpop.detect(&view, &ctx).expect("fpop should detect");

        assert_eq!(fpop_result.breakpoints, pelt_result.breakpoints);
        assert_eq!(fpop_result.change_points, pelt_result.change_points);
    }

    #[test]
    fn penalty_path_matches_pelt_primary_solution() {
        let values = vec![
            0.0, 0.0, 0.0, 0.0, 8.0, 8.0, 8.0, 8.0, -2.0, -2.0, -2.0, -2.0,
        ];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = constraints_with_min_segment_len(2);
        let ctx = ExecutionContext::new(&constraints);

        let stopping = Stopping::PenaltyPath(vec![Penalty::Manual(0.2), Penalty::Manual(50.0)]);

        let pelt = Pelt::new(
            CostL2Mean::default(),
            PeltConfig {
                stopping: stopping.clone(),
                params_per_segment: 2,
                cancel_check_every: 32,
            },
        )
        .expect("pelt config should be valid");
        let pelt_result = pelt.detect(&view, &ctx).expect("pelt should detect");

        let fpop = Fpop::new(
            CostL2Mean::default(),
            FpopConfig {
                stopping,
                params_per_segment: 2,
                cancel_check_every: 32,
            },
        )
        .expect("fpop config should be valid");
        let fpop_result = fpop.detect(&view, &ctx).expect("fpop should detect");

        assert_eq!(fpop_result.breakpoints, pelt_result.breakpoints);
    }
}
