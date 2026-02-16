// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{CpdError, OfflineChangePointResult, TimeSeriesView};
use cpd_doctor::{PipelineSpec, execute_pipeline};

/// Executes a [`PipelineSpec`] against an input view.
pub fn run_pipeline(
    x: &TimeSeriesView<'_>,
    pipeline: &PipelineSpec,
) -> Result<OfflineChangePointResult, CpdError> {
    execute_pipeline(x, pipeline)
}

/// Parses a JSON pipeline spec and executes it against an input view.
#[cfg(feature = "serde")]
pub fn run_pipeline_json(
    x: &TimeSeriesView<'_>,
    pipeline_json: &str,
) -> Result<OfflineChangePointResult, CpdError> {
    let pipeline: PipelineSpec = serde_json::from_str(pipeline_json)
        .map_err(|err| CpdError::invalid_input(format!("invalid pipeline JSON: {err}")))?;
    run_pipeline(x, &pipeline)
}

/// CLI namespace placeholder.
pub fn crate_name() -> &'static str {
    let _ = (
        cpd_core::crate_name(),
        cpd_doctor::crate_name(),
        cpd_offline::crate_name(),
    );
    "cpd-cli"
}

#[cfg(test)]
mod tests {
    use super::run_pipeline;
    use cpd_core::{Constraints, MemoryLayout, MissingPolicy, Stopping, TimeIndex, TimeSeriesView};
    use cpd_doctor::{CostConfig, DetectorConfig, OfflineDetectorConfig, PipelineSpec};

    fn univariate(values: &[f64]) -> TimeSeriesView<'_> {
        TimeSeriesView::from_f64(
            values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect("test view should be valid")
    }

    #[test]
    fn run_pipeline_executes_valid_spec() {
        let values = vec![0.0, 0.0, 0.0, 7.0, 7.0, 7.0, -2.0, -2.0, -2.0];
        let view = univariate(&values);
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

        let result = run_pipeline(&view, &pipeline).expect("pipeline should execute");
        assert_eq!(result.breakpoints, vec![3, 6, 9]);
    }
}
