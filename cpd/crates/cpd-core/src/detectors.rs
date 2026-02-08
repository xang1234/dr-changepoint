// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::results::OfflineChangePointResult;
use crate::{CpdError, ExecutionContext, TimeIndex, TimeSeriesView};

/// Per-step output emitted by online detectors.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct OnlineStepResult {
    pub t: usize,
    pub p_change: f64,
    pub alert: bool,
    pub alert_reason: Option<String>,
    pub run_length_mode: usize,
    pub run_length_mean: f64,
    pub processing_latency_us: Option<u64>,
}

/// Offline detector contract: full series in, full result out.
pub trait OfflineDetector {
    fn detect(
        &self,
        x: &TimeSeriesView<'_>,
        ctx: &ExecutionContext<'_>,
    ) -> Result<OfflineChangePointResult, CpdError>;
}

fn uniform_timestamp(t: usize, t0_ns: i64, dt_ns: i64) -> Result<i64, CpdError> {
    let t_i64 = i64::try_from(t).map_err(|_| {
        CpdError::invalid_input(format!(
            "time index overflow: t={t} does not fit into i64"
        ))
    })?;
    let delta = dt_ns.checked_mul(t_i64).ok_or_else(|| {
        CpdError::invalid_input(format!(
            "uniform timestamp overflow: dt_ns={dt_ns}, t={t}"
        ))
    })?;
    t0_ns.checked_add(delta).ok_or_else(|| {
        CpdError::invalid_input(format!(
            "uniform timestamp overflow: t0_ns={t0_ns}, delta={delta}"
        ))
    })
}

fn extract_step_f64(x: &TimeSeriesView<'_>, t: usize) -> Result<Vec<f64>, CpdError> {
    let mut step = Vec::with_capacity(x.d);

    match x.values {
        crate::DTypeView::F32(values) => match x.layout {
            crate::MemoryLayout::CContiguous => {
                let row_start = t
                    .checked_mul(x.d)
                    .ok_or_else(|| CpdError::invalid_input("row index overflow"))?;
                for j in 0..x.d {
                    step.push(f64::from(values[row_start + j]));
                }
            }
            crate::MemoryLayout::FContiguous => {
                for j in 0..x.d {
                    let idx = j
                        .checked_mul(x.n)
                        .and_then(|base| base.checked_add(t))
                        .ok_or_else(|| CpdError::invalid_input("column-major index overflow"))?;
                    step.push(f64::from(values[idx]));
                }
            }
            crate::MemoryLayout::Strided {
                row_stride,
                col_stride,
            } => {
                for j in 0..x.d {
                    let t_isize = isize::try_from(t).map_err(|_| {
                        CpdError::invalid_input(format!(
                            "time index {t} does not fit into isize for strided access"
                        ))
                    })?;
                    let j_isize = isize::try_from(j).map_err(|_| {
                        CpdError::invalid_input(format!(
                            "dimension index {j} does not fit into isize for strided access"
                        ))
                    })?;
                    let idx = t_isize
                        .checked_mul(row_stride)
                        .and_then(|left| j_isize.checked_mul(col_stride).and_then(|right| left.checked_add(right)))
                        .ok_or_else(|| {
                            CpdError::invalid_input(format!(
                                "strided index overflow at t={t}, j={j}, row_stride={row_stride}, col_stride={col_stride}"
                            ))
                        })?;
                    let idx_usize = usize::try_from(idx).map_err(|_| {
                        CpdError::invalid_input(format!(
                            "strided index negative at t={t}, j={j}: idx={idx}"
                        ))
                    })?;
                    let value = values.get(idx_usize).ok_or_else(|| {
                        CpdError::invalid_input(format!(
                            "strided index out of bounds at t={t}, j={j}: idx={idx_usize}, len={}",
                            values.len()
                        ))
                    })?;
                    step.push(f64::from(*value));
                }
            }
        },
        crate::DTypeView::F64(values) => match x.layout {
            crate::MemoryLayout::CContiguous => {
                let row_start = t
                    .checked_mul(x.d)
                    .ok_or_else(|| CpdError::invalid_input("row index overflow"))?;
                for j in 0..x.d {
                    step.push(values[row_start + j]);
                }
            }
            crate::MemoryLayout::FContiguous => {
                for j in 0..x.d {
                    let idx = j
                        .checked_mul(x.n)
                        .and_then(|base| base.checked_add(t))
                        .ok_or_else(|| CpdError::invalid_input("column-major index overflow"))?;
                    step.push(values[idx]);
                }
            }
            crate::MemoryLayout::Strided {
                row_stride,
                col_stride,
            } => {
                for j in 0..x.d {
                    let t_isize = isize::try_from(t).map_err(|_| {
                        CpdError::invalid_input(format!(
                            "time index {t} does not fit into isize for strided access"
                        ))
                    })?;
                    let j_isize = isize::try_from(j).map_err(|_| {
                        CpdError::invalid_input(format!(
                            "dimension index {j} does not fit into isize for strided access"
                        ))
                    })?;
                    let idx = t_isize
                        .checked_mul(row_stride)
                        .and_then(|left| j_isize.checked_mul(col_stride).and_then(|right| left.checked_add(right)))
                        .ok_or_else(|| {
                            CpdError::invalid_input(format!(
                                "strided index overflow at t={t}, j={j}, row_stride={row_stride}, col_stride={col_stride}"
                            ))
                        })?;
                    let idx_usize = usize::try_from(idx).map_err(|_| {
                        CpdError::invalid_input(format!(
                            "strided index negative at t={t}, j={j}: idx={idx}"
                        ))
                    })?;
                    let value = values.get(idx_usize).ok_or_else(|| {
                        CpdError::invalid_input(format!(
                            "strided index out of bounds at t={t}, j={j}: idx={idx_usize}, len={}",
                            values.len()
                        ))
                    })?;
                    step.push(*value);
                }
            }
        },
    }

    Ok(step)
}

/// Online detector contract: stateful incremental update.
pub trait OnlineDetector {
    type State: Clone + std::fmt::Debug;

    fn reset(&mut self);
    fn update(
        &mut self,
        x_t: &[f64],
        t_ns: Option<i64>,
        ctx: &ExecutionContext<'_>,
    ) -> Result<OnlineStepResult, CpdError>;
    fn save_state(&self) -> Self::State;
    fn load_state(&mut self, state: &Self::State);

    /// Default batched streaming path implemented on top of `update`.
    fn update_many(
        &mut self,
        x: &TimeSeriesView<'_>,
        ctx: &ExecutionContext<'_>,
    ) -> Result<Vec<OnlineStepResult>, CpdError> {
        let mut out = Vec::with_capacity(x.n);
        for t in 0..x.n {
            ctx.check_cancelled_every(t, 1)?;

            let x_t = extract_step_f64(x, t)?;
            let t_ns = match x.time {
                TimeIndex::None => None,
                TimeIndex::Uniform { t0_ns, dt_ns } => Some(uniform_timestamp(t, t0_ns, dt_ns)?),
                TimeIndex::Explicit(ts) => Some(ts[t]),
            };

            out.push(self.update(&x_t, t_ns, ctx)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{OfflineDetector, OnlineDetector, OnlineStepResult};
    use crate::{
        Constraints, Diagnostics, DTypeView, ExecutionContext, MemoryLayout, MissingPolicy,
        OfflineChangePointResult, TimeIndex, TimeSeriesView,
    };
    use std::borrow::Cow;

    struct MockOfflineDetector;

    impl OfflineDetector for MockOfflineDetector {
        fn detect(
            &self,
            x: &TimeSeriesView<'_>,
            _ctx: &ExecutionContext<'_>,
        ) -> Result<OfflineChangePointResult, crate::CpdError> {
            let diagnostics = Diagnostics {
                n: x.n,
                d: x.d,
                algorithm: Cow::Borrowed("mock-offline"),
                cost_model: Cow::Borrowed("none"),
                ..Diagnostics::default()
            };
            OfflineChangePointResult::new(x.n, vec![x.n], diagnostics)
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    struct MockState {
        updates_seen: usize,
    }

    struct MockOnlineDetector {
        state: MockState,
        recorded_inputs: Vec<Vec<f64>>,
        recorded_timestamps: Vec<Option<i64>>,
    }

    impl MockOnlineDetector {
        fn new() -> Self {
            Self {
                state: MockState { updates_seen: 0 },
                recorded_inputs: vec![],
                recorded_timestamps: vec![],
            }
        }
    }

    impl OnlineDetector for MockOnlineDetector {
        type State = MockState;

        fn reset(&mut self) {
            self.state.updates_seen = 0;
            self.recorded_inputs.clear();
            self.recorded_timestamps.clear();
        }

        fn update(
            &mut self,
            x_t: &[f64],
            t_ns: Option<i64>,
            _ctx: &ExecutionContext<'_>,
        ) -> Result<OnlineStepResult, crate::CpdError> {
            let t = self.state.updates_seen;
            self.state.updates_seen += 1;
            self.recorded_inputs.push(x_t.to_vec());
            self.recorded_timestamps.push(t_ns);
            Ok(OnlineStepResult {
                t,
                p_change: if x_t.is_empty() { 0.0 } else { x_t[0] },
                alert: false,
                alert_reason: None,
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

    fn ctx() -> ExecutionContext<'static> {
        static CONSTRAINTS: Constraints = Constraints {
            min_segment_len: 2,
            max_change_points: None,
            max_depth: None,
            candidate_splits: None,
            jump: 1,
            time_budget_ms: None,
            max_cost_evals: None,
            memory_budget_bytes: None,
            max_cache_bytes: None,
            cache_policy: crate::CachePolicy::Full,
            degradation_plan: Vec::new(),
            allow_algorithm_fallback: false,
        };
        ExecutionContext::new(&CONSTRAINTS)
    }

    #[test]
    fn offline_detector_trait_shape_sanity() {
        let detector = MockOfflineDetector;
        let values = [1.0_f64, 2.0, 3.0];
        let view = TimeSeriesView::new(
            DTypeView::F64(&values),
            3,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect("view should be valid");
        let result = detector.detect(&view, &ctx()).expect("detect should succeed");
        assert_eq!(result.breakpoints, vec![3]);
    }

    #[test]
    fn online_detector_state_lifecycle_and_reset() {
        let mut detector = MockOnlineDetector::new();
        detector.state.updates_seen = 5;
        let saved = detector.save_state();
        assert_eq!(saved, MockState { updates_seen: 5 });

        detector.reset();
        assert_eq!(detector.state.updates_seen, 0);

        detector.load_state(&saved);
        assert_eq!(detector.state.updates_seen, 5);
    }

    #[test]
    fn update_many_c_contiguous_univariate_and_multivariate() {
        let mut detector = MockOnlineDetector::new();
        let values = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = TimeSeriesView::new(
            DTypeView::F64(&values),
            3,
            2,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect("view should be valid");

        let out = detector
            .update_many(&view, &ctx())
            .expect("batch update should succeed");
        assert_eq!(out.len(), 3);
        assert_eq!(detector.recorded_inputs, vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        assert_eq!(detector.recorded_timestamps, vec![None, None, None]);
    }

    #[test]
    fn update_many_f_contiguous_path() {
        let mut detector = MockOnlineDetector::new();
        // n=3, d=2 in F-order: [x00, x10, x20, x01, x11, x21]
        let values = [1.0_f64, 3.0, 5.0, 2.0, 4.0, 6.0];
        let view = TimeSeriesView::new(
            DTypeView::F64(&values),
            3,
            2,
            MemoryLayout::FContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect("view should be valid");

        detector
            .update_many(&view, &ctx())
            .expect("batch update should succeed");
        assert_eq!(detector.recorded_inputs, vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
    }

    #[test]
    fn update_many_strided_path_with_positive_strides() {
        let mut detector = MockOnlineDetector::new();
        // Valid strided view with n*d=6 values and element strides row=1, col=3.
        // logical points: t0=[10,20], t1=[30,40], t2=[50,60]
        let values = [10.0_f64, 30.0, 50.0, 20.0, 40.0, 60.0];
        let view = TimeSeriesView::new(
            DTypeView::F64(&values),
            3,
            2,
            MemoryLayout::Strided {
                row_stride: 1,
                col_stride: 3,
            },
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect("view should be valid");

        detector
            .update_many(&view, &ctx())
            .expect("strided update should succeed");
        assert_eq!(detector.recorded_inputs, vec![vec![10.0, 20.0], vec![30.0, 40.0], vec![50.0, 60.0]]);
    }

    #[test]
    fn update_many_time_index_mapping_none_uniform_explicit() {
        let mut detector_none = MockOnlineDetector::new();
        let values = [1.0_f64, 2.0, 3.0];
        let view_none = TimeSeriesView::new(
            DTypeView::F64(&values),
            3,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect("view none valid");
        detector_none
            .update_many(&view_none, &ctx())
            .expect("none update succeeds");
        assert_eq!(detector_none.recorded_timestamps, vec![None, None, None]);

        let mut detector_uniform = MockOnlineDetector::new();
        let view_uniform = TimeSeriesView::new(
            DTypeView::F64(&values),
            3,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::Uniform {
                t0_ns: 1_000,
                dt_ns: 5,
            },
            MissingPolicy::Error,
        )
        .expect("view uniform valid");
        detector_uniform
            .update_many(&view_uniform, &ctx())
            .expect("uniform update succeeds");
        assert_eq!(
            detector_uniform.recorded_timestamps,
            vec![Some(1_000), Some(1_005), Some(1_010)]
        );

        let mut detector_explicit = MockOnlineDetector::new();
        let ts = [7_i64, 11_i64, 13_i64];
        let view_explicit = TimeSeriesView::new(
            DTypeView::F64(&values),
            3,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::Explicit(&ts),
            MissingPolicy::Error,
        )
        .expect("view explicit valid");
        detector_explicit
            .update_many(&view_explicit, &ctx())
            .expect("explicit update succeeds");
        assert_eq!(
            detector_explicit.recorded_timestamps,
            vec![Some(7), Some(11), Some(13)]
        );
    }

    #[test]
    fn update_many_strided_out_of_bounds_returns_invalid_input() {
        let mut detector = MockOnlineDetector::new();
        let values = [1.0_f64, 2.0, 3.0, 4.0];
        let view = TimeSeriesView::new(
            DTypeView::F64(&values),
            2,
            2,
            MemoryLayout::Strided {
                row_stride: 3,
                col_stride: 2,
            },
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect("view should be valid");

        let err = detector
            .update_many(&view, &ctx())
            .expect_err("out-of-bounds strided access must fail");
        assert!(err.to_string().contains("strided index out of bounds"));
    }

    #[test]
    fn update_many_uniform_timestamp_overflow_returns_invalid_input() {
        let mut detector = MockOnlineDetector::new();
        let values = [1.0_f64, 2.0];
        let view = TimeSeriesView::new(
            DTypeView::F64(&values),
            2,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::Uniform {
                t0_ns: i64::MAX - 1,
                dt_ns: 2,
            },
            MissingPolicy::Error,
        )
        .expect("view should be valid");

        let err = detector
            .update_many(&view, &ctx())
            .expect_err("overflow must fail");
        assert!(err.to_string().contains("uniform timestamp overflow"));
    }

    #[test]
    fn update_many_cancellation_propagates_cancelled_error() {
        let constraints = Constraints::default();
        let cancel = crate::CancelToken::new();
        let ctx = ExecutionContext::new(&constraints).with_cancel(&cancel);
        let mut detector = MockOnlineDetector::new();
        let values = [1.0_f64, 2.0, 3.0, 4.0];
        let view = TimeSeriesView::new(
            DTypeView::F64(&values),
            4,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect("view should be valid");

        cancel.cancel();
        let err = detector
            .update_many(&view, &ctx)
            .expect_err("cancelled token should stop iteration");
        assert_eq!(err.to_string(), "cancelled");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn online_step_result_serde_roundtrip() {
        let step = OnlineStepResult {
            t: 5,
            p_change: 0.42,
            alert: true,
            alert_reason: Some("threshold".to_string()),
            run_length_mode: 3,
            run_length_mean: 2.75,
            processing_latency_us: Some(120),
        };

        let encoded = serde_json::to_string(&step).expect("serialize step");
        let decoded: OnlineStepResult = serde_json::from_str(&encoded).expect("deserialize step");
        assert_eq!(decoded, step);
    }
}
