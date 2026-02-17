// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]
// PyO3-generated wrappers can trip false-positive clippy::useless_conversion diagnostics.
#![allow(clippy::useless_conversion)]

mod error_map;
mod numpy_interop;
mod result_objects;

use crate::error_map::cpd_error_to_pyerr;
use crate::numpy_interop::{DTypePolicy, OwnedSeries, parse_numpy_series};
use cpd_core::{
    CachePolicy, Constraints, CpdError, DegradationStep, ExecutionContext, MissingPolicy,
    OfflineChangePointResult as CoreOfflineChangePointResult, OfflineDetector, OnlineDetector,
    Penalty, ReproMode, Stopping, TimeSeriesView,
};
use cpd_costs::{
    CostCosine, CostL1Median, CostL2Mean, CostNormalFullCov, CostNormalMeanVar, CostRank,
};
use cpd_doctor::{
    CostConfig as DoctorCostConfig, DetectorConfig as DoctorDetectorConfig,
    OfflineDetectorConfig as DoctorOfflineDetectorConfig, PipelineSpec as DoctorPipelineSpec,
    execute_pipeline_with_repro_mode as execute_doctor_pipeline_with_repro_mode,
};
use cpd_offline::{
    BinSeg as OfflineBinSeg, BinSegConfig, Fpop as OfflineFpop, FpopConfig, Pelt as OfflinePelt,
    PeltConfig, WbsConfig, WbsIntervalStrategy,
};
use cpd_online::{
    AlertPolicy, BocpdConfig, BocpdDetector, BocpdState, ConstantHazard, CusumConfig,
    CusumDetector, CusumState, GeometricHazard, HazardSpec, LateDataPolicy, ObservationModel,
    ObservationStats, PageHinkleyConfig, PageHinkleyDetector, PageHinkleyState,
};
#[cfg(feature = "serde")]
use cpd_online::{
    BOCPD_DETECTOR_ID, CheckpointEnvelope, PayloadCodec, decode_checkpoint_envelope,
    encode_checkpoint_envelope, load_bocpd_checkpoint, load_bocpd_checkpoint_file,
    load_cusum_checkpoint, load_cusum_checkpoint_file, load_page_hinkley_checkpoint,
    load_page_hinkley_checkpoint_file, load_state_from_checkpoint_envelope,
    load_state_from_checkpoint_file, save_bocpd_checkpoint, save_bocpd_checkpoint_file,
    save_cusum_checkpoint, save_cusum_checkpoint_file, save_page_hinkley_checkpoint,
    save_page_hinkley_checkpoint_file,
};
#[cfg(feature = "preprocess")]
use cpd_preprocess::{
    DeseasonalizeConfig, DeseasonalizeMethod, DetrendConfig, DetrendMethod, PreprocessConfig,
    PreprocessPipeline, RobustScaleConfig, WinsorizeConfig,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
#[cfg(feature = "serde")]
use pyo3::types::PyBytes;
use pyo3::types::{PyAnyMethods, PyDict, PyList, PyModule};
use result_objects::{
    PyDiagnostics, PyOfflineChangePointResult, PyOnlineStepResult, PyPruningStats, PySegmentStats,
};
#[cfg(feature = "serde")]
use std::path::PathBuf;

fn parse_sequence(values: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    values.extract::<Vec<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "expected a sequence of float values for smoke detection",
        )
    })
}

fn parse_owned_series(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<OwnedSeries> {
    let numpy = PyModule::import(py, "numpy")?;
    let as_array = numpy.call_method1("asarray", (x,))?;
    let parsed = parse_numpy_series(
        py,
        &as_array,
        None,
        MissingPolicy::Error,
        DTypePolicy::KeepInput,
    )?;
    parsed.into_owned().map_err(cpd_error_to_pyerr)
}

#[cfg(feature = "serde")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PyCheckpointFormat {
    Bytes,
    Json,
}

#[cfg(feature = "serde")]
impl PyCheckpointFormat {
    fn parse(raw: &str) -> PyResult<Self> {
        match raw.to_ascii_lowercase().as_str() {
            "bytes" => Ok(Self::Bytes),
            "json" => Ok(Self::Json),
            _ => Err(PyValueError::new_err(format!(
                "unsupported checkpoint format '{raw}'; expected one of: 'bytes', 'json'"
            ))),
        }
    }

    fn infer(raw: Option<&str>, state: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Some(raw) = raw {
            return Self::parse(raw);
        }
        if state.downcast::<PyDict>().is_ok() {
            return Ok(Self::Json);
        }
        Ok(Self::Bytes)
    }

    fn payload_codec(self) -> PayloadCodec {
        match self {
            Self::Bytes => PayloadCodec::Bincode,
            Self::Json => PayloadCodec::Json,
        }
    }
}

#[cfg(feature = "serde")]
fn parse_checkpoint_path(
    path: Option<&Bound<'_, PyAny>>,
    context: &str,
) -> PyResult<Option<PathBuf>> {
    let Some(path) = path else {
        return Ok(None);
    };
    if path.is_none() {
        return Ok(None);
    }

    let os = PyModule::import(path.py(), "os")?;
    let fspath = os.call_method1("fspath", (path,))?;
    let raw: String = fspath.extract().map_err(|_| {
        PyTypeError::new_err(format!("{context} path must be str or os.PathLike[str]"))
    })?;
    if raw.is_empty() {
        return Err(PyValueError::new_err(format!(
            "{context} path must be non-empty"
        )));
    }
    Ok(Some(PathBuf::from(raw)))
}

#[cfg(feature = "serde")]
fn checkpoint_output(
    py: Python<'_>,
    envelope: &CheckpointEnvelope,
    format: PyCheckpointFormat,
) -> PyResult<Py<PyAny>> {
    let encoded = encode_checkpoint_envelope(envelope).map_err(cpd_error_to_pyerr)?;
    match format {
        PyCheckpointFormat::Bytes => Ok(PyBytes::new(py, &encoded).into_any().unbind()),
        PyCheckpointFormat::Json => {
            let json = PyModule::import(py, "json")?;
            let value = json.call_method1("loads", (PyBytes::new(py, &encoded),))?;
            Ok(value.unbind())
        }
    }
}

#[cfg(feature = "serde")]
fn decode_checkpoint_input(
    py: Python<'_>,
    state: &Bound<'_, PyAny>,
    format: PyCheckpointFormat,
) -> PyResult<CheckpointEnvelope> {
    let encoded = match format {
        PyCheckpointFormat::Bytes => state.extract::<Vec<u8>>().map_err(|_| {
            PyTypeError::new_err("checkpoint bytes input must be a bytes-like value")
        })?,
        PyCheckpointFormat::Json => {
            let json = PyModule::import(py, "json")?;
            let dumped = json.call_method1("dumps", (state,))?;
            let dumped: String = dumped.extract()?;
            dumped.into_bytes()
        }
    };
    decode_checkpoint_envelope(&encoded).map_err(cpd_error_to_pyerr)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PyCostModel {
    Cosine,
    L1Median,
    L2,
    Normal,
    NormalFullCov,
    Rank,
}

impl PyCostModel {
    fn parse(model: &str) -> PyResult<Self> {
        match model.to_ascii_lowercase().as_str() {
            "cosine" => Ok(Self::Cosine),
            "l1" | "l1_median" => Ok(Self::L1Median),
            "l2" => Ok(Self::L2),
            "normal" => Ok(Self::Normal),
            "normal_full_cov" | "normal_fullcov" | "normalfullcov" => Ok(Self::NormalFullCov),
            "rank" => Ok(Self::Rank),
            _ => Err(PyValueError::new_err(format!(
                "unsupported model '{model}'; expected one of: 'cosine', 'l1_median', 'l2', 'normal', 'normal_full_cov', 'rank'"
            ))),
        }
    }

    fn cost_model_name(self) -> &'static str {
        match self {
            Self::Cosine => "cosine",
            Self::L1Median => "l1_median",
            Self::L2 => "l2",
            Self::Normal => "normal",
            Self::NormalFullCov => "normal_full_cov",
            Self::Rank => "rank",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PyDetectorKind {
    Pelt,
    Binseg,
    Fpop,
}

impl PyDetectorKind {
    fn parse(detector: &str) -> PyResult<Self> {
        match detector.to_ascii_lowercase().as_str() {
            "pelt" => Ok(Self::Pelt),
            "binseg" => Ok(Self::Binseg),
            "fpop" => Ok(Self::Fpop),
            _ => Err(PyValueError::new_err(format!(
                "unsupported detector '{detector}'; expected one of: 'pelt', 'binseg', 'fpop'"
            ))),
        }
    }
}

fn parse_repro_mode(repro_mode: &str) -> PyResult<ReproMode> {
    match repro_mode.to_ascii_lowercase().as_str() {
        "fast" => Ok(ReproMode::Fast),
        "balanced" => Ok(ReproMode::Balanced),
        "strict" => Ok(ReproMode::Strict),
        _ => Err(PyValueError::new_err(format!(
            "unsupported repro_mode '{repro_mode}'; expected one of: 'fast', 'balanced', 'strict'"
        ))),
    }
}

fn required_dict_key<'py>(
    dict: &Bound<'py, PyDict>,
    key: &str,
    context: &str,
) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?.ok_or_else(|| {
        PyValueError::new_err(format!("{context} requires key '{key}' with a valid value"))
    })
}

fn reject_unknown_keys(dict: &Bound<'_, PyDict>, context: &str, allowed: &[&str]) -> PyResult<()> {
    for (key_obj, _) in dict.iter() {
        let key: String = key_obj
            .extract()
            .map_err(|_| PyTypeError::new_err(format!("{context} keys must be strings")))?;
        if !allowed.contains(&key.as_str()) {
            return Err(PyValueError::new_err(format!(
                "unsupported {context} key '{key}'"
            )));
        }
    }
    Ok(())
}

fn parse_alert_policy(
    value: Option<&Bound<'_, PyAny>>,
    default_policy: AlertPolicy,
) -> PyResult<AlertPolicy> {
    let Some(value) = value else {
        return Ok(default_policy);
    };
    if value.is_none() {
        return Ok(default_policy);
    }

    let dict = value
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("alert_policy must be a dict"))?;

    let mut threshold = default_policy.threshold;
    let mut hysteresis = default_policy.hysteresis;
    let mut cooldown_steps = default_policy.cooldown_steps;
    let mut min_run_length = default_policy.min_run_length;
    let mut saw_cooldown = false;
    let mut saw_cooldown_steps = false;

    for (key_obj, value_obj) in dict.iter() {
        let key: String = key_obj
            .extract()
            .map_err(|_| PyTypeError::new_err("alert_policy keys must be strings"))?;
        match key.as_str() {
            "threshold" => {
                threshold = value_obj
                    .extract::<f64>()
                    .map_err(|_| PyTypeError::new_err("alert_policy.threshold must be a float"))?
            }
            "hysteresis" => {
                hysteresis = value_obj
                    .extract::<f64>()
                    .map_err(|_| PyTypeError::new_err("alert_policy.hysteresis must be a float"))?
            }
            "cooldown" => {
                if saw_cooldown_steps {
                    return Err(PyValueError::new_err(
                        "alert_policy accepts only one of cooldown or cooldown_steps",
                    ));
                }
                cooldown_steps = value_obj.extract::<usize>().map_err(|_| {
                    PyTypeError::new_err("alert_policy.cooldown must be an integer")
                })?;
                saw_cooldown = true;
            }
            "cooldown_steps" => {
                if saw_cooldown {
                    return Err(PyValueError::new_err(
                        "alert_policy accepts only one of cooldown or cooldown_steps",
                    ));
                }
                cooldown_steps = value_obj.extract::<usize>().map_err(|_| {
                    PyTypeError::new_err("alert_policy.cooldown_steps must be an integer")
                })?;
                saw_cooldown_steps = true;
            }
            "min_run_length" => {
                min_run_length = value_obj.extract::<usize>().map_err(|_| {
                    PyTypeError::new_err("alert_policy.min_run_length must be an integer")
                })?
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported alert_policy key '{key}'"
                )));
            }
        }
    }

    let policy = AlertPolicy::new(threshold, hysteresis, cooldown_steps, min_run_length);
    policy.validate().map_err(cpd_error_to_pyerr)?;
    Ok(policy)
}

fn parse_overflow_policy(value: &str) -> PyResult<cpd_online::OverflowPolicy> {
    match value.to_ascii_lowercase().as_str() {
        "drop_oldest" | "dropoldest" => Ok(cpd_online::OverflowPolicy::DropOldest),
        "drop_newest" | "dropnewest" => Ok(cpd_online::OverflowPolicy::DropNewest),
        "error" => Ok(cpd_online::OverflowPolicy::Error),
        _ => Err(PyValueError::new_err(format!(
            "unsupported late_data_policy.on_overflow '{value}'; expected one of: 'drop_oldest', 'drop_newest', 'error'"
        ))),
    }
}

fn parse_late_data_policy(value: Option<&Bound<'_, PyAny>>) -> PyResult<LateDataPolicy> {
    let Some(value) = value else {
        return Ok(LateDataPolicy::Reject);
    };
    if value.is_none() {
        return Ok(LateDataPolicy::Reject);
    }

    if let Ok(named) = value.extract::<String>() {
        return match named.to_ascii_lowercase().as_str() {
            "reject" => Ok(LateDataPolicy::Reject),
            _ => Err(PyValueError::new_err(format!(
                "late_data_policy as string supports only 'reject'; use dict form for buffered policies"
            ))),
        };
    }

    let dict = value
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("late_data_policy must be a string or dict"))?;
    let kind_obj = required_dict_key(dict, "kind", "late_data_policy")?;
    let kind: String = kind_obj
        .extract()
        .map_err(|_| PyTypeError::new_err("late_data_policy.kind must be a string"))?;

    match kind.to_ascii_lowercase().as_str() {
        "reject" => {
            reject_unknown_keys(dict, "late_data_policy", &["kind"])?;
            Ok(LateDataPolicy::Reject)
        }
        "buffer_within_window" => {
            reject_unknown_keys(
                dict,
                "late_data_policy",
                &["kind", "max_delay_ns", "max_buffer_items", "on_overflow"],
            )?;
            let max_delay_ns = required_dict_key(dict, "max_delay_ns", "late_data_policy")?
                .extract::<i64>()
                .map_err(|_| {
                    PyTypeError::new_err("late_data_policy.max_delay_ns must be an integer")
                })?;
            let max_buffer_items = required_dict_key(dict, "max_buffer_items", "late_data_policy")?
                .extract::<usize>()
                .map_err(|_| {
                    PyTypeError::new_err("late_data_policy.max_buffer_items must be an integer")
                })?;
            let on_overflow = required_dict_key(dict, "on_overflow", "late_data_policy")?
                .extract::<String>()
                .map_err(|_| {
                    PyTypeError::new_err("late_data_policy.on_overflow must be a string")
                })?;
            let policy = LateDataPolicy::BufferWithinWindow {
                max_delay_ns,
                max_buffer_items,
                on_overflow: parse_overflow_policy(&on_overflow)?,
            };
            policy.validate().map_err(cpd_error_to_pyerr)?;
            Ok(policy)
        }
        "reorder_by_timestamp" => {
            reject_unknown_keys(
                dict,
                "late_data_policy",
                &["kind", "max_delay_ns", "max_buffer_items", "on_overflow"],
            )?;
            let max_delay_ns = required_dict_key(dict, "max_delay_ns", "late_data_policy")?
                .extract::<i64>()
                .map_err(|_| {
                    PyTypeError::new_err("late_data_policy.max_delay_ns must be an integer")
                })?;
            let max_buffer_items = required_dict_key(dict, "max_buffer_items", "late_data_policy")?
                .extract::<usize>()
                .map_err(|_| {
                    PyTypeError::new_err("late_data_policy.max_buffer_items must be an integer")
                })?;
            let on_overflow = required_dict_key(dict, "on_overflow", "late_data_policy")?
                .extract::<String>()
                .map_err(|_| {
                    PyTypeError::new_err("late_data_policy.on_overflow must be a string")
                })?;
            let policy = LateDataPolicy::ReorderByTimestamp {
                max_delay_ns,
                max_buffer_items,
                on_overflow: parse_overflow_policy(&on_overflow)?,
            };
            policy.validate().map_err(cpd_error_to_pyerr)?;
            Ok(policy)
        }
        _ => Err(PyValueError::new_err(format!(
            "unsupported late_data_policy.kind '{kind}'; expected one of: 'reject', 'buffer_within_window', 'reorder_by_timestamp'"
        ))),
    }
}

fn parse_bocpd_observation_model(model: &str) -> PyResult<ObservationModel> {
    match model.to_ascii_lowercase().as_str() {
        "gaussian_nig" | "gaussian" => Ok(ObservationModel::Gaussian {
            prior: cpd_online::GaussianNigPrior::default(),
        }),
        "poisson_gamma" | "poisson" => Ok(ObservationModel::Poisson {
            prior: cpd_online::PoissonGammaPrior::default(),
        }),
        "bernoulli_beta" | "bernoulli" => Ok(ObservationModel::Bernoulli {
            prior: cpd_online::BernoulliBetaPrior::default(),
        }),
        _ => Err(PyValueError::new_err(format!(
            "unsupported model '{model}'; expected one of: 'gaussian_nig', 'poisson_gamma', 'bernoulli_beta'"
        ))),
    }
}

fn parse_bocpd_hazard(value: Option<&Bound<'_, PyAny>>) -> PyResult<HazardSpec> {
    let Some(value) = value else {
        return Ok(HazardSpec::default());
    };
    if value.is_none() {
        return Ok(HazardSpec::default());
    }

    if let Ok(p_change) = value.extract::<f64>() {
        let hazard = ConstantHazard::new(p_change).map_err(cpd_error_to_pyerr)?;
        return Ok(HazardSpec::Constant(hazard));
    }

    let dict = value
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("hazard must be a float or dict"))?;
    let kind_obj = required_dict_key(dict, "kind", "hazard")?;
    let kind: String = kind_obj
        .extract()
        .map_err(|_| PyTypeError::new_err("hazard.kind must be a string"))?;

    match kind.to_ascii_lowercase().as_str() {
        "constant" => {
            reject_unknown_keys(dict, "hazard", &["kind", "p_change"])?;
            let p_change = required_dict_key(dict, "p_change", "hazard")?
                .extract::<f64>()
                .map_err(|_| PyTypeError::new_err("hazard.p_change must be a float"))?;
            Ok(HazardSpec::Constant(
                ConstantHazard::new(p_change).map_err(cpd_error_to_pyerr)?,
            ))
        }
        "geometric" => {
            reject_unknown_keys(dict, "hazard", &["kind", "mean_run_length"])?;
            let mean_run_length = required_dict_key(dict, "mean_run_length", "hazard")?
                .extract::<f64>()
                .map_err(|_| PyTypeError::new_err("hazard.mean_run_length must be a float"))?;
            Ok(HazardSpec::Geometric(
                GeometricHazard::new(mean_run_length).map_err(cpd_error_to_pyerr)?,
            ))
        }
        _ => Err(PyValueError::new_err(format!(
            "unsupported hazard.kind '{kind}'; expected one of: 'constant', 'geometric'"
        ))),
    }
}

fn bocpd_model_name(model: &ObservationModel) -> &'static str {
    match model {
        ObservationModel::Gaussian { .. } => "gaussian_nig",
        ObservationModel::Poisson { .. } => "poisson_gamma",
        ObservationModel::Bernoulli { .. } => "bernoulli_beta",
    }
}

fn bocpd_hazard_name(hazard: &HazardSpec) -> &'static str {
    match hazard {
        HazardSpec::Constant(_) => "constant",
        HazardSpec::Geometric(_) => "geometric",
    }
}

fn bocpd_state_matches_observation(state: &BocpdState, observation: &ObservationModel) -> bool {
    !state.run_stats.is_empty()
        && state.run_stats.iter().all(|stats| {
            matches!(
                (observation, stats),
                (
                    ObservationModel::Gaussian { .. },
                    ObservationStats::Gaussian { .. }
                ) | (
                    ObservationModel::Poisson { .. },
                    ObservationStats::Poisson { .. }
                ) | (
                    ObservationModel::Bernoulli { .. },
                    ObservationStats::Bernoulli { .. }
                )
            )
        })
}

fn online_update_scalar<D>(
    detector: &mut D,
    x_t: f64,
    t_ns: Option<i64>,
) -> PyResult<PyOnlineStepResult>
where
    D: OnlineDetector + Send,
{
    let constraints = Constraints::default();
    let ctx = ExecutionContext::new(&constraints);
    detector
        .update(&[x_t], t_ns, &ctx)
        .map(Into::into)
        .map_err(cpd_error_to_pyerr)
}

const UPDATE_MANY_GIL_RELEASE_MIN_WORK_ITEMS: usize = 16;

fn should_release_gil_for_update_many(n_samples: usize, n_dims: usize) -> bool {
    n_samples.saturating_mul(n_dims.max(1)) >= UPDATE_MANY_GIL_RELEASE_MIN_WORK_ITEMS
}

fn online_update_many<D>(
    py: Python<'_>,
    detector: &mut D,
    x_batch: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyOnlineStepResult>>
where
    D: OnlineDetector + Send,
{
    let owned = parse_owned_series(py, x_batch)?;
    let steps = if should_release_gil_for_update_many(owned.n_samples(), owned.n_dims()) {
        py.allow_threads(|| {
            let view = owned.view()?;
            let constraints = Constraints::default();
            let ctx = ExecutionContext::new(&constraints);
            detector.update_many(&view, &ctx)
        })
        .map_err(cpd_error_to_pyerr)?
    } else {
        let view = owned.view().map_err(cpd_error_to_pyerr)?;
        let constraints = Constraints::default();
        let ctx = ExecutionContext::new(&constraints);
        detector
            .update_many(&view, &ctx)
            .map_err(cpd_error_to_pyerr)?
    };

    Ok(steps.into_iter().map(Into::into).collect())
}

fn parse_cache_policy(value: &Bound<'_, PyAny>) -> PyResult<CachePolicy> {
    if let Ok(named) = value.extract::<String>() {
        return match named.to_ascii_lowercase().as_str() {
            "full" => Ok(CachePolicy::Full),
            "budgeted" | "approximate" => Err(PyValueError::new_err(
                "constraints.cache_policy as string only supports 'full'; use dict form for other policies",
            )),
            _ => Err(PyValueError::new_err(format!(
                "unsupported constraints.cache_policy '{named}'; expected 'full' or dict with kind"
            ))),
        };
    }

    let dict = value
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("constraints.cache_policy must be a string or dict"))?;
    let kind_obj = required_dict_key(dict, "kind", "constraints.cache_policy")?;
    let kind: String = kind_obj
        .extract()
        .map_err(|_| PyTypeError::new_err("constraints.cache_policy.kind must be a string"))?;

    match kind.to_ascii_lowercase().as_str() {
        "full" => Ok(CachePolicy::Full),
        "budgeted" => {
            let max_bytes = required_dict_key(dict, "max_bytes", "constraints.cache_policy")?
                .extract::<usize>()
                .map_err(|_| {
                    PyTypeError::new_err("constraints.cache_policy.max_bytes must be an integer")
                })?;
            Ok(CachePolicy::Budgeted { max_bytes })
        }
        "approximate" => {
            let max_bytes = required_dict_key(dict, "max_bytes", "constraints.cache_policy")?
                .extract::<usize>()
                .map_err(|_| {
                    PyTypeError::new_err("constraints.cache_policy.max_bytes must be an integer")
                })?;
            let error_tolerance =
                required_dict_key(dict, "error_tolerance", "constraints.cache_policy")?
                    .extract::<f64>()
                    .map_err(|_| {
                        PyTypeError::new_err(
                            "constraints.cache_policy.error_tolerance must be a float",
                        )
                    })?;
            Ok(CachePolicy::Approximate {
                max_bytes,
                error_tolerance,
            })
        }
        _ => Err(PyValueError::new_err(format!(
            "unsupported constraints.cache_policy.kind '{kind}'; expected one of: 'full', 'budgeted', 'approximate'"
        ))),
    }
}

fn parse_degradation_step(value: &Bound<'_, PyAny>) -> PyResult<DegradationStep> {
    if let Ok(named) = value.extract::<String>() {
        return match named.to_ascii_lowercase().as_str() {
            "disable_uncertainty_bands" => Ok(DegradationStep::DisableUncertaintyBands),
            _ => Err(PyValueError::new_err(format!(
                "unsupported constraints.degradation_plan step '{named}'"
            ))),
        };
    }

    let dict = value.downcast::<PyDict>().map_err(|_| {
        PyTypeError::new_err(
            "constraints.degradation_plan entries must be strings or dicts with 'kind'",
        )
    })?;

    let kind_obj = required_dict_key(dict, "kind", "constraints.degradation_plan step")?;
    let kind: String = kind_obj.extract().map_err(|_| {
        PyTypeError::new_err("constraints.degradation_plan step kind must be a string")
    })?;

    match kind.to_ascii_lowercase().as_str() {
        "increase_jump" => {
            let factor = required_dict_key(dict, "factor", "constraints.degradation_plan step")?
                .extract::<usize>()
                .map_err(|_| {
                    PyTypeError::new_err(
                        "constraints.degradation_plan increase_jump.factor must be an integer",
                    )
                })?;
            let max_jump = required_dict_key(
                dict,
                "max_jump",
                "constraints.degradation_plan step",
            )?
            .extract::<usize>()
            .map_err(|_| {
                PyTypeError::new_err(
                    "constraints.degradation_plan increase_jump.max_jump must be an integer",
                )
            })?;
            Ok(DegradationStep::IncreaseJump { factor, max_jump })
        }
        "disable_uncertainty_bands" => Ok(DegradationStep::DisableUncertaintyBands),
        "switch_cache_policy" => {
            let cache_policy =
                required_dict_key(dict, "cache_policy", "constraints.degradation_plan step")?;
            Ok(DegradationStep::SwitchCachePolicy(parse_cache_policy(
                &cache_policy,
            )?))
        }
        _ => Err(PyValueError::new_err(format!(
            "unsupported constraints.degradation_plan step kind '{kind}'"
        ))),
    }
}

fn parse_degradation_plan(value: &Bound<'_, PyAny>) -> PyResult<Vec<DegradationStep>> {
    if value.is_none() {
        return Ok(vec![]);
    }

    let plan = value
        .downcast::<PyList>()
        .map_err(|_| PyTypeError::new_err("constraints.degradation_plan must be a list"))?;
    let mut out = Vec::with_capacity(plan.len());
    for step in plan.iter() {
        out.push(parse_degradation_step(&step)?);
    }
    Ok(out)
}

fn resolve_stopping(pen: Option<f64>, n_bkps: Option<usize>, context: &str) -> PyResult<Stopping> {
    match (pen, n_bkps) {
        (Some(_), Some(_)) => Err(PyValueError::new_err(format!(
            "{context} requires exactly one of pen or n_bkps; got both"
        ))),
        (None, None) => Err(PyValueError::new_err(format!(
            "{context} requires exactly one of pen or n_bkps; got neither"
        ))),
        (Some(beta), None) => {
            if !beta.is_finite() || beta <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "{context} with pen=... requires a finite value > 0.0; got {beta}"
                )));
            }
            Ok(Stopping::Penalized(Penalty::Manual(beta)))
        }
        (None, Some(k)) => {
            if k == 0 {
                return Err(PyValueError::new_err(format!(
                    "{context} with n_bkps=... requires n_bkps >= 1; got 0"
                )));
            }
            Ok(Stopping::KnownK(k))
        }
    }
}

fn parse_constraints(constraints: Option<&Bound<'_, PyAny>>) -> PyResult<Constraints> {
    let Some(constraints) = constraints else {
        return Ok(Constraints::default());
    };

    let dict = constraints
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("constraints must be a dict"))?;
    let mut out = Constraints::default();

    for (key_obj, value_obj) in dict.iter() {
        let key: String = key_obj
            .extract()
            .map_err(|_| PyTypeError::new_err("constraints keys must be strings"))?;

        match key.as_str() {
            "min_segment_len" => out.min_segment_len = value_obj.extract::<usize>()?,
            "jump" => out.jump = value_obj.extract::<usize>()?,
            "max_change_points" => out.max_change_points = value_obj.extract::<Option<usize>>()?,
            "max_depth" => out.max_depth = value_obj.extract::<Option<usize>>()?,
            "candidate_splits" => {
                out.candidate_splits = value_obj.extract::<Option<Vec<usize>>>()?
            }
            "time_budget_ms" => out.time_budget_ms = value_obj.extract::<Option<u64>>()?,
            "max_cost_evals" => out.max_cost_evals = value_obj.extract::<Option<usize>>()?,
            "memory_budget_bytes" => {
                out.memory_budget_bytes = value_obj.extract::<Option<usize>>()?
            }
            "max_cache_bytes" => out.max_cache_bytes = value_obj.extract::<Option<usize>>()?,
            "cache_policy" => out.cache_policy = parse_cache_policy(&value_obj)?,
            "degradation_plan" => out.degradation_plan = parse_degradation_plan(&value_obj)?,
            "allow_algorithm_fallback" => {
                out.allow_algorithm_fallback = value_obj.extract::<bool>()?
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported constraints key '{key}'"
                )));
            }
        }
    }

    Ok(out)
}

fn parse_penalty(value: &Bound<'_, PyAny>) -> PyResult<Penalty> {
    if let Ok(named) = value.extract::<String>() {
        return match named.to_ascii_lowercase().as_str() {
            "bic" => Ok(Penalty::BIC),
            "aic" => Ok(Penalty::AIC),
            _ => Err(PyValueError::new_err(format!(
                "unsupported stopping.penalty '{named}'; expected 'bic', 'aic', or a positive float"
            ))),
        };
    }

    let beta = value.extract::<f64>().map_err(|_| {
        PyTypeError::new_err("stopping.penalty must be 'bic', 'aic', or a positive float")
    })?;

    if !beta.is_finite() || beta <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "stopping.penalty numeric value must be finite and > 0.0; got {beta}"
        )));
    }
    Ok(Penalty::Manual(beta))
}

fn parse_stopping(stopping: Option<&Bound<'_, PyAny>>) -> PyResult<Stopping> {
    let Some(stopping) = stopping else {
        return Ok(Stopping::Penalized(Penalty::BIC));
    };

    let dict = stopping
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("stopping must be a dict"))?;

    let mut n_bkps: Option<usize> = None;
    let mut pen: Option<f64> = None;
    let mut penalty: Option<Penalty> = None;

    for (key_obj, value_obj) in dict.iter() {
        let key: String = key_obj
            .extract()
            .map_err(|_| PyTypeError::new_err("stopping keys must be strings"))?;
        match key.as_str() {
            "n_bkps" => n_bkps = value_obj.extract::<Option<usize>>()?,
            "pen" => pen = value_obj.extract::<Option<f64>>()?,
            "penalty" => penalty = Some(parse_penalty(&value_obj)?),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported stopping key '{key}'"
                )));
            }
        }
    }

    if pen.is_some() && penalty.is_some() {
        return Err(PyValueError::new_err(
            "stopping accepts at most one of 'pen' or 'penalty'",
        ));
    }

    let penalty = match (pen, penalty) {
        (Some(beta), None) => {
            if !beta.is_finite() || beta <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "stopping.pen must be finite and > 0.0; got {beta}"
                )));
            }
            Some(Penalty::Manual(beta))
        }
        (None, Some(parsed)) => Some(parsed),
        (None, None) => None,
        (Some(_), Some(_)) => unreachable!(),
    };

    match (n_bkps, penalty) {
        (Some(_), Some(_)) => Err(PyValueError::new_err(
            "stopping requires exactly one of n_bkps or (pen/penalty); got both",
        )),
        (None, None) => Err(PyValueError::new_err(
            "stopping requires one of n_bkps, pen, or penalty",
        )),
        (Some(k), None) => {
            if k == 0 {
                return Err(PyValueError::new_err(
                    "stopping.n_bkps must be >= 1 when provided",
                ));
            }
            Ok(Stopping::KnownK(k))
        }
        (None, Some(parsed)) => Ok(Stopping::Penalized(parsed)),
    }
}

#[cfg(feature = "preprocess")]
fn parse_preprocess(preprocess: Option<&Bound<'_, PyAny>>) -> PyResult<Option<PreprocessPipeline>> {
    let Some(preprocess) = preprocess else {
        return Ok(None);
    };
    if preprocess.is_none() {
        return Ok(None);
    }

    let dict = preprocess
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("preprocess must be a dict"))?;
    let mut config = PreprocessConfig::default();

    for (key_obj, value_obj) in dict.iter() {
        let key: String = key_obj
            .extract()
            .map_err(|_| PyTypeError::new_err("preprocess keys must be strings"))?;
        match key.as_str() {
            "detrend" => config.detrend = parse_detrend_config(&value_obj)?,
            "deseasonalize" => config.deseasonalize = parse_deseasonalize_config(&value_obj)?,
            "winsorize" => config.winsorize = parse_winsorize_config(&value_obj)?,
            "robust_scale" => config.robust_scale = parse_robust_scale_config(&value_obj)?,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported preprocess key '{key}'"
                )));
            }
        }
    }

    let has_stage = config.detrend.is_some()
        || config.deseasonalize.is_some()
        || config.winsorize.is_some()
        || config.robust_scale.is_some();
    if !has_stage {
        return Ok(None);
    }

    PreprocessPipeline::new(config)
        .map(Some)
        .map_err(cpd_error_to_pyerr)
}

#[cfg(not(feature = "preprocess"))]
fn parse_preprocess(preprocess: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    if let Some(preprocess) = preprocess
        && !preprocess.is_none()
    {
        return Err(PyValueError::new_err(
            "preprocess requires cpd-python built with preprocess feature; rebuild with --features extension-module,preprocess",
        ));
    }
    Ok(())
}

fn parse_pipeline_interval_strategy(raw: &str) -> PyResult<WbsIntervalStrategy> {
    match raw.to_ascii_lowercase().as_str() {
        "random" => Ok(WbsIntervalStrategy::Random),
        "deterministic_grid" | "deterministicgrid" => Ok(WbsIntervalStrategy::DeterministicGrid),
        "stratified" => Ok(WbsIntervalStrategy::Stratified),
        _ => Err(PyValueError::new_err(format!(
            "unsupported pipeline.detector.interval_strategy '{raw}'; expected one of: 'random', 'deterministic_grid', 'stratified'"
        ))),
    }
}

fn parse_pipeline_penalty_compat(value: &Bound<'_, PyAny>) -> PyResult<Penalty> {
    if let Ok(named) = value.extract::<String>() {
        return match named.to_ascii_lowercase().as_str() {
            "bic" => Ok(Penalty::BIC),
            "aic" => Ok(Penalty::AIC),
            _ => Err(PyValueError::new_err(format!(
                "unsupported penalty '{named}'; expected one of: 'bic', 'aic', or {{'Manual': <positive-float>}}"
            ))),
        };
    }

    if let Ok(beta) = value.extract::<f64>() {
        if !beta.is_finite() || beta <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "manual penalty value must be finite and > 0.0; got {beta}"
            )));
        }
        return Ok(Penalty::Manual(beta));
    }

    let dict = value.downcast::<PyDict>().map_err(|_| {
        PyTypeError::new_err(
            "penalty must be 'bic', 'aic', a positive float, or {'Manual': <positive-float>}",
        )
    })?;
    if dict.len() != 1 {
        return Err(PyValueError::new_err(
            "penalty dict form must contain exactly one key: 'Manual'",
        ));
    }

    let (key_obj, value_obj) = dict
        .iter()
        .next()
        .expect("penalty dict len checked to be one");
    let key: String = key_obj
        .extract()
        .map_err(|_| PyTypeError::new_err("penalty dict keys must be strings"))?;
    match key.as_str() {
        "Manual" => {
            let beta = value_obj
                .extract::<f64>()
                .map_err(|_| PyTypeError::new_err("penalty['Manual'] must be a positive float"))?;
            if !beta.is_finite() || beta <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "penalty['Manual'] must be finite and > 0.0; got {beta}"
                )));
            }
            Ok(Penalty::Manual(beta))
        }
        _ => Err(PyValueError::new_err(format!(
            "unsupported penalty dict key '{key}'; expected 'Manual'"
        ))),
    }
}

fn parse_pipeline_stopping_compat(stopping: Option<&Bound<'_, PyAny>>) -> PyResult<Stopping> {
    let Some(stopping) = stopping else {
        return Ok(Stopping::Penalized(Penalty::BIC));
    };
    let dict = stopping
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("pipeline.stopping must be a dict"))?;

    let has_legacy_shape = dict.get_item("n_bkps")?.is_some()
        || dict.get_item("pen")?.is_some()
        || dict.get_item("penalty")?.is_some();
    if has_legacy_shape {
        return parse_stopping(Some(stopping));
    }

    if dict.len() != 1 {
        return Err(PyValueError::new_err(
            "pipeline.stopping serde form must have exactly one key: 'KnownK', 'Penalized', or 'PenaltyPath'",
        ));
    }

    let (key_obj, value_obj) = dict
        .iter()
        .next()
        .expect("stopping dict len checked to be one");
    let key: String = key_obj
        .extract()
        .map_err(|_| PyTypeError::new_err("pipeline.stopping keys must be strings"))?;
    match key.as_str() {
        "KnownK" => {
            let k = value_obj.extract::<usize>().map_err(|_| {
                PyTypeError::new_err("pipeline.stopping['KnownK'] must be an integer")
            })?;
            if k == 0 {
                return Err(PyValueError::new_err(
                    "pipeline.stopping['KnownK'] must be >= 1",
                ));
            }
            Ok(Stopping::KnownK(k))
        }
        "Penalized" => Ok(Stopping::Penalized(parse_pipeline_penalty_compat(
            &value_obj,
        )?)),
        "PenaltyPath" => {
            let penalties = value_obj.downcast::<PyList>().map_err(|_| {
                PyTypeError::new_err("pipeline.stopping['PenaltyPath'] must be a list")
            })?;
            if penalties.is_empty() {
                return Err(PyValueError::new_err(
                    "pipeline.stopping['PenaltyPath'] must not be empty",
                ));
            }
            let mut path = Vec::with_capacity(penalties.len());
            for penalty in penalties.iter() {
                path.push(parse_pipeline_penalty_compat(&penalty)?);
            }
            Ok(Stopping::PenaltyPath(path))
        }
        _ => Err(PyValueError::new_err(format!(
            "unsupported pipeline.stopping serde key '{key}'; expected one of: 'KnownK', 'Penalized', 'PenaltyPath'"
        ))),
    }
}

fn parse_pipeline_pelt_config(dict: &Bound<'_, PyDict>, context: &str) -> PyResult<PeltConfig> {
    let mut config = PeltConfig::default();
    for (key_obj, value_obj) in dict.iter() {
        let key: String = key_obj
            .extract()
            .map_err(|_| PyTypeError::new_err(format!("{context} keys must be strings")))?;
        match key.as_str() {
            "kind" => {}
            "stopping" => config.stopping = parse_pipeline_stopping_compat(Some(&value_obj))?,
            "params_per_segment" => config.params_per_segment = value_obj.extract::<usize>()?,
            "cancel_check_every" => config.cancel_check_every = value_obj.extract::<usize>()?,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported {context} key '{key}' for detector kind='pelt'"
                )));
            }
        }
    }
    Ok(config)
}

fn parse_pipeline_binseg_config(dict: &Bound<'_, PyDict>, context: &str) -> PyResult<BinSegConfig> {
    let mut config = BinSegConfig::default();
    for (key_obj, value_obj) in dict.iter() {
        let key: String = key_obj
            .extract()
            .map_err(|_| PyTypeError::new_err(format!("{context} keys must be strings")))?;
        match key.as_str() {
            "kind" => {}
            "stopping" => config.stopping = parse_pipeline_stopping_compat(Some(&value_obj))?,
            "params_per_segment" => config.params_per_segment = value_obj.extract::<usize>()?,
            "cancel_check_every" => config.cancel_check_every = value_obj.extract::<usize>()?,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported {context} key '{key}' for detector kind='binseg'"
                )));
            }
        }
    }
    Ok(config)
}

fn parse_pipeline_fpop_config(dict: &Bound<'_, PyDict>, context: &str) -> PyResult<FpopConfig> {
    let mut config = FpopConfig::default();
    for (key_obj, value_obj) in dict.iter() {
        let key: String = key_obj
            .extract()
            .map_err(|_| PyTypeError::new_err(format!("{context} keys must be strings")))?;
        match key.as_str() {
            "kind" => {}
            "stopping" => config.stopping = parse_pipeline_stopping_compat(Some(&value_obj))?,
            "params_per_segment" => config.params_per_segment = value_obj.extract::<usize>()?,
            "cancel_check_every" => config.cancel_check_every = value_obj.extract::<usize>()?,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported {context} key '{key}' for detector kind='fpop'"
                )));
            }
        }
    }
    Ok(config)
}

fn parse_pipeline_wbs_config(dict: &Bound<'_, PyDict>, context: &str) -> PyResult<WbsConfig> {
    let mut config = WbsConfig::default();
    for (key_obj, value_obj) in dict.iter() {
        let key: String = key_obj
            .extract()
            .map_err(|_| PyTypeError::new_err(format!("{context} keys must be strings")))?;
        match key.as_str() {
            "kind" => {}
            "stopping" => config.stopping = parse_pipeline_stopping_compat(Some(&value_obj))?,
            "params_per_segment" => config.params_per_segment = value_obj.extract::<usize>()?,
            "cancel_check_every" => config.cancel_check_every = value_obj.extract::<usize>()?,
            "num_intervals" => config.num_intervals = value_obj.extract::<Option<usize>>()?,
            "interval_strategy" => {
                let raw = value_obj.extract::<String>().map_err(|_| {
                    PyTypeError::new_err("pipeline.detector.interval_strategy must be a string")
                })?;
                config.interval_strategy = parse_pipeline_interval_strategy(&raw)?;
            }
            "seed" => config.seed = value_obj.extract::<u64>()?,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported {context} key '{key}' for detector kind='wbs'"
                )));
            }
        }
    }
    Ok(config)
}

fn parse_pipeline_detector_kind(
    kind: &str,
    dict: Option<&Bound<'_, PyDict>>,
) -> PyResult<DoctorOfflineDetectorConfig> {
    match kind.to_ascii_lowercase().as_str() {
        "pelt" => match dict {
            Some(dict) => Ok(DoctorOfflineDetectorConfig::Pelt(
                parse_pipeline_pelt_config(dict, "pipeline.detector")?,
            )),
            None => Ok(DoctorOfflineDetectorConfig::Pelt(PeltConfig::default())),
        },
        "binseg" => match dict {
            Some(dict) => Ok(DoctorOfflineDetectorConfig::BinSeg(
                parse_pipeline_binseg_config(dict, "pipeline.detector")?,
            )),
            None => Ok(DoctorOfflineDetectorConfig::BinSeg(BinSegConfig::default())),
        },
        "fpop" => match dict {
            Some(dict) => Ok(DoctorOfflineDetectorConfig::Fpop(
                parse_pipeline_fpop_config(dict, "pipeline.detector")?,
            )),
            None => Ok(DoctorOfflineDetectorConfig::Fpop(FpopConfig::default())),
        },
        "wbs" => match dict {
            Some(dict) => Ok(DoctorOfflineDetectorConfig::Wbs(parse_pipeline_wbs_config(
                dict,
                "pipeline.detector",
            )?)),
            None => Ok(DoctorOfflineDetectorConfig::Wbs(WbsConfig::default())),
        },
        _ => Err(PyValueError::new_err(format!(
            "unsupported pipeline.detector.kind '{kind}'; expected one of: 'pelt', 'binseg', 'fpop', 'wbs'"
        ))),
    }
}

fn parse_pipeline_detector_serde(
    value: &Bound<'_, PyAny>,
) -> PyResult<DoctorOfflineDetectorConfig> {
    let dict = value
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("pipeline.detector serde form must be a dict"))?;
    if dict.len() != 1 {
        return Err(PyValueError::new_err(
            "pipeline.detector serde form must contain exactly one key: 'Offline' or 'Online'",
        ));
    }

    if let Some(offline_value) = dict.get_item("Offline")? {
        let offline = offline_value
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("pipeline.detector['Offline'] must be a dict"))?;
        if offline.len() != 1 {
            return Err(PyValueError::new_err(
                "pipeline.detector['Offline'] must contain exactly one key: 'Pelt', 'BinSeg', 'Fpop', or 'Wbs'",
            ));
        }

        let (variant_obj, config_obj) = offline
            .iter()
            .next()
            .expect("offline detector dict len checked to be one");
        let variant: String = variant_obj.extract().map_err(|_| {
            PyTypeError::new_err("pipeline.detector['Offline'] keys must be strings")
        })?;
        let config = config_obj.downcast::<PyDict>().map_err(|_| {
            PyTypeError::new_err(format!(
                "pipeline.detector['Offline']['{variant}'] must be a dict"
            ))
        })?;
        return match variant.to_ascii_lowercase().as_str() {
            "pelt" => Ok(DoctorOfflineDetectorConfig::Pelt(
                parse_pipeline_pelt_config(&config, "pipeline.detector['Offline']['Pelt']")?,
            )),
            "binseg" => Ok(DoctorOfflineDetectorConfig::BinSeg(
                parse_pipeline_binseg_config(&config, "pipeline.detector['Offline']['BinSeg']")?,
            )),
            "fpop" => Ok(DoctorOfflineDetectorConfig::Fpop(
                parse_pipeline_fpop_config(&config, "pipeline.detector['Offline']['Fpop']")?,
            )),
            "wbs" => Ok(DoctorOfflineDetectorConfig::Wbs(parse_pipeline_wbs_config(
                &config,
                "pipeline.detector['Offline']['Wbs']",
            )?)),
            _ => Err(PyValueError::new_err(format!(
                "unsupported pipeline.detector['Offline'] variant '{variant}'; expected one of: 'Pelt', 'BinSeg', 'Fpop', 'Wbs'"
            ))),
        };
    }

    if dict.get_item("Online")?.is_some() {
        return Err(PyValueError::new_err(
            "detect_offline does not accept online pipeline detectors",
        ));
    }

    Err(PyValueError::new_err(
        "unsupported pipeline.detector serde form; expected one of: {'Offline': {...}} or {'Online': {...}}",
    ))
}

fn parse_pipeline_detector(value: &Bound<'_, PyAny>) -> PyResult<DoctorOfflineDetectorConfig> {
    if let Ok(kind) = value.extract::<String>() {
        return parse_pipeline_detector_kind(&kind, None);
    }

    let dict = value
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("pipeline.detector must be a string or dict"))?;
    if dict.get_item("kind")?.is_some() {
        let kind_value = required_dict_key(dict, "kind", "pipeline.detector")?;
        let kind: String = kind_value
            .extract()
            .map_err(|_| PyTypeError::new_err("pipeline.detector.kind must be a string"))?;
        return parse_pipeline_detector_kind(&kind, Some(dict));
    }

    parse_pipeline_detector_serde(value)
}

fn pipeline_detector_stopping(detector: &DoctorOfflineDetectorConfig) -> Stopping {
    match detector {
        DoctorOfflineDetectorConfig::Pelt(config) => config.stopping.clone(),
        DoctorOfflineDetectorConfig::BinSeg(config) => config.stopping.clone(),
        DoctorOfflineDetectorConfig::Fpop(config) => config.stopping.clone(),
        DoctorOfflineDetectorConfig::Wbs(config) => config.stopping.clone(),
    }
}

fn pipeline_detector_seed(detector: &DoctorOfflineDetectorConfig) -> Option<u64> {
    match detector {
        DoctorOfflineDetectorConfig::Wbs(config) => Some(config.seed),
        DoctorOfflineDetectorConfig::Pelt(_)
        | DoctorOfflineDetectorConfig::BinSeg(_)
        | DoctorOfflineDetectorConfig::Fpop(_) => None,
    }
}

fn apply_pipeline_controls(
    detector: &mut DoctorOfflineDetectorConfig,
    stopping: &Stopping,
    seed: Option<u64>,
) -> PyResult<()> {
    match detector {
        DoctorOfflineDetectorConfig::Pelt(config) => {
            config.stopping = stopping.clone();
            if seed.is_some() {
                return Err(PyValueError::new_err(
                    "pipeline.seed is only supported for detector='wbs'",
                ));
            }
        }
        DoctorOfflineDetectorConfig::BinSeg(config) => {
            config.stopping = stopping.clone();
            if seed.is_some() {
                return Err(PyValueError::new_err(
                    "pipeline.seed is only supported for detector='wbs'",
                ));
            }
        }
        DoctorOfflineDetectorConfig::Fpop(config) => {
            config.stopping = stopping.clone();
            if seed.is_some() {
                return Err(PyValueError::new_err(
                    "pipeline.seed is only supported for detector='wbs'",
                ));
            }
        }
        DoctorOfflineDetectorConfig::Wbs(config) => {
            config.stopping = stopping.clone();
            if let Some(seed_value) = seed {
                config.seed = seed_value;
            }
        }
    }
    Ok(())
}

fn parse_pipeline_cost(value: &Bound<'_, PyAny>) -> PyResult<DoctorCostConfig> {
    let raw = value
        .extract::<String>()
        .map_err(|_| PyTypeError::new_err("pipeline.cost must be a string"))?;
    match raw.to_ascii_lowercase().as_str() {
        "cosine" => Ok(DoctorCostConfig::Cosine),
        "l1" | "l1_median" | "l1median" => Ok(DoctorCostConfig::L1Median),
        "l2" => Ok(DoctorCostConfig::L2),
        "normal" => Ok(DoctorCostConfig::Normal),
        "normal_full_cov" | "normal_fullcov" | "normalfullcov" => {
            Ok(DoctorCostConfig::NormalFullCov)
        }
        "nig" => Ok(DoctorCostConfig::Nig),
        "rank" => Ok(DoctorCostConfig::Rank),
        "none" => Ok(DoctorCostConfig::None),
        _ => Err(PyValueError::new_err(format!(
            "unsupported pipeline.cost '{raw}'; expected one of: 'cosine', 'l1_median', 'l2', 'normal', 'normal_full_cov', 'nig', 'rank', 'none'"
        ))),
    }
}

fn parse_pipeline_spec(pipeline: &Bound<'_, PyAny>) -> PyResult<DoctorPipelineSpec> {
    let dict = pipeline
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("pipeline must be a dict"))?;
    reject_unknown_keys(
        dict,
        "pipeline",
        &[
            "detector",
            "cost",
            "constraints",
            "stopping",
            "preprocess",
            "seed",
        ],
    )?;

    let detector_value = required_dict_key(dict, "detector", "pipeline")?;
    let mut detector = parse_pipeline_detector(&detector_value)?;

    let cost = match dict.get_item("cost")? {
        Some(value) if !value.is_none() => parse_pipeline_cost(&value)?,
        _ => DoctorCostConfig::L2,
    };
    if matches!(cost, DoctorCostConfig::None) {
        return Err(PyValueError::new_err(
            "detect_offline requires pipeline.cost to be one of: 'cosine', 'l1_median', 'l2', 'normal', 'normal_full_cov', 'nig', 'rank'",
        ));
    }
    if matches!(&detector, DoctorOfflineDetectorConfig::Fpop(_))
        && !matches!(cost, DoctorCostConfig::L2)
    {
        return Err(PyValueError::new_err(
            "pipeline.detector='fpop' requires pipeline.cost='l2'",
        ));
    }
    let constraints = parse_constraints(dict.get_item("constraints")?.as_ref())?;
    let stopping = match dict.get_item("stopping")? {
        Some(value) if !value.is_none() => parse_pipeline_stopping_compat(Some(&value))?,
        _ => pipeline_detector_stopping(&detector),
    };
    let seed =
        match dict.get_item("seed")? {
            Some(value) if !value.is_none() => Some(value.extract::<u64>().map_err(|_| {
                PyTypeError::new_err("pipeline.seed must be an integer when provided")
            })?),
            _ => pipeline_detector_seed(&detector),
        };
    apply_pipeline_controls(&mut detector, &stopping, seed)?;

    #[cfg(feature = "preprocess")]
    let preprocess = match dict.get_item("preprocess")? {
        Some(value) if !value.is_none() => {
            parse_preprocess(Some(&value))?.map(|p| p.config().clone())
        }
        _ => None,
    };
    #[cfg(not(feature = "preprocess"))]
    let preprocess = {
        parse_preprocess(dict.get_item("preprocess")?.as_ref())?;
        None
    };

    Ok(DoctorPipelineSpec {
        detector: DoctorDetectorConfig::Offline(detector),
        cost,
        preprocess,
        constraints,
        stopping: Some(stopping),
        seed,
    })
}

#[cfg(feature = "preprocess")]
fn parse_detrend_config(value: &Bound<'_, PyAny>) -> PyResult<Option<DetrendConfig>> {
    if value.is_none() {
        return Ok(None);
    }

    let dict = value
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("preprocess.detrend must be a dict or None"))?;

    let mut method: Option<String> = None;
    let mut degree: Option<usize> = None;

    for (key_obj, value_obj) in dict.iter() {
        let key: String = key_obj
            .extract()
            .map_err(|_| PyTypeError::new_err("preprocess.detrend keys must be strings"))?;
        match key.as_str() {
            "method" => {
                method = Some(value_obj.extract::<String>().map_err(|_| {
                    PyTypeError::new_err("preprocess.detrend.method must be a string")
                })?)
            }
            "degree" => {
                degree = Some(value_obj.extract::<usize>().map_err(|_| {
                    PyTypeError::new_err("preprocess.detrend.degree must be an integer")
                })?)
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported preprocess.detrend key '{key}'"
                )));
            }
        }
    }

    let method = method.ok_or_else(|| {
        PyValueError::new_err("preprocess.detrend requires key 'method' with a valid value")
    })?;

    let method = match method.to_ascii_lowercase().as_str() {
        "linear" => {
            if degree.is_some() {
                return Err(PyValueError::new_err(
                    "preprocess.detrend.method='linear' does not accept key 'degree'",
                ));
            }
            DetrendMethod::Linear
        }
        "polynomial" => {
            let degree = degree.ok_or_else(|| {
                PyValueError::new_err(
                    "preprocess.detrend.method='polynomial' requires key 'degree'",
                )
            })?;
            DetrendMethod::Polynomial { degree }
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported preprocess.detrend.method '{method}'; expected one of: 'linear', 'polynomial'"
            )));
        }
    };

    Ok(Some(DetrendConfig { method }))
}

#[cfg(feature = "preprocess")]
fn parse_deseasonalize_config(value: &Bound<'_, PyAny>) -> PyResult<Option<DeseasonalizeConfig>> {
    if value.is_none() {
        return Ok(None);
    }

    let dict = value
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("preprocess.deseasonalize must be a dict or None"))?;

    let mut method: Option<String> = None;
    let mut period: Option<usize> = None;

    for (key_obj, value_obj) in dict.iter() {
        let key: String = key_obj
            .extract()
            .map_err(|_| PyTypeError::new_err("preprocess.deseasonalize keys must be strings"))?;
        match key.as_str() {
            "method" => {
                method = Some(value_obj.extract::<String>().map_err(|_| {
                    PyTypeError::new_err("preprocess.deseasonalize.method must be a string")
                })?)
            }
            "period" => {
                period = Some(value_obj.extract::<usize>().map_err(|_| {
                    PyTypeError::new_err("preprocess.deseasonalize.period must be an integer")
                })?)
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported preprocess.deseasonalize key '{key}'"
                )));
            }
        }
    }

    let method = method.ok_or_else(|| {
        PyValueError::new_err("preprocess.deseasonalize requires key 'method' with a valid value")
    })?;
    let period = period.ok_or_else(|| {
        PyValueError::new_err("preprocess.deseasonalize requires key 'period' with a valid value")
    })?;

    let method = match method.to_ascii_lowercase().as_str() {
        "differencing" => DeseasonalizeMethod::Differencing { period },
        "stl_like" => DeseasonalizeMethod::StlLike { period },
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported preprocess.deseasonalize.method '{method}'; expected one of: 'differencing', 'stl_like'"
            )));
        }
    };

    Ok(Some(DeseasonalizeConfig { method }))
}

#[cfg(feature = "preprocess")]
fn parse_winsorize_config(value: &Bound<'_, PyAny>) -> PyResult<Option<WinsorizeConfig>> {
    if value.is_none() {
        return Ok(None);
    }

    let dict = value
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("preprocess.winsorize must be a dict or None"))?;

    let mut config = WinsorizeConfig::default();
    for (key_obj, value_obj) in dict.iter() {
        let key: String = key_obj
            .extract()
            .map_err(|_| PyTypeError::new_err("preprocess.winsorize keys must be strings"))?;
        match key.as_str() {
            "lower_quantile" => {
                config.lower_quantile = value_obj.extract::<f64>().map_err(|_| {
                    PyTypeError::new_err("preprocess.winsorize.lower_quantile must be a float")
                })?;
            }
            "upper_quantile" => {
                config.upper_quantile = value_obj.extract::<f64>().map_err(|_| {
                    PyTypeError::new_err("preprocess.winsorize.upper_quantile must be a float")
                })?;
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported preprocess.winsorize key '{key}'"
                )));
            }
        }
    }

    Ok(Some(config))
}

#[cfg(feature = "preprocess")]
fn parse_robust_scale_config(value: &Bound<'_, PyAny>) -> PyResult<Option<RobustScaleConfig>> {
    if value.is_none() {
        return Ok(None);
    }

    let dict = value
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("preprocess.robust_scale must be a dict or None"))?;

    let mut config = RobustScaleConfig::default();
    for (key_obj, value_obj) in dict.iter() {
        let key: String = key_obj
            .extract()
            .map_err(|_| PyTypeError::new_err("preprocess.robust_scale keys must be strings"))?;
        match key.as_str() {
            "mad_epsilon" => {
                config.mad_epsilon = value_obj.extract::<f64>().map_err(|_| {
                    PyTypeError::new_err("preprocess.robust_scale.mad_epsilon must be a float")
                })?;
            }
            "normal_consistency" => {
                config.normal_consistency = value_obj.extract::<f64>().map_err(|_| {
                    PyTypeError::new_err(
                        "preprocess.robust_scale.normal_consistency must be a float",
                    )
                })?;
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported preprocess.robust_scale key '{key}'"
                )));
            }
        }
    }

    Ok(Some(config))
}

#[cfg(feature = "preprocess")]
fn preprocess_notes_from_reports(reports: &[cpd_preprocess::StepReport]) -> Vec<String> {
    reports
        .iter()
        .map(|report| {
            if report.notes.is_empty() {
                format!("preprocess:{} applied", report.step)
            } else {
                format!("preprocess:{} {}", report.step, report.notes.join("; "))
            }
        })
        .collect()
}

fn detect_with_view(
    detector: PyDetectorKind,
    model: PyCostModel,
    view: &TimeSeriesView<'_>,
    constraints: &Constraints,
    stopping: Stopping,
    repro_mode: ReproMode,
) -> Result<CoreOfflineChangePointResult, CpdError> {
    let ctx = ExecutionContext::new(constraints).with_repro_mode(repro_mode);

    match (detector, model) {
        (PyDetectorKind::Pelt, PyCostModel::Cosine) => {
            let config = PeltConfig {
                stopping,
                params_per_segment: 2,
                cancel_check_every: 1000,
            };
            let detector = OfflinePelt::new(CostCosine::new(repro_mode), config)?;
            detector.detect(view, &ctx)
        }
        (PyDetectorKind::Pelt, PyCostModel::L1Median) => {
            let config = PeltConfig {
                stopping,
                params_per_segment: 2,
                cancel_check_every: 1000,
            };
            let detector = OfflinePelt::new(CostL1Median::new(repro_mode), config)?;
            detector.detect(view, &ctx)
        }
        (PyDetectorKind::Pelt, PyCostModel::L2) => {
            let config = PeltConfig {
                stopping,
                params_per_segment: 2,
                cancel_check_every: 1000,
            };
            let detector = OfflinePelt::new(CostL2Mean::new(repro_mode), config)?;
            detector.detect(view, &ctx)
        }
        (PyDetectorKind::Pelt, PyCostModel::Normal) => {
            let config = PeltConfig {
                stopping,
                params_per_segment: PeltConfig::default().params_per_segment,
                cancel_check_every: 1000,
            };
            let detector = OfflinePelt::new(CostNormalMeanVar::new(repro_mode), config)?;
            detector.detect(view, &ctx)
        }
        (PyDetectorKind::Pelt, PyCostModel::NormalFullCov) => {
            let config = PeltConfig {
                stopping,
                params_per_segment: PeltConfig::default().params_per_segment,
                cancel_check_every: 1000,
            };
            let detector = OfflinePelt::new(CostNormalFullCov::new(repro_mode), config)?;
            detector.detect(view, &ctx)
        }
        (PyDetectorKind::Pelt, PyCostModel::Rank) => {
            let config = PeltConfig {
                stopping,
                params_per_segment: 2,
                cancel_check_every: 1000,
            };
            let detector = OfflinePelt::new(CostRank::new(repro_mode), config)?;
            detector.detect(view, &ctx)
        }
        (PyDetectorKind::Binseg, PyCostModel::L2) => {
            let config = BinSegConfig {
                stopping,
                params_per_segment: 2,
                cancel_check_every: 1000,
            };
            let detector = OfflineBinSeg::new(CostL2Mean::new(repro_mode), config)?;
            detector.detect(view, &ctx)
        }
        (PyDetectorKind::Binseg, PyCostModel::Cosine) => {
            let config = BinSegConfig {
                stopping,
                params_per_segment: 2,
                cancel_check_every: 1000,
            };
            let detector = OfflineBinSeg::new(CostCosine::new(repro_mode), config)?;
            detector.detect(view, &ctx)
        }
        (PyDetectorKind::Binseg, PyCostModel::L1Median) => {
            let config = BinSegConfig {
                stopping,
                params_per_segment: 2,
                cancel_check_every: 1000,
            };
            let detector = OfflineBinSeg::new(CostL1Median::new(repro_mode), config)?;
            detector.detect(view, &ctx)
        }
        (PyDetectorKind::Binseg, PyCostModel::Normal) => {
            let config = BinSegConfig {
                stopping,
                params_per_segment: BinSegConfig::default().params_per_segment,
                cancel_check_every: 1000,
            };
            let detector = OfflineBinSeg::new(CostNormalMeanVar::new(repro_mode), config)?;
            detector.detect(view, &ctx)
        }
        (PyDetectorKind::Binseg, PyCostModel::NormalFullCov) => {
            let config = BinSegConfig {
                stopping,
                params_per_segment: BinSegConfig::default().params_per_segment,
                cancel_check_every: 1000,
            };
            let detector = OfflineBinSeg::new(CostNormalFullCov::new(repro_mode), config)?;
            detector.detect(view, &ctx)
        }
        (PyDetectorKind::Binseg, PyCostModel::Rank) => {
            let config = BinSegConfig {
                stopping,
                params_per_segment: 2,
                cancel_check_every: 1000,
            };
            let detector = OfflineBinSeg::new(CostRank::new(repro_mode), config)?;
            detector.detect(view, &ctx)
        }
        (PyDetectorKind::Fpop, PyCostModel::L2) => {
            let config = FpopConfig {
                stopping,
                params_per_segment: 2,
                cancel_check_every: 1000,
            };
            let detector = OfflineFpop::new(CostL2Mean::new(repro_mode), config)?;
            detector.detect(view, &ctx)
        }
        (
            PyDetectorKind::Fpop,
            PyCostModel::Cosine
            | PyCostModel::L1Median
            | PyCostModel::Normal
            | PyCostModel::NormalFullCov
            | PyCostModel::Rank,
        ) => Err(CpdError::invalid_input(
            "detector='fpop' requires cost='l2'",
        )),
    }
}

fn detect_with_spec(
    detector: PyDetectorKind,
    model: PyCostModel,
    series: &OwnedSeries,
    constraints: &Constraints,
    stopping: Stopping,
    repro_mode: ReproMode,
) -> Result<CoreOfflineChangePointResult, CpdError> {
    let view = series.view()?;
    detect_with_view(detector, model, &view, constraints, stopping, repro_mode)
}

#[cfg(feature = "fuzzing")]
pub fn fuzz_detect_offline_numpy_case(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    time: Option<&Bound<'_, PyAny>>,
    detector: &str,
    cost: &str,
    pen: Option<f64>,
) -> PyResult<()> {
    let detector = PyDetectorKind::parse(detector)?;
    let cost = PyCostModel::parse(cost)?;
    let parsed = parse_numpy_series(py, x, time, MissingPolicy::Error, DTypePolicy::KeepInput)?;
    let owned = parsed.into_owned().map_err(cpd_error_to_pyerr)?;

    let stopping = match pen {
        Some(beta) if beta.is_finite() && beta > 0.0 => Stopping::Penalized(Penalty::Manual(beta)),
        _ => Stopping::Penalized(Penalty::BIC),
    };

    let _ = py
        .allow_threads(|| {
            detect_with_spec(
                detector,
                cost,
                &owned,
                &Constraints::default(),
                stopping,
                ReproMode::Balanced,
            )
        })
        .map_err(cpd_error_to_pyerr)?;

    Ok(())
}

/// High-level ruptures-like Python interface for offline PELT detection.
#[pyclass(module = "cpd._cpd_rs", name = "Pelt")]
#[derive(Clone, Debug)]
pub struct PyPelt {
    model: PyCostModel,
    min_segment_len: usize,
    jump: usize,
    max_change_points: Option<usize>,
    fitted: Option<OwnedSeries>,
}

#[pymethods]
impl PyPelt {
    #[new]
    #[pyo3(signature = (model = "l2", min_segment_len = 2, jump = 1, max_change_points = None))]
    fn new(
        model: &str,
        min_segment_len: usize,
        jump: usize,
        max_change_points: Option<usize>,
    ) -> PyResult<Self> {
        let model = PyCostModel::parse(model)?;
        if min_segment_len == 0 {
            return Err(PyValueError::new_err("min_segment_len must be >= 1; got 0"));
        }
        if jump == 0 {
            return Err(PyValueError::new_err("jump must be >= 1; got 0"));
        }

        Ok(Self {
            model,
            min_segment_len,
            jump,
            max_change_points,
            fitted: None,
        })
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.fitted = Some(parse_owned_series(py, x)?);
        Ok(slf)
    }

    #[pyo3(signature = (*, pen = None, n_bkps = None))]
    fn predict(
        &self,
        py: Python<'_>,
        pen: Option<f64>,
        n_bkps: Option<usize>,
    ) -> PyResult<PyOfflineChangePointResult> {
        let stopping = resolve_stopping(pen, n_bkps, "predict()")?;
        let fitted = self.fitted.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("fit(...) must be called before predict(...)")
        })?;
        let constraints = Constraints {
            min_segment_len: self.min_segment_len,
            jump: self.jump,
            max_change_points: self.max_change_points,
            ..Constraints::default()
        };

        let mut result = py
            .allow_threads(|| {
                detect_with_spec(
                    PyDetectorKind::Pelt,
                    self.model,
                    fitted,
                    &constraints,
                    stopping,
                    ReproMode::Balanced,
                )
            })
            .map_err(cpd_error_to_pyerr)?;

        for note in fitted.diagnostics() {
            result.diagnostics.notes.push(format!("fit: {note}"));
        }

        Ok(result.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "Pelt(model='{}', min_segment_len={}, jump={}, max_change_points={:?}, fitted={})",
            self.model.cost_model_name(),
            self.min_segment_len,
            self.jump,
            self.max_change_points,
            self.fitted.is_some()
        )
    }
}

/// High-level ruptures-like Python interface for offline BinSeg detection.
#[pyclass(module = "cpd._cpd_rs", name = "Binseg")]
#[derive(Clone, Debug)]
pub struct PyBinseg {
    model: PyCostModel,
    min_segment_len: usize,
    jump: usize,
    max_change_points: Option<usize>,
    max_depth: Option<usize>,
    fitted: Option<OwnedSeries>,
}

#[pymethods]
impl PyBinseg {
    #[new]
    #[pyo3(signature = (model = "l2", min_segment_len = 2, jump = 1, max_change_points = None, max_depth = None))]
    fn new(
        model: &str,
        min_segment_len: usize,
        jump: usize,
        max_change_points: Option<usize>,
        max_depth: Option<usize>,
    ) -> PyResult<Self> {
        let model = PyCostModel::parse(model)?;
        if min_segment_len == 0 {
            return Err(PyValueError::new_err("min_segment_len must be >= 1; got 0"));
        }
        if jump == 0 {
            return Err(PyValueError::new_err("jump must be >= 1; got 0"));
        }
        if matches!(max_depth, Some(0)) {
            return Err(PyValueError::new_err(
                "max_depth must be >= 1 when provided",
            ));
        }

        Ok(Self {
            model,
            min_segment_len,
            jump,
            max_change_points,
            max_depth,
            fitted: None,
        })
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.fitted = Some(parse_owned_series(py, x)?);
        Ok(slf)
    }

    #[pyo3(signature = (*, pen = None, n_bkps = None))]
    fn predict(
        &self,
        py: Python<'_>,
        pen: Option<f64>,
        n_bkps: Option<usize>,
    ) -> PyResult<PyOfflineChangePointResult> {
        let stopping = resolve_stopping(pen, n_bkps, "predict()")?;
        let fitted = self.fitted.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("fit(...) must be called before predict(...)")
        })?;
        let constraints = Constraints {
            min_segment_len: self.min_segment_len,
            jump: self.jump,
            max_change_points: self.max_change_points,
            max_depth: self.max_depth,
            ..Constraints::default()
        };

        let mut result = py
            .allow_threads(|| {
                detect_with_spec(
                    PyDetectorKind::Binseg,
                    self.model,
                    fitted,
                    &constraints,
                    stopping,
                    ReproMode::Balanced,
                )
            })
            .map_err(cpd_error_to_pyerr)?;

        for note in fitted.diagnostics() {
            result.diagnostics.notes.push(format!("fit: {note}"));
        }

        Ok(result.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "Binseg(model='{}', min_segment_len={}, jump={}, max_change_points={:?}, max_depth={:?}, fitted={})",
            self.model.cost_model_name(),
            self.min_segment_len,
            self.jump,
            self.max_change_points,
            self.max_depth,
            self.fitted.is_some()
        )
    }
}

/// High-level ruptures-like Python interface for offline FPOP detection.
#[pyclass(module = "cpd._cpd_rs", name = "Fpop")]
#[derive(Clone, Debug)]
pub struct PyFpop {
    min_segment_len: usize,
    jump: usize,
    max_change_points: Option<usize>,
    fitted: Option<OwnedSeries>,
}

#[pymethods]
impl PyFpop {
    #[new]
    #[pyo3(signature = (min_segment_len = 2, jump = 1, max_change_points = None))]
    fn new(
        min_segment_len: usize,
        jump: usize,
        max_change_points: Option<usize>,
    ) -> PyResult<Self> {
        if min_segment_len == 0 {
            return Err(PyValueError::new_err("min_segment_len must be >= 1; got 0"));
        }
        if jump == 0 {
            return Err(PyValueError::new_err("jump must be >= 1; got 0"));
        }

        Ok(Self {
            min_segment_len,
            jump,
            max_change_points,
            fitted: None,
        })
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.fitted = Some(parse_owned_series(py, x)?);
        Ok(slf)
    }

    #[pyo3(signature = (*, pen = None, n_bkps = None))]
    fn predict(
        &self,
        py: Python<'_>,
        pen: Option<f64>,
        n_bkps: Option<usize>,
    ) -> PyResult<PyOfflineChangePointResult> {
        let stopping = resolve_stopping(pen, n_bkps, "predict()")?;
        let fitted = self.fitted.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("fit(...) must be called before predict(...)")
        })?;
        let constraints = Constraints {
            min_segment_len: self.min_segment_len,
            jump: self.jump,
            max_change_points: self.max_change_points,
            ..Constraints::default()
        };

        let mut result = py
            .allow_threads(|| {
                detect_with_spec(
                    PyDetectorKind::Fpop,
                    PyCostModel::L2,
                    fitted,
                    &constraints,
                    stopping,
                    ReproMode::Balanced,
                )
            })
            .map_err(cpd_error_to_pyerr)?;

        for note in fitted.diagnostics() {
            result.diagnostics.notes.push(format!("fit: {note}"));
        }

        Ok(result.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "Fpop(min_segment_len={}, jump={}, max_change_points={:?}, fitted={})",
            self.min_segment_len,
            self.jump,
            self.max_change_points,
            self.fitted.is_some()
        )
    }
}

#[pyclass(module = "cpd._cpd_rs", name = "_BocpdState", frozen)]
#[derive(Clone, Debug)]
pub struct PyBocpdState {
    state: BocpdState,
}

impl From<BocpdState> for PyBocpdState {
    fn from(state: BocpdState) -> Self {
        Self { state }
    }
}

#[pyclass(module = "cpd._cpd_rs", name = "_CusumState", frozen)]
#[derive(Clone, Debug)]
pub struct PyCusumState {
    state: CusumState,
}

impl From<CusumState> for PyCusumState {
    fn from(state: CusumState) -> Self {
        Self { state }
    }
}

#[pyclass(module = "cpd._cpd_rs", name = "_PageHinkleyState", frozen)]
#[derive(Clone, Debug)]
pub struct PyPageHinkleyState {
    state: PageHinkleyState,
}

impl From<PageHinkleyState> for PyPageHinkleyState {
    fn from(state: PageHinkleyState) -> Self {
        Self { state }
    }
}

/// Stateful BOCPD detector for streaming updates.
#[pyclass(module = "cpd._cpd_rs", name = "Bocpd")]
#[derive(Clone, Debug)]
pub struct PyBocpd {
    detector: BocpdDetector,
}

#[pymethods]
impl PyBocpd {
    #[new]
    #[pyo3(signature = (model = "gaussian_nig", hazard = None, max_run_length = 2_000, alert_policy = None, late_data_policy = None))]
    fn new(
        model: &str,
        hazard: Option<&Bound<'_, PyAny>>,
        max_run_length: usize,
        alert_policy: Option<&Bound<'_, PyAny>>,
        late_data_policy: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut config = BocpdConfig::default();
        config.observation = parse_bocpd_observation_model(model)?;
        config.hazard = parse_bocpd_hazard(hazard)?;
        config.max_run_length = max_run_length;
        config.alert_policy = parse_alert_policy(alert_policy, config.alert_policy)?;
        config.late_data_policy = parse_late_data_policy(late_data_policy)?;
        let detector = BocpdDetector::new(config).map_err(cpd_error_to_pyerr)?;
        Ok(Self { detector })
    }

    #[pyo3(signature = (x_t, t_ns = None))]
    fn update(&mut self, x_t: f64, t_ns: Option<i64>) -> PyResult<PyOnlineStepResult> {
        online_update_scalar(&mut self.detector, x_t, t_ns)
    }

    fn update_many(
        &mut self,
        py: Python<'_>,
        x_batch: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<PyOnlineStepResult>> {
        online_update_many(py, &mut self.detector, x_batch)
    }

    fn reset(&mut self) {
        self.detector.reset();
    }

    #[cfg(feature = "serde")]
    #[pyo3(signature = (*, format = "bytes", path = None))]
    fn save_state(
        &self,
        py: Python<'_>,
        format: &str,
        path: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let format = PyCheckpointFormat::parse(format)?;
        let path = parse_checkpoint_path(path, "checkpoint")?;
        let payload_codec = format.payload_codec();
        if let Some(path) = path {
            save_bocpd_checkpoint_file(&self.detector, &path, payload_codec)
                .map_err(cpd_error_to_pyerr)?;
            return Ok(py.None());
        }

        let envelope =
            save_bocpd_checkpoint(&self.detector, payload_codec).map_err(cpd_error_to_pyerr)?;
        checkpoint_output(py, &envelope, format)
    }

    #[cfg(not(feature = "serde"))]
    fn save_state(&self) -> PyBocpdState {
        self.detector.save_state().into()
    }

    #[cfg(feature = "serde")]
    #[pyo3(signature = (state = None, *, format = None, path = None))]
    fn load_state(
        &mut self,
        py: Python<'_>,
        state: Option<&Bound<'_, PyAny>>,
        format: Option<&str>,
        path: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let path = parse_checkpoint_path(path, "checkpoint")?;
        if path.is_some() && state.is_some() {
            return Err(PyValueError::new_err(
                "load_state accepts either 'state' or 'path', not both",
            ));
        }

        if let Some(path) = path {
            if format.is_some() {
                return Err(PyValueError::new_err(
                    "load_state does not accept 'format' when loading from path",
                ));
            }
            let decoded_state: BocpdState =
                load_state_from_checkpoint_file(&path, BOCPD_DETECTOR_ID)
                    .map_err(cpd_error_to_pyerr)?;
            let observation = &self.detector.config().observation;
            if !bocpd_state_matches_observation(&decoded_state, observation) {
                return Err(PyValueError::new_err(format!(
                    "incompatible Bocpd state for model '{}': state run_stats variant does not match detector observation model",
                    bocpd_model_name(observation)
                )));
            }
            load_bocpd_checkpoint_file(&mut self.detector, &path).map_err(cpd_error_to_pyerr)?;
            return Ok(());
        }

        let state = state.ok_or_else(|| {
            PyValueError::new_err("load_state requires checkpoint 'state' bytes/json or 'path'")
        })?;

        if let Ok(legacy_state) = state.extract::<PyRef<'_, PyBocpdState>>() {
            let observation = &self.detector.config().observation;
            if !bocpd_state_matches_observation(&legacy_state.state, observation) {
                return Err(PyValueError::new_err(format!(
                    "incompatible Bocpd state for model '{}': state run_stats variant does not match detector observation model",
                    bocpd_model_name(observation)
                )));
            }
            self.detector.load_state(&legacy_state.state);
            return Ok(());
        }

        let format = PyCheckpointFormat::infer(format, state)?;
        let envelope = decode_checkpoint_input(py, state, format)?;
        let decoded_state: BocpdState =
            load_state_from_checkpoint_envelope(&envelope, BOCPD_DETECTOR_ID)
                .map_err(cpd_error_to_pyerr)?;
        let observation = &self.detector.config().observation;
        if !bocpd_state_matches_observation(&decoded_state, observation) {
            return Err(PyValueError::new_err(format!(
                "incompatible Bocpd state for model '{}': state run_stats variant does not match detector observation model",
                bocpd_model_name(observation)
            )));
        }
        load_bocpd_checkpoint(&mut self.detector, &envelope).map_err(cpd_error_to_pyerr)?;
        Ok(())
    }

    #[cfg(not(feature = "serde"))]
    fn load_state(&mut self, state: &PyBocpdState) -> PyResult<()> {
        let observation = &self.detector.config().observation;
        if !bocpd_state_matches_observation(&state.state, observation) {
            return Err(PyValueError::new_err(format!(
                "incompatible Bocpd state for model '{}': state run_stats variant does not match detector observation model",
                bocpd_model_name(observation)
            )));
        }
        self.detector.load_state(&state.state);
        Ok(())
    }

    fn __repr__(&self) -> String {
        let config = self.detector.config();
        format!(
            "Bocpd(model='{}', hazard='{}', max_run_length={})",
            bocpd_model_name(&config.observation),
            bocpd_hazard_name(&config.hazard),
            config.max_run_length
        )
    }
}

/// Stateful CUSUM detector for streaming updates.
#[pyclass(module = "cpd._cpd_rs", name = "Cusum")]
#[derive(Clone, Debug)]
pub struct PyCusum {
    detector: CusumDetector,
}

#[pymethods]
impl PyCusum {
    #[new]
    #[pyo3(signature = (drift = 0.0, threshold = 8.0, target_mean = 0.0, alert_policy = None, late_data_policy = None))]
    fn new(
        drift: f64,
        threshold: f64,
        target_mean: f64,
        alert_policy: Option<&Bound<'_, PyAny>>,
        late_data_policy: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut config = CusumConfig::default();
        config.drift = drift;
        config.threshold = threshold;
        config.target_mean = target_mean;
        config.alert_policy = parse_alert_policy(alert_policy, config.alert_policy)?;
        config.late_data_policy = parse_late_data_policy(late_data_policy)?;
        let detector = CusumDetector::new(config).map_err(cpd_error_to_pyerr)?;
        Ok(Self { detector })
    }

    #[pyo3(signature = (x_t, t_ns = None))]
    fn update(&mut self, x_t: f64, t_ns: Option<i64>) -> PyResult<PyOnlineStepResult> {
        online_update_scalar(&mut self.detector, x_t, t_ns)
    }

    fn update_many(
        &mut self,
        py: Python<'_>,
        x_batch: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<PyOnlineStepResult>> {
        online_update_many(py, &mut self.detector, x_batch)
    }

    fn reset(&mut self) {
        self.detector.reset();
    }

    #[cfg(feature = "serde")]
    #[pyo3(signature = (*, format = "bytes", path = None))]
    fn save_state(
        &self,
        py: Python<'_>,
        format: &str,
        path: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let format = PyCheckpointFormat::parse(format)?;
        let path = parse_checkpoint_path(path, "checkpoint")?;
        let payload_codec = format.payload_codec();
        if let Some(path) = path {
            save_cusum_checkpoint_file(&self.detector, &path, payload_codec)
                .map_err(cpd_error_to_pyerr)?;
            return Ok(py.None());
        }

        let envelope =
            save_cusum_checkpoint(&self.detector, payload_codec).map_err(cpd_error_to_pyerr)?;
        checkpoint_output(py, &envelope, format)
    }

    #[cfg(not(feature = "serde"))]
    fn save_state(&self) -> PyCusumState {
        self.detector.save_state().into()
    }

    #[cfg(feature = "serde")]
    #[pyo3(signature = (state = None, *, format = None, path = None))]
    fn load_state(
        &mut self,
        py: Python<'_>,
        state: Option<&Bound<'_, PyAny>>,
        format: Option<&str>,
        path: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let path = parse_checkpoint_path(path, "checkpoint")?;
        if path.is_some() && state.is_some() {
            return Err(PyValueError::new_err(
                "load_state accepts either 'state' or 'path', not both",
            ));
        }

        if let Some(path) = path {
            if format.is_some() {
                return Err(PyValueError::new_err(
                    "load_state does not accept 'format' when loading from path",
                ));
            }
            load_cusum_checkpoint_file(&mut self.detector, &path).map_err(cpd_error_to_pyerr)?;
            return Ok(());
        }

        let state = state.ok_or_else(|| {
            PyValueError::new_err("load_state requires checkpoint 'state' bytes/json or 'path'")
        })?;

        if let Ok(legacy_state) = state.extract::<PyRef<'_, PyCusumState>>() {
            self.detector.load_state(&legacy_state.state);
            return Ok(());
        }

        let format = PyCheckpointFormat::infer(format, state)?;
        let envelope = decode_checkpoint_input(py, state, format)?;
        load_cusum_checkpoint(&mut self.detector, &envelope).map_err(cpd_error_to_pyerr)?;
        Ok(())
    }

    #[cfg(not(feature = "serde"))]
    fn load_state(&mut self, state: &PyCusumState) {
        self.detector.load_state(&state.state);
    }

    fn __repr__(&self) -> String {
        let config = self.detector.config();
        format!(
            "Cusum(drift={}, threshold={}, target_mean={})",
            config.drift, config.threshold, config.target_mean
        )
    }
}

/// Stateful Page-Hinkley detector for streaming updates.
#[pyclass(module = "cpd._cpd_rs", name = "PageHinkley")]
#[derive(Clone, Debug)]
pub struct PyPageHinkley {
    detector: PageHinkleyDetector,
}

#[pymethods]
impl PyPageHinkley {
    #[new]
    #[pyo3(signature = (delta = 0.01, threshold = 8.0, initial_mean = 0.0, alert_policy = None, late_data_policy = None))]
    fn new(
        delta: f64,
        threshold: f64,
        initial_mean: f64,
        alert_policy: Option<&Bound<'_, PyAny>>,
        late_data_policy: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut config = PageHinkleyConfig::default();
        config.delta = delta;
        config.threshold = threshold;
        config.initial_mean = initial_mean;
        config.alert_policy = parse_alert_policy(alert_policy, config.alert_policy)?;
        config.late_data_policy = parse_late_data_policy(late_data_policy)?;
        let detector = PageHinkleyDetector::new(config).map_err(cpd_error_to_pyerr)?;
        Ok(Self { detector })
    }

    #[pyo3(signature = (x_t, t_ns = None))]
    fn update(&mut self, x_t: f64, t_ns: Option<i64>) -> PyResult<PyOnlineStepResult> {
        online_update_scalar(&mut self.detector, x_t, t_ns)
    }

    fn update_many(
        &mut self,
        py: Python<'_>,
        x_batch: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<PyOnlineStepResult>> {
        online_update_many(py, &mut self.detector, x_batch)
    }

    fn reset(&mut self) {
        self.detector.reset();
    }

    #[cfg(feature = "serde")]
    #[pyo3(signature = (*, format = "bytes", path = None))]
    fn save_state(
        &self,
        py: Python<'_>,
        format: &str,
        path: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let format = PyCheckpointFormat::parse(format)?;
        let path = parse_checkpoint_path(path, "checkpoint")?;
        let payload_codec = format.payload_codec();
        if let Some(path) = path {
            save_page_hinkley_checkpoint_file(&self.detector, &path, payload_codec)
                .map_err(cpd_error_to_pyerr)?;
            return Ok(py.None());
        }

        let envelope = save_page_hinkley_checkpoint(&self.detector, payload_codec)
            .map_err(cpd_error_to_pyerr)?;
        checkpoint_output(py, &envelope, format)
    }

    #[cfg(not(feature = "serde"))]
    fn save_state(&self) -> PyPageHinkleyState {
        self.detector.save_state().into()
    }

    #[cfg(feature = "serde")]
    #[pyo3(signature = (state = None, *, format = None, path = None))]
    fn load_state(
        &mut self,
        py: Python<'_>,
        state: Option<&Bound<'_, PyAny>>,
        format: Option<&str>,
        path: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let path = parse_checkpoint_path(path, "checkpoint")?;
        if path.is_some() && state.is_some() {
            return Err(PyValueError::new_err(
                "load_state accepts either 'state' or 'path', not both",
            ));
        }

        if let Some(path) = path {
            if format.is_some() {
                return Err(PyValueError::new_err(
                    "load_state does not accept 'format' when loading from path",
                ));
            }
            load_page_hinkley_checkpoint_file(&mut self.detector, &path)
                .map_err(cpd_error_to_pyerr)?;
            return Ok(());
        }

        let state = state.ok_or_else(|| {
            PyValueError::new_err("load_state requires checkpoint 'state' bytes/json or 'path'")
        })?;

        if let Ok(legacy_state) = state.extract::<PyRef<'_, PyPageHinkleyState>>() {
            self.detector.load_state(&legacy_state.state);
            return Ok(());
        }

        let format = PyCheckpointFormat::infer(format, state)?;
        let envelope = decode_checkpoint_input(py, state, format)?;
        load_page_hinkley_checkpoint(&mut self.detector, &envelope).map_err(cpd_error_to_pyerr)?;
        Ok(())
    }

    #[cfg(not(feature = "serde"))]
    fn load_state(&mut self, state: &PyPageHinkleyState) {
        self.detector.load_state(&state.state);
    }

    fn __repr__(&self) -> String {
        let config = self.detector.config();
        format!(
            "PageHinkley(delta={}, threshold={}, initial_mean={})",
            config.delta, config.threshold, config.initial_mean
        )
    }
}

/// Low-level power-user API for fully-specified offline detection.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (x, *, pipeline = None, detector = "pelt", cost = "l2", constraints = None, stopping = None, preprocess = None, repro_mode = "balanced", return_diagnostics = true))]
fn detect_offline(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    pipeline: Option<&Bound<'_, PyAny>>,
    detector: &str,
    cost: &str,
    constraints: Option<&Bound<'_, PyAny>>,
    stopping: Option<&Bound<'_, PyAny>>,
    preprocess: Option<&Bound<'_, PyAny>>,
    repro_mode: &str,
    return_diagnostics: bool,
) -> PyResult<PyOfflineChangePointResult> {
    let pipeline = match pipeline {
        Some(value) if !value.is_none() => Some(value),
        _ => None,
    };
    let constraints = match constraints {
        Some(value) if !value.is_none() => Some(value),
        _ => None,
    };
    let stopping = match stopping {
        Some(value) if !value.is_none() => Some(value),
        _ => None,
    };
    let preprocess = match preprocess {
        Some(value) if !value.is_none() => Some(value),
        _ => None,
    };

    if pipeline.is_some() {
        let custom_legacy_args = !detector.eq_ignore_ascii_case("pelt")
            || !cost.eq_ignore_ascii_case("l2")
            || constraints.is_some()
            || stopping.is_some()
            || preprocess.is_some();
        if custom_legacy_args {
            return Err(PyValueError::new_err(
                "detect_offline accepts either pipeline=... or explicit detector/cost/constraints/stopping/preprocess arguments, not both",
            ));
        }
    }

    let repro_mode = parse_repro_mode(repro_mode)?;
    let owned = parse_owned_series(py, x)?;

    if let Some(pipeline) = pipeline {
        let pipeline = parse_pipeline_spec(pipeline)?;
        let mut result = py
            .allow_threads(|| {
                let view = owned.view()?;
                execute_doctor_pipeline_with_repro_mode(&view, &pipeline, repro_mode)
            })
            .map_err(cpd_error_to_pyerr)?;

        if !return_diagnostics {
            result.diagnostics.notes.clear();
            result.diagnostics.warnings.clear();
        }
        for note in owned.diagnostics() {
            result.diagnostics.notes.push(format!("input: {note}"));
        }
        return Ok(result.into());
    }

    let detector = PyDetectorKind::parse(detector)?;
    let cost = PyCostModel::parse(cost)?;
    let constraints = parse_constraints(constraints)?;
    let stopping = parse_stopping(stopping)?;
    #[cfg(feature = "preprocess")]
    let preprocess_pipeline = parse_preprocess(preprocess)?;
    #[cfg(not(feature = "preprocess"))]
    parse_preprocess(preprocess)?;

    #[cfg(feature = "preprocess")]
    let (mut result, preprocess_notes) = py
        .allow_threads(|| {
            let view = owned.view()?;
            if let Some(pipeline) = preprocess_pipeline.as_ref() {
                let preprocessed = pipeline.apply(&view)?;
                let notes = preprocess_notes_from_reports(preprocessed.reports());
                let preprocessed_view = preprocessed.as_view()?;
                let result = detect_with_view(
                    detector,
                    cost,
                    &preprocessed_view,
                    &constraints,
                    stopping,
                    repro_mode,
                )?;
                Ok((result, notes))
            } else {
                let result =
                    detect_with_view(detector, cost, &view, &constraints, stopping, repro_mode)?;
                Ok((result, vec![]))
            }
        })
        .map_err(cpd_error_to_pyerr)?;

    #[cfg(not(feature = "preprocess"))]
    let (mut result, preprocess_notes): (CoreOfflineChangePointResult, Vec<String>) = py
        .allow_threads(|| {
            let result =
                detect_with_spec(detector, cost, &owned, &constraints, stopping, repro_mode)?;
            Ok((result, vec![]))
        })
        .map_err(cpd_error_to_pyerr)?;

    if !return_diagnostics {
        result.diagnostics.notes.clear();
        result.diagnostics.warnings.clear();
    }

    for note in preprocess_notes {
        result.diagnostics.notes.push(note);
    }

    for note in owned.diagnostics() {
        result.diagnostics.notes.push(format!("input: {note}"));
    }

    Ok(result.into())
}

/// Minimal placeholder detector used to validate Python packaging + wheel smoke tests.
#[pyclass(module = "cpd._cpd_rs")]
#[derive(Debug, Default)]
pub struct SmokeDetector {
    n: usize,
}

impl SmokeDetector {
    fn set_fitted_len(&mut self, n: usize) {
        self.n = n;
    }

    fn predicted_breakpoints(&self) -> Vec<usize> {
        vec![self.n]
    }
}

#[pymethods]
impl SmokeDetector {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        values: &Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let parsed = parse_sequence(values)?;
        slf.set_fitted_len(parsed.len());
        Ok(slf)
    }

    fn predict(&self) -> Vec<usize> {
        self.predicted_breakpoints()
    }
}

/// One-shot placeholder smoke detector function.
#[pyfunction]
fn smoke_detect(values: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let parsed = parse_sequence(values)?;
    Ok(vec![parsed.len()])
}

/// Python extension module entrypoint.
#[pymodule]
fn _cpd_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<PyPruningStats>()?;
    module.add_class::<PySegmentStats>()?;
    module.add_class::<PyDiagnostics>()?;
    module.add_class::<PyOfflineChangePointResult>()?;
    module.add_class::<PyOnlineStepResult>()?;
    module.add_class::<PyPelt>()?;
    module.add_class::<PyBinseg>()?;
    module.add_class::<PyFpop>()?;
    module.add_class::<PyBocpd>()?;
    module.add_class::<PyCusum>()?;
    module.add_class::<PyPageHinkley>()?;
    module.add_class::<PyBocpdState>()?;
    module.add_class::<PyCusumState>()?;
    module.add_class::<PyPageHinkleyState>()?;
    module.add_class::<SmokeDetector>()?;
    module.add_function(wrap_pyfunction!(detect_offline, module)?)?;
    module.add_function(wrap_pyfunction!(smoke_detect, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        _cpd_rs, PyBinseg, PyFpop, PyPelt, SmokeDetector, UPDATE_MANY_GIL_RELEASE_MIN_WORK_ITEMS,
        parse_pipeline_spec, should_release_gil_for_update_many, smoke_detect,
    };
    use pyo3::Python;
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyList, PyModule};
    use std::ffi::CString;
    use std::sync::Once;

    fn with_python<F, R>(f: F) -> R
    where
        F: for<'py> FnOnce(Python<'py>) -> R,
    {
        static INIT: Once = Once::new();
        INIT.call_once(pyo3::prepare_freethreaded_python);
        Python::with_gil(f)
    }

    fn run_python<'py>(
        py: Python<'py>,
        code: &str,
        globals: Option<&pyo3::Bound<'py, PyDict>>,
        locals: Option<&pyo3::Bound<'py, PyDict>>,
    ) -> pyo3::PyResult<()> {
        let code = CString::new(code).expect("python snippet should not contain NUL bytes");
        py.run(code.as_c_str(), globals, locals)
    }

    #[test]
    fn smoke_detector_rust_path_is_deterministic() {
        let mut detector = SmokeDetector::default();
        detector.set_fitted_len(3);
        assert_eq!(detector.predicted_breakpoints(), vec![3]);
    }

    #[test]
    fn smoke_detect_function_returns_terminal_breakpoint() {
        with_python(|py| {
            let values =
                PyList::new(py, [0.0, 1.0, 2.0, 3.0]).expect("python list should be created");
            let out = smoke_detect(values.as_any()).expect("smoke_detect should succeed");
            assert_eq!(out, vec![4]);
        });
    }

    #[test]
    fn update_many_gil_policy_uses_work_item_cutoff() {
        let cutoff = UPDATE_MANY_GIL_RELEASE_MIN_WORK_ITEMS;
        assert!(
            !should_release_gil_for_update_many(cutoff.saturating_sub(1), 1),
            "batch below cutoff should keep the GIL"
        );
        assert!(
            should_release_gil_for_update_many(cutoff, 1),
            "batch at cutoff should release the GIL"
        );
        assert!(
            should_release_gil_for_update_many(cutoff + 1, 1),
            "batch above cutoff should release the GIL"
        );
        assert!(
            should_release_gil_for_update_many(cutoff / 2, 2),
            "multivariate work at same total item count should release the GIL"
        );
    }

    #[test]
    fn module_registration_exposes_public_api() {
        with_python(|py| {
            let module = PyModule::new(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let version_attr = module
                .getattr("__version__")
                .expect("__version__ should be exported");
            let version: String = version_attr
                .extract()
                .expect("__version__ should be string");
            assert_eq!(version, env!("CARGO_PKG_VERSION"));

            module
                .getattr("SmokeDetector")
                .expect("SmokeDetector should be exported");
            module
                .getattr("smoke_detect")
                .expect("smoke_detect should be exported");
            module
                .getattr("PruningStats")
                .expect("PruningStats should be exported");
            module
                .getattr("SegmentStats")
                .expect("SegmentStats should be exported");
            module
                .getattr("Diagnostics")
                .expect("Diagnostics should be exported");
            module
                .getattr("OfflineChangePointResult")
                .expect("OfflineChangePointResult should be exported");
            module
                .getattr("OnlineStepResult")
                .expect("OnlineStepResult should be exported");
            module.getattr("Pelt").expect("Pelt should be exported");
            module.getattr("Binseg").expect("Binseg should be exported");
            module.getattr("Fpop").expect("Fpop should be exported");
            module.getattr("Bocpd").expect("Bocpd should be exported");
            module.getattr("Cusum").expect("Cusum should be exported");
            module
                .getattr("PageHinkley")
                .expect("PageHinkley should be exported");
            module
                .getattr("detect_offline")
                .expect("detect_offline should be exported");
        });
    }

    #[test]
    fn pelt_fit_predict_penalized_roundtrip() {
        with_python(|py| {
            let module = PyModule::new(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            run_python(py,
                "import numpy as np\nresult = cpd_rs.Pelt(model='l2').fit(np.array([0.,0.,0.,0.,0.,10.,10.,10.,10.,10.], dtype=np.float64)).predict(pen=1.0)",
                None,
                Some(&locals),
            )
            .expect("pelt penalized call should succeed");

            let result = locals
                .get_item("result")
                .expect("locals lookup should succeed")
                .expect("result should exist");
            let breakpoints: Vec<usize> = result
                .getattr("breakpoints")
                .expect("breakpoints attribute should exist")
                .extract()
                .expect("breakpoints should extract as Vec[usize]");
            assert_eq!(breakpoints, vec![5, 10]);
        });
    }

    #[test]
    fn pelt_predict_known_k_maps_to_knownk_stopping() {
        with_python(|py| {
            let module = PyModule::new(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            run_python(py,
                "import numpy as np\nresult = cpd_rs.Pelt(model='l2', min_segment_len=2).fit(np.array([0.,0.,0.,0.,10.,10.,10.,10.,-5.,-5.,-5.,-5.], dtype=np.float64)).predict(n_bkps=2)",
                None,
                Some(&locals),
            )
            .expect("known-k call should succeed");

            let result = locals
                .get_item("result")
                .expect("locals lookup should succeed")
                .expect("result should exist");
            let breakpoints: Vec<usize> = result
                .getattr("breakpoints")
                .expect("breakpoints attribute should exist")
                .extract()
                .expect("breakpoints should extract as Vec[usize]");
            assert_eq!(breakpoints, vec![4, 8, 12]);
        });
    }

    #[test]
    fn binseg_fit_predict_penalized_roundtrip() {
        with_python(|py| {
            let module = PyModule::new(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            run_python(py,
                "import numpy as np\nresult = cpd_rs.Binseg(model='l2', min_segment_len=2).fit(np.array([0.,0.,0.,0.,0.,10.,10.,10.,10.,10.], dtype=np.float64)).predict(pen=1.0)",
                None,
                Some(&locals),
            )
            .expect("binseg penalized call should succeed");

            let result = locals
                .get_item("result")
                .expect("locals lookup should succeed")
                .expect("result should exist");
            let breakpoints: Vec<usize> = result
                .getattr("breakpoints")
                .expect("breakpoints attribute should exist")
                .extract()
                .expect("breakpoints should extract as Vec[usize]");
            assert_eq!(breakpoints, vec![5, 10]);
        });
    }

    #[test]
    fn fpop_fit_predict_penalized_roundtrip() {
        with_python(|py| {
            let module = PyModule::new(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            run_python(
                py,
                "import numpy as np\nresult = cpd_rs.Fpop(min_segment_len=2).fit(np.array([0.,0.,0.,0.,0.,10.,10.,10.,10.,10.], dtype=np.float64)).predict(pen=1.0)",
                None,
                Some(&locals),
            )
            .expect("fpop penalized call should succeed");

            let result = locals
                .get_item("result")
                .expect("locals lookup should succeed")
                .expect("result should exist");
            let breakpoints: Vec<usize> = result
                .getattr("breakpoints")
                .expect("breakpoints attribute should exist")
                .extract()
                .expect("breakpoints should extract as Vec[usize]");
            assert_eq!(breakpoints, vec![5, 10]);
        });
    }

    #[test]
    fn fpop_rejects_non_l2_cost_in_detect_offline() {
        with_python(|py| {
            let module = PyModule::new(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            run_python(
                py,
                "import numpy as np\nx = np.array([0.,0.,0.,0.,10.,10.,10.,10.], dtype=np.float64)\ntry:\n    cpd_rs.detect_offline(x, detector='fpop', cost='normal', stopping={'n_bkps': 1})\n    raise AssertionError('expected fpop non-l2 rejection')\nexcept ValueError as exc:\n    assert \"requires cost='l2'\" in str(exc)",
                None,
                Some(&locals),
            )
            .expect("fpop should reject non-l2 cost");
        });
    }

    #[test]
    fn detect_offline_supports_normal_full_cov_cost() {
        with_python(|py| {
            let module = PyModule::new(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            run_python(
                py,
                "import numpy as np\nx = np.array([[0., 0.], [0.1, 0.1], [0.2, 0.2], [5.0, 5.1], [5.2, 5.2], [5.3, 5.4]], dtype=np.float64)\nresult = cpd_rs.detect_offline(x, detector='pelt', cost='normal_full_cov', stopping={'n_bkps': 1})",
                None,
                Some(&locals),
            )
            .expect("normal_full_cov should run");

            let result = locals
                .get_item("result")
                .expect("locals lookup should succeed")
                .expect("result should exist");
            let diagnostics = result
                .getattr("diagnostics")
                .expect("diagnostics should exist");
            let cost_model: String = diagnostics
                .getattr("cost_model")
                .expect("cost_model should exist")
                .extract()
                .expect("cost_model should extract");
            assert_eq!(cost_model, "normal_full_cov");
        });
    }

    #[test]
    fn detect_offline_supports_normal_full_cov_cost_with_binseg() {
        with_python(|py| {
            let module = PyModule::new(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            run_python(
                py,
                "import numpy as np\nx = np.array([[0., 0.], [0.1, 0.1], [0.2, 0.2], [5.0, 5.1], [5.2, 5.2], [5.3, 5.4]], dtype=np.float64)\nresult = cpd_rs.detect_offline(x, detector='binseg', cost='normal_full_cov', stopping={'n_bkps': 1})",
                None,
                Some(&locals),
            )
            .expect("normal_full_cov should run");

            let result = locals
                .get_item("result")
                .expect("locals lookup should succeed")
                .expect("result should exist");
            let diagnostics = result
                .getattr("diagnostics")
                .expect("diagnostics should exist");
            let cost_model: String = diagnostics
                .getattr("cost_model")
                .expect("cost_model should exist")
                .extract()
                .expect("cost_model should extract");
            assert_eq!(cost_model, "normal_full_cov");
        });
    }

    #[test]
    fn detect_offline_pipeline_rejects_fpop_with_non_l2_cost() {
        with_python(|py| {
            let pipeline = PyDict::new(py);
            let detector = PyDict::new(py);
            detector
                .set_item("kind", "fpop")
                .expect("detector.kind should be set");
            pipeline
                .set_item("detector", &detector)
                .expect("pipeline.detector should be set");
            pipeline
                .set_item("cost", "normal")
                .expect("pipeline.cost should be set");
            let stopping = PyDict::new(py);
            stopping
                .set_item("n_bkps", 1)
                .expect("stopping.n_bkps should be set");
            pipeline
                .set_item("stopping", &stopping)
                .expect("pipeline.stopping should be set");

            let err =
                parse_pipeline_spec(pipeline.as_any()).expect_err("pipeline parse should fail");
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(
                err.to_string()
                    .contains("pipeline.detector='fpop' requires pipeline.cost='l2'"),
                "unexpected error message: {err}"
            );
        });
    }

    #[test]
    fn detect_offline_binseg_matches_class_api_for_known_k() {
        with_python(|py| {
            let module = PyModule::new(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            run_python(py,
                "import numpy as np\nx = np.array([0.,0.,0.,0.,10.,10.,10.,10.,-5.,-5.,-5.,-5.], dtype=np.float64)\nclass_result = cpd_rs.Binseg(model='l2', min_segment_len=2).fit(x).predict(n_bkps=2)\nlow_result = cpd_rs.detect_offline(x, detector='binseg', cost='l2', constraints={'min_segment_len': 2}, stopping={'n_bkps': 2}, repro_mode='balanced')",
                None,
                Some(&locals),
            )
            .expect("detect_offline and class calls should succeed");

            let class_result = locals
                .get_item("class_result")
                .expect("locals lookup should succeed")
                .expect("class_result should exist");
            let low_result = locals
                .get_item("low_result")
                .expect("locals lookup should succeed")
                .expect("low_result should exist");

            let class_breakpoints: Vec<usize> = class_result
                .getattr("breakpoints")
                .expect("class breakpoints should exist")
                .extract()
                .expect("class breakpoints should extract");
            let low_breakpoints: Vec<usize> = low_result
                .getattr("breakpoints")
                .expect("low breakpoints should exist")
                .extract()
                .expect("low breakpoints should extract");
            assert_eq!(class_breakpoints, low_breakpoints);
        });
    }

    #[test]
    fn detect_offline_pipeline_spec_matches_explicit_arguments() {
        with_python(|py| {
            let module = PyModule::new(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            run_python(
                py,
                "import numpy as np\nx = np.array([0.,0.,0.,0.,10.,10.,10.,10.,-5.,-5.,-5.,-5.], dtype=np.float64)\npipeline = {'detector': {'kind': 'pelt'}, 'cost': 'l2', 'constraints': {'min_segment_len': 2}, 'stopping': {'n_bkps': 2}}\npipeline_result = cpd_rs.detect_offline(x, pipeline=pipeline)\nexplicit_result = cpd_rs.detect_offline(x, detector='pelt', cost='l2', constraints={'min_segment_len': 2}, stopping={'n_bkps': 2})",
                None,
                Some(&locals),
            )
            .expect("pipeline and explicit calls should succeed");

            let pipeline_result = locals
                .get_item("pipeline_result")
                .expect("locals lookup should succeed")
                .expect("pipeline_result should exist");
            let explicit_result = locals
                .get_item("explicit_result")
                .expect("locals lookup should succeed")
                .expect("explicit_result should exist");

            let pipeline_breakpoints: Vec<usize> = pipeline_result
                .getattr("breakpoints")
                .expect("pipeline breakpoints should exist")
                .extract()
                .expect("pipeline breakpoints should extract");
            let explicit_breakpoints: Vec<usize> = explicit_result
                .getattr("breakpoints")
                .expect("explicit breakpoints should exist")
                .extract()
                .expect("explicit breakpoints should extract");
            assert_eq!(pipeline_breakpoints, explicit_breakpoints);
        });
    }

    #[test]
    fn detect_offline_pipeline_spec_accepts_rust_serde_shape() {
        with_python(|py| {
            let module = PyModule::new(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            run_python(
                py,
                "import numpy as np\nx = np.array([0.,0.,0.,0.,10.,10.,10.,10.,-5.,-5.,-5.,-5.], dtype=np.float64)\npipeline = {'detector': {'Offline': {'Pelt': {'stopping': {'KnownK': 2}, 'params_per_segment': 2, 'cancel_check_every': 1000}}}, 'cost': 'L2', 'constraints': {'min_segment_len': 2}, 'stopping': {'KnownK': 2}, 'seed': None}\npipeline_result = cpd_rs.detect_offline(x, pipeline=pipeline)\nexplicit_result = cpd_rs.detect_offline(x, detector='pelt', cost='l2', constraints={'min_segment_len': 2}, stopping={'n_bkps': 2})",
                None,
                Some(&locals),
            )
            .expect("serde-shaped pipeline and explicit calls should succeed");

            let pipeline_result = locals
                .get_item("pipeline_result")
                .expect("locals lookup should succeed")
                .expect("pipeline_result should exist");
            let explicit_result = locals
                .get_item("explicit_result")
                .expect("locals lookup should succeed")
                .expect("explicit_result should exist");

            let pipeline_breakpoints: Vec<usize> = pipeline_result
                .getattr("breakpoints")
                .expect("pipeline breakpoints should exist")
                .extract()
                .expect("pipeline breakpoints should extract");
            let explicit_breakpoints: Vec<usize> = explicit_result
                .getattr("breakpoints")
                .expect("explicit breakpoints should exist")
                .extract()
                .expect("explicit breakpoints should extract");
            assert_eq!(pipeline_breakpoints, explicit_breakpoints);
        });
    }

    #[test]
    fn detect_offline_pipeline_spec_accepts_rust_serde_l1median_cost() {
        with_python(|py| {
            let module = PyModule::new(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            run_python(
                py,
                "import numpy as np\nx = np.array([0.,0.,0.,0.,10.,10.,10.,10.,-5.,-5.,-5.,-5.], dtype=np.float64)\npipeline = {'detector': {'Offline': {'Pelt': {'stopping': {'KnownK': 2}, 'params_per_segment': 2, 'cancel_check_every': 1000}}}, 'cost': 'L1Median', 'constraints': {'min_segment_len': 2}, 'stopping': {'KnownK': 2}, 'seed': None}\npipeline_result = cpd_rs.detect_offline(x, pipeline=pipeline)\nexplicit_result = cpd_rs.detect_offline(x, detector='pelt', cost='l1_median', constraints={'min_segment_len': 2}, stopping={'n_bkps': 2})",
                None,
                Some(&locals),
            )
            .expect("serde-shaped L1 pipeline and explicit calls should succeed");

            let pipeline_result = locals
                .get_item("pipeline_result")
                .expect("locals lookup should succeed")
                .expect("pipeline_result should exist");
            let explicit_result = locals
                .get_item("explicit_result")
                .expect("locals lookup should succeed")
                .expect("explicit_result should exist");

            let pipeline_breakpoints: Vec<usize> = pipeline_result
                .getattr("breakpoints")
                .expect("pipeline breakpoints should exist")
                .extract()
                .expect("pipeline breakpoints should extract");
            let explicit_breakpoints: Vec<usize> = explicit_result
                .getattr("breakpoints")
                .expect("explicit breakpoints should exist")
                .extract()
                .expect("explicit breakpoints should extract");
            assert_eq!(pipeline_breakpoints, explicit_breakpoints);
        });
    }

    #[test]
    fn detect_offline_pipeline_spec_uses_requested_repro_mode() {
        with_python(|py| {
            let module = PyModule::new(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            run_python(
                py,
                "import numpy as np\nx = np.array([0.,0.,0.,0.,10.,10.,10.,10.,-5.,-5.,-5.,-5.], dtype=np.float64)\npipeline = {'detector': {'kind': 'pelt'}, 'cost': 'l2', 'constraints': {'min_segment_len': 2}, 'stopping': {'n_bkps': 2}}\npipeline_result = cpd_rs.detect_offline(x, pipeline=pipeline, repro_mode='fast')\nexplicit_result = cpd_rs.detect_offline(x, detector='pelt', cost='l2', constraints={'min_segment_len': 2}, stopping={'n_bkps': 2}, repro_mode='fast')\nassert not any('currently uses balanced reproducibility mode' in note for note in pipeline_result.diagnostics.notes)",
                None,
                Some(&locals),
            )
            .expect("pipeline repro mode should match explicit path and avoid balanced-mode note");

            let pipeline_result = locals
                .get_item("pipeline_result")
                .expect("locals lookup should succeed")
                .expect("pipeline_result should exist");
            let explicit_result = locals
                .get_item("explicit_result")
                .expect("locals lookup should succeed")
                .expect("explicit_result should exist");

            let pipeline_breakpoints: Vec<usize> = pipeline_result
                .getattr("breakpoints")
                .expect("pipeline breakpoints should exist")
                .extract()
                .expect("pipeline breakpoints should extract");
            let explicit_breakpoints: Vec<usize> = explicit_result
                .getattr("breakpoints")
                .expect("explicit breakpoints should exist")
                .extract()
                .expect("explicit breakpoints should extract");
            assert_eq!(pipeline_breakpoints, explicit_breakpoints);
        });
    }

    #[test]
    #[cfg(not(feature = "preprocess"))]
    fn detect_offline_rejects_preprocess_without_feature() {
        with_python(|py| {
            let module = PyModule::new(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            run_python(py,
                "import numpy as np\nx = np.array([0.,0.,0.,0.,10.,10.,10.,10.], dtype=np.float64)\ntry:\n    cpd_rs.detect_offline(x, detector='pelt', cost='l2', stopping={'n_bkps': 1}, preprocess={'winsorize': {}})\n    raise AssertionError('expected preprocess feature error')\nexcept ValueError as exc:\n    assert 'preprocess feature' in str(exc)",
                None,
                Some(&locals),
            )
            .expect("preprocess should require feature flag");
        });
    }

    #[test]
    #[cfg(feature = "preprocess")]
    fn detect_offline_applies_preprocess_and_emits_notes() {
        with_python(|py| {
            let module = PyModule::new(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            run_python(py,
                "import numpy as np\nx = np.array([0.,0.,0.,0.,10.,10.,10.,10.], dtype=np.float64)\nresult = cpd_rs.detect_offline(x, detector='pelt', cost='l2', stopping={'n_bkps': 1}, preprocess={'winsorize': {}, 'robust_scale': {}})",
                None,
                Some(&locals),
            )
            .expect("detect_offline with preprocess should succeed");

            let result = locals
                .get_item("result")
                .expect("locals lookup should succeed")
                .expect("result should exist");
            let breakpoints: Vec<usize> = result
                .getattr("breakpoints")
                .expect("breakpoints attribute should exist")
                .extract()
                .expect("breakpoints should extract");
            assert_eq!(breakpoints, vec![4, 8]);

            let diagnostics = result
                .getattr("diagnostics")
                .expect("diagnostics should exist");
            let notes: Vec<String> = diagnostics
                .getattr("notes")
                .expect("notes should exist")
                .extract()
                .expect("notes should extract");
            assert!(
                notes
                    .iter()
                    .any(|note| note.starts_with("preprocess:winsorize"))
            );
            assert!(
                notes
                    .iter()
                    .any(|note| note.starts_with("preprocess:robust_scale"))
            );
        });
    }

    #[test]
    #[cfg(feature = "preprocess")]
    fn detect_offline_rejects_invalid_preprocess_shape() {
        with_python(|py| {
            let module = PyModule::new(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            run_python(py,
                "import numpy as np\nx = np.array([0.,0.,0.,0.,10.,10.,10.,10.], dtype=np.float64)\ntry:\n    cpd_rs.detect_offline(x, detector='pelt', cost='l2', stopping={'n_bkps': 1}, preprocess={'unknown_stage': {}})\n    raise AssertionError('expected unsupported preprocess key')\nexcept ValueError as exc:\n    assert \"unsupported preprocess key 'unknown_stage'\" in str(exc)\n\ntry:\n    cpd_rs.detect_offline(x, detector='pelt', cost='l2', stopping={'n_bkps': 1}, preprocess={'detrend': {'method': 'bad'}})\n    raise AssertionError('expected unsupported preprocess method')\nexcept ValueError as exc:\n    assert 'unsupported preprocess.detrend.method' in str(exc)",
                None,
                Some(&locals),
            )
            .expect("invalid preprocess shape should fail clearly");
        });
    }

    #[test]
    fn pelt_rejects_invalid_model_name() {
        with_python(|py| {
            let err = PyPelt::new("bad-model", 2, 1, None).expect_err("invalid model must fail");
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err.to_string().contains("unsupported model"));
        });
    }

    #[test]
    fn binseg_rejects_invalid_model_name() {
        with_python(|py| {
            let err =
                PyBinseg::new("bad-model", 2, 1, None, None).expect_err("invalid model must fail");
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err.to_string().contains("unsupported model"));
        });
    }

    #[test]
    fn fpop_rejects_invalid_constructor_values() {
        with_python(|py| {
            let min_seg_err = PyFpop::new(0, 1, None).expect_err("min_segment_len=0 must fail");
            assert!(min_seg_err.is_instance_of::<PyValueError>(py));
            assert!(min_seg_err.to_string().contains("min_segment_len"));

            let jump_err = PyFpop::new(2, 0, None).expect_err("jump=0 must fail");
            assert!(jump_err.is_instance_of::<PyValueError>(py));
            assert!(jump_err.to_string().contains("jump"));
        });
    }

    #[test]
    fn pelt_rejects_invalid_predict_argument_combinations() {
        with_python(|py| {
            let module = PyModule::new(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            run_python(py,
                "import numpy as np\np = cpd_rs.Pelt(model='l2').fit(np.array([0.,0.,1.,1.], dtype=np.float64))",
                None,
                Some(&locals),
            )
            .expect("fit should succeed");

            let pelt = locals
                .get_item("p")
                .expect("locals lookup should succeed")
                .expect("p should exist");

            let both_kwargs = PyDict::new(py);
            both_kwargs
                .set_item("pen", 1.0)
                .expect("set pen should work");
            both_kwargs
                .set_item("n_bkps", 1)
                .expect("set n_bkps should work");
            let both_err = pelt
                .call_method("predict", (), Some(&both_kwargs))
                .expect_err("pen + n_bkps should fail");
            assert!(both_err.is_instance_of::<PyValueError>(py));
            assert!(both_err.to_string().contains("exactly one"));

            let none_err = pelt
                .call_method0("predict")
                .expect_err("no args should fail");
            assert!(none_err.is_instance_of::<PyValueError>(py));
            assert!(none_err.to_string().contains("exactly one"));
        });
    }

    #[test]
    fn pelt_predict_before_fit_is_clear_error() {
        with_python(|py| {
            let module = PyModule::new(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let pelt = module
                .getattr("Pelt")
                .expect("Pelt should be exported")
                .call0()
                .expect("constructor should succeed");

            let kwargs = PyDict::new(py);
            kwargs.set_item("pen", 1.0).expect("set pen should succeed");
            let err = pelt
                .call_method("predict", (), Some(&kwargs))
                .expect_err("predict before fit should fail");
            assert!(err.is_instance_of::<PyRuntimeError>(py));
            assert!(err.to_string().contains("fit(...) must be called"));
        });
    }
}
