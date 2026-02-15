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
    OfflineChangePointResult as CoreOfflineChangePointResult, OfflineDetector, Penalty, ReproMode,
    Stopping, TimeSeriesView,
};
use cpd_costs::{CostL2Mean, CostNormalMeanVar};
use cpd_offline::{BinSeg as OfflineBinSeg, BinSegConfig, Pelt as OfflinePelt, PeltConfig};
#[cfg(feature = "preprocess")]
use cpd_preprocess::{
    DeseasonalizeConfig, DeseasonalizeMethod, DetrendConfig, DetrendMethod, PreprocessConfig,
    PreprocessPipeline, RobustScaleConfig, WinsorizeConfig,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyList, PyModule};
use result_objects::{PyDiagnostics, PyOfflineChangePointResult, PyPruningStats, PySegmentStats};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PyCostModel {
    L2,
    Normal,
}

impl PyCostModel {
    fn parse(model: &str) -> PyResult<Self> {
        match model.to_ascii_lowercase().as_str() {
            "l2" => Ok(Self::L2),
            "normal" => Ok(Self::Normal),
            _ => Err(PyValueError::new_err(format!(
                "unsupported model '{model}'; expected one of: 'l2', 'normal'"
            ))),
        }
    }

    fn cost_model_name(self) -> &'static str {
        match self {
            Self::L2 => "l2",
            Self::Normal => "normal",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PyDetectorKind {
    Pelt,
    Binseg,
}

impl PyDetectorKind {
    fn parse(detector: &str) -> PyResult<Self> {
        match detector.to_ascii_lowercase().as_str() {
            "pelt" => Ok(Self::Pelt),
            "binseg" => Ok(Self::Binseg),
            _ => Err(PyValueError::new_err(format!(
                "unsupported detector '{detector}'; expected one of: 'pelt', 'binseg'"
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
                params_per_segment: 3,
                cancel_check_every: 1000,
            };
            let detector = OfflinePelt::new(CostNormalMeanVar::new(repro_mode), config)?;
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
        (PyDetectorKind::Binseg, PyCostModel::Normal) => {
            let config = BinSegConfig {
                stopping,
                params_per_segment: 3,
                cancel_check_every: 1000,
            };
            let detector = OfflineBinSeg::new(CostNormalMeanVar::new(repro_mode), config)?;
            detector.detect(view, &ctx)
        }
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

/// Low-level power-user API for fully-specified offline detection.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (x, *, detector = "pelt", cost = "l2", constraints = None, stopping = None, preprocess = None, repro_mode = "balanced", return_diagnostics = true))]
fn detect_offline(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    detector: &str,
    cost: &str,
    constraints: Option<&Bound<'_, PyAny>>,
    stopping: Option<&Bound<'_, PyAny>>,
    preprocess: Option<&Bound<'_, PyAny>>,
    repro_mode: &str,
    return_diagnostics: bool,
) -> PyResult<PyOfflineChangePointResult> {
    let detector = PyDetectorKind::parse(detector)?;
    let cost = PyCostModel::parse(cost)?;
    let constraints = parse_constraints(constraints)?;
    let stopping = parse_stopping(stopping)?;
    #[cfg(feature = "preprocess")]
    let preprocess_pipeline = parse_preprocess(preprocess)?;
    #[cfg(not(feature = "preprocess"))]
    parse_preprocess(preprocess)?;
    let repro_mode = parse_repro_mode(repro_mode)?;
    let owned = parse_owned_series(py, x)?;

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
    module.add_class::<PyPelt>()?;
    module.add_class::<PyBinseg>()?;
    module.add_class::<SmokeDetector>()?;
    module.add_function(wrap_pyfunction!(detect_offline, module)?)?;
    module.add_function(wrap_pyfunction!(smoke_detect, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{_cpd_rs, PyBinseg, PyPelt, SmokeDetector, smoke_detect};
    use pyo3::Python;
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyList, PyModule};
    use std::sync::Once;

    fn with_python<F, R>(f: F) -> R
    where
        F: for<'py> FnOnce(Python<'py>) -> R,
    {
        static INIT: Once = Once::new();
        INIT.call_once(pyo3::prepare_freethreaded_python);
        Python::with_gil(f)
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
            let values = PyList::new_bound(py, [0.0, 1.0, 2.0, 3.0]);
            let out = smoke_detect(values.as_any()).expect("smoke_detect should succeed");
            assert_eq!(out, vec![4]);
        });
    }

    #[test]
    fn module_registration_exposes_public_api() {
        with_python(|py| {
            let module = PyModule::new_bound(py, "_cpd_rs").expect("module should be created");
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
            module.getattr("Pelt").expect("Pelt should be exported");
            module.getattr("Binseg").expect("Binseg should be exported");
            module
                .getattr("detect_offline")
                .expect("detect_offline should be exported");
        });
    }

    #[test]
    fn pelt_fit_predict_penalized_roundtrip() {
        with_python(|py| {
            let module = PyModule::new_bound(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new_bound(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            py.run_bound(
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
            let module = PyModule::new_bound(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new_bound(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            py.run_bound(
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
            let module = PyModule::new_bound(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new_bound(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            py.run_bound(
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
    fn detect_offline_binseg_matches_class_api_for_known_k() {
        with_python(|py| {
            let module = PyModule::new_bound(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new_bound(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            py.run_bound(
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
    #[cfg(not(feature = "preprocess"))]
    fn detect_offline_rejects_preprocess_without_feature() {
        with_python(|py| {
            let module = PyModule::new_bound(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new_bound(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            py.run_bound(
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
            let module = PyModule::new_bound(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new_bound(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            py.run_bound(
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
            let module = PyModule::new_bound(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new_bound(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            py.run_bound(
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
    fn pelt_rejects_invalid_predict_argument_combinations() {
        with_python(|py| {
            let module = PyModule::new_bound(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let locals = PyDict::new_bound(py);
            locals
                .set_item("cpd_rs", &module)
                .expect("locals should accept module");
            py.run_bound(
                "import numpy as np\np = cpd_rs.Pelt(model='l2').fit(np.array([0.,0.,1.,1.], dtype=np.float64))",
                None,
                Some(&locals),
            )
            .expect("fit should succeed");

            let pelt = locals
                .get_item("p")
                .expect("locals lookup should succeed")
                .expect("p should exist");

            let both_kwargs = PyDict::new_bound(py);
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
            let module = PyModule::new_bound(py, "_cpd_rs").expect("module should be created");
            _cpd_rs(&module).expect("module registration should succeed");

            let pelt = module
                .getattr("Pelt")
                .expect("Pelt should be exported")
                .call0()
                .expect("constructor should succeed");

            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("pen", 1.0).expect("set pen should succeed");
            let err = pelt
                .call_method("predict", (), Some(&kwargs))
                .expect_err("predict before fit should fail");
            assert!(err.is_instance_of::<PyRuntimeError>(py));
            assert!(err.to_string().contains("fit(...) must be called"));
        });
    }
}
