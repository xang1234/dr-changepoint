// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

#[cfg(feature = "serde")]
use crate::error_map::cpd_error_to_pyerr;
#[cfg(feature = "serde")]
use cpd_core::OfflineChangePointResultWire;
use cpd_core::{
    DIAGNOSTICS_SCHEMA_VERSION, Diagnostics as CoreDiagnostics,
    OfflineChangePointResult as CoreOfflineChangePointResult,
    OnlineStepResult as CoreOnlineStepResult, PruningStats as CorePruningStats, ReproMode,
    SegmentStats as CoreSegmentStats,
};
#[cfg(not(feature = "serde"))]
use pyo3::exceptions::PyNotImplementedError;
use pyo3::exceptions::{PyImportError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyModule};
#[cfg(feature = "serde")]
use serde_json::{Map as JsonMap, Value as JsonValue};
#[cfg(feature = "serde")]
use std::borrow::Cow;

fn repro_mode_to_str(mode: ReproMode) -> &'static str {
    match mode {
        ReproMode::Strict => "strict",
        ReproMode::Balanced => "balanced",
        ReproMode::Fast => "fast",
    }
}

#[cfg(feature = "serde")]
fn repro_mode_from_str(value: &str) -> ReproMode {
    match value {
        "strict" | "Strict" => ReproMode::Strict,
        "balanced" | "Balanced" => ReproMode::Balanced,
        "fast" | "Fast" => ReproMode::Fast,
        _ => ReproMode::Balanced,
    }
}

fn derive_change_points(n: usize, breakpoints: &[usize]) -> Vec<usize> {
    breakpoints.iter().copied().filter(|&bp| bp < n).collect()
}

fn import_pyplot<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    PyModule::import(py, "matplotlib.pyplot").map_err(|err| {
        PyImportError::new_err(format!(
            "plot() requires optional dependency 'matplotlib'. Install with: python -m pip install matplotlib ({err})"
        ))
    })
}

fn build_segment_mean_summary(
    n: usize,
    d: usize,
    segments: &[PySegmentStats],
) -> PyResult<Vec<Vec<f64>>> {
    if d == 0 {
        return Err(PyValueError::new_err(
            "plot() could not infer dimensionality from diagnostics/segments",
        ));
    }

    let mut summary = vec![vec![f64::NAN; n]; d];
    for (idx, segment) in segments.iter().enumerate() {
        if segment.end < segment.start {
            return Err(PyValueError::new_err(format!(
                "plot() cannot summarize segments[{idx}] with end < start"
            )));
        }
        if segment.end > n {
            return Err(PyValueError::new_err(format!(
                "plot() cannot summarize segments[{idx}] with end={} beyond diagnostics.n={n}",
                segment.end
            )));
        }

        for dim in 0..d {
            let value = segment
                .mean
                .as_ref()
                .and_then(|mean| mean.get(dim))
                .copied()
                .unwrap_or(f64::NAN);
            for slot in &mut summary[dim][segment.start..segment.end] {
                *slot = value;
            }
        }
    }
    Ok(summary)
}

#[pyclass(module = "cpd._cpd_rs", name = "PruningStats", frozen)]
#[derive(Clone, Debug)]
pub struct PyPruningStats {
    candidates_considered: usize,
    candidates_pruned: usize,
}

#[pymethods]
impl PyPruningStats {
    #[getter]
    fn candidates_considered(&self) -> usize {
        self.candidates_considered
    }

    #[getter]
    fn candidates_pruned(&self) -> usize {
        self.candidates_pruned
    }

    fn __repr__(&self) -> String {
        format!(
            "PruningStats(candidates_considered={}, candidates_pruned={})",
            self.candidates_considered, self.candidates_pruned
        )
    }
}

impl From<CorePruningStats> for PyPruningStats {
    fn from(stats: CorePruningStats) -> Self {
        Self {
            candidates_considered: stats.candidates_considered,
            candidates_pruned: stats.candidates_pruned,
        }
    }
}

impl PyPruningStats {
    #[cfg(feature = "serde")]
    fn to_core(&self) -> CorePruningStats {
        CorePruningStats {
            candidates_considered: self.candidates_considered,
            candidates_pruned: self.candidates_pruned,
        }
    }
}

#[pyclass(module = "cpd._cpd_rs", name = "SegmentStats", frozen)]
#[derive(Clone, Debug)]
pub struct PySegmentStats {
    start: usize,
    end: usize,
    mean: Option<Vec<f64>>,
    variance: Option<Vec<f64>>,
    count: usize,
    missing_count: usize,
}

#[pymethods]
impl PySegmentStats {
    #[getter]
    fn start(&self) -> usize {
        self.start
    }

    #[getter]
    fn end(&self) -> usize {
        self.end
    }

    #[getter]
    fn mean(&self) -> Option<Vec<f64>> {
        self.mean.clone()
    }

    #[getter]
    fn variance(&self) -> Option<Vec<f64>> {
        self.variance.clone()
    }

    #[getter]
    fn count(&self) -> usize {
        self.count
    }

    #[getter]
    fn missing_count(&self) -> usize {
        self.missing_count
    }

    fn __repr__(&self) -> String {
        format!(
            "SegmentStats(start={}, end={}, count={}, missing_count={})",
            self.start, self.end, self.count, self.missing_count
        )
    }
}

impl From<CoreSegmentStats> for PySegmentStats {
    fn from(stats: CoreSegmentStats) -> Self {
        Self {
            start: stats.start,
            end: stats.end,
            mean: stats.mean,
            variance: stats.variance,
            count: stats.count,
            missing_count: stats.missing_count,
        }
    }
}

impl PySegmentStats {
    #[cfg(feature = "serde")]
    fn to_core(&self) -> CoreSegmentStats {
        CoreSegmentStats {
            start: self.start,
            end: self.end,
            mean: self.mean.clone(),
            variance: self.variance.clone(),
            count: self.count,
            missing_count: self.missing_count,
        }
    }
}

#[pyclass(module = "cpd._cpd_rs", name = "Diagnostics", frozen)]
#[derive(Clone, Debug)]
pub struct PyDiagnostics {
    n: usize,
    d: usize,
    schema_version: u32,
    engine_version: Option<String>,
    runtime_ms: Option<u64>,
    notes: Vec<String>,
    warnings: Vec<String>,
    algorithm: String,
    cost_model: String,
    seed: Option<u64>,
    repro_mode: String,
    thread_count: Option<usize>,
    blas_backend: Option<String>,
    cpu_features: Option<Vec<String>>,
    #[cfg_attr(not(feature = "serde"), allow(dead_code))]
    params_json_text: Option<String>,
    pruning_stats: Option<PyPruningStats>,
    missing_policy_applied: Option<String>,
    missing_fraction: Option<f64>,
    effective_sample_count: Option<usize>,
}

#[pymethods]
impl PyDiagnostics {
    #[getter]
    fn n(&self) -> usize {
        self.n
    }

    #[getter]
    fn d(&self) -> usize {
        self.d
    }

    #[getter]
    fn schema_version(&self) -> u32 {
        self.schema_version
    }

    #[getter]
    fn engine_version(&self) -> Option<String> {
        self.engine_version.clone()
    }

    #[getter]
    fn runtime_ms(&self) -> Option<u64> {
        self.runtime_ms
    }

    #[getter]
    fn notes(&self) -> Vec<String> {
        self.notes.clone()
    }

    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    #[getter]
    fn algorithm(&self) -> String {
        self.algorithm.clone()
    }

    #[getter]
    fn cost_model(&self) -> String {
        self.cost_model.clone()
    }

    #[getter]
    fn seed(&self) -> Option<u64> {
        self.seed
    }

    #[getter]
    fn repro_mode(&self) -> String {
        self.repro_mode.clone()
    }

    #[getter]
    fn thread_count(&self) -> Option<usize> {
        self.thread_count
    }

    #[getter]
    fn blas_backend(&self) -> Option<String> {
        self.blas_backend.clone()
    }

    #[getter]
    fn cpu_features(&self) -> Option<Vec<String>> {
        self.cpu_features.clone()
    }

    #[getter]
    fn params_json<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        #[cfg(feature = "serde")]
        if let Some(serialized) = &self.params_json_text {
            let json = PyModule::import(py, "json")?;
            let value = json.call_method1("loads", (serialized,))?;
            return Ok(value.into_py(py));
        }

        Ok(py.None())
    }

    #[getter]
    fn pruning_stats<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyPruningStats>>> {
        self.pruning_stats
            .clone()
            .map(|stats| Py::new(py, stats))
            .transpose()
    }

    #[getter]
    fn missing_policy_applied(&self) -> Option<String> {
        self.missing_policy_applied.clone()
    }

    #[getter]
    fn missing_fraction(&self) -> Option<f64> {
        self.missing_fraction
    }

    #[getter]
    fn effective_sample_count(&self) -> Option<usize> {
        self.effective_sample_count
    }

    fn __repr__(&self) -> String {
        format!(
            "Diagnostics(n={}, d={}, algorithm='{}', cost_model='{}', repro_mode='{}')",
            self.n, self.d, self.algorithm, self.cost_model, self.repro_mode
        )
    }
}

impl From<CoreDiagnostics> for PyDiagnostics {
    fn from(diagnostics: CoreDiagnostics) -> Self {
        #[cfg(feature = "serde")]
        let params_json_text = diagnostics
            .params_json
            .and_then(|value| serde_json::to_string(&value).ok());
        #[cfg(not(feature = "serde"))]
        let params_json_text = None;

        let normalized_schema_version = if diagnostics.schema_version == 0 {
            DIAGNOSTICS_SCHEMA_VERSION
        } else {
            diagnostics.schema_version
        };

        let normalized_engine_version = diagnostics
            .engine_version
            .or_else(|| Some(env!("CARGO_PKG_VERSION").to_string()));

        Self {
            n: diagnostics.n,
            d: diagnostics.d,
            schema_version: normalized_schema_version,
            engine_version: normalized_engine_version,
            runtime_ms: diagnostics.runtime_ms,
            notes: diagnostics.notes,
            warnings: diagnostics.warnings,
            algorithm: diagnostics.algorithm.into_owned(),
            cost_model: diagnostics.cost_model.into_owned(),
            seed: diagnostics.seed,
            repro_mode: repro_mode_to_str(diagnostics.repro_mode).to_string(),
            thread_count: diagnostics.thread_count,
            blas_backend: diagnostics.blas_backend,
            cpu_features: diagnostics.cpu_features,
            params_json_text,
            pruning_stats: diagnostics.pruning_stats.map(Into::into),
            missing_policy_applied: diagnostics.missing_policy_applied,
            missing_fraction: diagnostics.missing_fraction,
            effective_sample_count: diagnostics.effective_sample_count,
        }
    }
}

impl PyDiagnostics {
    #[cfg(feature = "serde")]
    fn to_core(&self) -> PyResult<CoreDiagnostics> {
        let params_json = match &self.params_json_text {
            Some(serialized) => Some(serde_json::from_str(serialized).map_err(|err| {
                PyValueError::new_err(format!("failed to parse diagnostics.params_json: {err}"))
            })?),
            None => None,
        };

        Ok(CoreDiagnostics {
            n: self.n,
            d: self.d,
            schema_version: self.schema_version,
            engine_version: self.engine_version.clone(),
            runtime_ms: self.runtime_ms,
            notes: self.notes.clone(),
            warnings: self.warnings.clone(),
            algorithm: Cow::Owned(self.algorithm.clone()),
            cost_model: Cow::Owned(self.cost_model.clone()),
            seed: self.seed,
            repro_mode: repro_mode_from_str(&self.repro_mode),
            thread_count: self.thread_count,
            blas_backend: self.blas_backend.clone(),
            cpu_features: self.cpu_features.clone(),
            params_json,
            pruning_stats: self.pruning_stats.as_ref().map(PyPruningStats::to_core),
            missing_policy_applied: self.missing_policy_applied.clone(),
            missing_fraction: self.missing_fraction,
            effective_sample_count: self.effective_sample_count,
        })
    }
}

#[pyclass(module = "cpd._cpd_rs", name = "OnlineStepResult", frozen)]
#[derive(Clone, Debug)]
pub struct PyOnlineStepResult {
    t: usize,
    p_change: f64,
    alert: bool,
    alert_reason: Option<String>,
    run_length_mode: usize,
    run_length_mean: f64,
    processing_latency_us: Option<u64>,
}

#[pymethods]
impl PyOnlineStepResult {
    #[getter]
    fn t(&self) -> usize {
        self.t
    }

    #[getter]
    fn p_change(&self) -> f64 {
        self.p_change
    }

    #[getter]
    fn alert(&self) -> bool {
        self.alert
    }

    #[getter]
    fn alert_reason(&self) -> Option<String> {
        self.alert_reason.clone()
    }

    #[getter]
    fn run_length_mode(&self) -> usize {
        self.run_length_mode
    }

    #[getter]
    fn run_length_mean(&self) -> f64 {
        self.run_length_mean
    }

    #[getter]
    fn processing_latency_us(&self) -> Option<u64> {
        self.processing_latency_us
    }

    fn __repr__(&self) -> String {
        format!(
            "OnlineStepResult(t={}, p_change={:.6}, alert={}, run_length_mode={}, run_length_mean={:.3})",
            self.t, self.p_change, self.alert, self.run_length_mode, self.run_length_mean
        )
    }
}

impl From<CoreOnlineStepResult> for PyOnlineStepResult {
    fn from(step: CoreOnlineStepResult) -> Self {
        Self {
            t: step.t,
            p_change: step.p_change,
            alert: step.alert,
            alert_reason: step.alert_reason,
            run_length_mode: step.run_length_mode,
            run_length_mean: step.run_length_mean,
            processing_latency_us: step.processing_latency_us,
        }
    }
}

#[pyclass(module = "cpd._cpd_rs", name = "OfflineChangePointResult", frozen)]
#[derive(Clone, Debug)]
pub struct PyOfflineChangePointResult {
    breakpoints: Vec<usize>,
    change_points: Vec<usize>,
    scores: Option<Vec<f64>>,
    segments: Option<Vec<PySegmentStats>>,
    diagnostics: PyDiagnostics,
    #[cfg(feature = "serde")]
    result_unknown_fields: JsonMap<String, JsonValue>,
    #[cfg(feature = "serde")]
    diagnostics_unknown_fields: JsonMap<String, JsonValue>,
}

#[pymethods]
impl PyOfflineChangePointResult {
    #[getter]
    fn breakpoints(&self) -> Vec<usize> {
        self.breakpoints.clone()
    }

    #[getter]
    fn change_points(&self) -> Vec<usize> {
        self.change_points.clone()
    }

    #[getter]
    fn scores(&self) -> Option<Vec<f64>> {
        self.scores.clone()
    }

    #[getter]
    fn segments<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Py<PySegmentStats>>>> {
        self.segments
            .clone()
            .map(|segments| {
                segments
                    .into_iter()
                    .map(|segment| Py::new(py, segment))
                    .collect()
            })
            .transpose()
    }

    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDiagnostics>> {
        Py::new(py, self.diagnostics.clone())
    }

    #[cfg(feature = "serde")]
    fn to_json(&self) -> PyResult<String> {
        let core = self.to_core()?;
        let wire = OfflineChangePointResultWire::from_runtime_with_unknown(
            core,
            self.result_unknown_fields.clone(),
            self.diagnostics_unknown_fields.clone(),
        );
        serde_json::to_string(&wire)
            .map_err(|err| PyValueError::new_err(format!("failed to serialize result: {err}")))
    }

    #[cfg(not(feature = "serde"))]
    fn to_json(&self) -> PyResult<String> {
        Err(PyNotImplementedError::new_err(
            "to_json() requires cpd-python built with serde feature",
        ))
    }

    #[cfg(feature = "serde")]
    #[staticmethod]
    fn from_json(payload: &str) -> PyResult<Self> {
        if payload.trim().is_empty() {
            return Err(PyValueError::new_err(
                "from_json() requires a non-empty JSON string payload",
            ));
        }

        let wire: OfflineChangePointResultWire = serde_json::from_str(payload).map_err(|err| {
            PyValueError::new_err(format!(
                "invalid OfflineChangePointResult JSON payload: {err}"
            ))
        })?;
        let (core, result_unknown_fields, diagnostics_unknown_fields) =
            wire.into_runtime_parts().map_err(cpd_error_to_pyerr)?;
        let mut parsed = Self::from(core);
        parsed.result_unknown_fields = result_unknown_fields;
        parsed.diagnostics_unknown_fields = diagnostics_unknown_fields;
        Ok(parsed)
    }

    #[cfg(not(feature = "serde"))]
    #[staticmethod]
    fn from_json(_payload: &str) -> PyResult<Self> {
        Err(PyNotImplementedError::new_err(
            "from_json() requires cpd-python built with serde feature",
        ))
    }

    #[pyo3(signature = (values=None, *, ax=None, title=None, breakpoint_color="crimson", breakpoint_style="--", line_width=1.5, show_legend=true))]
    fn plot(
        &self,
        py: Python<'_>,
        values: Option<&Bound<'_, PyAny>>,
        ax: Option<&Bound<'_, PyAny>>,
        title: Option<&str>,
        breakpoint_color: &str,
        breakpoint_style: &str,
        line_width: f64,
        show_legend: bool,
    ) -> PyResult<Py<PyAny>> {
        if !line_width.is_finite() || line_width <= 0.0 {
            return Err(PyValueError::new_err(
                "plot() requires line_width to be a finite positive float",
            ));
        }

        let pyplot = import_pyplot(py)?;
        let diagnostics_n = self.diagnostics.n;
        let diagnostics_d = self.diagnostics.d;

        let series_by_dimension: Vec<Vec<f64>> = if let Some(raw_values) = values {
            let numpy = PyModule::import(py, "numpy")?;
            let array = numpy.call_method1("asarray", (raw_values,))?;
            let ndim: usize = array.getattr("ndim")?.extract()?;
            match ndim {
                1 => {
                    if diagnostics_d > 1 {
                        return Err(PyValueError::new_err(format!(
                            "plot() received univariate values, but diagnostics.d={diagnostics_d}; expected multivariate values with matching dimensions"
                        )));
                    }
                    let series: Vec<f64> = array.extract().map_err(|_| {
                        PyTypeError::new_err(
                            "plot() values must be numeric and convertible to float",
                        )
                    })?;
                    if series.len() != diagnostics_n {
                        return Err(PyValueError::new_err(format!(
                            "plot() values length must equal diagnostics.n; got values_len={}, diagnostics.n={diagnostics_n}",
                            series.len()
                        )));
                    }
                    vec![series]
                }
                2 => {
                    let shape: Vec<usize> = array.getattr("shape")?.extract()?;
                    let rows_len = *shape.first().ok_or_else(|| {
                        PyValueError::new_err("plot() could not determine rows from 2D input shape")
                    })?;
                    let dims = *shape.get(1).ok_or_else(|| {
                        PyValueError::new_err(
                            "plot() could not determine dimensions from 2D input shape",
                        )
                    })?;
                    if rows_len != diagnostics_n {
                        return Err(PyValueError::new_err(format!(
                            "plot() values rows must equal diagnostics.n; got rows={}, diagnostics.n={diagnostics_n}",
                            rows_len
                        )));
                    }
                    if dims == 0 {
                        return Err(PyValueError::new_err(
                            "plot() values must contain at least one column",
                        ));
                    }
                    if diagnostics_d > 0 && dims != diagnostics_d {
                        return Err(PyValueError::new_err(format!(
                            "plot() values column count must equal diagnostics.d; got dims={dims}, diagnostics.d={diagnostics_d}"
                        )));
                    }
                    let rows: Vec<Vec<f64>> = array.extract().map_err(|_| {
                        PyTypeError::new_err(
                            "plot() values must be numeric and convertible to a 2D float array",
                        )
                    })?;
                    if rows.len() != rows_len {
                        return Err(PyValueError::new_err(format!(
                            "plot() values rows do not match input shape; got rows={}, shape_rows={rows_len}",
                            rows.len()
                        )));
                    }
                    if rows_len == 0 {
                        vec![Vec::new(); dims]
                    } else {
                        for (row_idx, row) in rows.iter().enumerate() {
                            if row.len() != dims {
                                return Err(PyValueError::new_err(format!(
                                    "plot() values row {row_idx} has length {}, expected {dims}",
                                    row.len()
                                )));
                            }
                        }
                        let mut by_dim = vec![Vec::with_capacity(rows.len()); dims];
                        for row in rows {
                            for (dim, value) in row.into_iter().enumerate() {
                                by_dim[dim].push(value);
                            }
                        }
                        by_dim
                    }
                }
                _ => Err(PyValueError::new_err(format!(
                    "plot() expects 1D or 2D input values; got ndim={ndim}"
                )))?,
            }
        } else {
            let segments = self.segments.as_ref().ok_or_else(|| {
                PyValueError::new_err("plot() requires values when result.segments are unavailable")
            })?;
            let inferred_d = if diagnostics_d > 0 {
                diagnostics_d
            } else {
                segments
                    .iter()
                    .filter_map(|segment| segment.mean.as_ref().map(Vec::len))
                    .max()
                    .unwrap_or(0)
            };
            build_segment_mean_summary(diagnostics_n, inferred_d, segments)?
        };

        let n_axes = series_by_dimension.len();
        if n_axes == 0 {
            return Err(PyValueError::new_err(
                "plot() could not infer any plottable dimensions",
            ));
        }

        let (figure, axes): (Py<PyAny>, Vec<Py<PyAny>>) = if let Some(provided_ax) = ax {
            if n_axes != 1 {
                return Err(PyValueError::new_err(
                    "plot(ax=...) is only supported for univariate data; omit ax for multivariate plotting",
                ));
            }
            let figure = provided_ax.getattr("figure").map_err(|_| {
                PyTypeError::new_err(
                    "plot(ax=...) requires a matplotlib Axes-like object with a 'figure' attribute",
                )
            })?;
            (figure.unbind(), vec![provided_ax.clone().unbind()])
        } else {
            let kwargs = PyDict::new(py);
            kwargs.set_item("sharex", n_axes > 1)?;
            kwargs.set_item("squeeze", false)?;
            kwargs.set_item("figsize", (10.0f64, (n_axes as f64 * 2.75).max(3.5)))?;

            let subplots = pyplot.call_method("subplots", (n_axes, 1usize), Some(&kwargs))?;
            let fig = subplots.get_item(0)?;
            let axes_grid = subplots.get_item(1)?;
            let axes: Vec<Py<PyAny>> = axes_grid
                .call_method0("ravel")?
                .call_method0("tolist")?
                .extract()?;
            (fig.unbind(), axes)
        };

        for (dim, (axis, series)) in axes.iter().zip(series_by_dimension.iter()).enumerate() {
            let axis = axis.bind(py);
            let kwargs = PyDict::new(py);
            kwargs.set_item("linewidth", line_width)?;
            if n_axes == 1 {
                kwargs.set_item("label", "signal")?;
            } else {
                kwargs.set_item("label", format!("signal[{dim}]"))?;
            }
            axis.call_method("plot", (series.clone(),), Some(&kwargs))?;

            if n_axes > 1 {
                axis.call_method1("set_ylabel", (format!("x[{dim}]"),))?;
            }
        }

        for (axis_index, axis) in axes.iter().enumerate() {
            let axis = axis.bind(py);
            for (cp_index, change_point) in self.change_points.iter().enumerate() {
                let kwargs = PyDict::new(py);
                kwargs.set_item("color", breakpoint_color)?;
                kwargs.set_item("linestyle", breakpoint_style)?;
                kwargs.set_item("linewidth", 1.25f64)?;
                if axis_index == 0 && cp_index == 0 {
                    kwargs.set_item("label", "change-point")?;
                }
                axis.call_method("axvline", (*change_point,), Some(&kwargs))?;
            }
        }

        if let Some(title) = title {
            axes[0].bind(py).call_method1("set_title", (title,))?;
        }
        axes[n_axes - 1]
            .bind(py)
            .call_method1("set_xlabel", ("t",))?;

        if show_legend {
            for axis in &axes {
                axis.bind(py).call_method1("legend", ("best",))?;
            }
        }

        figure.bind(py).call_method0("tight_layout")?;
        Ok(figure)
    }

    fn __repr__(&self) -> String {
        format!(
            "OfflineChangePointResult(breakpoints={:?}, change_points={:?}, scores_len={}, segments_len={})",
            self.breakpoints,
            self.change_points,
            self.scores.as_ref().map_or(0, Vec::len),
            self.segments.as_ref().map_or(0, Vec::len)
        )
    }
}

impl From<CoreOfflineChangePointResult> for PyOfflineChangePointResult {
    fn from(result: CoreOfflineChangePointResult) -> Self {
        let diagnostics = PyDiagnostics::from(result.diagnostics);
        let n = if diagnostics.n == 0 {
            result.breakpoints.last().copied().unwrap_or(0)
        } else {
            diagnostics.n
        };
        let change_points = derive_change_points(n, &result.breakpoints);

        Self {
            breakpoints: result.breakpoints,
            change_points,
            scores: result.scores,
            segments: result
                .segments
                .map(|segments| segments.into_iter().map(Into::into).collect()),
            diagnostics,
            #[cfg(feature = "serde")]
            result_unknown_fields: JsonMap::new(),
            #[cfg(feature = "serde")]
            diagnostics_unknown_fields: JsonMap::new(),
        }
    }
}

impl PyOfflineChangePointResult {
    #[cfg(feature = "serde")]
    fn to_core(&self) -> PyResult<CoreOfflineChangePointResult> {
        Ok(CoreOfflineChangePointResult {
            breakpoints: self.breakpoints.clone(),
            change_points: self.change_points.clone(),
            scores: self.scores.clone(),
            segments: self
                .segments
                .as_ref()
                .map(|segments| segments.iter().map(PySegmentStats::to_core).collect()),
            diagnostics: self.diagnostics.to_core()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PyDiagnostics, PyOfflineChangePointResult, PyOnlineStepResult, PyPruningStats,
        PySegmentStats, repro_mode_to_str,
    };
    use cpd_core::{
        DIAGNOSTICS_SCHEMA_VERSION, Diagnostics as CoreDiagnostics,
        OfflineChangePointResult as CoreOfflineChangePointResult,
        OnlineStepResult as CoreOnlineStepResult, PruningStats as CorePruningStats, ReproMode,
        SegmentStats as CoreSegmentStats,
    };
    #[cfg(not(feature = "serde"))]
    use pyo3::exceptions::PyNotImplementedError;
    #[cfg(feature = "serde")]
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::{PyAnyMethods, PyDict, PyList};
    #[cfg(feature = "serde")]
    use serde_json::Value;
    use std::borrow::Cow;
    use std::ffi::CString;
    use std::sync::Once;

    #[cfg(feature = "serde")]
    const OFFLINE_RESULT_FIXTURE_JSON: &str =
        include_str!("../tests/fixtures/offline_result_v1.json");
    #[cfg(feature = "serde")]
    const OFFLINE_RESULT_V2_ADDITIVE_FIXTURE_JSON: &str =
        include_str!("../../../tests/fixtures/migrations/result/offline_result.v2.additive.json");

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
    ) -> PyResult<()> {
        let code = CString::new(code).expect("python snippet should not contain NUL bytes");
        py.run(code.as_c_str(), globals, locals)
    }

    fn sample_core_result() -> CoreOfflineChangePointResult {
        let diagnostics = CoreDiagnostics {
            n: 100,
            d: 2,
            schema_version: 0,
            engine_version: None,
            runtime_ms: Some(123),
            notes: vec!["run complete".to_string()],
            warnings: vec!["none".to_string()],
            algorithm: Cow::Owned("pelt".to_string()),
            cost_model: Cow::Owned("l2_mean".to_string()),
            seed: Some(7),
            repro_mode: ReproMode::Balanced,
            thread_count: Some(4),
            blas_backend: Some("openblas".to_string()),
            cpu_features: Some(vec!["avx2".to_string(), "fma".to_string()]),
            #[cfg(feature = "serde")]
            params_json: Some(serde_json::json!({
                "jump": 5,
                "min_segment_len": 10
            })),
            pruning_stats: Some(CorePruningStats {
                candidates_considered: 150,
                candidates_pruned: 120,
            }),
            missing_policy_applied: Some("Ignore".to_string()),
            missing_fraction: Some(0.1),
            effective_sample_count: Some(90),
        };

        CoreOfflineChangePointResult {
            breakpoints: vec![40, 100],
            change_points: vec![3, 7, 9],
            scores: Some(vec![0.75]),
            segments: Some(vec![
                CoreSegmentStats {
                    start: 0,
                    end: 40,
                    mean: Some(vec![1.0, 2.0]),
                    variance: Some(vec![0.1, 0.2]),
                    count: 40,
                    missing_count: 2,
                },
                CoreSegmentStats {
                    start: 40,
                    end: 100,
                    mean: Some(vec![3.0, 4.0]),
                    variance: Some(vec![0.3, 0.4]),
                    count: 60,
                    missing_count: 3,
                },
            ]),
            diagnostics,
        }
    }

    fn sample_online_step() -> CoreOnlineStepResult {
        CoreOnlineStepResult {
            t: 12,
            p_change: 0.42,
            alert: true,
            alert_reason: Some("threshold crossed".to_string()),
            run_length_mode: 3,
            run_length_mean: 2.75,
            processing_latency_us: Some(120),
        }
    }

    fn sample_univariate_result() -> CoreOfflineChangePointResult {
        let diagnostics = CoreDiagnostics {
            n: 100,
            d: 1,
            schema_version: DIAGNOSTICS_SCHEMA_VERSION,
            engine_version: Some("test-engine".to_string()),
            runtime_ms: Some(5),
            notes: vec!["summary-only".to_string()],
            warnings: vec![],
            algorithm: Cow::Owned("pelt".to_string()),
            cost_model: Cow::Owned("l2".to_string()),
            seed: Some(3),
            repro_mode: ReproMode::Balanced,
            thread_count: None,
            blas_backend: None,
            cpu_features: None,
            #[cfg(feature = "serde")]
            params_json: None,
            pruning_stats: None,
            missing_policy_applied: None,
            missing_fraction: None,
            effective_sample_count: None,
        };
        CoreOfflineChangePointResult {
            breakpoints: vec![50, 100],
            change_points: vec![50],
            scores: Some(vec![0.5]),
            segments: Some(vec![
                CoreSegmentStats {
                    start: 0,
                    end: 50,
                    mean: Some(vec![0.0]),
                    variance: Some(vec![1.0]),
                    count: 50,
                    missing_count: 0,
                },
                CoreSegmentStats {
                    start: 50,
                    end: 100,
                    mean: Some(vec![5.0]),
                    variance: Some(vec![1.5]),
                    count: 50,
                    missing_count: 0,
                },
            ]),
            diagnostics,
        }
    }

    fn sample_empty_univariate_result() -> CoreOfflineChangePointResult {
        let diagnostics = CoreDiagnostics {
            n: 0,
            d: 1,
            schema_version: DIAGNOSTICS_SCHEMA_VERSION,
            engine_version: Some("test-engine".to_string()),
            runtime_ms: Some(0),
            notes: vec!["empty".to_string()],
            warnings: vec![],
            algorithm: Cow::Owned("pelt".to_string()),
            cost_model: Cow::Owned("l2".to_string()),
            seed: None,
            repro_mode: ReproMode::Balanced,
            thread_count: None,
            blas_backend: None,
            cpu_features: None,
            #[cfg(feature = "serde")]
            params_json: None,
            pruning_stats: None,
            missing_policy_applied: None,
            missing_fraction: None,
            effective_sample_count: Some(0),
        };
        CoreOfflineChangePointResult {
            breakpoints: vec![],
            change_points: vec![],
            scores: Some(vec![]),
            segments: Some(vec![]),
            diagnostics,
        }
    }

    #[test]
    fn breakpoints_drive_change_points_derivation() {
        let py_result = PyOfflineChangePointResult::from(sample_core_result());
        assert_eq!(py_result.breakpoints, vec![40, 100]);
        assert_eq!(py_result.change_points, vec![40]);
    }

    #[test]
    fn diagnostics_normalizes_schema_and_engine_version() {
        let py_result = PyOfflineChangePointResult::from(sample_core_result());
        assert_eq!(
            py_result.diagnostics.schema_version,
            DIAGNOSTICS_SCHEMA_VERSION
        );
        assert_eq!(
            py_result.diagnostics.engine_version,
            Some(env!("CARGO_PKG_VERSION").to_string())
        );
        assert_eq!(
            py_result.diagnostics.repro_mode,
            repro_mode_to_str(ReproMode::Balanced)
        );
    }

    #[test]
    fn python_properties_are_accessible_and_typed() {
        with_python(|py| {
            let py_result = Py::new(py, PyOfflineChangePointResult::from(sample_core_result()))
                .expect("result object should be constructible");
            let result_any = py_result.bind(py);

            let breakpoints: Vec<usize> = result_any
                .getattr("breakpoints")
                .expect("breakpoints should be exported")
                .extract()
                .expect("breakpoints should be list[int]");
            let change_points: Vec<usize> = result_any
                .getattr("change_points")
                .expect("change_points should be exported")
                .extract()
                .expect("change_points should be list[int]");
            let scores: Option<Vec<f64>> = result_any
                .getattr("scores")
                .expect("scores should be exported")
                .extract()
                .expect("scores should be optional list[float]");

            assert_eq!(breakpoints, vec![40, 100]);
            assert_eq!(change_points, vec![40]);
            assert_eq!(scores, Some(vec![0.75]));

            let diagnostics = result_any
                .getattr("diagnostics")
                .expect("diagnostics should be exported");
            let algorithm: String = diagnostics
                .getattr("algorithm")
                .expect("algorithm should be exported")
                .extract()
                .expect("algorithm should be a string");
            let missing_fraction: Option<f64> = diagnostics
                .getattr("missing_fraction")
                .expect("missing_fraction should be exported")
                .extract()
                .expect("missing_fraction should be optional float");
            assert_eq!(algorithm, "pelt");
            assert_eq!(missing_fraction, Some(0.1));

            let pruning = diagnostics
                .getattr("pruning_stats")
                .expect("pruning_stats should be exported");
            assert!(!pruning.is_none());
            let considered: usize = pruning
                .getattr("candidates_considered")
                .expect("candidates_considered should be exported")
                .extract()
                .expect("candidates_considered should be int");
            assert_eq!(considered, 150);

            let segments_any = result_any
                .getattr("segments")
                .expect("segments should be exported");
            let segments = segments_any
                .downcast::<PyList>()
                .expect("segments should be list[SegmentStats]");
            assert_eq!(segments.len(), 2);
            let first = segments.get_item(0).expect("first segment should exist");
            let first_count: usize = first
                .getattr("count")
                .expect("segment count should be exported")
                .extract()
                .expect("segment count should be int");
            assert_eq!(first_count, 40);

            let params_json = diagnostics
                .getattr("params_json")
                .expect("params_json should be exported");
            #[cfg(feature = "serde")]
            {
                let as_repr: String = params_json
                    .repr()
                    .expect("repr should succeed")
                    .extract()
                    .expect("repr should be string");
                assert!(as_repr.contains("min_segment_len"));
            }
            #[cfg(not(feature = "serde"))]
            {
                assert!(params_json.is_none());
            }
        });
    }

    #[test]
    fn repr_is_human_readable() {
        with_python(|py| {
            let py_result = Py::new(py, PyOfflineChangePointResult::from(sample_core_result()))
                .expect("result object should be constructible");
            let result_repr: String = py_result
                .bind(py)
                .repr()
                .expect("repr should succeed")
                .extract()
                .expect("repr should be string");
            assert!(result_repr.contains("OfflineChangePointResult"));
            assert!(result_repr.contains("breakpoints"));

            let diagnostics = py_result
                .bind(py)
                .getattr("diagnostics")
                .expect("diagnostics should be exported");
            let diagnostics_repr: String = diagnostics
                .repr()
                .expect("repr should succeed")
                .extract()
                .expect("repr should be string");
            assert!(diagnostics_repr.contains("Diagnostics"));

            let pruning = diagnostics
                .getattr("pruning_stats")
                .expect("pruning_stats should be exported");
            let pruning_repr: String = pruning
                .repr()
                .expect("repr should succeed")
                .extract()
                .expect("repr should be string");
            assert!(pruning_repr.contains("PruningStats"));

            let segments_any = py_result
                .bind(py)
                .getattr("segments")
                .expect("segments should be exported");
            let segments = segments_any
                .downcast::<PyList>()
                .expect("segments should be list[SegmentStats]");
            let segment_repr: String = segments
                .get_item(0)
                .expect("segment should exist")
                .repr()
                .expect("repr should succeed")
                .extract()
                .expect("repr should be string");
            assert!(segment_repr.contains("SegmentStats"));
        });
    }

    #[test]
    fn online_step_result_properties_and_repr_are_stable() {
        with_python(|py| {
            let py_step = Py::new(py, PyOnlineStepResult::from(sample_online_step()))
                .expect("online step should be constructible");
            let step_any = py_step.bind(py);

            let t: usize = step_any
                .getattr("t")
                .expect("t should be exported")
                .extract()
                .expect("t should be int");
            let p_change: f64 = step_any
                .getattr("p_change")
                .expect("p_change should be exported")
                .extract()
                .expect("p_change should be float");
            let alert: bool = step_any
                .getattr("alert")
                .expect("alert should be exported")
                .extract()
                .expect("alert should be bool");
            let reason: Option<String> = step_any
                .getattr("alert_reason")
                .expect("alert_reason should be exported")
                .extract()
                .expect("alert_reason should be optional string");
            let run_length_mode: usize = step_any
                .getattr("run_length_mode")
                .expect("run_length_mode should be exported")
                .extract()
                .expect("run_length_mode should be int");
            let run_length_mean: f64 = step_any
                .getattr("run_length_mean")
                .expect("run_length_mean should be exported")
                .extract()
                .expect("run_length_mean should be float");
            let processing_latency_us: Option<u64> = step_any
                .getattr("processing_latency_us")
                .expect("processing_latency_us should be exported")
                .extract()
                .expect("processing_latency_us should be optional int");

            assert_eq!(t, 12);
            assert!((p_change - 0.42).abs() < f64::EPSILON);
            assert!(alert);
            assert_eq!(reason.as_deref(), Some("threshold crossed"));
            assert_eq!(run_length_mode, 3);
            assert!((run_length_mean - 2.75).abs() < f64::EPSILON);
            assert_eq!(processing_latency_us, Some(120));

            let repr: String = step_any
                .repr()
                .expect("repr should succeed")
                .extract()
                .expect("repr should be string");
            assert!(repr.contains("OnlineStepResult"));
            assert!(repr.contains("p_change"));
        });
    }

    #[test]
    #[cfg(not(feature = "serde"))]
    fn to_json_requires_serde_feature() {
        with_python(|py| {
            let py_result = Py::new(py, PyOfflineChangePointResult::from(sample_core_result()))
                .expect("result object should be constructible");

            let err = py_result
                .bind(py)
                .call_method0("to_json")
                .expect_err("to_json should fail without serde feature");
            assert!(err.is_instance_of::<PyNotImplementedError>(py));
            assert!(
                err.to_string()
                    .contains("to_json() requires cpd-python built with serde feature")
            );
        });
    }

    #[test]
    #[cfg(feature = "serde")]
    fn to_json_roundtrip_preserves_core_payload() {
        with_python(|py| {
            let py_result = Py::new(py, PyOfflineChangePointResult::from(sample_core_result()))
                .expect("result object should be constructible");

            let json_payload: String = py_result
                .bind(py)
                .call_method0("to_json")
                .expect("to_json should succeed")
                .extract()
                .expect("to_json should return string");

            let decoded: CoreOfflineChangePointResult =
                serde_json::from_str(&json_payload).expect("json should deserialize");

            assert_eq!(decoded.breakpoints, vec![40, 100]);
            assert_eq!(decoded.change_points, vec![40]);
            assert_eq!(decoded.scores, Some(vec![0.75]));
            assert_eq!(
                decoded.diagnostics.schema_version,
                DIAGNOSTICS_SCHEMA_VERSION
            );
            assert_eq!(
                decoded.diagnostics.engine_version,
                Some(env!("CARGO_PKG_VERSION").to_string())
            );
            assert_eq!(decoded.diagnostics.algorithm, "pelt");
            assert_eq!(decoded.diagnostics.cost_model, "l2_mean");
            assert!(decoded.diagnostics.pruning_stats.is_some());
            assert_eq!(
                decoded.diagnostics.params_json,
                Some(serde_json::json!({
                    "jump": 5,
                    "min_segment_len": 10
                }))
            );
        });
    }

    #[test]
    #[cfg(feature = "serde")]
    fn serde_fixture_deserializes_and_validates() {
        let decoded: CoreOfflineChangePointResult =
            serde_json::from_str(OFFLINE_RESULT_FIXTURE_JSON)
                .expect("fixture should deserialize as core result");
        decoded
            .validate(decoded.diagnostics.n)
            .expect("fixture should validate");
        assert_eq!(decoded.breakpoints, vec![40, 100]);
        assert_eq!(decoded.change_points, vec![40]);
        assert_eq!(
            decoded.diagnostics.schema_version,
            DIAGNOSTICS_SCHEMA_VERSION
        );
    }

    #[test]
    #[cfg(feature = "serde")]
    fn to_json_matches_versioned_fixture_shape() {
        with_python(|py| {
            let py_result = Py::new(py, PyOfflineChangePointResult::from(sample_core_result()))
                .expect("result object should be constructible");

            let json_payload: String = py_result
                .bind(py)
                .call_method0("to_json")
                .expect("to_json should succeed")
                .extract()
                .expect("to_json should return string");

            let generated: Value =
                serde_json::from_str(&json_payload).expect("generated JSON should parse");
            let mut fixture: Value = serde_json::from_str(OFFLINE_RESULT_FIXTURE_JSON)
                .expect("fixture JSON should parse");

            // Keep fixture stable across version bumps while asserting full shape equality.
            if let Some(diagnostics) = fixture
                .get_mut("diagnostics")
                .and_then(serde_json::Value::as_object_mut)
            {
                diagnostics.insert(
                    "engine_version".to_string(),
                    Value::String(env!("CARGO_PKG_VERSION").to_string()),
                );
            }

            assert_eq!(generated, fixture);
        });
    }

    #[test]
    #[cfg(not(feature = "serde"))]
    fn from_json_requires_serde_feature() {
        with_python(|py| {
            let py_result = Py::new(py, PyOfflineChangePointResult::from(sample_core_result()))
                .expect("result object should be constructible");
            let result_type = py_result
                .bind(py)
                .getattr("__class__")
                .expect("result class should be available");

            let err = result_type
                .call_method1("from_json", ("{}",))
                .expect_err("from_json should fail without serde feature");
            assert!(err.is_instance_of::<PyNotImplementedError>(py));
            assert!(
                err.to_string()
                    .contains("from_json() requires cpd-python built with serde feature")
            );
        });
    }

    #[test]
    #[cfg(feature = "serde")]
    fn from_json_roundtrip_preserves_breakpoints_and_diagnostics() {
        with_python(|py| {
            let py_result = Py::new(py, PyOfflineChangePointResult::from(sample_core_result()))
                .expect("result object should be constructible");
            let result_type = py_result
                .bind(py)
                .getattr("__class__")
                .expect("result class should be available");

            let json_payload: String = py_result
                .bind(py)
                .call_method0("to_json")
                .expect("to_json should succeed")
                .extract()
                .expect("to_json should return string");
            let roundtrip = result_type
                .call_method1("from_json", (json_payload,))
                .expect("from_json should succeed");

            let breakpoints: Vec<usize> = roundtrip
                .getattr("breakpoints")
                .expect("breakpoints should be available")
                .extract()
                .expect("breakpoints should be typed");
            assert_eq!(breakpoints, vec![40, 100]);

            let diagnostics = roundtrip
                .getattr("diagnostics")
                .expect("diagnostics should be available");
            let algorithm: String = diagnostics
                .getattr("algorithm")
                .expect("algorithm should be available")
                .extract()
                .expect("algorithm should be a string");
            let schema_version: u32 = diagnostics
                .getattr("schema_version")
                .expect("schema_version should be available")
                .extract()
                .expect("schema_version should be int");
            assert_eq!(algorithm, "pelt");
            assert_eq!(schema_version, DIAGNOSTICS_SCHEMA_VERSION);
        });
    }

    #[test]
    #[cfg(feature = "serde")]
    fn from_json_reports_structural_parse_errors() {
        with_python(|py| {
            let py_result = Py::new(py, PyOfflineChangePointResult::from(sample_core_result()))
                .expect("result object should be constructible");
            let result_type = py_result
                .bind(py)
                .getattr("__class__")
                .expect("result class should be available");

            let err = result_type
                .call_method1("from_json", ("{\"breakpoints\": [1]}",))
                .expect_err("missing required fields should fail");
            assert!(err.is_instance_of::<PyValueError>(py));
            let message = err.to_string();
            assert!(message.contains("invalid OfflineChangePointResult JSON payload"));
            assert!(message.contains("missing field"));
        });
    }

    #[test]
    #[cfg(feature = "serde")]
    fn from_json_reports_schema_version_and_semantic_validation_errors() {
        let mut unsupported: Value =
            serde_json::from_str(OFFLINE_RESULT_FIXTURE_JSON).expect("fixture should parse");
        unsupported["diagnostics"]["schema_version"] = Value::from(99u32);

        let mut invalid_semantics: Value =
            serde_json::from_str(OFFLINE_RESULT_FIXTURE_JSON).expect("fixture should parse");
        invalid_semantics["change_points"] = Value::Array(vec![Value::from(1u64)]);

        with_python(|py| {
            let py_result = Py::new(py, PyOfflineChangePointResult::from(sample_core_result()))
                .expect("result object should be constructible");
            let result_type = py_result
                .bind(py)
                .getattr("__class__")
                .expect("result class should be available");

            let unsupported_payload =
                serde_json::to_string(&unsupported).expect("payload should serialize");
            let err = result_type
                .call_method1("from_json", (unsupported_payload,))
                .expect_err("unsupported schema should fail");
            assert!(err.is_instance_of::<PyValueError>(py));
            let message = err.to_string();
            assert!(message.contains("schema_version=99"));
            assert!(message.contains("supported versions are 1..=2"));

            let invalid_payload =
                serde_json::to_string(&invalid_semantics).expect("payload should serialize");
            let err = result_type
                .call_method1("from_json", (invalid_payload,))
                .expect_err("semantic mismatch should fail");
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(
                err.to_string()
                    .contains("change_points must equal breakpoints excluding n")
            );
        });
    }

    #[test]
    #[cfg(feature = "serde")]
    fn from_json_to_json_roundtrip_preserves_additive_unknown_fields() {
        with_python(|py| {
            let py_result = Py::new(py, PyOfflineChangePointResult::from(sample_core_result()))
                .expect("result object should be constructible");
            let result_type = py_result
                .bind(py)
                .getattr("__class__")
                .expect("result class should be available");

            let parsed = result_type
                .call_method1("from_json", (OFFLINE_RESULT_V2_ADDITIVE_FIXTURE_JSON,))
                .expect("from_json should accept additive v2 fixture");
            let reserialized: String = parsed
                .call_method0("to_json")
                .expect("to_json should succeed")
                .extract()
                .expect("to_json should return a JSON string");

            let payload: Value =
                serde_json::from_str(&reserialized).expect("payload should parse as JSON");
            let result_flag = payload
                .get("future_result_flag")
                .and_then(Value::as_str)
                .expect("result unknown field should roundtrip");
            let diag_source = payload
                .get("diagnostics")
                .and_then(Value::as_object)
                .and_then(|diagnostics| diagnostics.get("future_diagnostics_flag"))
                .and_then(Value::as_object)
                .and_then(|extra| extra.get("source"))
                .and_then(Value::as_str)
                .expect("diagnostics unknown field should roundtrip");
            assert_eq!(result_flag, "additive-v2");
            assert_eq!(diag_source, "v2");
        });
    }

    #[test]
    fn plot_reports_missing_matplotlib_with_clear_error() {
        with_python(|py| {
            let py_result = Py::new(
                py,
                PyOfflineChangePointResult::from(sample_univariate_result()),
            )
            .expect("result object should be constructible");
            let locals = PyDict::new(py);
            locals
                .set_item("result", py_result)
                .expect("locals should accept result");

            run_python(
                py,
                r#"
import builtins

_orig_import = builtins.__import__
def _blocked(name, *args, **kwargs):
    if name == "matplotlib" or name.startswith("matplotlib."):
        raise ImportError("blocked for plot test")
    return _orig_import(name, *args, **kwargs)

builtins.__import__ = _blocked
try:
    try:
        result.plot()
        raise AssertionError("plot() should require matplotlib")
    except ImportError as exc:
        assert "plot() requires optional dependency 'matplotlib'" in str(exc)
finally:
    builtins.__import__ = _orig_import
"#,
                None,
                Some(&locals),
            )
            .expect("missing matplotlib path should be explicit");
        });
    }

    #[test]
    fn plot_supports_univariate_and_multivariate_paths_with_fake_backend() {
        with_python(|py| {
            let univariate = Py::new(
                py,
                PyOfflineChangePointResult::from(sample_univariate_result()),
            )
            .expect("univariate result should be constructible");
            let multivariate = Py::new(py, PyOfflineChangePointResult::from(sample_core_result()))
                .expect("multivariate result should be constructible");
            let locals = PyDict::new(py);
            locals
                .set_item("univariate", univariate)
                .expect("locals should accept univariate");
            locals
                .set_item("multivariate", multivariate)
                .expect("locals should accept multivariate");

            run_python(
                py,
                r#"
import sys
import types
import numpy as np

class _FakeAxis:
    def __init__(self):
        self.lines = []
        self.vlines = []
        self.xlabel = None
        self.ylabel = None
        self.title = None
        self.legend_calls = 0
        self.figure = None
    def plot(self, y, **kwargs):
        self.lines.append((list(y), dict(kwargs)))
    def axvline(self, x, **kwargs):
        self.vlines.append((int(x), dict(kwargs)))
    def set_title(self, title):
        self.title = title
    def set_ylabel(self, ylabel):
        self.ylabel = ylabel
    def set_xlabel(self, xlabel):
        self.xlabel = xlabel
    def legend(self, *_args, **_kwargs):
        self.legend_calls += 1

class _FakeFigure:
    def __init__(self):
        self.axes = []
        self.tight_layout_calls = 0
    def tight_layout(self):
        self.tight_layout_calls += 1

def _subplots(rows, cols=1, **_kwargs):
    fig = _FakeFigure()
    axes_grid = np.empty((rows, cols), dtype=object)
    for r in range(rows):
        for c in range(cols):
            axis = _FakeAxis()
            axis.figure = fig
            axes_grid[r, c] = axis
            fig.axes.append(axis)
    return fig, axes_grid

pyplot = types.ModuleType("matplotlib.pyplot")
pyplot.subplots = _subplots
matplotlib = types.ModuleType("matplotlib")
matplotlib.pyplot = pyplot
_prev_matplotlib = sys.modules.get("matplotlib")
_prev_pyplot = sys.modules.get("matplotlib.pyplot")
sys.modules["matplotlib"] = matplotlib
sys.modules["matplotlib.pyplot"] = pyplot

try:
    uni_fig = univariate.plot(values=np.linspace(0.0, 1.0, 100), title="univariate")
    assert len(uni_fig.axes) == 1
    assert len(uni_fig.axes[0].lines) == 1
    assert [x for x, _ in uni_fig.axes[0].vlines] == [50]
    assert uni_fig.axes[0].title == "univariate"
    assert uni_fig.tight_layout_calls == 1

    multi_fig = multivariate.plot()
    assert len(multi_fig.axes) == 2
    assert all(len(axis.lines) == 1 for axis in multi_fig.axes)
    assert all([x for x, _ in axis.vlines] == [40] for axis in multi_fig.axes)
    assert multi_fig.axes[0].ylabel == "x[0]"
    assert multi_fig.axes[1].ylabel == "x[1]"
    assert multi_fig.axes[1].xlabel == "t"
    assert multi_fig.tight_layout_calls == 1
finally:
    if _prev_matplotlib is None:
        sys.modules.pop("matplotlib", None)
    else:
        sys.modules["matplotlib"] = _prev_matplotlib
    if _prev_pyplot is None:
        sys.modules.pop("matplotlib.pyplot", None)
    else:
        sys.modules["matplotlib.pyplot"] = _prev_pyplot
"#,
                None,
                Some(&locals),
            )
            .expect("fake backend should exercise plot paths");
        });
    }

    #[test]
    fn plot_accepts_empty_2d_values_when_diagnostics_n_is_zero() {
        with_python(|py| {
            let empty = Py::new(
                py,
                PyOfflineChangePointResult::from(sample_empty_univariate_result()),
            )
            .expect("empty result should be constructible");
            let locals = PyDict::new(py);
            locals
                .set_item("empty_result", empty)
                .expect("locals should accept empty result");

            run_python(
                py,
                r#"
import sys
import types
import numpy as np

class _FakeAxis:
    def __init__(self):
        self.lines = []
        self.vlines = []
        self.figure = None
    def plot(self, y, **kwargs):
        self.lines.append((list(y), dict(kwargs)))
    def axvline(self, x, **kwargs):
        self.vlines.append((int(x), dict(kwargs)))
    def set_title(self, *_args, **_kwargs):
        pass
    def set_ylabel(self, *_args, **_kwargs):
        pass
    def set_xlabel(self, *_args, **_kwargs):
        pass
    def legend(self, *_args, **_kwargs):
        pass

class _FakeFigure:
    def __init__(self):
        self.axes = []
        self.tight_layout_calls = 0
    def tight_layout(self):
        self.tight_layout_calls += 1

def _subplots(rows, cols=1, **_kwargs):
    fig = _FakeFigure()
    axes_grid = np.empty((rows, cols), dtype=object)
    for r in range(rows):
        for c in range(cols):
            axis = _FakeAxis()
            axis.figure = fig
            axes_grid[r, c] = axis
            fig.axes.append(axis)
    return fig, axes_grid

pyplot = types.ModuleType("matplotlib.pyplot")
pyplot.subplots = _subplots
matplotlib = types.ModuleType("matplotlib")
matplotlib.pyplot = pyplot
_prev_matplotlib = sys.modules.get("matplotlib")
_prev_pyplot = sys.modules.get("matplotlib.pyplot")
sys.modules["matplotlib"] = matplotlib
sys.modules["matplotlib.pyplot"] = pyplot
try:
    fig = empty_result.plot(values=np.empty((0, 1), dtype=float))
    assert len(fig.axes) == 1
    assert fig.axes[0].lines[0][0] == []
finally:
    if _prev_matplotlib is None:
        sys.modules.pop("matplotlib", None)
    else:
        sys.modules["matplotlib"] = _prev_matplotlib
    if _prev_pyplot is None:
        sys.modules.pop("matplotlib.pyplot", None)
    else:
        sys.modules["matplotlib.pyplot"] = _prev_pyplot
"#,
                None,
                Some(&locals),
            )
            .expect("n=0 2D values should be accepted");
        });
    }

    #[test]
    fn py_wrappers_convert_from_core_types() {
        let stats = PyPruningStats::from(CorePruningStats {
            candidates_considered: 10,
            candidates_pruned: 9,
        });
        assert_eq!(stats.candidates_considered, 10);
        assert_eq!(stats.candidates_pruned, 9);

        let segment = PySegmentStats::from(CoreSegmentStats {
            start: 0,
            end: 5,
            mean: None,
            variance: None,
            count: 5,
            missing_count: 0,
        });
        assert_eq!(segment.start, 0);
        assert_eq!(segment.end, 5);

        let diagnostics = PyDiagnostics::from(CoreDiagnostics {
            n: 5,
            d: 1,
            schema_version: DIAGNOSTICS_SCHEMA_VERSION,
            engine_version: Some("x.y.z".to_string()),
            runtime_ms: Some(1),
            notes: vec![],
            warnings: vec![],
            algorithm: Cow::Borrowed("binseg"),
            cost_model: Cow::Borrowed("normal_mean_var"),
            seed: None,
            repro_mode: ReproMode::Fast,
            thread_count: None,
            blas_backend: None,
            cpu_features: None,
            #[cfg(feature = "serde")]
            params_json: None,
            pruning_stats: None,
            missing_policy_applied: None,
            missing_fraction: None,
            effective_sample_count: None,
        });
        assert_eq!(diagnostics.algorithm, "binseg");
        assert_eq!(diagnostics.repro_mode, "fast");
    }
}
