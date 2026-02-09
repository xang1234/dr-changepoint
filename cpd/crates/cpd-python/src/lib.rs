// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]

mod error_map;
mod numpy_interop;
mod result_objects;

use crate::error_map::cpd_error_to_pyerr;
use crate::numpy_interop::{DTypePolicy, OwnedSeries, parse_numpy_series};
use cpd_core::{
    Constraints, CpdError, ExecutionContext, MissingPolicy,
    OfflineChangePointResult as CoreOfflineChangePointResult, OfflineDetector, Penalty, Stopping,
};
use cpd_costs::{CostL2Mean, CostNormalMeanVar};
use cpd_offline::{Pelt as OfflinePelt, PeltConfig};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyModule};
use result_objects::{PyDiagnostics, PyOfflineChangePointResult, PyPruningStats, PySegmentStats};

fn parse_sequence(values: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    values.extract::<Vec<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "expected a sequence of float values for smoke detection",
        )
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PyPeltModel {
    L2,
    Normal,
}

impl PyPeltModel {
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

    fn params_per_segment(self) -> usize {
        match self {
            Self::L2 => 2,
            Self::Normal => 3,
        }
    }
}

fn resolve_stopping(pen: Option<f64>, n_bkps: Option<usize>) -> PyResult<Stopping> {
    match (pen, n_bkps) {
        (Some(_), Some(_)) => Err(PyValueError::new_err(
            "predict() requires exactly one of pen or n_bkps; got both",
        )),
        (None, None) => Err(PyValueError::new_err(
            "predict() requires exactly one of pen or n_bkps; got neither",
        )),
        (Some(beta), None) => {
            if !beta.is_finite() || beta <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "predict(pen=...) requires a finite value > 0.0; got {beta}"
                )));
            }
            Ok(Stopping::Penalized(Penalty::Manual(beta)))
        }
        (None, Some(k)) => {
            if k == 0 {
                return Err(PyValueError::new_err(
                    "predict(n_bkps=...) requires n_bkps >= 1; got 0",
                ));
            }
            Ok(Stopping::KnownK(k))
        }
    }
}

fn detect_with_model(
    model: PyPeltModel,
    series: &OwnedSeries,
    constraints: &Constraints,
    stopping: Stopping,
) -> Result<CoreOfflineChangePointResult, CpdError> {
    let view = series.view()?;
    let ctx = ExecutionContext::new(constraints);
    let config = PeltConfig {
        stopping,
        params_per_segment: model.params_per_segment(),
        cancel_check_every: 1000,
    };

    match model {
        PyPeltModel::L2 => {
            let detector = OfflinePelt::new(CostL2Mean::default(), config)?;
            detector.detect(&view, &ctx)
        }
        PyPeltModel::Normal => {
            let detector = OfflinePelt::new(CostNormalMeanVar::default(), config)?;
            detector.detect(&view, &ctx)
        }
    }
}

/// High-level ruptures-like Python interface for offline PELT detection.
#[pyclass(module = "cpd._cpd_rs", name = "Pelt")]
#[derive(Clone, Debug)]
pub struct PyPelt {
    model: PyPeltModel,
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
        let model = PyPeltModel::parse(model)?;
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
        let numpy = PyModule::import_bound(py, "numpy")?;
        let as_array = numpy.call_method1("asarray", (x,))?;
        let parsed = parse_numpy_series(
            py,
            &as_array,
            None,
            MissingPolicy::Error,
            DTypePolicy::KeepInput,
        )?;
        let owned = parsed.into_owned().map_err(cpd_error_to_pyerr)?;
        slf.fitted = Some(owned);
        Ok(slf)
    }

    #[pyo3(signature = (*, pen = None, n_bkps = None))]
    fn predict(
        &self,
        py: Python<'_>,
        pen: Option<f64>,
        n_bkps: Option<usize>,
    ) -> PyResult<PyOfflineChangePointResult> {
        let stopping = resolve_stopping(pen, n_bkps)?;
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
            .allow_threads(|| detect_with_model(self.model, fitted, &constraints, stopping))
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
    module.add_class::<SmokeDetector>()?;
    module.add_function(wrap_pyfunction!(smoke_detect, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{_cpd_rs, PyPelt, SmokeDetector, smoke_detect};
    use pyo3::Python;
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyList, PyModule};

    #[test]
    fn smoke_detector_rust_path_is_deterministic() {
        let mut detector = SmokeDetector::default();
        detector.set_fitted_len(3);
        assert_eq!(detector.predicted_breakpoints(), vec![3]);
    }

    #[test]
    fn smoke_detect_function_returns_terminal_breakpoint() {
        Python::with_gil(|py| {
            let values = PyList::new_bound(py, [0.0, 1.0, 2.0, 3.0]);
            let out = smoke_detect(values.as_any()).expect("smoke_detect should succeed");
            assert_eq!(out, vec![4]);
        });
    }

    #[test]
    fn module_registration_exposes_public_api() {
        Python::with_gil(|py| {
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
        });
    }

    #[test]
    fn pelt_fit_predict_penalized_roundtrip() {
        Python::with_gil(|py| {
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
        Python::with_gil(|py| {
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
    fn pelt_rejects_invalid_model_name() {
        Python::with_gil(|py| {
            let err = PyPelt::new("bad-model", 2, 1, None).expect_err("invalid model must fail");
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err.to_string().contains("unsupported model"));
        });
    }

    #[test]
    fn pelt_rejects_invalid_predict_argument_combinations() {
        Python::with_gil(|py| {
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
        Python::with_gil(|py| {
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
