// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::types::PyModule;

fn parse_sequence(values: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    values.extract::<Vec<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "expected a sequence of float values for smoke detection",
        )
    })
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
    module.add_class::<SmokeDetector>()?;
    module.add_function(wrap_pyfunction!(smoke_detect, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{_cpd_rs, SmokeDetector, smoke_detect};
    use pyo3::Python;
    use pyo3::types::{PyAnyMethods, PyList, PyModule};

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
        });
    }
}
