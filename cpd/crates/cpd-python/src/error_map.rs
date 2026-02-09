// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![allow(dead_code)]

use cpd_core::CpdError;
use pyo3::PyErr;
use pyo3::exceptions::{PyFloatingPointError, PyNotImplementedError, PyRuntimeError, PyValueError};

/// Maps core cpd-rs errors into Python exception classes.
pub(crate) fn cpd_error_to_pyerr(err: CpdError) -> PyErr {
    match err {
        CpdError::InvalidInput(msg) => PyValueError::new_err(msg),
        CpdError::NumericalIssue(msg) => PyFloatingPointError::new_err(msg),
        CpdError::NotSupported(msg) => PyNotImplementedError::new_err(msg),
        CpdError::ResourceLimit(msg) => PyRuntimeError::new_err(msg),
        CpdError::Cancelled => PyRuntimeError::new_err("cancelled"),
    }
}
