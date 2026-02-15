// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![allow(dead_code)]

use crate::error_map::cpd_error_to_pyerr;
use cpd_core::{CpdError, DTypeView, MemoryLayout, MissingPolicy, TimeIndex, TimeSeriesView};
use numpy::datetime::{Datetime, units};
use numpy::ndarray::IxDyn;
use numpy::{
    PyArrayDescr, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn,
    PyUntypedArray, PyUntypedArrayMethods, dtype,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DTypePolicy {
    KeepInput,
    ForceF64,
}

#[derive(Debug)]
enum ValueStorage<'py> {
    BorrowedF32(PyReadonlyArrayDyn<'py, f32>),
    BorrowedF64(PyReadonlyArrayDyn<'py, f64>),
    OwnedF32(Vec<f32>),
    OwnedF64(Vec<f64>),
}

#[derive(Clone, Debug)]
enum OwnedValueStorage {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

#[derive(Debug)]
enum TimeStorage<'py> {
    None,
    BorrowedI64(PyReadonlyArrayDyn<'py, i64>),
    OwnedI64(Vec<i64>),
}

#[derive(Clone, Debug)]
enum OwnedTimeStorage {
    None,
    Explicit(Vec<i64>),
}

#[derive(Debug)]
pub(crate) struct ParsedSeries<'py> {
    values: ValueStorage<'py>,
    time: TimeStorage<'py>,
    n: usize,
    d: usize,
    missing_policy: MissingPolicy,
    diagnostics: Vec<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct OwnedSeries {
    values: OwnedValueStorage,
    time: OwnedTimeStorage,
    n: usize,
    d: usize,
    missing_policy: MissingPolicy,
    diagnostics: Vec<String>,
}

impl<'py> ParsedSeries<'py> {
    pub(crate) fn view(&self) -> Result<TimeSeriesView<'_>, CpdError> {
        enum Values<'a> {
            F32(&'a [f32]),
            F64(&'a [f64]),
        }

        let values = match &self.values {
            ValueStorage::BorrowedF32(values) => Values::F32(values.as_slice().map_err(|_| {
                CpdError::invalid_input("internal error: expected contiguous borrowed f32 values")
            })?),
            ValueStorage::BorrowedF64(values) => Values::F64(values.as_slice().map_err(|_| {
                CpdError::invalid_input("internal error: expected contiguous borrowed f64 values")
            })?),
            ValueStorage::OwnedF32(values) => Values::F32(values.as_slice()),
            ValueStorage::OwnedF64(values) => Values::F64(values.as_slice()),
        };

        let time_index = match &self.time {
            TimeStorage::None => TimeIndex::None,
            TimeStorage::BorrowedI64(time) => {
                TimeIndex::Explicit(time.as_slice().map_err(|_| {
                    CpdError::invalid_input(
                        "internal error: expected contiguous borrowed time index",
                    )
                })?)
            }
            TimeStorage::OwnedI64(time) => TimeIndex::Explicit(time.as_slice()),
        };

        let dtype_view = match values {
            Values::F32(values) => DTypeView::F32(values),
            Values::F64(values) => DTypeView::F64(values),
        };

        TimeSeriesView::new(
            dtype_view,
            self.n,
            self.d,
            MemoryLayout::CContiguous,
            None,
            time_index,
            self.missing_policy,
        )
    }

    pub(crate) fn diagnostics(&self) -> &[String] {
        &self.diagnostics
    }

    pub(crate) fn into_owned(self) -> Result<OwnedSeries, CpdError> {
        let values = match self.values {
            ValueStorage::BorrowedF32(values) => OwnedValueStorage::F32(
                values
                    .as_slice()
                    .map_err(|_| {
                        CpdError::invalid_input(
                            "internal error: expected contiguous borrowed f32 values",
                        )
                    })?
                    .to_vec(),
            ),
            ValueStorage::BorrowedF64(values) => OwnedValueStorage::F64(
                values
                    .as_slice()
                    .map_err(|_| {
                        CpdError::invalid_input(
                            "internal error: expected contiguous borrowed f64 values",
                        )
                    })?
                    .to_vec(),
            ),
            ValueStorage::OwnedF32(values) => OwnedValueStorage::F32(values),
            ValueStorage::OwnedF64(values) => OwnedValueStorage::F64(values),
        };

        let time = match self.time {
            TimeStorage::None => OwnedTimeStorage::None,
            TimeStorage::BorrowedI64(time) => OwnedTimeStorage::Explicit(
                time.as_slice()
                    .map_err(|_| {
                        CpdError::invalid_input(
                            "internal error: expected contiguous borrowed time index",
                        )
                    })?
                    .to_vec(),
            ),
            TimeStorage::OwnedI64(time) => OwnedTimeStorage::Explicit(time),
        };

        Ok(OwnedSeries {
            values,
            time,
            n: self.n,
            d: self.d,
            missing_policy: self.missing_policy,
            diagnostics: self.diagnostics,
        })
    }
}

impl OwnedSeries {
    pub(crate) fn view(&self) -> Result<TimeSeriesView<'_>, CpdError> {
        let values = match &self.values {
            OwnedValueStorage::F32(values) => DTypeView::F32(values.as_slice()),
            OwnedValueStorage::F64(values) => DTypeView::F64(values.as_slice()),
        };

        let time_index = match &self.time {
            OwnedTimeStorage::None => TimeIndex::None,
            OwnedTimeStorage::Explicit(time) => TimeIndex::Explicit(time.as_slice()),
        };

        TimeSeriesView::new(
            values,
            self.n,
            self.d,
            MemoryLayout::CContiguous,
            None,
            time_index,
            self.missing_policy,
        )
    }

    pub(crate) fn diagnostics(&self) -> &[String] {
        &self.diagnostics
    }
}

pub(crate) fn parse_numpy_series<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    time: Option<&Bound<'py, PyAny>>,
    missing_policy: MissingPolicy,
    dtype_policy: DTypePolicy,
) -> PyResult<ParsedSeries<'py>> {
    let x_array = x
        .downcast::<PyUntypedArray>()
        .map_err(|_| PyTypeError::new_err("expected numpy.ndarray for x"))?;

    let (n, d) = parse_shape(x_array)?;
    let mut diagnostics = Vec::new();

    let values = parse_numeric_storage(py, x_array, n, d, dtype_policy, &mut diagnostics)?;
    let time_storage = parse_time_storage(py, time, n, &mut diagnostics)?;

    let parsed = ParsedSeries {
        values,
        time: time_storage,
        n,
        d,
        missing_policy,
        diagnostics,
    };

    parsed.view().map_err(cpd_error_to_pyerr)?;
    Ok(parsed)
}

fn parse_shape(array: &Bound<'_, PyUntypedArray>) -> PyResult<(usize, usize)> {
    match array.ndim() {
        1 => Ok((array.shape()[0], 1)),
        2 => Ok((array.shape()[0], array.shape()[1])),
        ndim => Err(PyValueError::new_err(format!(
            "expected shape (n,) or (n, d), got ndim={ndim}"
        ))),
    }
}

fn parse_numeric_storage<'py>(
    py: Python<'py>,
    x_array: &Bound<'py, PyUntypedArray>,
    n: usize,
    d: usize,
    dtype_policy: DTypePolicy,
    diagnostics: &mut Vec<String>,
) -> PyResult<ValueStorage<'py>> {
    let dtype_descr = x_array.dtype();

    if dtype_descr.is_equiv_to(&dtype::<f64>(py)) {
        return parse_f64_storage(x_array, n, d, diagnostics);
    }
    if dtype_descr.is_equiv_to(&dtype::<f32>(py)) {
        return parse_f32_storage(x_array, n, d, dtype_policy, diagnostics);
    }

    Err(PyTypeError::new_err(format!(
        "expected float32 or float64, got {}",
        dtype_name(&dtype_descr)
    )))
}

fn parse_f64_storage<'py>(
    x_array: &Bound<'py, PyUntypedArray>,
    n: usize,
    d: usize,
    diagnostics: &mut Vec<String>,
) -> PyResult<ValueStorage<'py>> {
    let array = x_array.downcast::<PyArrayDyn<f64>>().map_err(|_| {
        PyRuntimeError::new_err("internal error: failed to downcast float64 ndarray")
    })?;

    if array.is_c_contiguous() {
        return Ok(ValueStorage::BorrowedF64(array.readonly()));
    }

    let readonly = array.readonly();
    let copied = copy_f64_from_readonly(&readonly, n, d);
    if array.is_fortran_contiguous() && d > 1 {
        diagnostics.push("copied from F-contiguous to C-contiguous layout".to_string());
    } else {
        diagnostics.push("copied from strided to contiguous layout".to_string());
    }
    Ok(ValueStorage::OwnedF64(copied))
}

fn parse_f32_storage<'py>(
    x_array: &Bound<'py, PyUntypedArray>,
    n: usize,
    d: usize,
    dtype_policy: DTypePolicy,
    diagnostics: &mut Vec<String>,
) -> PyResult<ValueStorage<'py>> {
    let array = x_array.downcast::<PyArrayDyn<f32>>().map_err(|_| {
        PyRuntimeError::new_err("internal error: failed to downcast float32 ndarray")
    })?;

    if array.is_c_contiguous() {
        let readonly = array.readonly();
        return match dtype_policy {
            DTypePolicy::KeepInput => Ok(ValueStorage::BorrowedF32(readonly)),
            DTypePolicy::ForceF64 => {
                diagnostics
                    .push("upcast from float32 to float64 (cost model requires f64)".to_string());
                Ok(ValueStorage::OwnedF64(
                    readonly
                        .as_slice()
                        .map_err(|_| {
                            PyRuntimeError::new_err(
                                "internal error: expected contiguous float32 data",
                            )
                        })?
                        .iter()
                        .copied()
                        .map(f64::from)
                        .collect(),
                ))
            }
        };
    }

    let readonly = array.readonly();
    let is_fortran = array.is_fortran_contiguous() && d > 1;
    if is_fortran {
        diagnostics.push("copied from F-contiguous to C-contiguous layout".to_string());
    } else {
        diagnostics.push("copied from strided to contiguous layout".to_string());
    }

    match dtype_policy {
        DTypePolicy::KeepInput => Ok(ValueStorage::OwnedF32(copy_f32_from_readonly(
            &readonly, n, d,
        ))),
        DTypePolicy::ForceF64 => {
            diagnostics
                .push("upcast from float32 to float64 (cost model requires f64)".to_string());
            Ok(ValueStorage::OwnedF64(copy_f32_as_f64_from_readonly(
                &readonly, n, d,
            )))
        }
    }
}

fn parse_time_storage<'py>(
    py: Python<'py>,
    time: Option<&Bound<'py, PyAny>>,
    n: usize,
    diagnostics: &mut Vec<String>,
) -> PyResult<TimeStorage<'py>> {
    let Some(time_any) = time else {
        return Ok(TimeStorage::None);
    };

    let time_array = time_any
        .downcast::<PyUntypedArray>()
        .map_err(|_| PyTypeError::new_err("expected numpy.ndarray for time index"))?;

    if time_array.ndim() != 1 {
        return Err(PyValueError::new_err(format!(
            "time index must be 1D with length n={n}, got ndim={}",
            time_array.ndim()
        )));
    }
    if time_array.shape()[0] != n {
        return Err(PyValueError::new_err(format!(
            "time index length mismatch: got {}, expected n={n}",
            time_array.shape()[0]
        )));
    }

    let dtype_descr = time_array.dtype();
    if dtype_descr.is_equiv_to(&dtype::<i64>(py)) {
        return parse_i64_time_storage(time_array, diagnostics);
    }
    if dtype_descr.is_equiv_to(&dtype::<Datetime<units::Nanoseconds>>(py)) {
        return parse_datetime_ns_time_storage(time_array, n, diagnostics);
    }

    let message = format!(
        "expected int64 or datetime64[ns] for time index, got {}",
        dtype_name(&dtype_descr)
    );
    Err(PyTypeError::new_err(message))
}

fn parse_i64_time_storage<'py>(
    time_array: &Bound<'py, PyUntypedArray>,
    diagnostics: &mut Vec<String>,
) -> PyResult<TimeStorage<'py>> {
    let array = time_array.downcast::<PyArrayDyn<i64>>().map_err(|_| {
        PyRuntimeError::new_err("internal error: failed to downcast int64 time index")
    })?;

    if array.is_contiguous() {
        return Ok(TimeStorage::BorrowedI64(array.readonly()));
    }

    diagnostics.push("copied time index from strided to contiguous layout".to_string());
    let readonly = array.readonly();
    Ok(TimeStorage::OwnedI64(copy_i64_from_readonly(&readonly)))
}

fn parse_datetime_ns_time_storage<'py>(
    time_array: &Bound<'py, PyUntypedArray>,
    n: usize,
    diagnostics: &mut Vec<String>,
) -> PyResult<TimeStorage<'py>> {
    let array = time_array
        .downcast::<PyArrayDyn<Datetime<units::Nanoseconds>>>()
        .map_err(|_| {
            PyRuntimeError::new_err("internal error: failed to downcast datetime64[ns] time index")
        })?;

    if !array.is_contiguous() {
        diagnostics.push("copied time index from strided to contiguous layout".to_string());
    }
    let readonly = array.readonly();
    Ok(TimeStorage::OwnedI64(
        copy_datetime_ns_as_i64_from_readonly(&readonly, n),
    ))
}

fn copy_f64_from_readonly(readonly: &PyReadonlyArrayDyn<'_, f64>, n: usize, d: usize) -> Vec<f64> {
    let view = readonly.as_array();
    let mut out = Vec::with_capacity(n.saturating_mul(d));
    if d == 1 {
        for row in 0..n {
            let value = view
                .get(IxDyn(&[row]))
                .expect("validated ndarray shape for 1D copy");
            out.push(*value);
        }
    } else {
        for row in 0..n {
            for col in 0..d {
                let value = view
                    .get(IxDyn(&[row, col]))
                    .expect("validated ndarray shape for 2D copy");
                out.push(*value);
            }
        }
    }
    out
}

fn copy_f32_from_readonly(readonly: &PyReadonlyArrayDyn<'_, f32>, n: usize, d: usize) -> Vec<f32> {
    let view = readonly.as_array();
    let mut out = Vec::with_capacity(n.saturating_mul(d));
    if d == 1 {
        for row in 0..n {
            let value = view
                .get(IxDyn(&[row]))
                .expect("validated ndarray shape for 1D copy");
            out.push(*value);
        }
    } else {
        for row in 0..n {
            for col in 0..d {
                let value = view
                    .get(IxDyn(&[row, col]))
                    .expect("validated ndarray shape for 2D copy");
                out.push(*value);
            }
        }
    }
    out
}

fn copy_f32_as_f64_from_readonly(
    readonly: &PyReadonlyArrayDyn<'_, f32>,
    n: usize,
    d: usize,
) -> Vec<f64> {
    let view = readonly.as_array();
    let mut out = Vec::with_capacity(n.saturating_mul(d));
    if d == 1 {
        for row in 0..n {
            let value = view
                .get(IxDyn(&[row]))
                .expect("validated ndarray shape for 1D copy");
            out.push(f64::from(*value));
        }
    } else {
        for row in 0..n {
            for col in 0..d {
                let value = view
                    .get(IxDyn(&[row, col]))
                    .expect("validated ndarray shape for 2D copy");
                out.push(f64::from(*value));
            }
        }
    }
    out
}

fn copy_i64_from_readonly(readonly: &PyReadonlyArrayDyn<'_, i64>) -> Vec<i64> {
    readonly.as_array().iter().copied().collect()
}

fn copy_datetime_ns_as_i64_from_readonly(
    readonly: &PyReadonlyArrayDyn<'_, Datetime<units::Nanoseconds>>,
    n: usize,
) -> Vec<i64> {
    let view = readonly.as_array();
    let mut out = Vec::with_capacity(n);
    for row in 0..n {
        let value = view
            .get(IxDyn(&[row]))
            .expect("validated ndarray shape for datetime64[ns] copy");
        out.push(i64::from(*value));
    }
    out
}

fn dtype_name(dtype: &Bound<'_, PyArrayDescr>) -> String {
    dtype
        .str()
        .and_then(|repr| repr.extract::<String>())
        .unwrap_or_else(|_| "<unknown dtype>".to_string())
}

#[cfg(test)]
mod tests {
    use super::{DTypePolicy, parse_numpy_series};
    use cpd_core::{DTypeView, MissingPolicy, TimeIndex, TimeSeriesView};
    use numpy::{PyArrayDyn, PyArrayMethods};
    use pyo3::exceptions::{PyTypeError, PyValueError};
    use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyModule};
    use pyo3::{PyErr, Python};
    use std::sync::Once;

    fn with_python<F, R>(f: F) -> R
    where
        F: for<'py> FnOnce(Python<'py>) -> R,
    {
        static INIT: Once = Once::new();
        INIT.call_once(pyo3::prepare_freethreaded_python);
        Python::with_gil(f)
    }

    fn eval_numpy<'py>(py: Python<'py>, expr: &str) -> pyo3::Bound<'py, pyo3::PyAny> {
        let np = PyModule::import_bound(py, "numpy").expect("numpy should import");
        let locals = PyDict::new_bound(py);
        locals.set_item("np", np).expect("locals should accept np");
        py.eval_bound(expr, None, Some(&locals))
            .expect("python expression should evaluate")
    }

    fn view_f64_ptr(view: &TimeSeriesView<'_>) -> *const f64 {
        match view.values {
            DTypeView::F64(values) => values.as_ptr(),
            DTypeView::F32(_) => panic!("expected f64 data"),
        }
    }

    fn view_f32_ptr(view: &TimeSeriesView<'_>) -> *const f32 {
        match view.values {
            DTypeView::F32(values) => values.as_ptr(),
            DTypeView::F64(_) => panic!("expected f32 data"),
        }
    }

    fn err_contains(err: &PyErr, text: &str) {
        assert!(
            err.to_string().contains(text),
            "error did not contain {text:?}: {}",
            err
        );
    }

    #[test]
    fn c_contiguous_f64_1d_zero_copy() {
        with_python(|py| {
            let x = eval_numpy(py, "np.array([1.0, 2.0, 3.0], dtype=np.float64)");
            let parsed =
                parse_numpy_series(py, &x, None, MissingPolicy::Error, DTypePolicy::KeepInput)
                    .expect("parsing should succeed");
            let view = parsed.view().expect("view should be valid");

            let array = x
                .downcast::<PyArrayDyn<f64>>()
                .expect("x should be downcastable to float64");
            let source_ptr = array
                .readonly()
                .as_slice()
                .expect("source should be contiguous")
                .as_ptr();
            assert_eq!(view_f64_ptr(&view), source_ptr);
            assert!(parsed.diagnostics().is_empty());
        });
    }

    #[test]
    fn c_contiguous_f64_2d_zero_copy() {
        with_python(|py| {
            let x = eval_numpy(
                py,
                "np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64, order='C')",
            );
            let parsed =
                parse_numpy_series(py, &x, None, MissingPolicy::Error, DTypePolicy::KeepInput)
                    .expect("parsing should succeed");
            let view = parsed.view().expect("view should be valid");

            assert_eq!(view.n, 2);
            assert_eq!(view.d, 2);

            let array = x
                .downcast::<PyArrayDyn<f64>>()
                .expect("x should be downcastable to float64");
            let source_ptr = array
                .readonly()
                .as_slice()
                .expect("source should be contiguous")
                .as_ptr();
            assert_eq!(view_f64_ptr(&view), source_ptr);
            assert!(parsed.diagnostics().is_empty());
        });
    }

    #[test]
    fn c_contiguous_f32_keep_input_is_zero_copy() {
        with_python(|py| {
            let x = eval_numpy(py, "np.array([1.0, 2.0, 3.0], dtype=np.float32)");
            let parsed =
                parse_numpy_series(py, &x, None, MissingPolicy::Error, DTypePolicy::KeepInput)
                    .expect("parsing should succeed");
            let view = parsed.view().expect("view should be valid");

            let array = x
                .downcast::<PyArrayDyn<f32>>()
                .expect("x should be downcastable to float32");
            let source_ptr = array
                .readonly()
                .as_slice()
                .expect("source should be contiguous")
                .as_ptr();
            assert_eq!(view_f32_ptr(&view), source_ptr);
            assert!(parsed.diagnostics().is_empty());
        });
    }

    #[test]
    fn c_contiguous_f32_force_f64_upcasts_with_note() {
        with_python(|py| {
            let x = eval_numpy(py, "np.array([1.0, 2.0, 3.0], dtype=np.float32)");
            let parsed =
                parse_numpy_series(py, &x, None, MissingPolicy::Error, DTypePolicy::ForceF64)
                    .expect("parsing should succeed");
            let view = parsed.view().expect("view should be valid");

            match view.values {
                DTypeView::F64(values) => assert_eq!(values, &[1.0, 2.0, 3.0]),
                DTypeView::F32(_) => panic!("force-f64 should not keep f32"),
            }
            assert!(
                parsed
                    .diagnostics()
                    .iter()
                    .any(|note| note.contains("upcast from float32 to float64"))
            );
        });
    }

    #[test]
    fn f_contiguous_f64_copies_with_note() {
        with_python(|py| {
            let x = eval_numpy(
                py,
                "np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))",
            );
            let parsed =
                parse_numpy_series(py, &x, None, MissingPolicy::Error, DTypePolicy::KeepInput)
                    .expect("parsing should succeed");
            let view = parsed.view().expect("view should be valid");

            let source = x
                .downcast::<PyArrayDyn<f64>>()
                .expect("x should be float64");
            let source_ptr = source
                .readonly()
                .as_slice()
                .expect("fortran array should still be contiguous")
                .as_ptr();
            assert_ne!(view_f64_ptr(&view), source_ptr);
            assert!(
                parsed
                    .diagnostics()
                    .iter()
                    .any(|note| note.contains("copied from F-contiguous to C-contiguous layout"))
            );
        });
    }

    #[test]
    fn strided_f64_copies_with_note() {
        with_python(|py| {
            let x = eval_numpy(py, "np.arange(10, dtype=np.float64)[::2]");
            let parsed =
                parse_numpy_series(py, &x, None, MissingPolicy::Error, DTypePolicy::KeepInput)
                    .expect("parsing should succeed");
            let view = parsed.view().expect("view should be valid");

            let source = x
                .downcast::<PyArrayDyn<f64>>()
                .expect("x should be float64");
            let source_ptr = source.readonly().as_array().as_ptr();
            assert_ne!(view_f64_ptr(&view), source_ptr);
            assert!(
                parsed
                    .diagnostics()
                    .iter()
                    .any(|note| note.contains("copied from strided to contiguous layout"))
            );
        });
    }

    #[test]
    fn invalid_dtype_is_rejected() {
        with_python(|py| {
            let x = eval_numpy(py, "np.array([1, 2, 3], dtype=np.int64)");
            let err =
                parse_numpy_series(py, &x, None, MissingPolicy::Error, DTypePolicy::KeepInput)
                    .expect_err("int dtype must be rejected");
            assert!(err.is_instance_of::<PyTypeError>(py));
            err_contains(&err, "expected float32 or float64");
        });
    }

    #[test]
    fn nans_with_missing_policy_error_are_rejected() {
        with_python(|py| {
            let x = eval_numpy(py, "np.array([1.0, np.nan, 3.0], dtype=np.float64)");
            let err =
                parse_numpy_series(py, &x, None, MissingPolicy::Error, DTypePolicy::KeepInput)
                    .expect_err("nan + MissingPolicy::Error should fail");
            assert!(err.is_instance_of::<PyValueError>(py));
            err_contains(&err, "MissingPolicy::Error");
        });
    }

    #[test]
    fn shape_rejection_for_invalid_ndim() {
        with_python(|py| {
            let scalar = eval_numpy(py, "np.array(1.0, dtype=np.float64)");
            let scalar_err = parse_numpy_series(
                py,
                &scalar,
                None,
                MissingPolicy::Error,
                DTypePolicy::KeepInput,
            )
            .expect_err("0D input must fail");
            assert!(scalar_err.is_instance_of::<PyValueError>(py));
            err_contains(&scalar_err, "shape (n,) or (n, d)");

            let cube = eval_numpy(py, "np.zeros((2, 2, 2), dtype=np.float64)");
            let cube_err = parse_numpy_series(
                py,
                &cube,
                None,
                MissingPolicy::Error,
                DTypePolicy::KeepInput,
            )
            .expect_err("3D input must fail");
            assert!(cube_err.is_instance_of::<PyValueError>(py));
            err_contains(&cube_err, "shape (n,) or (n, d)");
        });
    }

    #[test]
    fn time_index_int64_explicit() {
        with_python(|py| {
            let x = eval_numpy(py, "np.array([0.1, 0.2, 0.3], dtype=np.float64)");
            let t = eval_numpy(py, "np.array([10, 20, 30], dtype=np.int64)");
            let parsed = parse_numpy_series(
                py,
                &x,
                Some(&t),
                MissingPolicy::Error,
                DTypePolicy::KeepInput,
            )
            .expect("parsing should succeed");
            let view = parsed.view().expect("view should be valid");

            match view.time {
                TimeIndex::Explicit(timestamps) => assert_eq!(timestamps, &[10, 20, 30]),
                TimeIndex::None | TimeIndex::Uniform { .. } => {
                    panic!("expected explicit time index")
                }
            }
        });
    }

    #[test]
    fn time_index_datetime64_ns_explicit() {
        with_python(|py| {
            let x = eval_numpy(py, "np.array([0.1, 0.2, 0.3], dtype=np.float64)");
            let t = eval_numpy(
                py,
                "np.array([1, 2, 3], dtype=np.int64).view('datetime64[ns]')",
            );
            let parsed = parse_numpy_series(
                py,
                &x,
                Some(&t),
                MissingPolicy::Error,
                DTypePolicy::KeepInput,
            )
            .expect("parsing should succeed");
            let view = parsed.view().expect("view should be valid");

            match view.time {
                TimeIndex::Explicit(timestamps) => assert_eq!(timestamps, &[1, 2, 3]),
                TimeIndex::None | TimeIndex::Uniform { .. } => {
                    panic!("expected explicit time index")
                }
            }
        });
    }

    #[test]
    fn time_index_length_mismatch_rejected() {
        with_python(|py| {
            let x = eval_numpy(py, "np.array([0.1, 0.2, 0.3], dtype=np.float64)");
            let t = eval_numpy(py, "np.array([10, 20], dtype=np.int64)");
            let err = parse_numpy_series(
                py,
                &x,
                Some(&t),
                MissingPolicy::Error,
                DTypePolicy::KeepInput,
            )
            .expect_err("time length mismatch must fail");

            assert!(err.is_instance_of::<PyValueError>(py));
            err_contains(&err, "time index length mismatch");
        });
    }

    #[test]
    fn omitted_time_index_defaults_to_none() {
        with_python(|py| {
            let x = eval_numpy(py, "np.array([0.1, 0.2, 0.3], dtype=np.float64)");
            let parsed =
                parse_numpy_series(py, &x, None, MissingPolicy::Error, DTypePolicy::KeepInput)
                    .expect("parsing should succeed");
            let view = parsed.view().expect("view should be valid");

            match view.time {
                TimeIndex::None => {}
                TimeIndex::Uniform { .. } | TimeIndex::Explicit(_) => {
                    panic!("expected TimeIndex::None")
                }
            }
        });
    }

    #[test]
    fn into_owned_detaches_data_and_preserves_diagnostics() {
        with_python(|py| {
            let x = eval_numpy(
                py,
                "np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))",
            );
            let parsed =
                parse_numpy_series(py, &x, None, MissingPolicy::Error, DTypePolicy::KeepInput)
                    .expect("parsing should succeed");
            let owned = parsed.into_owned().expect("snapshot should succeed");
            let view = owned.view().expect("owned snapshot should validate");

            let source = x
                .downcast::<PyArrayDyn<f64>>()
                .expect("x should be float64");
            let source_ptr = source
                .readonly()
                .as_slice()
                .expect("fortran array should still be contiguous")
                .as_ptr();

            assert_ne!(view_f64_ptr(&view), source_ptr);
            assert!(
                owned
                    .diagnostics()
                    .iter()
                    .any(|note| note.contains("copied from F-contiguous to C-contiguous layout"))
            );
        });
    }
}
