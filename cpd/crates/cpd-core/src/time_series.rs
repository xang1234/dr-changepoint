// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::CpdError;

/// Missing-data handling strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissingPolicy {
    /// Fail if any missing value is present.
    Error,
    /// Impute missing values with zero.
    ImputeZero,
    /// Impute missing values with carry-forward value.
    ImputeLast,
    /// Allow missing-aware algorithms to ignore missing values.
    Ignore,
}

/// Optional time index metadata for samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeIndex<'a> {
    /// Implicit sample index (0..n-1).
    None,
    /// Uniformly sampled time index in Unix nanoseconds.
    Uniform { t0_ns: i64, dt_ns: i64 },
    /// Explicit timestamps in Unix nanoseconds, length must match `n`.
    Explicit(&'a [i64]),
}

/// Borrowed numeric data view over either f32 or f64 values.
#[derive(Clone, Copy, Debug)]
pub enum DTypeView<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
}

/// Memory layout metadata for underlying data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryLayout {
    CContiguous,
    FContiguous,
    Strided {
        row_stride: isize,
        col_stride: isize,
    },
}

/// Zero-copy time-series view over a borrowed numeric buffer.
#[derive(Clone, Copy, Debug)]
pub struct TimeSeriesView<'a> {
    pub values: DTypeView<'a>,
    pub n: usize,
    pub d: usize,
    pub layout: MemoryLayout,
    pub missing_mask: Option<&'a [u8]>,
    pub time: TimeIndex<'a>,
    pub missing: MissingPolicy,
}

impl<'a> TimeSeriesView<'a> {
    /// Constructs a validated `TimeSeriesView`.
    pub fn new(
        values: DTypeView<'a>,
        n: usize,
        d: usize,
        layout: MemoryLayout,
        missing_mask: Option<&'a [u8]>,
        time: TimeIndex<'a>,
        missing: MissingPolicy,
    ) -> Result<Self, CpdError> {
        if n == 0 {
            return Err(CpdError::invalid_input("n must be >= 1"));
        }
        if d == 0 {
            return Err(CpdError::invalid_input("d must be >= 1"));
        }

        let expected_len = n
            .checked_mul(d)
            .ok_or_else(|| CpdError::invalid_input("n*d overflow while validating shape"))?;

        let value_len = match values {
            DTypeView::F32(slice) => slice.len(),
            DTypeView::F64(slice) => slice.len(),
        };
        if value_len != expected_len {
            return Err(CpdError::invalid_input(format!(
                "value length mismatch: got {value_len}, expected {expected_len} (n={n}, d={d})"
            )));
        }

        if let Some(mask) = missing_mask {
            if mask.len() != expected_len {
                return Err(CpdError::invalid_input(format!(
                    "missing_mask length mismatch: got {}, expected {} (n={}, d={})",
                    mask.len(),
                    expected_len,
                    n,
                    d
                )));
            }
            if let Some((idx, val)) = mask
                .iter()
                .copied()
                .enumerate()
                .find(|(_, v)| *v != 0 && *v != 1)
            {
                return Err(CpdError::invalid_input(format!(
                    "missing_mask must contain only 0/1 bytes: index {idx} has {val}"
                )));
            }
        }

        match time {
            TimeIndex::None => {}
            TimeIndex::Uniform { dt_ns, .. } => {
                if dt_ns <= 0 {
                    return Err(CpdError::invalid_input(format!(
                        "Uniform time index requires dt_ns > 0, got {dt_ns}"
                    )));
                }
            }
            TimeIndex::Explicit(timestamps) => {
                if timestamps.len() != n {
                    return Err(CpdError::invalid_input(format!(
                        "Explicit time index length mismatch: got {}, expected n={}",
                        timestamps.len(),
                        n
                    )));
                }
            }
        }

        if let MemoryLayout::Strided {
            row_stride,
            col_stride,
        } = layout
        {
            if row_stride == 0 || col_stride == 0 {
                return Err(CpdError::invalid_input(format!(
                    "Strided layout requires non-zero strides: row_stride={row_stride}, col_stride={col_stride}"
                )));
            }
        }

        Ok(Self {
            values,
            n,
            d,
            layout,
            missing_mask,
            time,
            missing,
        })
    }

    /// Convenience constructor for f32-backed data.
    pub fn from_f32(
        values: &'a [f32],
        n: usize,
        d: usize,
        layout: MemoryLayout,
        missing_mask: Option<&'a [u8]>,
        time: TimeIndex<'a>,
        missing: MissingPolicy,
    ) -> Result<Self, CpdError> {
        Self::new(
            DTypeView::F32(values),
            n,
            d,
            layout,
            missing_mask,
            time,
            missing,
        )
    }

    /// Convenience constructor for f64-backed data.
    pub fn from_f64(
        values: &'a [f64],
        n: usize,
        d: usize,
        layout: MemoryLayout,
        missing_mask: Option<&'a [u8]>,
        time: TimeIndex<'a>,
        missing: MissingPolicy,
    ) -> Result<Self, CpdError> {
        Self::new(
            DTypeView::F64(values),
            n,
            d,
            layout,
            missing_mask,
            time,
            missing,
        )
    }

    /// Returns true when `d == 1`.
    pub fn is_univariate(&self) -> bool {
        self.d == 1
    }

    /// Returns true when `d > 1`.
    pub fn is_multivariate(&self) -> bool {
        self.d > 1
    }

    /// Returns true when the mask exists and contains at least one missing value.
    pub fn has_missing(&self) -> bool {
        self.n_missing() > 0
    }

    /// Counts missing values using the optional 0/1 mask.
    pub fn n_missing(&self) -> usize {
        self.missing_mask
            .map_or(0, |mask| mask.iter().copied().map(usize::from).sum())
    }
}

#[cfg(test)]
mod tests {
    use super::{DTypeView, MemoryLayout, MissingPolicy, TimeIndex, TimeSeriesView};

    fn base_mask(len: usize) -> Vec<u8> {
        vec![0; len]
    }

    #[test]
    fn from_f32_univariate_valid_case() {
        let data = [1.0_f32, 2.0, 3.0, 4.0];
        let view = TimeSeriesView::from_f32(
            &data,
            4,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect("from_f32 should succeed");

        assert!(view.is_univariate());
        assert!(!view.is_multivariate());
        assert_eq!(view.n_missing(), 0);
        assert!(!view.has_missing());
    }

    #[test]
    fn from_f64_multivariate_valid_case() {
        let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = TimeSeriesView::from_f64(
            &data,
            3,
            2,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect("from_f64 should succeed");

        assert!(!view.is_univariate());
        assert!(view.is_multivariate());
        match view.values {
            DTypeView::F64(slice) => assert_eq!(slice.len(), 6),
            _ => panic!("expected f64 view"),
        }
    }

    #[test]
    fn rejects_n_zero() {
        let data = [1.0_f64];
        let err = TimeSeriesView::from_f64(
            &data,
            0,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect_err("n=0 must fail");

        assert!(err.to_string().contains("n must be >= 1"));
    }

    #[test]
    fn rejects_d_zero() {
        let data = [1.0_f64];
        let err = TimeSeriesView::from_f64(
            &data,
            1,
            0,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect_err("d=0 must fail");

        assert!(err.to_string().contains("d must be >= 1"));
    }

    #[test]
    fn rejects_checked_mul_overflow() {
        let data: [f32; 0] = [];
        let err = TimeSeriesView::from_f32(
            &data,
            usize::MAX,
            2,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect_err("overflow must fail");

        assert!(err.to_string().contains("n*d overflow"));
    }

    #[test]
    fn rejects_value_length_mismatch_for_f32_and_f64() {
        let f32_data = [1.0_f32, 2.0, 3.0];
        let err_f32 = TimeSeriesView::from_f32(
            &f32_data,
            2,
            2,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect_err("f32 mismatch must fail");
        assert!(err_f32.to_string().contains("value length mismatch"));

        let f64_data = [1.0_f64, 2.0, 3.0];
        let err_f64 = TimeSeriesView::from_f64(
            &f64_data,
            2,
            2,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect_err("f64 mismatch must fail");
        assert!(err_f64.to_string().contains("value length mismatch"));
    }

    #[test]
    fn rejects_missing_mask_length_mismatch() {
        let data = [1.0_f64, 2.0, 3.0, 4.0];
        let mask = [0_u8, 1_u8, 0_u8];

        let err = TimeSeriesView::from_f64(
            &data,
            2,
            2,
            MemoryLayout::CContiguous,
            Some(&mask),
            TimeIndex::None,
            MissingPolicy::Ignore,
        )
        .expect_err("mask length mismatch must fail");

        assert!(err.to_string().contains("missing_mask length mismatch"));
    }

    #[test]
    fn rejects_invalid_mask_values() {
        let data = [1.0_f64, 2.0, 3.0, 4.0];
        let mask = [0_u8, 2_u8, 1_u8, 0_u8];

        let err = TimeSeriesView::from_f64(
            &data,
            2,
            2,
            MemoryLayout::CContiguous,
            Some(&mask),
            TimeIndex::None,
            MissingPolicy::Ignore,
        )
        .expect_err("non-binary mask must fail");

        assert!(err.to_string().contains("only 0/1 bytes"));
    }

    #[test]
    fn rejects_uniform_time_with_non_positive_dt() {
        let data = [1.0_f32, 2.0];

        let zero_dt = TimeSeriesView::from_f32(
            &data,
            2,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::Uniform { t0_ns: 0, dt_ns: 0 },
            MissingPolicy::Error,
        )
        .expect_err("dt=0 must fail");
        assert!(zero_dt.to_string().contains("dt_ns > 0"));

        let neg_dt = TimeSeriesView::from_f32(
            &data,
            2,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::Uniform {
                t0_ns: 0,
                dt_ns: -1,
            },
            MissingPolicy::Error,
        )
        .expect_err("dt<0 must fail");
        assert!(neg_dt.to_string().contains("dt_ns > 0"));
    }

    #[test]
    fn rejects_explicit_time_length_mismatch() {
        let data = [1.0_f64, 2.0, 3.0];
        let ts = [10_i64, 20_i64];

        let err = TimeSeriesView::from_f64(
            &data,
            3,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::Explicit(&ts),
            MissingPolicy::Error,
        )
        .expect_err("explicit time length mismatch must fail");

        assert!(
            err.to_string()
                .contains("Explicit time index length mismatch")
        );
    }

    #[test]
    fn rejects_strided_with_zero_stride() {
        let data = [1.0_f64, 2.0, 3.0, 4.0];

        let row_zero = TimeSeriesView::from_f64(
            &data,
            2,
            2,
            MemoryLayout::Strided {
                row_stride: 0,
                col_stride: 1,
            },
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect_err("row stride zero must fail");
        assert!(row_zero.to_string().contains("non-zero strides"));

        let col_zero = TimeSeriesView::from_f64(
            &data,
            2,
            2,
            MemoryLayout::Strided {
                row_stride: 1,
                col_stride: 0,
            },
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect_err("col stride zero must fail");
        assert!(col_zero.to_string().contains("non-zero strides"));
    }

    #[test]
    fn missing_helpers_cover_none_all_zero_and_mixed_masks() {
        let data = [1.0_f64, 2.0, 3.0, 4.0];

        let no_mask = TimeSeriesView::from_f64(
            &data,
            2,
            2,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect("no mask should succeed");
        assert_eq!(no_mask.n_missing(), 0);
        assert!(!no_mask.has_missing());

        let zero_mask = base_mask(4);
        let zero_mask_view = TimeSeriesView::from_f64(
            &data,
            2,
            2,
            MemoryLayout::CContiguous,
            Some(&zero_mask),
            TimeIndex::None,
            MissingPolicy::Ignore,
        )
        .expect("all-zero mask should succeed");
        assert_eq!(zero_mask_view.n_missing(), 0);
        assert!(!zero_mask_view.has_missing());

        let mixed_mask = [0_u8, 1_u8, 1_u8, 0_u8];
        let mixed_view = TimeSeriesView::from_f64(
            &data,
            2,
            2,
            MemoryLayout::CContiguous,
            Some(&mixed_mask),
            TimeIndex::None,
            MissingPolicy::Ignore,
        )
        .expect("mixed mask should succeed");
        assert_eq!(mixed_view.n_missing(), 2);
        assert!(mixed_view.has_missing());
    }

    #[test]
    fn univariate_and_multivariate_helpers_behave_as_expected() {
        let uni = [1.0_f32, 2.0, 3.0];
        let uni_view = TimeSeriesView::from_f32(
            &uni,
            3,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect("univariate should succeed");
        assert!(uni_view.is_univariate());
        assert!(!uni_view.is_multivariate());

        let multi = [1.0_f32, 2.0, 3.0, 4.0];
        let multi_view = TimeSeriesView::from_f32(
            &multi,
            2,
            2,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect("multivariate should succeed");
        assert!(!multi_view.is_univariate());
        assert!(multi_view.is_multivariate());
    }
}
