// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::CpdError;
use crate::missing::scan_nans;

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
    effective_missing_count: usize,
    total_value_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EffectiveMissingStats {
    effective_missing_count: usize,
    mask_missing_count: usize,
    nan_missing_count: usize,
    first_missing_index: Option<usize>,
}

fn effective_missing_stats(
    values: DTypeView<'_>,
    missing_mask: Option<&[u8]>,
) -> EffectiveMissingStats {
    let (nan_missing_count, nan_positions) = scan_nans(values);
    let first_nan_index = nan_positions.first().copied();

    if let Some(mask) = missing_mask {
        let mask_missing_count = mask.iter().copied().map(usize::from).sum();
        let first_mask_index = mask.iter().position(|&value| value == 1);
        let additional_nan_count = nan_positions.iter().filter(|&&idx| mask[idx] == 0).count();
        let first_missing_index = match (first_mask_index, first_nan_index) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            (None, None) => None,
        };

        EffectiveMissingStats {
            effective_missing_count: mask_missing_count + additional_nan_count,
            mask_missing_count,
            nan_missing_count,
            first_missing_index,
        }
    } else {
        EffectiveMissingStats {
            effective_missing_count: nan_missing_count,
            mask_missing_count: 0,
            nan_missing_count,
            first_missing_index: first_nan_index,
        }
    }
}

fn first_infinite_position(values: DTypeView<'_>) -> Option<(usize, &'static str)> {
    match values {
        DTypeView::F32(slice) => slice.iter().enumerate().find_map(|(idx, value)| {
            if value.is_infinite() {
                if value.is_sign_positive() {
                    Some((idx, "+inf"))
                } else {
                    Some((idx, "-inf"))
                }
            } else {
                None
            }
        }),
        DTypeView::F64(slice) => slice.iter().enumerate().find_map(|(idx, value)| {
            if value.is_infinite() {
                if value.is_sign_positive() {
                    Some((idx, "+inf"))
                } else {
                    Some((idx, "-inf"))
                }
            } else {
                None
            }
        }),
    }
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

        if let Some((idx, kind)) = first_infinite_position(values) {
            return Err(CpdError::invalid_input(format!(
                "non-finite value encountered: index={idx}, value={kind}; TimeSeriesView does not accept infinities"
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

        let missing_stats = effective_missing_stats(values, missing_mask);
        if matches!(missing, MissingPolicy::Error) && missing_stats.effective_missing_count > 0 {
            let first_missing_index = missing_stats.first_missing_index.unwrap_or(0);
            return Err(CpdError::invalid_input(format!(
                "missing data encountered with MissingPolicy::Error: effective_missing_count={}, mask_missing_count={}, nan_missing_count={}, first_missing_index={first_missing_index}",
                missing_stats.effective_missing_count,
                missing_stats.mask_missing_count,
                missing_stats.nan_missing_count
            )));
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
            effective_missing_count: missing_stats.effective_missing_count,
            total_value_count: expected_len,
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

    /// Returns true when the effective missing set (mask union NaNs) is non-empty.
    pub fn has_missing(&self) -> bool {
        self.n_missing() > 0
    }

    /// Counts missing values using `effective_missing = mask UNION NaN`.
    pub fn n_missing(&self) -> usize {
        self.effective_missing_count
    }

    /// Returns the effective fraction of missing values in `[0.0, 1.0]`.
    pub fn missing_fraction(&self) -> f64 {
        if self.total_value_count == 0 {
            0.0
        } else {
            self.effective_missing_count as f64 / self.total_value_count as f64
        }
    }

    /// Returns the number of non-missing values after applying union semantics.
    pub fn effective_sample_count(&self) -> usize {
        self.total_value_count
            .saturating_sub(self.effective_missing_count)
    }

    /// Returns the timestamp for a sample index.
    ///
    /// Returns `None` when:
    /// - no time index is attached,
    /// - `idx` is out of bounds, or
    /// - timestamp arithmetic overflows (uniform index).
    pub fn timestamp_at(&self, idx: usize) -> Option<i64> {
        if idx >= self.n {
            return None;
        }

        match self.time {
            TimeIndex::None => None,
            TimeIndex::Uniform { t0_ns, dt_ns } => {
                let idx_i64 = i64::try_from(idx).ok()?;
                let delta = dt_ns.checked_mul(idx_i64)?;
                t0_ns.checked_add(delta)
            }
            TimeIndex::Explicit(timestamps) => timestamps.get(idx).copied(),
        }
    }

    /// Returns elapsed time between two sample indices as `end - start`.
    ///
    /// Returns `None` when:
    /// - indices are reversed (`start_idx > end_idx`),
    /// - either index has no timestamp, or
    /// - subtraction overflows.
    pub fn duration_between(&self, start_idx: usize, end_idx: usize) -> Option<i64> {
        if start_idx > end_idx {
            return None;
        }

        let start = self.timestamp_at(start_idx)?;
        let end = self.timestamp_at(end_idx)?;
        end.checked_sub(start)
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
    fn timestamp_at_returns_none_for_missing_time_index() {
        let data = [1.0_f64, 2.0, 3.0];
        let view = TimeSeriesView::from_f64(
            &data,
            3,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect("view should be valid");

        assert_eq!(view.timestamp_at(0), None);
        assert_eq!(view.timestamp_at(2), None);
    }

    #[test]
    fn timestamp_at_uniform_handles_valid_out_of_range_and_overflow_cases() {
        let data = [1.0_f64, 2.0, 3.0];
        let view = TimeSeriesView::from_f64(
            &data,
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
        .expect("view should be valid");

        assert_eq!(view.timestamp_at(0), Some(1_000));
        assert_eq!(view.timestamp_at(1), Some(1_005));
        assert_eq!(view.timestamp_at(2), Some(1_010));
        assert_eq!(view.timestamp_at(3), None);

        let overflow_view = TimeSeriesView::from_f64(
            &data,
            3,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::Uniform {
                t0_ns: i64::MAX - 1,
                dt_ns: 2,
            },
            MissingPolicy::Error,
        )
        .expect("overflow test view should be valid");
        assert_eq!(overflow_view.timestamp_at(1), None);
    }

    #[test]
    fn timestamp_at_explicit_returns_values_and_none_out_of_range() {
        let data = [1.0_f64, 2.0, 3.0];
        let ts = [11_i64, 17_i64, 23_i64];
        let view = TimeSeriesView::from_f64(
            &data,
            3,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::Explicit(&ts),
            MissingPolicy::Error,
        )
        .expect("view should be valid");

        assert_eq!(view.timestamp_at(0), Some(11));
        assert_eq!(view.timestamp_at(2), Some(23));
        assert_eq!(view.timestamp_at(3), None);
    }

    #[test]
    fn duration_between_returns_none_for_missing_time_index() {
        let data = [1.0_f64, 2.0, 3.0];
        let view = TimeSeriesView::from_f64(
            &data,
            3,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect("view should be valid");

        assert_eq!(view.duration_between(0, 2), None);
    }

    #[test]
    fn duration_between_uniform_and_explicit_use_direct_difference() {
        let data = [1.0_f64, 2.0, 3.0, 4.0];
        let uniform = TimeSeriesView::from_f64(
            &data,
            4,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::Uniform {
                t0_ns: 1_000,
                dt_ns: 5,
            },
            MissingPolicy::Error,
        )
        .expect("uniform view should be valid");
        assert_eq!(uniform.duration_between(1, 3), Some(10));

        let ts = [10_i64, 14_i64, 27_i64, 39_i64];
        let explicit = TimeSeriesView::from_f64(
            &data,
            4,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::Explicit(&ts),
            MissingPolicy::Error,
        )
        .expect("explicit view should be valid");
        assert_eq!(explicit.duration_between(1, 3), Some(25));
    }

    #[test]
    fn duration_between_returns_none_for_reversed_or_invalid_indices() {
        let data = [1.0_f64, 2.0, 3.0];
        let ts = [10_i64, 20_i64, 30_i64];
        let view = TimeSeriesView::from_f64(
            &data,
            3,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::Explicit(&ts),
            MissingPolicy::Error,
        )
        .expect("view should be valid");

        assert_eq!(view.duration_between(2, 1), None);
        assert_eq!(view.duration_between(0, 3), None);
    }

    #[test]
    fn duration_between_allows_negative_and_returns_none_on_sub_overflow() {
        let data = [1.0_f64, 2.0, 3.0];
        let non_monotonic_ts = [10_i64, 5_i64, 7_i64];
        let non_monotonic = TimeSeriesView::from_f64(
            &data,
            3,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::Explicit(&non_monotonic_ts),
            MissingPolicy::Error,
        )
        .expect("non-monotonic view should be valid");
        assert_eq!(non_monotonic.duration_between(0, 1), Some(-5));

        let overflow_data = [1.0_f64, 2.0];
        let overflow_ts = [i64::MAX, i64::MIN];
        let overflow = TimeSeriesView::from_f64(
            &overflow_data,
            2,
            1,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::Explicit(&overflow_ts),
            MissingPolicy::Error,
        )
        .expect("overflow view should be valid");
        assert_eq!(overflow.duration_between(0, 1), None);
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
    fn missing_policy_error_rejects_nan_only_input() {
        let data = [1.0_f64, f64::NAN, 3.0, 4.0];
        let err = TimeSeriesView::from_f64(
            &data,
            2,
            2,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect_err("MissingPolicy::Error must reject NaN input");
        let msg = err.to_string();
        assert!(msg.contains("MissingPolicy::Error"));
        assert!(msg.contains("effective_missing_count=1"));
        assert!(msg.contains("first_missing_index=1"));
    }

    #[test]
    fn missing_policy_error_rejects_mask_only_missing_input() {
        let data = [1.0_f64, 2.0, 3.0, 4.0];
        let mask = [0_u8, 1_u8, 0_u8, 0_u8];
        let err = TimeSeriesView::from_f64(
            &data,
            2,
            2,
            MemoryLayout::CContiguous,
            Some(&mask),
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect_err("MissingPolicy::Error must reject explicit missing mask entries");
        let msg = err.to_string();
        assert!(msg.contains("effective_missing_count=1"));
        assert!(msg.contains("mask_missing_count=1"));
        assert!(msg.contains("first_missing_index=1"));
    }

    #[test]
    fn missing_policy_error_rejects_union_of_mask_and_nans() {
        let data = [1.0_f64, f64::NAN, 3.0, 4.0];
        let mask = [1_u8, 0_u8, 0_u8, 0_u8];
        let err = TimeSeriesView::from_f64(
            &data,
            2,
            2,
            MemoryLayout::CContiguous,
            Some(&mask),
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect_err("MissingPolicy::Error must reject effective missing union");
        let msg = err.to_string();
        assert!(msg.contains("effective_missing_count=2"));
        assert!(msg.contains("mask_missing_count=1"));
        assert!(msg.contains("nan_missing_count=1"));
        assert!(msg.contains("first_missing_index=0"));
    }

    #[test]
    fn non_error_missing_policies_accept_missing_inputs() {
        let data = [1.0_f64, f64::NAN, 3.0, 4.0];
        let mask = [1_u8, 0_u8, 0_u8, 0_u8];
        let policies = [
            MissingPolicy::ImputeZero,
            MissingPolicy::ImputeLast,
            MissingPolicy::Ignore,
        ];

        for policy in policies {
            let view = TimeSeriesView::from_f64(
                &data,
                2,
                2,
                MemoryLayout::CContiguous,
                Some(&mask),
                TimeIndex::None,
                policy,
            )
            .expect("non-error missing policy should accept missing data");

            assert_eq!(view.n_missing(), 2);
            assert!(view.has_missing());
        }
    }

    #[test]
    fn rejects_infinity_for_all_missing_policies() {
        let policies = [
            MissingPolicy::Error,
            MissingPolicy::ImputeZero,
            MissingPolicy::ImputeLast,
            MissingPolicy::Ignore,
        ];

        let plus_inf = [1.0_f64, f64::INFINITY, 3.0, 4.0];
        for policy in policies {
            let err = TimeSeriesView::from_f64(
                &plus_inf,
                2,
                2,
                MemoryLayout::CContiguous,
                None,
                TimeIndex::None,
                policy,
            )
            .expect_err("all policies must reject +inf");
            let msg = err.to_string();
            assert!(msg.contains("non-finite value"));
            assert!(msg.contains("value=+inf"));
        }

        let minus_inf = [1.0_f64, f64::NEG_INFINITY, 3.0, 4.0];
        for policy in policies {
            let err = TimeSeriesView::from_f64(
                &minus_inf,
                2,
                2,
                MemoryLayout::CContiguous,
                None,
                TimeIndex::None,
                policy,
            )
            .expect_err("all policies must reject -inf");
            let msg = err.to_string();
            assert!(msg.contains("non-finite value"));
            assert!(msg.contains("value=-inf"));
        }
    }

    #[test]
    fn union_semantics_count_disjoint_mask_and_nans() {
        let data = [1.0_f64, 2.0, f64::NAN, 4.0, 5.0, 6.0];
        let mask = [0_u8, 1_u8, 0_u8, 0_u8, 1_u8, 0_u8];
        let view = TimeSeriesView::from_f64(
            &data,
            3,
            2,
            MemoryLayout::CContiguous,
            Some(&mask),
            TimeIndex::None,
            MissingPolicy::Ignore,
        )
        .expect("Ignore should accept missing union semantics");

        assert_eq!(view.n_missing(), 3);
        assert!((view.missing_fraction() - 0.5).abs() < f64::EPSILON);
        assert_eq!(view.effective_sample_count(), 3);
    }

    #[test]
    fn missing_fraction_and_effective_sample_count_cover_no_missing_and_full_missing() {
        let clean = [1.0_f64, 2.0, 3.0, 4.0];
        let clean_view = TimeSeriesView::from_f64(
            &clean,
            2,
            2,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect("clean input should succeed");
        assert_eq!(clean_view.missing_fraction(), 0.0);
        assert_eq!(clean_view.effective_sample_count(), 4);

        let all_nan = [f64::NAN, f64::NAN, f64::NAN, f64::NAN];
        let all_missing_view = TimeSeriesView::from_f64(
            &all_nan,
            2,
            2,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Ignore,
        )
        .expect("Ignore should accept fully missing data");
        assert!((all_missing_view.missing_fraction() - 1.0).abs() < f64::EPSILON);
        assert_eq!(all_missing_view.effective_sample_count(), 0);
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
