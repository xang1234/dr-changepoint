// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

pub mod error;
pub mod time_series;

pub use error::CpdError;
pub use time_series::{DTypeView, MemoryLayout, MissingPolicy, TimeIndex, TimeSeriesView};

/// Core shared types and traits for cpd-rs.
pub fn crate_name() -> &'static str {
    "cpd-core"
}
