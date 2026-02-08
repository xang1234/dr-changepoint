// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

pub mod constraints;
pub mod control;
pub mod diagnostics;
pub mod error;
pub mod execution_context;
pub mod missing;
pub mod observability;
pub mod repro;
pub mod time_series;

pub use constraints::{
    CachePolicy, Constraints, DegradationStep, ValidatedConstraints, canonicalize_candidates,
    validate_constraints,
};
pub use control::{BudgetMode, BudgetStatus, CancelToken};
pub use diagnostics::{DIAGNOSTICS_SCHEMA_VERSION, Diagnostics, PruningStats};
pub use error::CpdError;
pub use execution_context::ExecutionContext;
pub use missing::{
    MissingRunStats, MissingSupport, build_missing_mask, check_missing_compatibility,
    compute_missing_run_stats, scan_nans,
};
pub use observability::{ProgressSink, TelemetrySink};
pub use repro::ReproMode;
pub use time_series::{DTypeView, MemoryLayout, MissingPolicy, TimeIndex, TimeSeriesView};

/// Core shared types and traits for cpd-rs.
pub fn crate_name() -> &'static str {
    "cpd-core"
}
