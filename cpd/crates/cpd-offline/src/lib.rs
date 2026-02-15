// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

pub mod binseg;
pub mod pelt;
#[cfg(feature = "serde")]
pub mod schema_migration;
pub mod wbs;

pub use binseg::{BinSeg, BinSegConfig};
pub use pelt::{Pelt, PeltConfig};
#[cfg(feature = "serde")]
pub use schema_migration::{BinSegConfigWire, PeltConfigWire, WbsConfigWire};
pub use wbs::{Wbs, WbsConfig, WbsIntervalStrategy};

/// Offline detector namespace placeholder.
pub fn crate_name() -> &'static str {
    let _ = (cpd_core::crate_name(), cpd_costs::crate_name());
    "cpd-offline"
}
