// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

/// Built-in cost model namespace placeholder.
pub fn crate_name() -> &'static str {
    let _ = cpd_core::crate_name();
    "cpd-costs"
}
