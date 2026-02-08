// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

/// Python bindings namespace placeholder. PyO3 wiring lands in CPD-deg.20.
pub fn crate_name() -> &'static str {
    let _ = (
        cpd_core::crate_name(),
        cpd_offline::crate_name(),
        cpd_online::crate_name(),
        cpd_doctor::crate_name(),
    );
    "cpd-python"
}
