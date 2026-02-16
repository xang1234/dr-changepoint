// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

pub mod baseline;
pub mod bocpd;
#[cfg(feature = "serde")]
pub mod checkpoint;
pub mod event_time;

pub use baseline::{
    CUSUM_DETECTOR_ID, CUSUM_STATE_SCHEMA_VERSION, CusumConfig, CusumDetector, CusumState,
    PAGE_HINKLEY_DETECTOR_ID, PAGE_HINKLEY_STATE_SCHEMA_VERSION, PageHinkleyConfig,
    PageHinkleyDetector, PageHinkleyState,
};
pub use bocpd::{
    BOCPD_DETECTOR_ID, BOCPD_STATE_SCHEMA_VERSION, BernoulliBetaPrior, BocpdConfig, BocpdDetector,
    BocpdState, ConstantHazard, GaussianNigPrior, GeometricHazard, HazardFunction, HazardSpec,
    ObservationModel, ObservationStats, PoissonGammaPrior,
};
#[cfg(feature = "serde")]
pub use checkpoint::{
    CHECKPOINT_MIGRATION_GUIDANCE_PATH, CURRENT_CHECKPOINT_SCHEMA_VERSION, CheckpointEnvelope,
    MIN_SUPPORTED_CHECKPOINT_SCHEMA_VERSION, PayloadCodec, decode_checkpoint_envelope,
    encode_checkpoint_envelope, load_bocpd_checkpoint, load_bocpd_checkpoint_file,
    load_cusum_checkpoint, load_cusum_checkpoint_file, load_page_hinkley_checkpoint,
    load_page_hinkley_checkpoint_file, load_state_from_checkpoint_envelope,
    load_state_from_checkpoint_file, save_bocpd_checkpoint, save_bocpd_checkpoint_file,
    save_cusum_checkpoint, save_cusum_checkpoint_file, save_page_hinkley_checkpoint,
    save_page_hinkley_checkpoint_file, save_state_to_checkpoint_envelope,
    save_state_to_checkpoint_file, validate_checkpoint_state_schema_version,
};
pub use event_time::{LateDataCounters, LateDataPolicy, OverflowPolicy};

/// Online detector namespace.
pub fn crate_name() -> &'static str {
    let _ = (cpd_core::crate_name(), cpd_costs::crate_name());
    "cpd-online"
}
