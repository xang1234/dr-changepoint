// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::baseline::{
    CUSUM_DETECTOR_ID, CUSUM_STATE_SCHEMA_VERSION, CusumDetector, CusumState,
    PAGE_HINKLEY_DETECTOR_ID, PAGE_HINKLEY_STATE_SCHEMA_VERSION, PageHinkleyDetector,
    PageHinkleyState,
};
use crate::bocpd::{BOCPD_DETECTOR_ID, BOCPD_STATE_SCHEMA_VERSION, BocpdDetector, BocpdState};
use cpd_core::{CpdError, MIGRATION_GUIDANCE_PATH, OnlineDetector};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

/// Current checkpoint state schema version emitted by runtime writers.
pub const CURRENT_CHECKPOINT_SCHEMA_VERSION: u32 = 1;
/// Minimum checkpoint state schema version accepted by runtime readers.
pub const MIN_SUPPORTED_CHECKPOINT_SCHEMA_VERSION: u32 = 0;
/// Migration guidance template referenced in unsupported-version errors.
pub const CHECKPOINT_MIGRATION_GUIDANCE_PATH: &str = MIGRATION_GUIDANCE_PATH;

/// Supported codec for detector state payload bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PayloadCodec {
    Json,
    Bincode,
}

/// Serialized checkpoint envelope for online detector state persistence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointEnvelope {
    pub detector_id: String,
    pub state_schema_version: u32,
    pub engine_fingerprint: String,
    pub created_at_ns: i64,
    pub payload_crc32: u32,
    pub payload_codec: PayloadCodec,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, Deserialize)]
struct LegacyCheckpointEnvelopeV0 {
    schema_version: u32,
    detector_id: String,
    engine_version: String,
    created_at_ns: i64,
    payload_crc32: String,
    payload_codec: String,
    payload: serde_json::Value,
}

impl LegacyCheckpointEnvelopeV0 {
    fn into_current(self) -> Result<CheckpointEnvelope, CpdError> {
        let payload_codec = parse_payload_codec(&self.payload_codec)?;
        let payload = match payload_codec {
            PayloadCodec::Json => serde_json::to_vec(&self.payload).map_err(|err| {
                CpdError::invalid_input(format!(
                    "legacy checkpoint payload serialization failed (codec=json): {err}"
                ))
            })?,
            PayloadCodec::Bincode => {
                serde_json::from_value::<Vec<u8>>(self.payload).map_err(|err| {
                    CpdError::invalid_input(format!(
                        "legacy checkpoint payload parse failed (codec=bincode): {err}"
                    ))
                })?
            }
        };

        let payload_crc32 = parse_legacy_crc32(&self.payload_crc32)?;

        let envelope = CheckpointEnvelope {
            detector_id: self.detector_id,
            state_schema_version: self.schema_version,
            engine_fingerprint: self.engine_version,
            created_at_ns: self.created_at_ns,
            payload_crc32,
            payload_codec,
            payload,
        };

        envelope.validate_metadata()?;
        envelope.verify_payload_crc32()?;
        Ok(envelope)
    }
}

impl CheckpointEnvelope {
    fn validate_metadata(&self) -> Result<(), CpdError> {
        if self.detector_id.trim().is_empty() {
            return Err(CpdError::invalid_input(
                "checkpoint detector_id must be non-empty",
            ));
        }
        if self.engine_fingerprint.trim().is_empty() {
            return Err(CpdError::invalid_input(
                "checkpoint engine_fingerprint must be non-empty",
            ));
        }
        if self.created_at_ns < 0 {
            return Err(CpdError::invalid_input(format!(
                "checkpoint created_at_ns must be >= 0; got {}",
                self.created_at_ns
            )));
        }
        validate_checkpoint_state_schema_version(self.state_schema_version)
    }

    fn verify_payload_crc32(&self) -> Result<(), CpdError> {
        let observed = crc32fast::hash(&self.payload);
        if observed != self.payload_crc32 {
            return Err(CpdError::invalid_input(format!(
                "checkpoint payload crc32 mismatch: expected=0x{:08x}, observed=0x{:08x}",
                self.payload_crc32, observed
            )));
        }
        Ok(())
    }
}

fn parse_payload_codec(raw: &str) -> Result<PayloadCodec, CpdError> {
    match raw {
        "json" => Ok(PayloadCodec::Json),
        "bincode" => Ok(PayloadCodec::Bincode),
        other => Err(CpdError::invalid_input(format!(
            "checkpoint payload_codec '{other}' is unsupported; expected one of: json, bincode"
        ))),
    }
}

fn parse_legacy_crc32(raw: &str) -> Result<u32, CpdError> {
    let normalized = raw.strip_prefix("0x").unwrap_or(raw);
    u32::from_str_radix(normalized, 16).map_err(|err| {
        CpdError::invalid_input(format!(
            "legacy checkpoint payload_crc32='{raw}' is invalid: {err}"
        ))
    })
}

fn checkpoint_engine_fingerprint() -> String {
    format!(
        "cpd-online/{}/{}-{}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}

fn now_unix_ns() -> Result<i64, CpdError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| {
            CpdError::resource_limit(format!(
                "system clock before UNIX epoch; cannot timestamp checkpoint: {err}"
            ))
        })?;
    i64::try_from(elapsed.as_nanos()).map_err(|_| {
        CpdError::resource_limit("system timestamp overflow while constructing checkpoint")
    })
}

fn io_resource_error(action: &str, path: &Path, err: std::io::Error) -> CpdError {
    CpdError::resource_limit(format!("{action} '{}': {err}", path.display()))
}

fn serialize_state_payload<State: Serialize>(
    state: &State,
    payload_codec: PayloadCodec,
) -> Result<Vec<u8>, CpdError> {
    match payload_codec {
        PayloadCodec::Json => serde_json::to_vec(state).map_err(|err| {
            CpdError::invalid_input(format!(
                "checkpoint payload serialization failed (codec=json): {err}"
            ))
        }),
        PayloadCodec::Bincode => bincode::serialize(state).map_err(|err| {
            CpdError::invalid_input(format!(
                "checkpoint payload serialization failed (codec=bincode): {err}"
            ))
        }),
    }
}

fn deserialize_state_payload<State: DeserializeOwned>(
    payload: &[u8],
    payload_codec: PayloadCodec,
) -> Result<State, CpdError> {
    match payload_codec {
        PayloadCodec::Json => serde_json::from_slice(payload).map_err(|err| {
            CpdError::invalid_input(format!(
                "checkpoint payload deserialization failed (codec=json): {err}"
            ))
        }),
        PayloadCodec::Bincode => bincode::deserialize(payload).map_err(|err| {
            CpdError::invalid_input(format!(
                "checkpoint payload deserialization failed (codec=bincode): {err}"
            ))
        }),
    }
}

fn write_checkpoint_file_atomic(path: &Path, encoded: &[u8]) -> Result<(), CpdError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path.file_name().ok_or_else(|| {
        CpdError::invalid_input(format!(
            "checkpoint path '{}' must include a file name",
            path.display()
        ))
    })?;
    let file_name = file_name.to_string_lossy();
    if file_name.is_empty() {
        return Err(CpdError::invalid_input(format!(
            "checkpoint path '{}' must include a non-empty file name",
            path.display()
        )));
    }

    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or_default();
    let temp_path = parent.join(format!("{file_name}.tmp-{}-{suffix}", process::id()));

    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temp_path)
        .map_err(|err| {
            io_resource_error("failed creating checkpoint temp file", &temp_path, err)
        })?;

    if let Err(err) = file.write_all(encoded) {
        let _ = std::fs::remove_file(&temp_path);
        return Err(io_resource_error(
            "failed writing checkpoint temp file",
            &temp_path,
            err,
        ));
    }

    if let Err(err) = file.sync_all() {
        let _ = std::fs::remove_file(&temp_path);
        return Err(io_resource_error(
            "failed fsync on checkpoint temp file",
            &temp_path,
            err,
        ));
    }

    if let Err(err) = std::fs::rename(&temp_path, path) {
        let _ = std::fs::remove_file(&temp_path);
        return Err(io_resource_error(
            "failed renaming checkpoint temp file",
            path,
            err,
        ));
    }

    Ok(())
}

/// Validates checkpoint schema version compatibility using N-1 read policy.
pub fn validate_checkpoint_state_schema_version(state_schema_version: u32) -> Result<(), CpdError> {
    if (MIN_SUPPORTED_CHECKPOINT_SCHEMA_VERSION..=CURRENT_CHECKPOINT_SCHEMA_VERSION)
        .contains(&state_schema_version)
    {
        return Ok(());
    }

    Err(CpdError::invalid_input(format!(
        "checkpoint state_schema_version={} is unsupported; supported versions are {}..={}. See {} for migration guidance.",
        state_schema_version,
        MIN_SUPPORTED_CHECKPOINT_SCHEMA_VERSION,
        CURRENT_CHECKPOINT_SCHEMA_VERSION,
        CHECKPOINT_MIGRATION_GUIDANCE_PATH
    )))
}

/// Serializes a checkpoint envelope to JSON bytes.
pub fn encode_checkpoint_envelope(envelope: &CheckpointEnvelope) -> Result<Vec<u8>, CpdError> {
    envelope.validate_metadata()?;
    envelope.verify_payload_crc32()?;
    serde_json::to_vec(envelope).map_err(|err| {
        CpdError::invalid_input(format!("checkpoint envelope serialization failed: {err}"))
    })
}

/// Deserializes checkpoint envelope JSON bytes, accepting canonical v1 and legacy v0.
pub fn decode_checkpoint_envelope(encoded: &[u8]) -> Result<CheckpointEnvelope, CpdError> {
    let value: serde_json::Value = serde_json::from_slice(encoded).map_err(|err| {
        CpdError::invalid_input(format!("checkpoint envelope JSON parse failed: {err}"))
    })?;

    let envelope = if value.get("state_schema_version").is_some() {
        serde_json::from_value::<CheckpointEnvelope>(value).map_err(|err| {
            CpdError::invalid_input(format!(
                "checkpoint envelope v1 parse failed (expected canonical fields): {err}"
            ))
        })?
    } else if value.get("schema_version").is_some() {
        serde_json::from_value::<LegacyCheckpointEnvelopeV0>(value)
            .map_err(|err| {
                CpdError::invalid_input(format!(
                    "checkpoint envelope legacy v0 parse failed: {err}"
                ))
            })?
            .into_current()?
    } else {
        return Err(CpdError::invalid_input(
            "checkpoint envelope must contain 'state_schema_version' (v1) or 'schema_version' (legacy v0)",
        ));
    };

    envelope.validate_metadata()?;
    envelope.verify_payload_crc32()?;
    Ok(envelope)
}

/// Saves detector state into a checkpoint envelope.
pub fn save_state_to_checkpoint_envelope<State: Serialize>(
    detector_id: &str,
    state_schema_version: u32,
    state: &State,
    payload_codec: PayloadCodec,
) -> Result<CheckpointEnvelope, CpdError> {
    if detector_id.trim().is_empty() {
        return Err(CpdError::invalid_input(
            "checkpoint detector_id must be non-empty",
        ));
    }

    validate_checkpoint_state_schema_version(state_schema_version)?;

    let payload = serialize_state_payload(state, payload_codec)?;
    let envelope = CheckpointEnvelope {
        detector_id: detector_id.to_string(),
        state_schema_version,
        engine_fingerprint: checkpoint_engine_fingerprint(),
        created_at_ns: now_unix_ns()?,
        payload_crc32: crc32fast::hash(&payload),
        payload_codec,
        payload,
    };
    envelope.validate_metadata()?;
    envelope.verify_payload_crc32()?;
    Ok(envelope)
}

/// Loads detector state from a checkpoint envelope after compatibility and CRC checks.
pub fn load_state_from_checkpoint_envelope<State: DeserializeOwned>(
    envelope: &CheckpointEnvelope,
    expected_detector_id: &str,
) -> Result<State, CpdError> {
    envelope.validate_metadata()?;
    envelope.verify_payload_crc32()?;
    if envelope.detector_id != expected_detector_id {
        return Err(CpdError::invalid_input(format!(
            "checkpoint detector mismatch: expected='{}', found='{}'",
            expected_detector_id, envelope.detector_id
        )));
    }

    deserialize_state_payload(&envelope.payload, envelope.payload_codec)
}

/// Saves detector state to `path` using atomic persistence (tmp + fsync + rename).
pub fn save_state_to_checkpoint_file<State: Serialize>(
    path: impl AsRef<Path>,
    detector_id: &str,
    state_schema_version: u32,
    state: &State,
    payload_codec: PayloadCodec,
) -> Result<CheckpointEnvelope, CpdError> {
    let path = path.as_ref();
    let envelope =
        save_state_to_checkpoint_envelope(detector_id, state_schema_version, state, payload_codec)?;
    let encoded = encode_checkpoint_envelope(&envelope)?;
    write_checkpoint_file_atomic(path, &encoded)?;
    Ok(envelope)
}

/// Loads detector state from checkpoint file at `path`.
pub fn load_state_from_checkpoint_file<State: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_detector_id: &str,
) -> Result<State, CpdError> {
    let path = path.as_ref();
    let encoded = std::fs::read(path)
        .map_err(|err| io_resource_error("failed reading checkpoint file", path, err))?;
    let envelope = decode_checkpoint_envelope(&encoded)?;
    load_state_from_checkpoint_envelope(&envelope, expected_detector_id)
}

/// Saves BOCPD detector state into a checkpoint envelope.
pub fn save_bocpd_checkpoint(
    detector: &BocpdDetector,
    payload_codec: PayloadCodec,
) -> Result<CheckpointEnvelope, CpdError> {
    let state = detector.save_state();
    save_state_to_checkpoint_envelope(
        BOCPD_DETECTOR_ID,
        BOCPD_STATE_SCHEMA_VERSION,
        &state,
        payload_codec,
    )
}

/// Saves BOCPD detector state to checkpoint file at `path`.
pub fn save_bocpd_checkpoint_file(
    detector: &BocpdDetector,
    path: impl AsRef<Path>,
    payload_codec: PayloadCodec,
) -> Result<CheckpointEnvelope, CpdError> {
    let state = detector.save_state();
    save_state_to_checkpoint_file(
        path,
        BOCPD_DETECTOR_ID,
        BOCPD_STATE_SCHEMA_VERSION,
        &state,
        payload_codec,
    )
}

/// Restores BOCPD detector state from a checkpoint envelope.
pub fn load_bocpd_checkpoint(
    detector: &mut BocpdDetector,
    envelope: &CheckpointEnvelope,
) -> Result<(), CpdError> {
    let state: BocpdState = load_state_from_checkpoint_envelope(envelope, BOCPD_DETECTOR_ID)?;
    state.validate()?;
    detector.load_state(&state);
    Ok(())
}

/// Restores BOCPD detector state from checkpoint file at `path`.
pub fn load_bocpd_checkpoint_file(
    detector: &mut BocpdDetector,
    path: impl AsRef<Path>,
) -> Result<(), CpdError> {
    let state: BocpdState = load_state_from_checkpoint_file(path, BOCPD_DETECTOR_ID)?;
    state.validate()?;
    detector.load_state(&state);
    Ok(())
}

/// Saves CUSUM detector state into a checkpoint envelope.
pub fn save_cusum_checkpoint(
    detector: &CusumDetector,
    payload_codec: PayloadCodec,
) -> Result<CheckpointEnvelope, CpdError> {
    let state = detector.save_state();
    save_state_to_checkpoint_envelope(
        CUSUM_DETECTOR_ID,
        CUSUM_STATE_SCHEMA_VERSION,
        &state,
        payload_codec,
    )
}

/// Saves CUSUM detector state to checkpoint file at `path`.
pub fn save_cusum_checkpoint_file(
    detector: &CusumDetector,
    path: impl AsRef<Path>,
    payload_codec: PayloadCodec,
) -> Result<CheckpointEnvelope, CpdError> {
    let state = detector.save_state();
    save_state_to_checkpoint_file(
        path,
        CUSUM_DETECTOR_ID,
        CUSUM_STATE_SCHEMA_VERSION,
        &state,
        payload_codec,
    )
}

/// Restores CUSUM detector state from a checkpoint envelope.
pub fn load_cusum_checkpoint(
    detector: &mut CusumDetector,
    envelope: &CheckpointEnvelope,
) -> Result<(), CpdError> {
    let state: CusumState = load_state_from_checkpoint_envelope(envelope, CUSUM_DETECTOR_ID)?;
    state.validate()?;
    detector.load_state(&state);
    Ok(())
}

/// Restores CUSUM detector state from checkpoint file at `path`.
pub fn load_cusum_checkpoint_file(
    detector: &mut CusumDetector,
    path: impl AsRef<Path>,
) -> Result<(), CpdError> {
    let state: CusumState = load_state_from_checkpoint_file(path, CUSUM_DETECTOR_ID)?;
    state.validate()?;
    detector.load_state(&state);
    Ok(())
}

/// Saves Page-Hinkley detector state into a checkpoint envelope.
pub fn save_page_hinkley_checkpoint(
    detector: &PageHinkleyDetector,
    payload_codec: PayloadCodec,
) -> Result<CheckpointEnvelope, CpdError> {
    let state = detector.save_state();
    save_state_to_checkpoint_envelope(
        PAGE_HINKLEY_DETECTOR_ID,
        PAGE_HINKLEY_STATE_SCHEMA_VERSION,
        &state,
        payload_codec,
    )
}

/// Saves Page-Hinkley detector state to checkpoint file at `path`.
pub fn save_page_hinkley_checkpoint_file(
    detector: &PageHinkleyDetector,
    path: impl AsRef<Path>,
    payload_codec: PayloadCodec,
) -> Result<CheckpointEnvelope, CpdError> {
    let state = detector.save_state();
    save_state_to_checkpoint_file(
        path,
        PAGE_HINKLEY_DETECTOR_ID,
        PAGE_HINKLEY_STATE_SCHEMA_VERSION,
        &state,
        payload_codec,
    )
}

/// Restores Page-Hinkley detector state from a checkpoint envelope.
pub fn load_page_hinkley_checkpoint(
    detector: &mut PageHinkleyDetector,
    envelope: &CheckpointEnvelope,
) -> Result<(), CpdError> {
    let state: PageHinkleyState =
        load_state_from_checkpoint_envelope(envelope, PAGE_HINKLEY_DETECTOR_ID)?;
    state.validate()?;
    detector.load_state(&state);
    Ok(())
}

/// Restores Page-Hinkley detector state from checkpoint file at `path`.
pub fn load_page_hinkley_checkpoint_file(
    detector: &mut PageHinkleyDetector,
    path: impl AsRef<Path>,
) -> Result<(), CpdError> {
    let state: PageHinkleyState = load_state_from_checkpoint_file(path, PAGE_HINKLEY_DETECTOR_ID)?;
    state.validate()?;
    detector.load_state(&state);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        CHECKPOINT_MIGRATION_GUIDANCE_PATH, PayloadCodec, decode_checkpoint_envelope,
        encode_checkpoint_envelope, load_bocpd_checkpoint, load_bocpd_checkpoint_file,
        load_cusum_checkpoint, load_cusum_checkpoint_file, load_page_hinkley_checkpoint_file,
        load_state_from_checkpoint_envelope, save_bocpd_checkpoint, save_bocpd_checkpoint_file,
        save_cusum_checkpoint, save_cusum_checkpoint_file, save_page_hinkley_checkpoint_file,
        save_state_to_checkpoint_envelope, validate_checkpoint_state_schema_version,
    };
    use crate::{
        BocpdConfig, BocpdDetector, CUSUM_DETECTOR_ID, CusumConfig, CusumDetector,
        PageHinkleyConfig, PageHinkleyDetector,
    };
    use cpd_core::{Constraints, ExecutionContext, OnlineDetector};
    use std::path::{Path, PathBuf};
    use std::process;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};

    const CHECKPOINT_V0_FIXTURE: &str = include_str!(
        "../../../tests/fixtures/migrations/checkpoint/online_detector_checkpoint.v0.json"
    );
    const CHECKPOINT_V1_FIXTURE: &str = include_str!(
        "../../../tests/fixtures/migrations/checkpoint/online_detector_checkpoint.v1.json"
    );

    fn ctx() -> ExecutionContext<'static> {
        static CONSTRAINTS: OnceLock<Constraints> = OnceLock::new();
        let constraints = CONSTRAINTS.get_or_init(Constraints::default);
        ExecutionContext::new(constraints)
    }

    fn make_bocpd() -> BocpdDetector {
        BocpdDetector::new(BocpdConfig::default()).expect("valid BOCPD config")
    }

    fn make_cusum() -> CusumDetector {
        CusumDetector::new(CusumConfig::default()).expect("valid CUSUM config")
    }

    fn make_page_hinkley() -> PageHinkleyDetector {
        PageHinkleyDetector::new(PageHinkleyConfig::default()).expect("valid Page-Hinkley config")
    }

    fn unique_temp_checkpoint_path(stem: &str) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let seq = NEXT.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("{stem}-{}-{seq}.json", process::id()))
    }

    fn remove_file_if_exists(path: &Path) {
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn envelope_roundtrip_supports_json_and_bincode_payload_codecs() {
        #[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
        struct MiniState {
            t: usize,
            score: f64,
        }

        let state = MiniState { t: 7, score: 0.42 };
        for codec in [PayloadCodec::Json, PayloadCodec::Bincode] {
            let envelope = save_state_to_checkpoint_envelope("mini", 1, &state, codec)
                .expect("envelope save should succeed");
            let encoded = encode_checkpoint_envelope(&envelope).expect("envelope should encode");
            let decoded =
                decode_checkpoint_envelope(&encoded).expect("envelope should decode after encode");
            assert_eq!(decoded, envelope);

            let restored: MiniState =
                load_state_from_checkpoint_envelope(&decoded, "mini").expect("state load succeeds");
            assert_eq!(restored, state);
        }
    }

    #[test]
    fn schema_compatibility_accepts_v1_and_legacy_v0_fixtures() {
        let v1 = decode_checkpoint_envelope(CHECKPOINT_V1_FIXTURE.as_bytes())
            .expect("v1 fixture should decode");
        assert_eq!(v1.state_schema_version, 1);

        let state_v1: crate::CusumState =
            load_state_from_checkpoint_envelope(&v1, CUSUM_DETECTOR_ID)
                .expect("v1 fixture should deserialize detector state");
        assert_eq!(state_v1.t, 3);

        let v0 = decode_checkpoint_envelope(CHECKPOINT_V0_FIXTURE.as_bytes())
            .expect("legacy v0 fixture should decode");
        assert_eq!(v0.state_schema_version, 0);

        let state_v0: crate::CusumState =
            load_state_from_checkpoint_envelope(&v0, CUSUM_DETECTOR_ID)
                .expect("v0 fixture should deserialize detector state");
        assert_eq!(state_v0.t, 3);
    }

    #[test]
    fn unsupported_schema_version_returns_actionable_error() {
        let err = validate_checkpoint_state_schema_version(99)
            .expect_err("unsupported schema should fail");
        let message = err.to_string();
        assert!(message.contains("state_schema_version=99"));
        assert!(message.contains(CHECKPOINT_MIGRATION_GUIDANCE_PATH));
    }

    #[test]
    fn corrupted_payload_and_truncated_envelope_fail_fast_with_invalid_input() {
        let mut detector = make_cusum();
        detector
            .update(&[1.0], None, &ctx())
            .expect("CUSUM update should succeed");

        let mut envelope = save_cusum_checkpoint(&detector, PayloadCodec::Json)
            .expect("save checkpoint should succeed");
        envelope.payload[0] ^= 0x01;

        let mut restored = make_cusum();
        let crc_err =
            load_cusum_checkpoint(&mut restored, &envelope).expect_err("crc mismatch must fail");
        assert!(crc_err.to_string().contains("crc32 mismatch"));

        let encoded = encode_checkpoint_envelope(
            &save_cusum_checkpoint(&detector, PayloadCodec::Json)
                .expect("checkpoint save should succeed"),
        )
        .expect("envelope encode should succeed");
        let truncated = &encoded[..encoded.len() / 2];
        let trunc_err = decode_checkpoint_envelope(truncated)
            .expect_err("truncated envelope must fail parsing");
        assert!(trunc_err.to_string().contains("JSON parse failed"));
    }

    #[test]
    fn detector_id_mismatch_fails_before_state_restore() {
        let detector = make_bocpd();
        let envelope = save_bocpd_checkpoint(&detector, PayloadCodec::Json)
            .expect("save checkpoint should succeed");

        let mut wrong_detector = make_cusum();
        let err = load_cusum_checkpoint(&mut wrong_detector, &envelope)
            .expect_err("detector mismatch must fail");
        assert!(err.to_string().contains("detector mismatch"));
    }

    #[test]
    fn bocpd_checkpoint_file_roundtrip_matches_uninterrupted_run() {
        let path = unique_temp_checkpoint_path("cpd-online-bocpd-checkpoint");
        remove_file_if_exists(&path);

        let mut baseline = make_bocpd();
        let mut first = make_bocpd();
        for i in 0..120 {
            let x = ((i as f64) * 0.07).sin();
            let lhs = baseline
                .update(&[x], None, &ctx())
                .expect("baseline update should succeed");
            let rhs = first
                .update(&[x], None, &ctx())
                .expect("first update should succeed");
            assert!((lhs.p_change - rhs.p_change).abs() < 1e-12);
            assert_eq!(lhs.alert, rhs.alert);
            assert_eq!(lhs.run_length_mode, rhs.run_length_mode);
        }

        save_bocpd_checkpoint_file(&first, &path, PayloadCodec::Bincode)
            .expect("checkpoint save should succeed");
        let mut restored = make_bocpd();
        load_bocpd_checkpoint_file(&mut restored, &path).expect("checkpoint load should succeed");

        for i in 120..260 {
            let x = if i % 53 < 11 {
                4.0
            } else {
                ((i as f64) * 0.03).cos()
            };
            let lhs = baseline
                .update(&[x], None, &ctx())
                .expect("baseline update should succeed");
            let rhs = restored
                .update(&[x], None, &ctx())
                .expect("restored update should succeed");
            assert!((lhs.p_change - rhs.p_change).abs() < 1e-12);
            assert_eq!(lhs.alert, rhs.alert);
            assert_eq!(lhs.run_length_mode, rhs.run_length_mode);
        }

        assert_eq!(baseline.save_state(), restored.save_state());
        remove_file_if_exists(&path);
    }

    #[test]
    fn cusum_checkpoint_file_roundtrip_matches_uninterrupted_run() {
        let path = unique_temp_checkpoint_path("cpd-online-cusum-checkpoint");
        remove_file_if_exists(&path);

        let mut baseline = make_cusum();
        let mut first = make_cusum();
        for i in 0..120 {
            let x = if i < 50 { 0.0 } else { 1.2 };
            let lhs = baseline
                .update(&[x], None, &ctx())
                .expect("baseline update should succeed");
            let rhs = first
                .update(&[x], None, &ctx())
                .expect("first update should succeed");
            assert!((lhs.p_change - rhs.p_change).abs() < 1e-12);
            assert_eq!(lhs.alert, rhs.alert);
            assert_eq!(lhs.run_length_mode, rhs.run_length_mode);
        }

        save_cusum_checkpoint_file(&first, &path, PayloadCodec::Json)
            .expect("checkpoint save should succeed");
        let mut restored = make_cusum();
        load_cusum_checkpoint_file(&mut restored, &path).expect("checkpoint load should succeed");

        for i in 120..260 {
            let x = if i % 37 < 9 { 2.0 } else { 0.1 };
            let lhs = baseline
                .update(&[x], None, &ctx())
                .expect("baseline update should succeed");
            let rhs = restored
                .update(&[x], None, &ctx())
                .expect("restored update should succeed");
            assert!((lhs.p_change - rhs.p_change).abs() < 1e-12);
            assert_eq!(lhs.alert, rhs.alert);
            assert_eq!(lhs.run_length_mode, rhs.run_length_mode);
        }

        assert_eq!(baseline.save_state(), restored.save_state());
        remove_file_if_exists(&path);
    }

    #[test]
    fn page_hinkley_checkpoint_file_roundtrip_matches_uninterrupted_run() {
        let path = unique_temp_checkpoint_path("cpd-online-page-hinkley-checkpoint");
        remove_file_if_exists(&path);

        let mut baseline = make_page_hinkley();
        let mut first = make_page_hinkley();
        for i in 0..120 {
            let x = if i < 50 { 0.0 } else { 1.5 };
            let lhs = baseline
                .update(&[x], None, &ctx())
                .expect("baseline update should succeed");
            let rhs = first
                .update(&[x], None, &ctx())
                .expect("first update should succeed");
            assert!((lhs.p_change - rhs.p_change).abs() < 1e-12);
            assert_eq!(lhs.alert, rhs.alert);
            assert_eq!(lhs.run_length_mode, rhs.run_length_mode);
        }

        save_page_hinkley_checkpoint_file(&first, &path, PayloadCodec::Json)
            .expect("checkpoint save should succeed");
        let mut restored = make_page_hinkley();
        load_page_hinkley_checkpoint_file(&mut restored, &path)
            .expect("checkpoint load should succeed");

        for i in 120..260 {
            let x = if i % 31 < 10 { 2.5 } else { 0.2 };
            let lhs = baseline
                .update(&[x], None, &ctx())
                .expect("baseline update should succeed");
            let rhs = restored
                .update(&[x], None, &ctx())
                .expect("restored update should succeed");
            assert!((lhs.p_change - rhs.p_change).abs() < 1e-12);
            assert_eq!(lhs.alert, rhs.alert);
            assert_eq!(lhs.run_length_mode, rhs.run_length_mode);
        }

        assert_eq!(baseline.save_state(), restored.save_state());
        remove_file_if_exists(&path);
    }

    #[test]
    fn atomic_save_writes_final_file_without_leaking_temp_files() {
        let path = unique_temp_checkpoint_path("cpd-online-atomic-save");
        remove_file_if_exists(&path);
        let parent = path
            .parent()
            .expect("temp checkpoint path must have a parent")
            .to_path_buf();
        let file_name = path
            .file_name()
            .expect("temp checkpoint path must have a file name")
            .to_string_lossy()
            .to_string();
        let temp_prefix = format!("{file_name}.tmp-");

        let detector = make_cusum();
        save_cusum_checkpoint_file(&detector, &path, PayloadCodec::Json)
            .expect("atomic checkpoint save should succeed");

        assert!(path.exists(), "final checkpoint path should exist");
        let dir_entries = std::fs::read_dir(&parent).expect("should read temp dir entries");
        for entry in dir_entries {
            let entry = entry.expect("directory entry should load");
            let name = entry.file_name();
            let name = name.to_string_lossy();
            assert!(
                !name.starts_with(&temp_prefix),
                "stale temp checkpoint file detected: {name}"
            );
        }

        let mut restored = make_cusum();
        load_cusum_checkpoint_file(&mut restored, &path).expect("load after atomic save succeeds");
        assert_eq!(restored.save_state(), detector.save_state());
        remove_file_if_exists(&path);
    }

    #[test]
    fn load_bocpd_checkpoint_rejects_payload_that_deserializes_but_fails_state_validation() {
        let mut detector = make_bocpd();
        detector
            .update(&[1.0], None, &ctx())
            .expect("update should succeed");
        let mut envelope = save_bocpd_checkpoint(&detector, PayloadCodec::Json)
            .expect("checkpoint save should succeed");

        let mut state: crate::BocpdState =
            load_state_from_checkpoint_envelope(&envelope, crate::BOCPD_DETECTOR_ID)
                .expect("state decode should succeed");
        state.log_run_probs.clear();
        envelope.payload =
            serde_json::to_vec(&state).expect("mutated state should serialize for test fixture");
        envelope.payload_crc32 = crc32fast::hash(&envelope.payload);

        let err = load_bocpd_checkpoint(&mut detector, &envelope)
            .expect_err("state validation should fail before load_state installation");
        assert!(
            err.to_string()
                .contains("requires at least one run-length probability")
        );
    }
}
