// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::{
    Constraints, CpdError, Diagnostics, OfflineChangePointResult, SegmentStats,
    validate_constraints_config,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Current schema version written by CPD runtime adapters.
pub const CURRENT_SCHEMA_VERSION: u32 = 1;
/// Maximum additive forward-compatible schema version accepted by readers.
pub const MAX_FORWARD_COMPAT_SCHEMA_VERSION: u32 = 2;
/// Path to migration guidance template included in schema-version errors.
pub const MIGRATION_GUIDANCE_PATH: &str = "cpd/docs/templates/schema_migration.md";

pub type UnknownFields = Map<String, Value>;

fn unsupported_schema_message(artifact: &str, schema_version: u32) -> String {
    format!(
        "{artifact} schema_version={schema_version} is unsupported; supported versions are {CURRENT_SCHEMA_VERSION}..={MAX_FORWARD_COMPAT_SCHEMA_VERSION}. See {MIGRATION_GUIDANCE_PATH} for migration guidance."
    )
}

/// Validates whether an artifact schema version is currently readable.
pub fn validate_schema_version(schema_version: u32, artifact: &str) -> Result<(), CpdError> {
    if (CURRENT_SCHEMA_VERSION..=MAX_FORWARD_COMPAT_SCHEMA_VERSION).contains(&schema_version) {
        return Ok(());
    }

    Err(CpdError::invalid_input(unsupported_schema_message(
        artifact,
        schema_version,
    )))
}

/// Wire format for versioned constraints config payloads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstraintsConfigWire {
    pub schema_version: u32,
    #[serde(flatten)]
    pub config: Constraints,
    #[serde(default, flatten)]
    pub unknown_fields: UnknownFields,
}

impl ConstraintsConfigWire {
    pub fn from_runtime(config: Constraints) -> Self {
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            config,
            unknown_fields: UnknownFields::new(),
        }
    }

    pub fn from_runtime_with_unknown(
        config: Constraints,
        schema_version: u32,
        unknown_fields: UnknownFields,
    ) -> Self {
        Self {
            schema_version,
            config,
            unknown_fields,
        }
    }

    pub fn into_runtime_parts(self) -> Result<(Constraints, UnknownFields), CpdError> {
        validate_schema_version(self.schema_version, "ConstraintsConfig")?;
        validate_constraints_config(&self.config)?;
        Ok((self.config, self.unknown_fields))
    }

    pub fn to_runtime(self) -> Result<Constraints, CpdError> {
        let (config, _) = self.into_runtime_parts()?;
        Ok(config)
    }
}

/// Wire format for diagnostics payloads that preserves unknown fields.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticsWire {
    #[serde(flatten)]
    pub diagnostics: Diagnostics,
    #[serde(default, flatten)]
    pub unknown_fields: UnknownFields,
}

impl DiagnosticsWire {
    pub fn from_runtime(diagnostics: Diagnostics) -> Self {
        Self {
            diagnostics,
            unknown_fields: UnknownFields::new(),
        }
    }

    pub fn from_runtime_with_unknown(
        diagnostics: Diagnostics,
        unknown_fields: UnknownFields,
    ) -> Self {
        Self {
            diagnostics,
            unknown_fields,
        }
    }

    pub fn schema_version(&self) -> u32 {
        self.diagnostics.schema_version
    }

    pub fn set_schema_version(&mut self, schema_version: u32) {
        self.diagnostics.schema_version = schema_version;
    }

    pub fn into_runtime_parts(self) -> Result<(Diagnostics, UnknownFields), CpdError> {
        validate_schema_version(self.diagnostics.schema_version, "Diagnostics")?;
        Ok((self.diagnostics, self.unknown_fields))
    }

    pub fn to_runtime(self) -> Result<Diagnostics, CpdError> {
        let (diagnostics, _) = self.into_runtime_parts()?;
        Ok(diagnostics)
    }
}

/// Wire format for versioned offline result payloads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OfflineChangePointResultWire {
    pub breakpoints: Vec<usize>,
    pub change_points: Vec<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scores: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<SegmentStats>>,
    pub diagnostics: DiagnosticsWire,
    #[serde(default, flatten)]
    pub unknown_fields: UnknownFields,
}

impl OfflineChangePointResultWire {
    pub fn from_runtime(result: OfflineChangePointResult) -> Self {
        Self {
            breakpoints: result.breakpoints,
            change_points: result.change_points,
            scores: result.scores,
            segments: result.segments,
            diagnostics: DiagnosticsWire::from_runtime(result.diagnostics),
            unknown_fields: UnknownFields::new(),
        }
    }

    pub fn from_runtime_with_unknown(
        result: OfflineChangePointResult,
        result_unknown_fields: UnknownFields,
        diagnostics_unknown_fields: UnknownFields,
    ) -> Self {
        Self {
            breakpoints: result.breakpoints,
            change_points: result.change_points,
            scores: result.scores,
            segments: result.segments,
            diagnostics: DiagnosticsWire::from_runtime_with_unknown(
                result.diagnostics,
                diagnostics_unknown_fields,
            ),
            unknown_fields: result_unknown_fields,
        }
    }

    pub fn schema_version(&self) -> u32 {
        self.diagnostics.schema_version()
    }

    pub fn set_schema_version(&mut self, schema_version: u32) {
        self.diagnostics.set_schema_version(schema_version);
    }

    pub fn into_runtime_parts(
        self,
    ) -> Result<(OfflineChangePointResult, UnknownFields, UnknownFields), CpdError> {
        let (diagnostics, diagnostics_unknown_fields) = self.diagnostics.into_runtime_parts()?;
        let result = OfflineChangePointResult {
            breakpoints: self.breakpoints,
            change_points: self.change_points,
            scores: self.scores,
            segments: self.segments,
            diagnostics,
        };
        result.validate(result.diagnostics.n)?;
        Ok((result, self.unknown_fields, diagnostics_unknown_fields))
    }

    pub fn to_runtime(self) -> Result<OfflineChangePointResult, CpdError> {
        let (result, _, _) = self.into_runtime_parts()?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CURRENT_SCHEMA_VERSION, ConstraintsConfigWire, DiagnosticsWire,
        MAX_FORWARD_COMPAT_SCHEMA_VERSION, MIGRATION_GUIDANCE_PATH, OfflineChangePointResultWire,
    };
    use crate::Constraints;
    use serde_json::{Value, json};

    const CONSTRAINTS_V1_FIXTURE: &str =
        include_str!("../../../tests/fixtures/migrations/config/constraints.v1.json");
    const CONSTRAINTS_V2_FIXTURE: &str =
        include_str!("../../../tests/fixtures/migrations/config/constraints.v2.additive.json");
    const RESULT_V1_FIXTURE: &str =
        include_str!("../../../tests/fixtures/migrations/result/offline_result.v1.json");
    const RESULT_V2_FIXTURE: &str =
        include_str!("../../../tests/fixtures/migrations/result/offline_result.v2.additive.json");
    const DIAGNOSTICS_V1_FIXTURE: &str =
        include_str!("../../../tests/fixtures/migrations/result/diagnostics.v1.json");
    const DIAGNOSTICS_V2_FIXTURE: &str =
        include_str!("../../../tests/fixtures/migrations/result/diagnostics.v2.additive.json");

    fn parse_json(raw: &str) -> Value {
        serde_json::from_str(raw).expect("fixture should parse")
    }

    #[test]
    fn constraints_v1_roundtrip_matches_fixture() {
        let wire: ConstraintsConfigWire = serde_json::from_str(CONSTRAINTS_V1_FIXTURE)
            .expect("v1 constraints should deserialize");
        let encoded = serde_json::to_value(&wire).expect("wire should serialize");
        assert_eq!(encoded, parse_json(CONSTRAINTS_V1_FIXTURE));
        assert_eq!(wire.schema_version, CURRENT_SCHEMA_VERSION);
    }

    #[test]
    fn diagnostics_v1_roundtrip_matches_fixture() {
        let wire: DiagnosticsWire = serde_json::from_str(DIAGNOSTICS_V1_FIXTURE)
            .expect("v1 diagnostics should deserialize");
        let encoded = serde_json::to_value(&wire).expect("wire should serialize");
        assert_eq!(encoded, parse_json(DIAGNOSTICS_V1_FIXTURE));
        assert_eq!(wire.schema_version(), CURRENT_SCHEMA_VERSION);
    }

    #[test]
    fn offline_result_v1_roundtrip_matches_fixture() {
        let wire: OfflineChangePointResultWire =
            serde_json::from_str(RESULT_V1_FIXTURE).expect("v1 result should deserialize");
        let encoded = serde_json::to_value(&wire).expect("wire should serialize");
        assert_eq!(encoded, parse_json(RESULT_V1_FIXTURE));
        assert_eq!(wire.schema_version(), CURRENT_SCHEMA_VERSION);
    }

    #[test]
    fn v1_reader_accepts_v2_additive_fixtures() {
        let constraints_wire: ConstraintsConfigWire = serde_json::from_str(CONSTRAINTS_V2_FIXTURE)
            .expect("v2 constraints should deserialize");
        let diagnostics_wire: DiagnosticsWire = serde_json::from_str(DIAGNOSTICS_V2_FIXTURE)
            .expect("v2 diagnostics should deserialize");
        let result_wire: OfflineChangePointResultWire =
            serde_json::from_str(RESULT_V2_FIXTURE).expect("v2 result should deserialize");

        assert_eq!(
            constraints_wire.schema_version,
            MAX_FORWARD_COMPAT_SCHEMA_VERSION
        );
        assert_eq!(
            diagnostics_wire.schema_version(),
            MAX_FORWARD_COMPAT_SCHEMA_VERSION
        );
        assert_eq!(
            result_wire.schema_version(),
            MAX_FORWARD_COMPAT_SCHEMA_VERSION
        );

        constraints_wire
            .to_runtime()
            .expect("v2 constraints should convert to runtime");
        diagnostics_wire
            .to_runtime()
            .expect("v2 diagnostics should convert to runtime");
        result_wire
            .to_runtime()
            .expect("v2 result should convert to runtime");
    }

    #[test]
    fn v2_reader_behavior_accepts_v1_and_fills_defaults() {
        let mut constraints_wire: ConstraintsConfigWire = serde_json::from_value(json!({
            "schema_version": 1,
            "min_segment_len": 2,
            "jump": 1,
            "cache_policy": "Full",
            "degradation_plan": [],
            "allow_algorithm_fallback": false
        }))
        .expect("minimal v1 constraints should deserialize");
        constraints_wire.schema_version = MAX_FORWARD_COMPAT_SCHEMA_VERSION;
        let constraints_value =
            serde_json::to_value(&constraints_wire).expect("constraints wire should serialize");
        assert_eq!(
            constraints_value.get("schema_version"),
            Some(&Value::from(MAX_FORWARD_COMPAT_SCHEMA_VERSION))
        );
        let constraints_runtime = constraints_wire
            .to_runtime()
            .expect("constraints to_runtime should validate and fill option defaults");
        assert!(constraints_runtime.max_change_points.is_none());
        assert!(constraints_runtime.max_depth.is_none());
        assert!(constraints_runtime.candidate_splits.is_none());
        assert!(constraints_runtime.time_budget_ms.is_none());
        assert!(constraints_runtime.max_cost_evals.is_none());
        assert!(constraints_runtime.memory_budget_bytes.is_none());
        assert!(constraints_runtime.max_cache_bytes.is_none());

        let mut diagnostics_wire: DiagnosticsWire = serde_json::from_value(json!({
            "n": 20,
            "d": 1,
            "schema_version": 1,
            "algorithm": "pelt",
            "cost_model": "l2_mean",
            "repro_mode": "Balanced",
            "notes": [],
            "warnings": []
        }))
        .expect("minimal v1 diagnostics should deserialize");
        diagnostics_wire.set_schema_version(MAX_FORWARD_COMPAT_SCHEMA_VERSION);
        let diagnostics_value =
            serde_json::to_value(&diagnostics_wire).expect("diagnostics wire should serialize");
        assert_eq!(
            diagnostics_value.get("schema_version"),
            Some(&Value::from(MAX_FORWARD_COMPAT_SCHEMA_VERSION))
        );
        let diagnostics_runtime = diagnostics_wire
            .to_runtime()
            .expect("diagnostics to_runtime should fill option defaults");
        assert!(diagnostics_runtime.engine_version.is_none());
        assert!(diagnostics_runtime.runtime_ms.is_none());
        assert!(diagnostics_runtime.seed.is_none());
        assert!(diagnostics_runtime.thread_count.is_none());
        assert!(diagnostics_runtime.blas_backend.is_none());
        assert!(diagnostics_runtime.cpu_features.is_none());
        assert!(diagnostics_runtime.pruning_stats.is_none());
        assert!(diagnostics_runtime.missing_policy_applied.is_none());
        assert!(diagnostics_runtime.missing_fraction.is_none());
        assert!(diagnostics_runtime.effective_sample_count.is_none());

        let mut result_wire: OfflineChangePointResultWire = serde_json::from_value(json!({
            "breakpoints": [20],
            "change_points": [],
            "diagnostics": {
                "n": 20,
                "d": 1,
                "schema_version": 1,
                "algorithm": "pelt",
                "cost_model": "l2_mean",
                "repro_mode": "Balanced",
                "notes": [],
                "warnings": []
            }
        }))
        .expect("minimal v1 result should deserialize");
        result_wire.set_schema_version(MAX_FORWARD_COMPAT_SCHEMA_VERSION);
        let result_value =
            serde_json::to_value(&result_wire).expect("result wire should serialize");
        let diagnostics_obj = result_value
            .get("diagnostics")
            .and_then(Value::as_object)
            .expect("diagnostics should be object");
        assert_eq!(
            diagnostics_obj.get("schema_version"),
            Some(&Value::from(MAX_FORWARD_COMPAT_SCHEMA_VERSION))
        );
        let result_runtime = result_wire
            .to_runtime()
            .expect("result to_runtime should fill defaults and validate shape");
        assert!(result_runtime.scores.is_none());
        assert!(result_runtime.segments.is_none());
    }

    #[test]
    fn unsupported_schema_version_returns_migration_guidance() {
        let err = ConstraintsConfigWire {
            schema_version: 99,
            config: ConstraintsConfigWire::from_runtime(Default::default()).config,
            unknown_fields: Default::default(),
        }
        .to_runtime()
        .expect_err("unsupported schema version should fail");
        let message = err.to_string();
        assert!(message.contains("schema_version=99"));
        assert!(message.contains(MIGRATION_GUIDANCE_PATH));
    }

    #[test]
    fn unknown_fields_roundtrip_for_result_and_diagnostics() {
        let wire: OfflineChangePointResultWire =
            serde_json::from_str(RESULT_V2_FIXTURE).expect("v2 result should deserialize");

        assert!(wire.unknown_fields.contains_key("future_result_flag"));
        assert!(
            wire.diagnostics
                .unknown_fields
                .contains_key("future_diagnostics_flag")
        );

        let (runtime, result_unknown, diagnostics_unknown) = wire
            .clone()
            .into_runtime_parts()
            .expect("wire should convert to runtime with unknown fields");

        let rebuilt = OfflineChangePointResultWire::from_runtime_with_unknown(
            runtime,
            result_unknown,
            diagnostics_unknown,
        );
        let rebuilt_value = serde_json::to_value(&rebuilt).expect("rebuilt wire should serialize");
        let source_value = parse_json(RESULT_V2_FIXTURE);
        assert_eq!(
            rebuilt_value.get("future_result_flag"),
            source_value.get("future_result_flag")
        );
        let rebuilt_diag = rebuilt_value
            .get("diagnostics")
            .and_then(Value::as_object)
            .expect("rebuilt diagnostics should be object");
        let source_diag = source_value
            .get("diagnostics")
            .and_then(Value::as_object)
            .expect("source diagnostics should be object");
        assert_eq!(
            rebuilt_diag.get("future_diagnostics_flag"),
            source_diag.get("future_diagnostics_flag")
        );
    }

    #[test]
    fn constraints_to_runtime_rejects_invalid_config_shape() {
        let err = ConstraintsConfigWire {
            schema_version: CURRENT_SCHEMA_VERSION,
            config: Constraints {
                jump: 0,
                ..Constraints::default()
            },
            unknown_fields: Default::default(),
        }
        .to_runtime()
        .expect_err("jump=0 must fail strict to_runtime validation");
        assert!(err.to_string().contains("constraints.jump"));
    }

    #[test]
    fn result_to_runtime_rejects_invalid_semantics() {
        let wire: OfflineChangePointResultWire = serde_json::from_value(json!({
            "breakpoints": [20],
            "change_points": [10],
            "diagnostics": {
                "n": 20,
                "d": 1,
                "schema_version": 1,
                "algorithm": "pelt",
                "cost_model": "l2_mean",
                "repro_mode": "Balanced",
                "notes": [],
                "warnings": []
            }
        }))
        .expect("wire should deserialize");
        let err = wire
            .to_runtime()
            .expect_err("invalid change_points must fail strict to_runtime validation");
        assert!(
            err.to_string()
                .contains("change_points must equal breakpoints excluding n")
        );
    }
}
