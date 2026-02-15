// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::{BinSegConfig, PeltConfig, WbsConfig, WbsIntervalStrategy};
use cpd_core::{
    CURRENT_SCHEMA_VERSION, CpdError, Stopping, validate_schema_version, validate_stopping,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub type UnknownFields = Map<String, Value>;

/// Wire format for versioned PELT config payloads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PeltConfigWire {
    pub schema_version: u32,
    pub stopping: Stopping,
    #[serde(default = "default_params_per_segment")]
    pub params_per_segment: usize,
    #[serde(default = "default_cancel_check_every")]
    pub cancel_check_every: usize,
    #[serde(default, flatten)]
    pub unknown_fields: UnknownFields,
}

impl PeltConfigWire {
    pub fn from_runtime(config: PeltConfig) -> Self {
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            stopping: config.stopping,
            params_per_segment: config.params_per_segment,
            cancel_check_every: config.cancel_check_every,
            unknown_fields: UnknownFields::new(),
        }
    }

    pub fn from_runtime_with_unknown(
        config: PeltConfig,
        schema_version: u32,
        unknown_fields: UnknownFields,
    ) -> Self {
        Self {
            schema_version,
            stopping: config.stopping,
            params_per_segment: config.params_per_segment,
            cancel_check_every: config.cancel_check_every,
            unknown_fields,
        }
    }

    pub fn into_runtime_parts(self) -> Result<(PeltConfig, UnknownFields), CpdError> {
        validate_schema_version(self.schema_version, "PeltConfig")?;
        let config = PeltConfig {
            stopping: self.stopping,
            params_per_segment: self.params_per_segment,
            cancel_check_every: self.cancel_check_every,
        };
        validate_pelt_config(&config)?;
        Ok((config, self.unknown_fields))
    }

    pub fn to_runtime(self) -> Result<PeltConfig, CpdError> {
        let (config, _) = self.into_runtime_parts()?;
        Ok(config)
    }
}

/// Wire format for versioned BinSeg config payloads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BinSegConfigWire {
    pub schema_version: u32,
    pub stopping: Stopping,
    #[serde(default = "default_params_per_segment")]
    pub params_per_segment: usize,
    #[serde(default = "default_cancel_check_every")]
    pub cancel_check_every: usize,
    #[serde(default, flatten)]
    pub unknown_fields: UnknownFields,
}

impl BinSegConfigWire {
    pub fn from_runtime(config: BinSegConfig) -> Self {
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            stopping: config.stopping,
            params_per_segment: config.params_per_segment,
            cancel_check_every: config.cancel_check_every,
            unknown_fields: UnknownFields::new(),
        }
    }

    pub fn from_runtime_with_unknown(
        config: BinSegConfig,
        schema_version: u32,
        unknown_fields: UnknownFields,
    ) -> Self {
        Self {
            schema_version,
            stopping: config.stopping,
            params_per_segment: config.params_per_segment,
            cancel_check_every: config.cancel_check_every,
            unknown_fields,
        }
    }

    pub fn into_runtime_parts(self) -> Result<(BinSegConfig, UnknownFields), CpdError> {
        validate_schema_version(self.schema_version, "BinSegConfig")?;
        let config = BinSegConfig {
            stopping: self.stopping,
            params_per_segment: self.params_per_segment,
            cancel_check_every: self.cancel_check_every,
        };
        validate_binseg_config(&config)?;
        Ok((config, self.unknown_fields))
    }

    pub fn to_runtime(self) -> Result<BinSegConfig, CpdError> {
        let (config, _) = self.into_runtime_parts()?;
        Ok(config)
    }
}

/// Wire format for versioned WBS config payloads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WbsConfigWire {
    pub schema_version: u32,
    pub stopping: Stopping,
    #[serde(default = "default_params_per_segment")]
    pub params_per_segment: usize,
    #[serde(default)]
    pub num_intervals: Option<usize>,
    #[serde(default = "default_wbs_interval_strategy")]
    pub interval_strategy: WbsIntervalStrategy,
    #[serde(default = "default_wbs_seed")]
    pub seed: u64,
    #[serde(default = "default_cancel_check_every")]
    pub cancel_check_every: usize,
    #[serde(default, flatten)]
    pub unknown_fields: UnknownFields,
}

impl WbsConfigWire {
    pub fn from_runtime(config: WbsConfig) -> Self {
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            stopping: config.stopping,
            params_per_segment: config.params_per_segment,
            num_intervals: config.num_intervals,
            interval_strategy: config.interval_strategy,
            seed: config.seed,
            cancel_check_every: config.cancel_check_every,
            unknown_fields: UnknownFields::new(),
        }
    }

    pub fn from_runtime_with_unknown(
        config: WbsConfig,
        schema_version: u32,
        unknown_fields: UnknownFields,
    ) -> Self {
        Self {
            schema_version,
            stopping: config.stopping,
            params_per_segment: config.params_per_segment,
            num_intervals: config.num_intervals,
            interval_strategy: config.interval_strategy,
            seed: config.seed,
            cancel_check_every: config.cancel_check_every,
            unknown_fields,
        }
    }

    pub fn into_runtime_parts(self) -> Result<(WbsConfig, UnknownFields), CpdError> {
        validate_schema_version(self.schema_version, "WbsConfig")?;
        let config = WbsConfig {
            stopping: self.stopping,
            params_per_segment: self.params_per_segment,
            num_intervals: self.num_intervals,
            interval_strategy: self.interval_strategy,
            seed: self.seed,
            cancel_check_every: self.cancel_check_every,
        };
        validate_wbs_config(&config)?;
        Ok((config, self.unknown_fields))
    }

    pub fn to_runtime(self) -> Result<WbsConfig, CpdError> {
        let (config, _) = self.into_runtime_parts()?;
        Ok(config)
    }
}

fn default_params_per_segment() -> usize {
    PeltConfig::default().params_per_segment
}

fn default_cancel_check_every() -> usize {
    PeltConfig::default().cancel_check_every
}

fn default_wbs_interval_strategy() -> WbsIntervalStrategy {
    WbsConfig::default().interval_strategy
}

fn default_wbs_seed() -> u64 {
    WbsConfig::default().seed
}

fn validate_pelt_config(config: &PeltConfig) -> Result<(), CpdError> {
    validate_stopping(&config.stopping)?;
    if config.params_per_segment == 0 {
        return Err(CpdError::invalid_input(
            "PeltConfig.params_per_segment must be >= 1; got 0",
        ));
    }
    Ok(())
}

fn validate_binseg_config(config: &BinSegConfig) -> Result<(), CpdError> {
    validate_stopping(&config.stopping)?;
    if config.params_per_segment == 0 {
        return Err(CpdError::invalid_input(
            "BinSegConfig.params_per_segment must be >= 1; got 0",
        ));
    }
    Ok(())
}

fn validate_wbs_config(config: &WbsConfig) -> Result<(), CpdError> {
    validate_stopping(&config.stopping)?;
    if config.params_per_segment == 0 {
        return Err(CpdError::invalid_input(
            "WbsConfig.params_per_segment must be >= 1; got 0",
        ));
    }
    if matches!(config.num_intervals, Some(0)) {
        return Err(CpdError::invalid_input(
            "WbsConfig.num_intervals must be >= 1 when provided; got 0",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{BinSegConfigWire, PeltConfigWire, WbsConfigWire};
    use cpd_core::{MAX_FORWARD_COMPAT_SCHEMA_VERSION, MIGRATION_GUIDANCE_PATH};
    use serde_json::{Value, json};

    const PELT_V1_FIXTURE: &str =
        include_str!("../../../tests/fixtures/migrations/config/pelt.v1.json");
    const PELT_V2_FIXTURE: &str =
        include_str!("../../../tests/fixtures/migrations/config/pelt.v2.additive.json");
    const BINSEG_V1_FIXTURE: &str =
        include_str!("../../../tests/fixtures/migrations/config/binseg.v1.json");
    const BINSEG_V2_FIXTURE: &str =
        include_str!("../../../tests/fixtures/migrations/config/binseg.v2.additive.json");
    const WBS_V1_FIXTURE: &str =
        include_str!("../../../tests/fixtures/migrations/config/wbs.v1.json");
    const WBS_V2_FIXTURE: &str =
        include_str!("../../../tests/fixtures/migrations/config/wbs.v2.additive.json");

    fn parse_json(raw: &str) -> Value {
        serde_json::from_str(raw).expect("fixture should parse")
    }

    #[test]
    fn pelt_v1_roundtrip_matches_fixture() {
        let wire: PeltConfigWire =
            serde_json::from_str(PELT_V1_FIXTURE).expect("v1 pelt should deserialize");
        let encoded = serde_json::to_value(&wire).expect("wire should serialize");
        assert_eq!(encoded, parse_json(PELT_V1_FIXTURE));
    }

    #[test]
    fn binseg_v1_roundtrip_matches_fixture() {
        let wire: BinSegConfigWire =
            serde_json::from_str(BINSEG_V1_FIXTURE).expect("v1 binseg should deserialize");
        let encoded = serde_json::to_value(&wire).expect("wire should serialize");
        assert_eq!(encoded, parse_json(BINSEG_V1_FIXTURE));
    }

    #[test]
    fn wbs_v1_roundtrip_matches_fixture() {
        let wire: WbsConfigWire =
            serde_json::from_str(WBS_V1_FIXTURE).expect("v1 wbs should deserialize");
        let encoded = serde_json::to_value(&wire).expect("wire should serialize");
        assert_eq!(encoded, parse_json(WBS_V1_FIXTURE));
    }

    #[test]
    fn v1_reader_accepts_v2_additive_fixtures() {
        let pelt_wire: PeltConfigWire =
            serde_json::from_str(PELT_V2_FIXTURE).expect("v2 pelt should deserialize");
        let binseg_wire: BinSegConfigWire =
            serde_json::from_str(BINSEG_V2_FIXTURE).expect("v2 binseg should deserialize");
        let wbs_wire: WbsConfigWire =
            serde_json::from_str(WBS_V2_FIXTURE).expect("v2 wbs should deserialize");

        assert_eq!(pelt_wire.schema_version, MAX_FORWARD_COMPAT_SCHEMA_VERSION);
        assert_eq!(
            binseg_wire.schema_version,
            MAX_FORWARD_COMPAT_SCHEMA_VERSION
        );
        assert_eq!(wbs_wire.schema_version, MAX_FORWARD_COMPAT_SCHEMA_VERSION);

        pelt_wire
            .to_runtime()
            .expect("v2 pelt should convert to runtime");
        binseg_wire
            .to_runtime()
            .expect("v2 binseg should convert to runtime");
        wbs_wire
            .to_runtime()
            .expect("v2 wbs should convert to runtime");
    }

    #[test]
    fn v2_reader_behavior_accepts_v1_and_fills_defaults() {
        let mut pelt_wire: PeltConfigWire = serde_json::from_value(json!({
            "schema_version": 1,
            "stopping": {"Penalized": "BIC"}
        }))
        .expect("minimal v1 pelt should deserialize");
        pelt_wire.schema_version = MAX_FORWARD_COMPAT_SCHEMA_VERSION;
        let pelt_value = serde_json::to_value(&pelt_wire).expect("wire should serialize");
        assert_eq!(
            pelt_value.get("schema_version"),
            Some(&Value::from(MAX_FORWARD_COMPAT_SCHEMA_VERSION))
        );
        let pelt_runtime = pelt_wire
            .to_runtime()
            .expect("pelt to_runtime should fill defaults");
        assert_eq!(pelt_runtime.params_per_segment, 2);
        assert_eq!(pelt_runtime.cancel_check_every, 1000);

        let mut binseg_wire: BinSegConfigWire = serde_json::from_value(json!({
            "schema_version": 1,
            "stopping": {"Penalized": "AIC"}
        }))
        .expect("minimal v1 binseg should deserialize");
        binseg_wire.schema_version = MAX_FORWARD_COMPAT_SCHEMA_VERSION;
        let binseg_value = serde_json::to_value(&binseg_wire).expect("wire should serialize");
        assert_eq!(
            binseg_value.get("schema_version"),
            Some(&Value::from(MAX_FORWARD_COMPAT_SCHEMA_VERSION))
        );
        let binseg_runtime = binseg_wire
            .to_runtime()
            .expect("binseg to_runtime should fill defaults");
        assert_eq!(binseg_runtime.params_per_segment, 2);
        assert_eq!(binseg_runtime.cancel_check_every, 1000);

        let mut wbs_wire: WbsConfigWire = serde_json::from_value(json!({
            "schema_version": 1,
            "stopping": {"Penalized": "BIC"}
        }))
        .expect("minimal v1 wbs should deserialize");
        wbs_wire.schema_version = MAX_FORWARD_COMPAT_SCHEMA_VERSION;
        let wbs_value = serde_json::to_value(&wbs_wire).expect("wire should serialize");
        assert_eq!(
            wbs_value.get("schema_version"),
            Some(&Value::from(MAX_FORWARD_COMPAT_SCHEMA_VERSION))
        );
        let wbs_runtime = wbs_wire
            .to_runtime()
            .expect("wbs to_runtime should fill defaults");
        assert_eq!(wbs_runtime.params_per_segment, 2);
        assert_eq!(wbs_runtime.cancel_check_every, 1000);
        assert_eq!(wbs_runtime.num_intervals, None);
        assert_eq!(wbs_runtime.seed, 0);
    }

    #[test]
    fn unsupported_schema_version_returns_migration_guidance() {
        let err = PeltConfigWire {
            schema_version: 99,
            stopping: PeltConfigWire::from_runtime(Default::default()).stopping,
            params_per_segment: 2,
            cancel_check_every: 1000,
            unknown_fields: Default::default(),
        }
        .to_runtime()
        .expect_err("unsupported schema version should fail");
        let message = err.to_string();
        assert!(message.contains("schema_version=99"));
        assert!(message.contains(MIGRATION_GUIDANCE_PATH));
    }

    #[test]
    fn unknown_fields_roundtrip_for_config_wires() {
        let pelt_wire: PeltConfigWire =
            serde_json::from_str(PELT_V2_FIXTURE).expect("v2 pelt should deserialize");
        assert!(pelt_wire.unknown_fields.contains_key("future_pelt_flag"));
        let (pelt_runtime, pelt_unknown) = pelt_wire
            .clone()
            .into_runtime_parts()
            .expect("pelt wire should convert to runtime");
        let rebuilt_pelt = PeltConfigWire::from_runtime_with_unknown(
            pelt_runtime,
            MAX_FORWARD_COMPAT_SCHEMA_VERSION,
            pelt_unknown,
        );
        let rebuilt_pelt_value =
            serde_json::to_value(&rebuilt_pelt).expect("rebuilt pelt wire should serialize");
        let source_pelt_value = parse_json(PELT_V2_FIXTURE);
        assert_eq!(
            rebuilt_pelt_value.get("future_pelt_flag"),
            source_pelt_value.get("future_pelt_flag")
        );

        let binseg_wire: BinSegConfigWire =
            serde_json::from_str(BINSEG_V2_FIXTURE).expect("v2 binseg should deserialize");
        assert!(
            binseg_wire
                .unknown_fields
                .contains_key("future_binseg_flag")
        );
        let (binseg_runtime, binseg_unknown) = binseg_wire
            .clone()
            .into_runtime_parts()
            .expect("binseg wire should convert to runtime");
        let rebuilt_binseg = BinSegConfigWire::from_runtime_with_unknown(
            binseg_runtime,
            MAX_FORWARD_COMPAT_SCHEMA_VERSION,
            binseg_unknown,
        );
        let rebuilt_binseg_value =
            serde_json::to_value(&rebuilt_binseg).expect("rebuilt binseg wire should serialize");
        let source_binseg_value = parse_json(BINSEG_V2_FIXTURE);
        assert_eq!(
            rebuilt_binseg_value.get("future_binseg_flag"),
            source_binseg_value.get("future_binseg_flag")
        );

        let wbs_wire: WbsConfigWire =
            serde_json::from_str(WBS_V2_FIXTURE).expect("v2 wbs should deserialize");
        assert!(wbs_wire.unknown_fields.contains_key("future_wbs_flag"));
        let (wbs_runtime, wbs_unknown) = wbs_wire
            .clone()
            .into_runtime_parts()
            .expect("wbs wire should convert to runtime");
        let rebuilt_wbs = WbsConfigWire::from_runtime_with_unknown(
            wbs_runtime,
            MAX_FORWARD_COMPAT_SCHEMA_VERSION,
            wbs_unknown,
        );
        let rebuilt_wbs_value =
            serde_json::to_value(&rebuilt_wbs).expect("rebuilt wbs wire should serialize");
        let source_wbs_value = parse_json(WBS_V2_FIXTURE);
        assert_eq!(
            rebuilt_wbs_value.get("future_wbs_flag"),
            source_wbs_value.get("future_wbs_flag")
        );
    }

    #[test]
    fn pelt_to_runtime_rejects_invalid_params_per_segment() {
        let wire: PeltConfigWire = serde_json::from_value(json!({
            "schema_version": 1,
            "stopping": {"Penalized": "BIC"},
            "params_per_segment": 0,
            "cancel_check_every": 1000
        }))
        .expect("wire should deserialize");
        let err = wire
            .to_runtime()
            .expect_err("params_per_segment=0 must fail strict to_runtime validation");
        assert!(err.to_string().contains("PeltConfig.params_per_segment"));
    }

    #[test]
    fn binseg_to_runtime_rejects_invalid_stopping() {
        let wire: BinSegConfigWire = serde_json::from_value(json!({
            "schema_version": 1,
            "stopping": {"KnownK": 0},
            "params_per_segment": 2,
            "cancel_check_every": 1000
        }))
        .expect("wire should deserialize");
        let err = wire
            .to_runtime()
            .expect_err("KnownK(0) must fail strict to_runtime validation");
        assert!(err.to_string().contains("KnownK"));
    }

    #[test]
    fn wbs_to_runtime_rejects_invalid_num_intervals() {
        let wire: WbsConfigWire = serde_json::from_value(json!({
            "schema_version": 1,
            "stopping": {"Penalized": "BIC"},
            "params_per_segment": 2,
            "num_intervals": 0,
            "interval_strategy": "Random",
            "seed": 123,
            "cancel_check_every": 1000
        }))
        .expect("wire should deserialize");
        let err = wire
            .to_runtime()
            .expect_err("num_intervals=0 must fail strict to_runtime validation");
        assert!(err.to_string().contains("WbsConfig.num_intervals"));
    }
}
