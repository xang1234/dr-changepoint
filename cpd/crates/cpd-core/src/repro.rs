// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

/// Reproducibility mode used to control determinism/performance trade-offs.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReproMode {
    Strict,
    #[default]
    Balanced,
    Fast,
}

#[cfg(test)]
mod tests {
    use super::ReproMode;

    #[test]
    fn repro_mode_default_is_balanced() {
        assert_eq!(ReproMode::default(), ReproMode::Balanced);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn repro_mode_serde_roundtrip() {
        for mode in [ReproMode::Strict, ReproMode::Balanced, ReproMode::Fast] {
            let encoded = serde_json::to_string(&mode).expect("repro mode should serialize");
            let decoded: ReproMode =
                serde_json::from_str(&encoded).expect("repro mode should deserialize");
            assert_eq!(decoded, mode);
        }
    }
}
