// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::CpdError;
use std::cmp::Ordering;

/// Overflow behavior for late-data buffering windows.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverflowPolicy {
    DropOldest,
    DropNewest,
    Error,
}

impl OverflowPolicy {
    /// Stable user-facing policy name for diagnostics and messages.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DropOldest => "DropOldest",
            Self::DropNewest => "DropNewest",
            Self::Error => "Error",
        }
    }
}

/// Late-data policy for event-time online updates.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LateDataPolicy {
    Reject,
    BufferWithinWindow {
        max_delay_ns: i64,
        max_buffer_items: usize,
        on_overflow: OverflowPolicy,
    },
    ReorderByTimestamp {
        max_delay_ns: i64,
        max_buffer_items: usize,
        on_overflow: OverflowPolicy,
    },
}

impl Default for LateDataPolicy {
    fn default() -> Self {
        Self::Reject
    }
}

impl LateDataPolicy {
    /// Stable user-facing policy name for diagnostics and messages.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Reject => "Reject",
            Self::BufferWithinWindow { .. } => "BufferWithinWindow",
            Self::ReorderByTimestamp { .. } => "ReorderByTimestamp",
        }
    }

    /// Validates policy parameters.
    pub fn validate(&self) -> Result<(), CpdError> {
        match self {
            Self::Reject => Ok(()),
            Self::BufferWithinWindow {
                max_delay_ns,
                max_buffer_items,
                ..
            }
            | Self::ReorderByTimestamp {
                max_delay_ns,
                max_buffer_items,
                ..
            } => {
                if *max_delay_ns <= 0 {
                    return Err(CpdError::invalid_input(format!(
                        "{}.max_delay_ns must be > 0; got {}",
                        self.as_str(),
                        max_delay_ns
                    )));
                }
                if *max_buffer_items == 0 {
                    return Err(CpdError::invalid_input(format!(
                        "{}.max_buffer_items must be > 0; got 0",
                        self.as_str()
                    )));
                }
                Ok(())
            }
        }
    }
}

/// Late-data counters carried in detector state and checkpoint payloads.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LateDataCounters {
    pub late_events: u64,
    pub buffered_events: u64,
    pub reordered_events: u64,
    pub dropped_oldest: u64,
    pub dropped_newest: u64,
    pub overflow_errors: u64,
}

/// Comparator for deterministic timestamp ordering with stable arrival tie-break.
pub fn compare_event_time_then_arrival(
    lhs_t_ns: i64,
    lhs_arrival_seq: u64,
    rhs_t_ns: i64,
    rhs_arrival_seq: u64,
) -> Ordering {
    lhs_t_ns
        .cmp(&rhs_t_ns)
        .then(lhs_arrival_seq.cmp(&rhs_arrival_seq))
}

#[cfg(test)]
mod tests {
    use super::{LateDataPolicy, OverflowPolicy, compare_event_time_then_arrival};

    #[test]
    fn late_data_policy_validation_rejects_non_positive_values() {
        let invalid_delay = LateDataPolicy::BufferWithinWindow {
            max_delay_ns: 0,
            max_buffer_items: 8,
            on_overflow: OverflowPolicy::Error,
        };
        let err_delay = invalid_delay
            .validate()
            .expect_err("zero delay should be rejected");
        assert!(err_delay.to_string().contains("max_delay_ns"));

        let invalid_capacity = LateDataPolicy::ReorderByTimestamp {
            max_delay_ns: 10,
            max_buffer_items: 0,
            on_overflow: OverflowPolicy::DropNewest,
        };
        let err_capacity = invalid_capacity
            .validate()
            .expect_err("zero capacity should be rejected");
        assert!(err_capacity.to_string().contains("max_buffer_items"));
    }

    #[test]
    fn compare_event_time_then_arrival_has_stable_tie_break() {
        assert_eq!(
            compare_event_time_then_arrival(10, 1, 11, 0),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            compare_event_time_then_arrival(10, 1, 10, 2),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            compare_event_time_then_arrival(10, 4, 10, 4),
            std::cmp::Ordering::Equal
        );
    }
}
