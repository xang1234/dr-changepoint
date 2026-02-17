// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_main]

#[path = "common.rs"]
mod common;

use cpd_core::{Constraints, ExecutionContext, OnlineDetector};
use cpd_online::{
    AlertPolicy, BernoulliBetaPrior, BocpdConfig, BocpdDetector, ConstantHazard, GaussianNigPrior,
    GeometricHazard, HazardSpec, LateDataPolicy, ObservationModel, OverflowPolicy,
    PoissonGammaPrior,
};
use libfuzzer_sys::fuzz_target;

#[derive(Clone, Copy)]
enum ModelKind {
    Gaussian,
    Poisson,
    Bernoulli,
}

fn build_hazard(kind_seed: u8, value_seed: u8) -> HazardSpec {
    match kind_seed % 2 {
        0 => {
            let p_change = 0.01 + (f64::from(value_seed) / 255.0) * 0.98;
            HazardSpec::Constant(
                ConstantHazard::new(p_change).expect("mapped constant hazard must be valid"),
            )
        }
        _ => {
            let mean_run_length = 2.0 + (f64::from(value_seed) / 255.0) * 510.0;
            HazardSpec::Geometric(
                GeometricHazard::new(mean_run_length)
                    .expect("mapped geometric hazard must be valid"),
            )
        }
    }
}

fn build_observation(
    kind_seed: u8,
    mu_seed: i16,
    a_seed: u8,
    b_seed: u8,
    c_seed: u8,
) -> (ObservationModel, ModelKind) {
    match kind_seed % 3 {
        0 => {
            let prior = GaussianNigPrior {
                mu0: f64::from(mu_seed) / 16.0,
                kappa0: 0.25 + f64::from(a_seed % 32) / 8.0,
                alpha0: 0.25 + f64::from(b_seed % 32) / 8.0,
                beta0: 0.25 + f64::from(c_seed % 32) / 8.0,
            };
            (ObservationModel::Gaussian { prior }, ModelKind::Gaussian)
        }
        1 => {
            let prior = PoissonGammaPrior {
                alpha: 0.25 + f64::from(a_seed % 64) / 8.0,
                beta: 0.25 + f64::from(b_seed % 64) / 8.0,
            };
            (ObservationModel::Poisson { prior }, ModelKind::Poisson)
        }
        _ => {
            let prior = BernoulliBetaPrior {
                alpha: 0.25 + f64::from(a_seed % 64) / 8.0,
                beta: 0.25 + f64::from(c_seed % 64) / 8.0,
            };
            (ObservationModel::Bernoulli { prior }, ModelKind::Bernoulli)
        }
    }
}

fn build_alert_policy(
    threshold_seed: u8,
    hysteresis_seed: u8,
    cooldown_seed: u8,
    min_run_seed: u8,
) -> AlertPolicy {
    let threshold = f64::from(threshold_seed) / 255.0;
    let hysteresis = threshold * (f64::from(hysteresis_seed) / 255.0);
    AlertPolicy::new(
        threshold,
        hysteresis,
        usize::from(cooldown_seed % 16),
        usize::from(min_run_seed % 16),
    )
}

fn build_overflow_policy(seed: u8) -> OverflowPolicy {
    match seed % 3 {
        0 => OverflowPolicy::DropOldest,
        1 => OverflowPolicy::DropNewest,
        _ => OverflowPolicy::Error,
    }
}

fn build_late_data_policy(
    seed: u8,
    delay_seed: u8,
    buffer_seed: u8,
    overflow_seed: u8,
) -> LateDataPolicy {
    let max_delay_ns = i64::try_from(common::bounded(delay_seed, 1, 128)).unwrap_or(1);
    let max_buffer_items = common::bounded(buffer_seed, 1, 12);
    let on_overflow = build_overflow_policy(overflow_seed);

    match seed % 3 {
        0 => LateDataPolicy::Reject,
        1 => LateDataPolicy::BufferWithinWindow {
            max_delay_ns,
            max_buffer_items,
            on_overflow,
        },
        _ => LateDataPolicy::ReorderByTimestamp {
            max_delay_ns,
            max_buffer_items,
            on_overflow,
        },
    }
}

fn build_observation_value(model: ModelKind, base: f64, mode_seed: u8, raw_seed: i16) -> f64 {
    let bounded = (f64::from(raw_seed) / 8.0).clamp(-1_000.0, 1_000.0);
    let magnitude = i64::from(raw_seed).abs() as u64;

    match model {
        ModelKind::Gaussian => match mode_seed % 7 {
            0 => base,
            1 => bounded,
            2 => 0.0,
            3 => f64::from(raw_seed),
            4 => f64::NAN,
            5 => f64::INFINITY,
            _ => f64::NEG_INFINITY,
        },
        ModelKind::Poisson => match mode_seed % 7 {
            0 => (magnitude % 64) as f64,
            1 => base.abs().round(),
            2 => -((magnitude % 8) as f64),
            3 => bounded,
            4 => f64::NAN,
            5 => f64::INFINITY,
            _ => f64::NEG_INFINITY,
        },
        ModelKind::Bernoulli => match mode_seed % 7 {
            0 => {
                if raw_seed & 1 == 0 {
                    0.0
                } else {
                    1.0
                }
            }
            1 => base,
            2 => 2.0,
            3 => -1.0,
            4 => bounded,
            5 => f64::NAN,
            _ => f64::INFINITY,
        },
    }
}

fn build_timestamp(current_ts: &mut i64, mode_seed: u8, delta_seed: i16) -> Option<i64> {
    let step = i64::from(delta_seed % 16);
    match mode_seed % 5 {
        0 => None,
        1 => {
            *current_ts = current_ts.saturating_add(step.abs().saturating_add(1));
            Some(*current_ts)
        }
        2 => {
            *current_ts = current_ts.saturating_sub(step.abs().saturating_add(1));
            Some(*current_ts)
        }
        3 => Some(*current_ts),
        _ => Some(i64::from(delta_seed)),
    }
}

fn build_cost_eval_budget(seed: u8) -> Option<usize> {
    match seed % 8 {
        0 => Some(0),
        1 => Some(1),
        2 => Some(usize::from((seed % 4).saturating_add(2))),
        _ => None,
    }
}

fn choose_dims(seed: u8, extra_seed: u8) -> usize {
    match seed % 8 {
        0 => 0,
        1 => 2,
        2 => 3,
        _ => {
            if extra_seed & 1 == 0 {
                1
            } else {
                4
            }
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let mut cursor = common::ByteCursor::new(data);

    let (observation, model_kind) = build_observation(
        cursor.next_u8(),
        cursor.next_i16(),
        cursor.next_u8(),
        cursor.next_u8(),
        cursor.next_u8(),
    );

    let config = BocpdConfig {
        hazard: build_hazard(cursor.next_u8(), cursor.next_u8()),
        observation,
        max_run_length: common::bounded(cursor.next_u8(), 1, 256),
        log_prob_threshold: if cursor.next_u8() & 1 == 0 {
            None
        } else {
            Some(-(f64::from(cursor.next_u8() % 80) / 2.0))
        },
        alert_policy: build_alert_policy(
            cursor.next_u8(),
            cursor.next_u8(),
            cursor.next_u8(),
            cursor.next_u8(),
        ),
        late_data_policy: build_late_data_policy(
            cursor.next_u8(),
            cursor.next_u8(),
            cursor.next_u8(),
            cursor.next_u8(),
        ),
    };

    let Ok(mut detector) = BocpdDetector::new(config) else {
        return;
    };

    let payload_len = common::bounded(cursor.next_u8(), 0, 192).saturating_mul(8);
    let mut values = common::decode_f64_chunks(&cursor.take_padded(payload_len), 192);
    if values.is_empty() {
        values.push(0.0);
    }

    let mut value_idx = 0usize;
    let mut current_ts = i64::from(cursor.next_i16());
    let steps = common::bounded(cursor.next_u8(), 1, 96);

    for _ in 0..steps {
        let op_seed = cursor.next_u8();

        if op_seed % 9 == 0 {
            detector.reset();
            continue;
        }

        if op_seed % 11 == 0 {
            let snapshot = detector.save_state();
            detector.load_state(&snapshot);
            continue;
        }

        let budget_seed = cursor.next_u8();
        let constraints = Constraints {
            max_cost_evals: build_cost_eval_budget(budget_seed),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);

        let base = values[value_idx % values.len()];
        value_idx = value_idx.wrapping_add(1);
        let raw_seed = cursor.next_i16();
        let x = build_observation_value(model_kind, base, cursor.next_u8(), raw_seed);

        let dims = choose_dims(cursor.next_u8(), cursor.next_u8());

        let mut x_t = Vec::with_capacity(dims);
        for dim in 0..dims {
            if dim == 0 {
                x_t.push(x);
            } else {
                let noise = f64::from(cursor.next_i16()) / 16.0;
                x_t.push((x + noise).clamp(-1_000.0, 1_000.0));
            }
        }

        let t_ns = build_timestamp(&mut current_ts, cursor.next_u8(), cursor.next_i16());
        let _ = detector.update(x_t.as_slice(), t_ns, &ctx);
    }
});
