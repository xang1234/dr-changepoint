// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

pub mod bernoulli;
pub mod l2;
pub mod model;
pub mod nig;
pub mod normal;
pub mod poisson;

pub use bernoulli::{BernoulliCache, CostBernoulli};
pub use cpd_core::MissingSupport;
pub use l2::{CostL2Mean, L2Cache};
pub use model::{CachedCost, CostModel};
pub use nig::{CostNIGMarginal, NIGCache, NIGPrior};
pub use normal::{CostNormalMeanVar, NormalCache};
pub use poisson::{CostPoissonRate, PoissonCache};

/// Built-in cost model namespace placeholder.
pub fn crate_name() -> &'static str {
    let _ = cpd_core::crate_name();
    "cpd-costs"
}
