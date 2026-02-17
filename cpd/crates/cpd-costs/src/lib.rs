// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

pub mod ar;
pub mod bernoulli;
pub mod cosine;
pub mod l1;
pub mod l2;
pub mod linear;
pub mod model;
pub mod nig;
pub mod normal;
pub mod poisson;
pub mod rank;

pub use ar::{ARCache, CostAR};
pub use bernoulli::{BernoulliCache, CostBernoulli};
pub use cosine::{CosineCache, CostCosine};
pub use cpd_core::MissingSupport;
pub use l1::{CostL1Median, L1MedianCache};
pub use l2::{CostL2Mean, L2Cache};
pub use linear::{CostLinear, LinearCache};
pub use model::{CachedCost, CostModel};
pub use nig::{CostNIGMarginal, NIGCache, NIGPrior};
pub use normal::{CostNormalFullCov, CostNormalMeanVar, NormalCache, NormalFullCovCache};
pub use poisson::{CostPoissonRate, PoissonCache};
pub use rank::{CostRank, RankCache};

/// Built-in cost model namespace placeholder.
pub fn crate_name() -> &'static str {
    let _ = cpd_core::crate_name();
    "cpd-costs"
}
