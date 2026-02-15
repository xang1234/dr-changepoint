// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{CachePolicy, DTypeView, MemoryLayout, MissingPolicy, TimeIndex, TimeSeriesView};
use cpd_costs::{CostL2Mean, CostModel, CostNIGMarginal, CostNormalMeanVar};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

const N: usize = 50_000;

fn generate_multivariate_values(n: usize, d: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(n * d);
    for t in 0..n {
        for dim in 0..d {
            let x = t as f64 + 1.0;
            let y = dim as f64 + 1.0;
            values.push((x * y) + (0.03 * x).sin() + (0.07 * y).cos());
        }
    }
    values
}

fn benchmark_multivariate_segment_scaling(c: &mut Criterion) {
    let l2_model = CostL2Mean::default();
    let normal_model = CostNormalMeanVar::default();
    let nig_model = CostNIGMarginal::default();

    let mut group = c.benchmark_group("cost_models_multivariate_segment");
    let start = N / 5;
    let end = (4 * N) / 5;

    for d in [1_usize, 4, 16] {
        let values = generate_multivariate_values(N, d);
        let view = TimeSeriesView::new(
            DTypeView::F64(&values),
            N,
            d,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .expect("benchmark view should be valid");

        let l2_cache = l2_model
            .precompute(&view, &CachePolicy::Full)
            .expect("L2 precompute should succeed");
        let normal_cache = normal_model
            .precompute(&view, &CachePolicy::Full)
            .expect("Normal precompute should succeed");
        let nig_cache = nig_model
            .precompute(&view, &CachePolicy::Full)
            .expect("NIG precompute should succeed");

        let l2_name = format!("l2_segment_cost_n5e4_d{d}");
        group.bench_function(l2_name, |b| {
            b.iter(|| l2_model.segment_cost(black_box(&l2_cache), black_box(start), black_box(end)))
        });

        let normal_name = format!("normal_segment_cost_n5e4_d{d}");
        group.bench_function(normal_name, |b| {
            b.iter(|| {
                normal_model.segment_cost(
                    black_box(&normal_cache),
                    black_box(start),
                    black_box(end),
                )
            })
        });

        let nig_name = format!("nig_segment_cost_n5e4_d{d}");
        group.bench_function(nig_name, |b| {
            b.iter(|| {
                nig_model.segment_cost(black_box(&nig_cache), black_box(start), black_box(end))
            })
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_multivariate_segment_scaling);
criterion_main!(benches);
