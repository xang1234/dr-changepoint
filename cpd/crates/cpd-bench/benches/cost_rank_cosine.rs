// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{CachePolicy, DTypeView, MemoryLayout, MissingPolicy, TimeIndex, TimeSeriesView};
use cpd_costs::{CostCosine, CostModel, CostRank};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

const N: usize = 100_000;
const GROUP_NAME: &str = "cost_rank_cosine_segment";
const BENCH_DIMS: [usize; 3] = [1, 8, 16];

fn generate_values(n: usize, d: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(n * d);
    for t in 0..n {
        for dim in 0..d {
            let x = t as f64 + 1.0;
            let y = dim as f64 + 1.0;
            values.push((0.013 * x).sin() + (0.071 * y).cos() + 0.003 * x * y);
        }
    }
    values
}

fn make_c_contiguous_view<'a>(values: &'a [f64], n: usize, d: usize) -> TimeSeriesView<'a> {
    TimeSeriesView::new(
        DTypeView::F64(values),
        n,
        d,
        MemoryLayout::CContiguous,
        None,
        TimeIndex::None,
        MissingPolicy::Error,
    )
    .expect("benchmark view should be valid")
}

fn benchmark_rank_cosine_segment_scaling(c: &mut Criterion) {
    let rank_model = CostRank::default();
    let cosine_model = CostCosine::default();
    let mut group = c.benchmark_group(GROUP_NAME);
    let start = N / 5;
    let end = (4 * N) / 5;

    for d in BENCH_DIMS {
        let values = generate_values(N, d);
        let view = make_c_contiguous_view(values.as_slice(), N, d);

        let rank_cache = rank_model
            .precompute(&view, &CachePolicy::Full)
            .expect("rank precompute should succeed");
        let cosine_cache = cosine_model
            .precompute(&view, &CachePolicy::Full)
            .expect("cosine precompute should succeed");

        group.bench_function(format!("rank_segment_cost_n1e5_d{d}"), |b| {
            b.iter(|| {
                rank_model.segment_cost(black_box(&rank_cache), black_box(start), black_box(end))
            })
        });

        group.bench_function(format!("cosine_segment_cost_n1e5_d{d}"), |b| {
            b.iter(|| {
                cosine_model.segment_cost(
                    black_box(&cosine_cache),
                    black_box(start),
                    black_box(end),
                )
            })
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_rank_cosine_segment_scaling);
criterion_main!(benches);
