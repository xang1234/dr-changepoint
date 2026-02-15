// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{
    Constraints, DTypeView, ExecutionContext, MemoryLayout, MissingPolicy, OfflineDetector,
    Penalty, Stopping, TimeIndex, TimeSeriesView,
};
use cpd_costs::CostL2Mean;
use cpd_offline::{Wbs, WbsConfig, WbsIntervalStrategy};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn make_view<'a>(values: &'a [f64], n: usize) -> TimeSeriesView<'a> {
    TimeSeriesView::new(
        DTypeView::F64(values),
        n,
        1,
        MemoryLayout::CContiguous,
        None,
        TimeIndex::None,
        MissingPolicy::Error,
    )
    .expect("benchmark view should be valid")
}

fn step_series(n: usize) -> Vec<f64> {
    let mut values = vec![0.0; n];
    for v in values.iter_mut().skip(n / 2) {
        *v = 5.0;
    }
    values
}

fn bench_wbs_l2(c: &mut Criterion, case_id: &str, n: usize, m: usize, penalty: f64) {
    let values = step_series(n);
    let view = make_view(&values, n);
    let constraints = Constraints {
        jump: 1,
        ..Constraints::default()
    };
    let ctx = ExecutionContext::new(&constraints);
    let detector = Wbs::new(
        CostL2Mean::default(),
        WbsConfig {
            stopping: Stopping::Penalized(Penalty::Manual(penalty)),
            params_per_segment: 2,
            num_intervals: Some(m),
            interval_strategy: WbsIntervalStrategy::Stratified,
            seed: 42,
            cancel_check_every: 1_000,
        },
    )
    .expect("detector config should be valid");

    c.bench_function(case_id, |b| {
        b.iter(|| {
            detector
                .detect(black_box(&view), black_box(&ctx))
                .expect("WBS L2 benchmark detect should succeed");
        })
    });
}

fn benchmark_wbs_l2_n1e5_m100(c: &mut Criterion) {
    const N: usize = 100_000;
    const M: usize = 100;
    bench_wbs_l2(c, "wbs_l2_n1e5_m100", N, M, 10.0);
}

criterion_group!(benches, benchmark_wbs_l2_n1e5_m100);
criterion_main!(benches);
