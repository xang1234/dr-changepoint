// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{
    Constraints, DTypeView, ExecutionContext, MemoryLayout, MissingPolicy, OfflineDetector,
    Penalty, Stopping, TimeIndex, TimeSeriesView,
};
use cpd_costs::CostL2Mean;
use cpd_offline::{Fpop, FpopConfig, Pelt, PeltConfig};
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

fn favorable_series(n: usize) -> Vec<f64> {
    let mut values = vec![0.0; n];
    let regime = n / 4;
    for v in values.iter_mut().skip(regime).take(regime) {
        *v = 6.0;
    }
    for v in values.iter_mut().skip(regime * 2).take(regime) {
        *v = -4.0;
    }
    values
}

fn bench_pelt_fpop_pair(
    c: &mut Criterion,
    case_suffix: &str,
    n: usize,
    jump: usize,
    min_segment_len: usize,
    penalty: f64,
) {
    let values = favorable_series(n);
    let view = make_view(&values, n);
    let constraints = Constraints {
        min_segment_len,
        jump,
        ..Constraints::default()
    };
    let ctx = ExecutionContext::new(&constraints);

    let pelt = Pelt::new(
        CostL2Mean::default(),
        PeltConfig {
            stopping: Stopping::Penalized(Penalty::Manual(penalty)),
            params_per_segment: 2,
            cancel_check_every: 1_000,
        },
    )
    .expect("PELT config should be valid");

    let fpop = Fpop::new(
        CostL2Mean::default(),
        FpopConfig {
            stopping: Stopping::Penalized(Penalty::Manual(penalty)),
            params_per_segment: 2,
            cancel_check_every: 1_000,
        },
    )
    .expect("FPOP config should be valid");

    c.bench_function(&format!("pelt_l2_{case_suffix}"), |b| {
        b.iter(|| {
            pelt.detect(black_box(&view), black_box(&ctx))
                .expect("PELT benchmark detect should succeed");
        })
    });

    c.bench_function(&format!("fpop_l2_{case_suffix}"), |b| {
        b.iter(|| {
            fpop.detect(black_box(&view), black_box(&ctx))
                .expect("FPOP benchmark detect should succeed");
        })
    });
}

fn benchmark_fpop_vs_pelt_n1e4(c: &mut Criterion) {
    bench_pelt_fpop_pair(c, "n1e4_jump1", 10_000, 1, 2, 12.0);
}

fn benchmark_fpop_vs_pelt_n1e5(c: &mut Criterion) {
    bench_pelt_fpop_pair(c, "n1e5_jump2", 100_000, 2, 8, 20.0);
}

criterion_group!(
    benches,
    benchmark_fpop_vs_pelt_n1e4,
    benchmark_fpop_vs_pelt_n1e5
);
criterion_main!(benches);
