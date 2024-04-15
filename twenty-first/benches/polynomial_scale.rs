use std::hint::black_box;

use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use rand::random;
use twenty_first::math::b_field_element::BFieldElement;
use twenty_first::math::other::random_elements;
use twenty_first::math::polynomial::Polynomial;
use twenty_first::math::x_field_element::XFieldElement;

fn various_scales(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul");

    let sizes = [1 << 10, 1 << 17];

    for size in sizes {
        bfes_w_bfe(
            &mut group,
            BenchmarkId::new("scale bfe-pol with bfe", size),
            size,
        );
    }

    for size in sizes {
        bfes_w_xfe(
            &mut group,
            BenchmarkId::new("scale bfe-pol with xfe", size),
            size,
        );
    }

    for size in sizes {
        xfes_w_bfe(
            &mut group,
            BenchmarkId::new("scale xfe-pol with bfe", size),
            size,
        );
    }

    for size in sizes {
        xfes_w_xfe(
            &mut group,
            BenchmarkId::new("scale xfe-pol with xfe", size),
            size,
        );
    }

    group.finish();
}

fn bfes_w_bfe(group: &mut BenchmarkGroup<WallTime>, bench_id: BenchmarkId, size: usize) {
    let bfe_pol: Polynomial<BFieldElement> = Polynomial::new(random_elements(size));
    let bfe: BFieldElement = random();

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| {
            let _ = black_box(bfe_pol.scale(bfe));
        })
    });
    group.sample_size(10);
}

fn bfes_w_xfe(group: &mut BenchmarkGroup<WallTime>, bench_id: BenchmarkId, size: usize) {
    let bfe_pol: Polynomial<BFieldElement> = Polynomial::new(random_elements(size));
    let xfe: XFieldElement = random();

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| {
            let _ = black_box(bfe_pol.scale(xfe));
        })
    });
    group.sample_size(10);
}

fn xfes_w_bfe(group: &mut BenchmarkGroup<WallTime>, bench_id: BenchmarkId, size: usize) {
    let xfe_pol: Polynomial<XFieldElement> = Polynomial::new(random_elements(size));
    let bfe: BFieldElement = random();

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| {
            let _ = black_box(xfe_pol.scale(bfe));
        })
    });
    group.sample_size(10);
}

fn xfes_w_xfe(group: &mut BenchmarkGroup<WallTime>, bench_id: BenchmarkId, size: usize) {
    let xfe_pol: Polynomial<XFieldElement> = Polynomial::new(random_elements(size));
    let xfe: XFieldElement = random();

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| {
            let _ = black_box(xfe_pol.scale(xfe));
        })
    });
    group.sample_size(10);
}

criterion_group!(benches, various_scales);
criterion_main!(benches);
