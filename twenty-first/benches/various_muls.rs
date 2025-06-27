use std::hint::black_box;

use criterion::BenchmarkGroup;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::measurement::WallTime;
use twenty_first::math::b_field_element::BFieldElement;
use twenty_first::math::other::random_elements;
use twenty_first::math::x_field_element::XFieldElement;

fn unsigned_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul");

    let sizes = [10, 100, 1000, 1_000_000];

    for size in sizes {
        nop(&mut group, BenchmarkId::new("nop", size), size);
    }

    for size in sizes {
        u32_mul(&mut group, BenchmarkId::new("(u32,u32)->u64", size), size);
    }

    for size in sizes {
        u64_mul(&mut group, BenchmarkId::new("(u64,u64)->u128", size), size);
    }

    for size in sizes {
        bfe_bfe_mul(&mut group, BenchmarkId::new("(BFE,BFE)->BFE", size), size);
    }

    for size in sizes {
        xfe_xfe_mul(&mut group, BenchmarkId::new("(XFE,XFE)->XFE", size), size);
    }

    for size in sizes {
        xfe_bfe_mul(&mut group, BenchmarkId::new("(XFE,BFE)->XFE", size), size);
    }

    group.finish();
}

fn nop(group: &mut BenchmarkGroup<WallTime>, bench_id: BenchmarkId, size: usize) {
    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| {
            for _ in 0..size {
                let () = black_box(());
            }
        })
    });
    group.sample_size(10);
}

fn u32_mul(group: &mut BenchmarkGroup<WallTime>, bench_id: BenchmarkId, size: usize) {
    let xs: Vec<u32> = random_elements(size);

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| {
            for i in 0..(size - 1) {
                let _ = black_box(xs[i] as u64 * xs[i + 1] as u64);
            }
        })
    });
    group.sample_size(10);
}

fn u64_mul(group: &mut BenchmarkGroup<WallTime>, bench_id: BenchmarkId, size: usize) {
    let xs: Vec<u64> = random_elements(size);

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| {
            for i in 0..(size - 1) {
                let _ = black_box(xs[i] as u128 * xs[i + 1] as u128);
            }
        })
    });
    group.sample_size(10);
}

fn bfe_bfe_mul(group: &mut BenchmarkGroup<WallTime>, bench_id: BenchmarkId, size: usize) {
    let xs: Vec<BFieldElement> = random_elements(size);

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| {
            for i in 0..(size - 1) {
                let _ = black_box(xs[i] * xs[i + 1]);
            }
        })
    });
    group.sample_size(10);
}

fn xfe_xfe_mul(group: &mut BenchmarkGroup<WallTime>, bench_id: BenchmarkId, size: usize) {
    let xs: Vec<XFieldElement> = random_elements(size);

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| {
            for i in 0..(size - 1) {
                let _ = black_box(xs[i] * xs[i + 1]);
            }
        })
    });
    group.sample_size(10);
}

fn xfe_bfe_mul(group: &mut BenchmarkGroup<WallTime>, bench_id: BenchmarkId, size: usize) {
    let xs: Vec<XFieldElement> = random_elements(size);
    let bs: Vec<BFieldElement> = random_elements(size);

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| {
            for i in 0..(size - 1) {
                let _ = black_box(xs[i] * bs[i]);
            }
        })
    });
    group.sample_size(10);
}

criterion_group!(benches, unsigned_mul);
criterion_main!(benches);
