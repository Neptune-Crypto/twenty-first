use criterion::criterion_group;
use criterion::criterion_main;
use criterion::measurement::WallTime;
use criterion::BenchmarkGroup;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use std::hint::black_box;
use twenty_first::shared_math::other::random_elements;

fn unsigned_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul");

    let log2_of_sizes: Vec<usize> = vec![10, 100, 1000, 1_000_000];

    for log2_of_size in log2_of_sizes.iter() {
        u32_mul(
            &mut group,
            BenchmarkId::new("(u32,u32)->u64", log2_of_size),
            *log2_of_size,
        );
    }

    for log2_of_size in log2_of_sizes.iter() {
        u64_mul(
            &mut group,
            BenchmarkId::new("(u64,u64)->u128", log2_of_size),
            *log2_of_size,
        );
    }

    group.finish();
}

fn u32_mul(group: &mut BenchmarkGroup<WallTime>, bench_id: BenchmarkId, log2_of_size: usize) {
    let size: usize = 1 << log2_of_size;
    let xs: Vec<u32> = random_elements(size);

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| {
            for i in 0..(size - 1) {
                let _ = black_box(|| {
                    let _ = xs[i] as u64 * xs[i + 1] as u64;
                });
            }
        })
    });
    group.sample_size(10);
}

fn u64_mul(group: &mut BenchmarkGroup<WallTime>, bench_id: BenchmarkId, log2_of_size: usize) {
    let size: usize = 1 << log2_of_size;
    let xs: Vec<u64> = random_elements(size);

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| {
            for i in 0..(size - 1) {
                let _ = black_box(|| {
                    let _ = xs[i] as u128 * xs[i + 1] as u128;
                });
            }
        })
    });
    group.sample_size(10);
}

criterion_group!(benches, unsigned_mul);
criterion_main!(benches);
