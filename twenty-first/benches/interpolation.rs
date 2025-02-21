use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use twenty_first::math::other::random_elements;
use twenty_first::prelude::*;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = interpolation<{ 1 << 10 }>,
              interpolation<{ 1 << 15 }>,
);

fn interpolation<const SIZE: usize>(c: &mut Criterion) {
    let log2_of_size = SIZE.ilog2();
    let mut group = c.benchmark_group(format!("Various Interpolations in 2^{log2_of_size} Points"));
    group.throughput(Throughput::Elements(u64::try_from(SIZE).unwrap()));

    let xs: Vec<BFieldElement> = random_elements(SIZE);
    let ys: Vec<BFieldElement> = random_elements(SIZE);

    if log2_of_size < 14 {
        let id = BenchmarkId::new("Lagrange", log2_of_size);
        let lagrange = || Polynomial::lagrange_interpolate(&xs, &ys);
        group.bench_function(id, |b| b.iter(lagrange));

        let id = BenchmarkId::new("Fast sequential", log2_of_size);
        group.bench_function(id, |b| b.iter(|| Polynomial::fast_interpolate(&xs, &ys)));

        let id = BenchmarkId::new("Dispatcher sequential", log2_of_size);
        group.bench_function(id, |b| b.iter(|| Polynomial::interpolate(&xs, &ys)));
    }

    let id = BenchmarkId::new("Fast parallel", log2_of_size);
    group.bench_function(id, |b| {
        b.iter(|| Polynomial::par_fast_interpolate(&xs, &ys))
    });

    let id = BenchmarkId::new("Dispatcher parallel", log2_of_size);
    group.bench_function(id, |b| b.iter(|| Polynomial::par_interpolate(&xs, &ys)));

    group.finish();
}
