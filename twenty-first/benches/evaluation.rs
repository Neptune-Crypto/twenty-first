use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

use twenty_first::math::other::random_elements;
use twenty_first::prelude::*;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = evaluation<{ 1 << 18 }, { 1 << 6 }>,
              evaluation<{ 1 << 18 }, { 1 << 7 }>,
              evaluation<{ 1 << 18 }, { 1 << 8 }>,
              evaluation<{ 1 << 19 }, { 1 << 6 }>,
              evaluation<{ 1 << 19 }, { 1 << 7 }>,
              evaluation<{ 1 << 19 }, { 1 << 8 }>,
              evaluation<{ 1 << 20 }, { 1 << 6 }>,
              evaluation<{ 1 << 20 }, { 1 << 7 }>,
              evaluation<{ 1 << 20 }, { 1 << 8 }>,
);

fn evaluation<const SIZE: usize, const NUM_POINTS: usize>(c: &mut Criterion) {
    let log2_of_size = SIZE.ilog2();
    let mut group = c.benchmark_group(format!(
        "Evaluation of degree-{} polynomial in {NUM_POINTS} Points",
        SIZE - 1
    ));
    group.sample_size(10);

    let poly = Polynomial::new(random_elements(SIZE));
    let eval_points: Vec<BFieldElement> = random_elements(NUM_POINTS);

    let id = BenchmarkId::new("Iterative", log2_of_size);
    group.bench_function(id, |b| {
        b.iter(|| poly.iterative_batch_evaluate(&eval_points))
    });

    // `vector_batch_evaluate` exists, but is super slow. Put it here if you plan to run benchmarks
    // during a coffee break.

    let id = BenchmarkId::new("Divide-and-Conquer", log2_of_size);
    group.bench_function(id, |b| {
        b.iter(|| poly.divide_and_conquer_batch_evaluate(&eval_points))
    });

    let id = BenchmarkId::new("Dispatcher", log2_of_size);
    group.bench_function(id, |b| b.iter(|| poly.batch_evaluate(&eval_points)));

    group.finish();
}
