use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use rayon::prelude::*;

use twenty_first::math::other::random_elements;
use twenty_first::prelude::*;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(50);
    targets = evaluation<{ 1 << 10 }>,
              evaluation<{ 1 << 14 }>,
              evaluation<{ 1 << 16 }>,
);

fn evaluation<const SIZE: usize>(c: &mut Criterion) {
    let log2_of_size = SIZE.ilog2();
    let mut group = c.benchmark_group(format!("Various Evaluations in 2^{log2_of_size} Points"));
    group.throughput(Throughput::Elements(u64::try_from(SIZE).unwrap()));

    let poly = Polynomial::new(random_elements(SIZE));
    let eval_points: Vec<BFieldElement> = random_elements(SIZE);

    let id = BenchmarkId::new("Parallel", log2_of_size);
    let par_eval = || -> Vec<_> { eval_points.par_iter().map(|&p| poly.evaluate(p)).collect() };
    group.bench_function(id, |b| b.iter(par_eval));

    // `vector_batch_evaluate` exists, but is super slow. Put it here if you plan to run benchmarks
    // during a coffee break.

    let id = BenchmarkId::new("Fast", log2_of_size);
    group.bench_function(id, |b| b.iter(|| poly.fast_evaluate(&eval_points)));

    let id = BenchmarkId::new("Faster of the two", log2_of_size);
    group.bench_function(id, |b| b.iter(|| poly.batch_evaluate(&eval_points)));

    group.finish();
}
