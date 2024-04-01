use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use rayon::prelude::*;

use twenty_first::math::ntt;
use twenty_first::math::other::random_elements;
use twenty_first::math::traits::PrimitiveRootOfUnity;
use twenty_first::prelude::*;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = evaluation<{ 1 << 10 }>,
              evaluation<{ 1 << 14 }>,
);

fn evaluation<const SIZE: usize>(c: &mut Criterion) {
    let log2_of_size = SIZE.ilog2();
    let mut group = c.benchmark_group(format!("Various Evaluations in 2^{log2_of_size} Points"));
    group.throughput(Throughput::Elements(u64::try_from(SIZE).unwrap()));

    let poly = Polynomial::new(random_elements(SIZE));
    let eval_points: Vec<BFieldElement> = random_elements(SIZE);

    let par_eval = || -> Vec<_> { eval_points.par_iter().map(|p| poly.evaluate(p)).collect() };
    group.bench_with_input(
        BenchmarkId::new("Parallel evaluate", log2_of_size),
        &log2_of_size,
        |b, _| b.iter(par_eval),
    );

    group.bench_with_input(
        BenchmarkId::new("Fast evaluate", log2_of_size),
        &log2_of_size,
        |b, _| b.iter(|| poly.fast_evaluate(&eval_points)),
    );

    // Note that NTT/iNTT can only handle inputs of length 2^k and the domain has to be a subgroup
    // of order 2^k whereas the other evaluation methods are generic.
    let primitive_root =
        BFieldElement::primitive_root_of_unity(u64::try_from(SIZE).unwrap()).unwrap();
    let mut coefficients = poly.coefficients;
    group.bench_with_input(
        BenchmarkId::new("Regular NTT", log2_of_size),
        &log2_of_size,
        |b, _| b.iter(|| ntt::intt(&mut coefficients, primitive_root, log2_of_size)),
    );

    group.finish();
}
