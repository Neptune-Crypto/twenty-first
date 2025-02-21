use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use rand::random;
use twenty_first::math::other::random_elements;
use twenty_first::prelude::*;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = poly_scalar_mul<1>,
              poly_scalar_mul<7>,
              poly_scalar_mul<13>,
);

fn poly_scalar_mul<const LOG2_DEG: usize>(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!(
        "Multiplication of Polynomial of Degree 2^{LOG2_DEG} with a Scalar"
    ));

    let mut poly = Polynomial::<BFieldElement>::new(random_elements((1 << LOG2_DEG) + 1));
    let scalar = random();

    let id = BenchmarkId::new("Mut", LOG2_DEG);
    group.bench_function(id, |b| b.iter(|| poly.scalar_mul_mut(scalar)));

    let id = BenchmarkId::new("Immut", LOG2_DEG);
    group.bench_function(id, |b| b.iter(|| poly.scalar_mul(scalar)));

    group.finish();
}
