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
    targets = poly_scale<5>,
              poly_scale<10>,
              poly_scale<15>,
              poly_scale<20>,
);

fn poly_scale<const LOG2_SIZE: usize>(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("Scale Polynomials of Degree 2^{LOG2_SIZE}"));

    let bfe_poly = Polynomial::<BFieldElement>::new(random_elements((1 << LOG2_SIZE) + 1));
    let xfe_poly = Polynomial::<XFieldElement>::new(random_elements((1 << LOG2_SIZE) + 1));
    let scalar = random();

    group.bench_function(BenchmarkId::new("bfe poly", LOG2_SIZE), |b| {
        b.iter(|| bfe_poly.scale(scalar))
    });
    group.bench_function(BenchmarkId::new("xfe poly", LOG2_SIZE), |b| {
        b.iter(|| xfe_poly.scale(scalar))
    });

    group.finish();
}
