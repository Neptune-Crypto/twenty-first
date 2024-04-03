use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

use twenty_first::math::other::random_elements;
use twenty_first::prelude::*;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = poly_div<12>,
              poly_div<13>,
              poly_div<14>,
              poly_div<15>,
              poly_div<16>,
              poly_div<17>,
              poly_div<18>,
);

fn poly_div<const LOG2_SIZE: usize>(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("Division of Polynomials of Degree 2^{LOG2_SIZE}"));

    let new_poly = || Polynomial::<BFieldElement>::new(random_elements((1 << LOG2_SIZE) + 1));
    let poly_0 = new_poly();
    let poly_1 = new_poly();

    let id = BenchmarkId::new("Naïve", LOG2_SIZE);
    group.bench_function(id, |b| b.iter(|| poly_0.clone() / poly_1.clone()));

    let id = BenchmarkId::new("Fast", LOG2_SIZE);
    group.bench_function(id, |b| b.iter(|| poly_0.fast_divide(&poly_1)));

    let id = BenchmarkId::new("Faster of the two", LOG2_SIZE);
    group.bench_function(id, |b| b.iter(|| poly_0.divide(&poly_1)));

    group.finish();
}
