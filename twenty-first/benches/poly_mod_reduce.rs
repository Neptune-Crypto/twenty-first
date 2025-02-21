use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use twenty_first::math::other::random_elements;
use twenty_first::prelude::*;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(50);
    targets = poly_mod_reduce<{ 1 << 20 }, {1<<4}>,
              poly_mod_reduce<{ 1 << 20 }, {1<<5}>,
              poly_mod_reduce<{ 1 << 20 }, {1<<6}>,
              poly_mod_reduce<{ 1 << 20 }, {1<<7}>,
              poly_mod_reduce<{ 1 << 20 }, {1<<8}>,
);

fn poly_mod_reduce<const SIZE_LHS: usize, const SIZE_RHS: usize>(c: &mut Criterion) {
    let log2_of_size = SIZE_LHS.ilog2();
    let mut group = c.benchmark_group(format!(
        "Modular reduction of degree {} by degree {}",
        SIZE_LHS - 1,
        SIZE_RHS - 1
    ));
    let lhs = Polynomial::new(random_elements::<BFieldElement>(SIZE_LHS));
    let rhs = Polynomial::new(random_elements::<BFieldElement>(SIZE_RHS));

    let id = BenchmarkId::new("long division", log2_of_size);
    group.bench_function(id, |b| b.iter(|| lhs.clone() % rhs.clone()));

    let id = BenchmarkId::new("fast reduce", log2_of_size);
    group.bench_function(id, |b| b.iter(|| lhs.fast_reduce(&rhs)));

    group.finish();
}
