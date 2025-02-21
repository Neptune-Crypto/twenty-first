use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use twenty_first::math::other::random_elements;
use twenty_first::prelude::*;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = poly_mul<7>,
              poly_mul<8>,
              poly_mul<9>,
);

fn poly_mul<const LOG2_SIZE: usize>(c: &mut Criterion) {
    let product_degree = LOG2_SIZE + 1;
    let mut group = c.benchmark_group(format!(
        "Multiplication of Polynomials of Degree 2^{LOG2_SIZE} (Product Degree: 2^{product_degree})"
    ));

    let new_poly = || Polynomial::<BFieldElement>::new(random_elements((1 << LOG2_SIZE) + 1));
    let poly_0 = new_poly();
    let poly_1 = new_poly();

    let id = BenchmarkId::new("Naïve", product_degree);
    group.bench_function(id, |b| b.iter(|| poly_0.naive_multiply(&poly_1)));

    let id = BenchmarkId::new("Fast", product_degree);
    group.bench_function(id, |b| b.iter(|| poly_0.fast_multiply(&poly_1)));

    let id = BenchmarkId::new("Faster of the two", product_degree);
    group.bench_function(id, |b| b.iter(|| poly_0.multiply(&poly_1)));

    group.finish();
}
