use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use twenty_first::math::b_field_element::BFieldElement;
use twenty_first::math::other::random_elements;
use twenty_first::math::polynomial::Polynomial;
use twenty_first::math::x_field_element::XFieldElement;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = coset_functions<10>, coset_functions<17>,
);

fn coset_functions<const LOG2_SIZE: usize>(c: &mut Criterion) {
    let size = 1 << LOG2_SIZE;
    let offset = BFieldElement::generator();

    let mut group = c.benchmark_group(format!("polynomial coset of degree 2^{LOG2_SIZE}"));
    group.throughput(Throughput::Elements(size));

    let size = 1 << LOG2_SIZE; // different type
    let bfe_poly = Polynomial::<BFieldElement>::new(random_elements(size));
    group.bench_function(BenchmarkId::new("coset-evaluate bfe-pol", size), |b| {
        b.iter(|| bfe_poly.fast_coset_evaluate(offset, size))
    });

    let xfe_poly = Polynomial::<XFieldElement>::new(random_elements(size));
    group.bench_function(BenchmarkId::new("coset-evaluate xfe-pol", size), |b| {
        b.iter(|| xfe_poly.fast_coset_evaluate(offset, size))
    });

    let bfe_values: Vec<BFieldElement> = random_elements(size);
    group.bench_function(BenchmarkId::new("coset-interpolate bfe-pol", size), |b| {
        b.iter(|| Polynomial::fast_coset_interpolate(offset, &bfe_values))
    });

    let xfe_values: Vec<XFieldElement> = random_elements(size);
    group.bench_function(BenchmarkId::new("coset-interpolate xfe-pol", size), |b| {
        b.iter(|| Polynomial::fast_coset_interpolate(offset, &xfe_values))
    });

    group.finish();
}
