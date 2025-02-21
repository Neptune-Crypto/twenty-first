use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use twenty_first::math::other::random_elements;
use twenty_first::prelude::*;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = coset_extrapolation<{ 1 << 18 }, { 1 << 8 }>,
              coset_extrapolation<{ 1 << 19 }, { 1 << 8 }>,
              coset_extrapolation<{ 1 << 20 }, { 1 << 8 }>,
              coset_extrapolation<{ 1 << 21 }, { 1 << 8 }>,
              coset_extrapolation<{ 1 << 22 }, { 1 << 8 }>,
              coset_extrapolation<{ 1 << 23 }, { 1 << 8 }>,
);

fn coset_extrapolation<const SIZE: usize, const NUM_POINTS: usize>(c: &mut Criterion) {
    let log2_of_size = SIZE.ilog2();
    let mut group = c.benchmark_group(format!(
        "Fast extrapolation of length-{SIZE} codeword in {NUM_POINTS} Points"
    ));

    let codeword = random_elements(SIZE);
    let offset = BFieldElement::new(7);
    let eval_points: Vec<BFieldElement> = random_elements(NUM_POINTS);

    let id = BenchmarkId::new("Fast Codeword Extrapolation", log2_of_size);
    group.bench_function(id, |b| {
        b.iter(|| Polynomial::<BFieldElement>::coset_extrapolate(offset, &codeword, &eval_points))
    });

    group.finish();
}
