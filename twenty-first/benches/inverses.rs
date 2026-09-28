use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use itertools::Itertools;
use twenty_first::math::b_field_element::BFieldElement;
use twenty_first::math::other::random_elements;
use twenty_first::math::traits::FiniteField;

/// Run with `cargo criterion --bench inverse`
fn inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("Inverses");
    group.sample_size(10); // runs
    let count = 1024 * 1024; // count of elements to be inverted per run

    let rnd_elems: Vec<BFieldElement> = random_elements(count);

    let inverse = BenchmarkId::new("Inverse", 0);
    group.bench_function(inverse, |bencher| {
        bencher.iter(|| {
            rnd_elems.iter().map(|x| x.inverse()).collect_vec();
        });
    });

    let batch_inverse = BenchmarkId::new("Batch inverse", 0);
    group.bench_function(batch_inverse, |bencher| {
        bencher.iter(|| BFieldElement::batch_inversion(rnd_elems.clone()));
    });

    let par_batch_inverse = BenchmarkId::new("Parallel batch inverse", 0);
    group.bench_function(par_batch_inverse, |bencher| {
        bencher.iter(|| BFieldElement::par_batch_inversion(rnd_elems.clone()));
    });

    group.finish();
}

criterion_group!(benches, inverse);
criterion_main!(benches);
