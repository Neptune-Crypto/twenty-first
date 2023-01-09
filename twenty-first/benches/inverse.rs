use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::other::random_elements;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::timing_reporter::TimingReporter;

/// Run with `cargo criterion --bench inverse`
fn inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("Inverses");
    group.sample_size(10); // runs
    let count = 1024 * 1024; // count of elements to be inversed per run

    let mut timer = TimingReporter::start();
    let rnd_elems: Vec<BFieldElement> = random_elements(count);

    timer.elapsed(&format!("Generate {} BFieldElements.", count));

    let inverse = BenchmarkId::new("Inverse", 0);
    group.bench_function(inverse, |bencher| {
        bencher.iter(|| {
            rnd_elems.iter().map(|x| x.inverse()).collect_vec();
        });
    });
    timer.elapsed("Winterfell");

    group.finish();
    let report = timer.finish();

    println!("{}", report);
}

criterion_group!(benches, inverse);
criterion_main!(benches);
