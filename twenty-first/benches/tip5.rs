use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use rand::RngCore;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::other::random_elements;
use twenty_first::shared_math::rescue_prime_regular::DIGEST_LENGTH;
use twenty_first::shared_math::tip5::Tip5;

fn bench_10(c: &mut Criterion) {
    let mut group = c.benchmark_group("tip5/hash_10");

    let size = 10;
    group.sample_size(100);

    let mut rng = rand::thread_rng();
    let single_element: [BFieldElement; 10] = (0..10)
        .into_iter()
        .map(|_| BFieldElement::new(rng.next_u64()))
        .collect_vec()
        .try_into()
        .unwrap();

    let tip5 = Tip5::new();

    group.bench_function(BenchmarkId::new("Tip5 / Hash 10", size), |bencher| {
        bencher.iter(|| tip5.hash_10(&single_element));
    });
}

fn bench_varlen(c: &mut Criterion) {
    let mut group = c.benchmark_group("tip5/hash_varlen");

    let size = 16_384;
    group.sample_size(50);
    let elements: Vec<BFieldElement> = random_elements(size);
    let tip5 = Tip5::new();

    group.bench_function(
        BenchmarkId::new("Tip5 / Hash Variable Length", size),
        |bencher| {
            bencher.iter(|| tip5.hash_varlen(&elements));
        },
    );
}

fn bench_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("tip5/parallel");

    let size = 65536;
    group.sample_size(50);
    let elements: Vec<[BFieldElement; 10]> = (0..size)
        .map(|_| random_elements(10).try_into().unwrap())
        .collect();
    let tip5 = Tip5::new();

    group.bench_function(BenchmarkId::new("Tip5 / Parallel Hash", size), |bencher| {
        bencher.iter(|| {
            elements
                .par_iter()
                .map(|e| tip5.hash_10(e))
                .collect::<Vec<[BFieldElement; DIGEST_LENGTH]>>()
        });
    });
}

criterion_group!(benches, bench_10, bench_varlen, bench_parallel);
criterion_main!(benches);
