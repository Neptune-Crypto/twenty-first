use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use rand::RngCore;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::other::random_elements;
use twenty_first::shared_math::rescue_prime_regular::{RescuePrimeRegular, DIGEST_LENGTH};

fn bench_10(c: &mut Criterion) {
    let mut group = c.benchmark_group("rescue_prime_regular/hash_10");

    let size = 10;
    group.sample_size(100);

    let mut rng = rand::thread_rng();
    let single_element: [BFieldElement; 10] = (0..10)
        .into_iter()
        .map(|_| BFieldElement::new(rng.next_u64()))
        .collect_vec()
        .try_into()
        .unwrap();

    group.bench_function(
        BenchmarkId::new("RescuePrimeRegular / Hash 10", size),
        |bencher| {
            bencher.iter(|| RescuePrimeRegular::hash_10(&single_element));
        },
    );
}

fn bench_varlen(c: &mut Criterion) {
    let mut group = c.benchmark_group("rescue_prime_regular/hash_varlen");

    let size = 16_384;
    group.sample_size(50);
    let elements: Vec<BFieldElement> = random_elements(size);

    group.bench_function(
        BenchmarkId::new("RescuePrimeRegular / Hash Variable Length", size),
        |bencher| {
            bencher.iter(|| RescuePrimeRegular::hash_varlen(&elements));
        },
    );
}

fn bench_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("rescue_prime_regular/parallel");

    let size = 65536;
    group.sample_size(50);
    let elements: Vec<[BFieldElement; 10]> = (0..size)
        .map(|_| random_elements(10).try_into().unwrap())
        .collect();

    group.bench_function(
        BenchmarkId::new("RescuePrimeRegular / Parallel Hash", size),
        |bencher| {
            bencher.iter(|| {
                elements
                    .par_iter()
                    .map(RescuePrimeRegular::hash_10)
                    .collect::<Vec<[BFieldElement; DIGEST_LENGTH]>>()
            });
        },
    );
}

criterion_group!(benches, bench_10, bench_varlen, bench_parallel);
criterion_main!(benches);
