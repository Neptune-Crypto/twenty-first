use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{thread_rng, Rng};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::digest::DIGEST_LENGTH;
use twenty_first::shared_math::other::random_elements;
use twenty_first::shared_math::tip5::Tip5;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;

fn bench_10(c: &mut Criterion) {
    let mut group = c.benchmark_group("tip5/hash_10");

    let size = 10;
    group.sample_size(100);

    let single_element: [BFieldElement; 10] = thread_rng().gen();
    group.bench_function(BenchmarkId::new("Tip5 / Hash 10", size), |bencher| {
        bencher.iter(|| Tip5::hash_10(&single_element));
    });
}

fn bench_pair(c: &mut Criterion) {
    let mut group = c.benchmark_group("tip5/hash_pair");

    let left = thread_rng().gen();
    let right = thread_rng().gen();

    group.bench_function(BenchmarkId::new("Tip5 / Hash Pair", "pair"), |bencher| {
        bencher.iter(|| Tip5::hash_pair(&left, &right));
    });
}

fn bench_varlen(c: &mut Criterion) {
    let mut group = c.benchmark_group("tip5/hash_varlen");

    let size = 16_384;
    group.sample_size(50);
    let elements: Vec<BFieldElement> = random_elements(size);

    group.bench_function(
        BenchmarkId::new("Tip5 / Hash Variable Length", size),
        |bencher| {
            bencher.iter(|| Tip5::hash_varlen(&elements));
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

    group.bench_function(BenchmarkId::new("Tip5 / Parallel Hash", size), |bencher| {
        bencher.iter(|| {
            elements
                .par_iter()
                .map(Tip5::hash_10)
                .collect::<Vec<[BFieldElement; DIGEST_LENGTH]>>()
        });
    });
}

criterion_group!(benches, bench_10, bench_pair, bench_varlen, bench_parallel);
criterion_main!(benches);
