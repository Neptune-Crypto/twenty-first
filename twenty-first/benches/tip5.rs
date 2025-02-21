use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use rand::random;
use rayon::prelude::*;
use twenty_first::math::other::random_elements;
use twenty_first::prelude::*;

fn bench_10(c: &mut Criterion) {
    let mut group = c.benchmark_group("tip5/hash_10");

    let size = 10;
    group.sample_size(100);

    let single_element: [BFieldElement; 10] = random();
    group.bench_function(BenchmarkId::new("Tip5 / Hash 10", size), |bencher| {
        bencher.iter(|| Tip5::hash_10(&single_element));
    });
}

fn bench_pair(c: &mut Criterion) {
    let mut group = c.benchmark_group("tip5/hash_pair");

    let left = random();
    let right = random();

    group.bench_function(BenchmarkId::new("Tip5 / Hash Pair", "pair"), |bencher| {
        bencher.iter(|| Tip5::hash_pair(left, right));
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
                .collect::<Vec<[BFieldElement; Digest::LEN]>>()
        });
    });
}

criterion_group!(benches, bench_10, bench_pair, bench_varlen, bench_parallel);
criterion_main!(benches);
