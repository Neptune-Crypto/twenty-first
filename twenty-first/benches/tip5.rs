use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use rand::random;
use rayon::prelude::*;
use twenty_first::math::other::random_elements;
use twenty_first::prelude::*;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(8));
    targets =
        hash_10,
        hash_10_x2,
        hash_10_many::<2>,
        hash_10_many::<200>,
        hash_pair,
        hash_varlen::<10>,
        hash_varlen::<16_384>,
        hash_parallel::<65_536>,
);

fn hash_10(c: &mut Criterion) {
    let input = random();
    c.bench_function("hash_10", |b| b.iter(|| Tip5::hash_10(&input)));
}

fn hash_10_x2(c: &mut Criterion) {
    let [left, right] = random();
    c.bench_function("hash_10_x2", |b| b.iter(|| Tip5::hash_10_x2(&left, &right)));
}

fn hash_10_many<const N: usize>(c: &mut Criterion) {
    let elements = random::<[_; N]>();
    c.benchmark_group("hash_10_many")
        .bench_function(BenchmarkId::new("N", N), |b| {
            b.iter(|| Tip5::hash_10_many(&elements))
        });
}

fn hash_pair(c: &mut Criterion) {
    let (left, right) = random();
    c.bench_function("hash_pair", |b| b.iter(|| Tip5::hash_pair(left, right)));
}

fn hash_varlen<const LEN: usize>(c: &mut Criterion) {
    let input = random_elements(LEN);
    c.benchmark_group("hash_varlen")
        .bench_function(BenchmarkId::new("len", LEN), |b| {
            b.iter(|| Tip5::hash_varlen(&input))
        });
}

fn hash_parallel<const LEN: usize>(c: &mut Criterion) {
    let input = (0..LEN).map(|_| random()).collect::<Vec<_>>();
    c.benchmark_group("hash_parallel")
        .bench_function(BenchmarkId::new("len", LEN), |b| {
            b.iter(|| input.par_iter().map(Tip5::hash_10).collect::<Vec<_>>());
        });
}
