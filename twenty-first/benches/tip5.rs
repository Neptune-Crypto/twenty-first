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
        hash_pair,
        hash_varlen::<10>,
        hash_varlen::<16_384>,
        hash_parallel::<65_536>,
        absorb_many::<64>,
        absorb_individually::<64>,
        hash_rows_of_bfe_columns::<400, 512>,
        hash_rows_of_xfe_columns::<88, 512>,
);

fn hash_10(c: &mut Criterion) {
    let input = random();
    c.bench_function("hash_10", |b| b.iter(|| Tip5::hash_10(&input)));
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

fn absorb_many<const NUM_SPONGES: usize>(c: &mut Criterion) {
    let mut sponges = vec![Tip5::init(); NUM_SPONGES];
    let inputs = (0..NUM_SPONGES).map(|_| random()).collect::<Vec<_>>();
    c.benchmark_group("absorb_many")
        .bench_function(BenchmarkId::new("sponges", NUM_SPONGES), |b| {
            b.iter(|| Tip5::absorb_many(&mut sponges, &inputs))
        });
}

/// The baseline for [`absorb_many`]: the same work, one sponge at a time.
fn absorb_individually<const NUM_SPONGES: usize>(c: &mut Criterion) {
    let mut sponges = vec![Tip5::init(); NUM_SPONGES];
    let inputs = (0..NUM_SPONGES).map(|_| random()).collect::<Vec<_>>();
    c.benchmark_group("absorb_individually").bench_function(
        BenchmarkId::new("sponges", NUM_SPONGES),
        |b| {
            b.iter(|| {
                for (sponge, &input) in sponges.iter_mut().zip(&inputs) {
                    sponge.absorb(input);
                }
            })
        },
    );
}

fn hash_rows_of_bfe_columns<const NUM_COLUMNS: usize, const NUM_ROWS: usize>(c: &mut Criterion) {
    let columns = (0..NUM_COLUMNS)
        .map(|_| random_elements::<BFieldElement>(NUM_ROWS))
        .collect::<Vec<_>>();
    let columns = columns.iter().map(Vec::as_slice).collect::<Vec<_>>();
    c.benchmark_group("hash_rows_of_bfe_columns")
        .bench_function(
            BenchmarkId::new("columns_x_rows", format!("{NUM_COLUMNS}x{NUM_ROWS}")),
            |b| b.iter(|| Tip5::hash_rows_of_columns(&columns)),
        );
}

fn hash_rows_of_xfe_columns<const NUM_COLUMNS: usize, const NUM_ROWS: usize>(c: &mut Criterion) {
    let columns = (0..NUM_COLUMNS)
        .map(|_| random_elements::<XFieldElement>(NUM_ROWS))
        .collect::<Vec<_>>();
    let columns = columns.iter().map(Vec::as_slice).collect::<Vec<_>>();
    c.benchmark_group("hash_rows_of_xfe_columns")
        .bench_function(
            BenchmarkId::new("columns_x_rows", format!("{NUM_COLUMNS}x{NUM_ROWS}")),
            |b| b.iter(|| Tip5::hash_rows_of_columns(&columns)),
        );
}
