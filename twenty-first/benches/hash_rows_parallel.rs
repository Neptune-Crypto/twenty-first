//! Hash the rows of a large, column-major table on all cores, comparing
//! [`Tip5::hash_rows_of_columns`] to transposing chunks of rows into a buffer
//! and hashing those with [`Tip5::hash_varlen_many`].
//!
//! The table is sized to exceed any CPU cache, so that this benchmark
//! captures the memory system as well as the permutation. It is therefore
//! slow, allocates on the order of a gigabyte, and is deliberately absent
//! from the CodSpeed workflow. Run it with
//!
//! ```sh
//! cargo bench --bench hash_rows_parallel
//! ```
//!
//! The number of rows can be set with the environment variable
//! `HASH_ROWS_BENCH_NUM_ROWS`; it defaults to 2^18.

use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use rayon::prelude::*;
use twenty_first::math::other::random_elements;
use twenty_first::math::x_field_element::as_flat_slice;
use twenty_first::prelude::*;

/// Mirrors the chunking in Triton VM's row hashing, where every chunk of
/// rows is one unit of parallel work.
const ROWS_PER_CHUNK: usize = 128;

const NUM_MAIN_COLUMNS: usize = 400;
const NUM_AUX_COLUMNS: usize = 88;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(20));
    targets = main_table, aux_table,
);

fn num_rows() -> usize {
    std::env::var("HASH_ROWS_BENCH_NUM_ROWS")
        .ok()
        .map(|rows| rows.parse().expect("number of rows must be an integer"))
        .unwrap_or(1 << 18)
}

fn main_table(c: &mut Criterion) {
    let num_rows = num_rows();
    let columns = (0..NUM_MAIN_COLUMNS)
        .map(|_| random_elements::<BFieldElement>(num_rows))
        .collect::<Vec<_>>();
    let columns = columns.iter().map(Vec::as_slice).collect::<Vec<_>>();

    let mut group = c.benchmark_group("main_table");
    let id = |approach| BenchmarkId::new(approach, format!("{NUM_MAIN_COLUMNS}x{num_rows}"));
    group.bench_function(id("transpose_then_hash_varlen_many"), |b| {
        b.iter(|| hash_chunks(&columns, transpose_then_hash_varlen_many))
    });
    group.bench_function(id("hash_rows_of_columns"), |b| {
        b.iter(|| hash_chunks(&columns, Tip5::hash_rows_of_columns))
    });
    group.finish();
}

fn aux_table(c: &mut Criterion) {
    let num_rows = num_rows();
    let columns = (0..NUM_AUX_COLUMNS)
        .map(|_| random_elements::<XFieldElement>(num_rows))
        .collect::<Vec<_>>();
    let columns = columns.iter().map(Vec::as_slice).collect::<Vec<_>>();

    let mut group = c.benchmark_group("aux_table");
    let id = |approach| BenchmarkId::new(approach, format!("{NUM_AUX_COLUMNS}x{num_rows}"));
    group.bench_function(id("transpose_then_hash_varlen_many"), |b| {
        b.iter(|| hash_chunks(&columns, transpose_then_hash_varlen_many_xfe))
    });
    group.bench_function(id("hash_rows_of_columns"), |b| {
        b.iter(|| hash_chunks(&columns, Tip5::hash_rows_of_columns))
    });
    group.finish();
}

/// Hash the rows of the table chunk by chunk, in parallel.
fn hash_chunks<T: Copy + Sync>(
    columns: &[&[T]],
    hash_rows: impl Fn(&[&[T]]) -> Vec<Digest> + Sync,
) -> Vec<Digest> {
    let num_rows = columns[0].len();
    (0..num_rows)
        .into_par_iter()
        .step_by(ROWS_PER_CHUNK)
        .flat_map_iter(|first_row| {
            let last_row = (first_row + ROWS_PER_CHUNK).min(num_rows);
            let chunk = columns
                .iter()
                .map(|column| &column[first_row..last_row])
                .collect::<Vec<_>>();
            hash_rows(&chunk)
        })
        .collect()
}

/// What Triton VM does: gather the rows into a row-major buffer, then hash.
fn transpose_then_hash_varlen_many(columns: &[&[BFieldElement]]) -> Vec<Digest> {
    let rows = transpose(columns);
    let rows = rows.chunks_exact(columns.len()).collect::<Vec<_>>();
    Tip5::hash_varlen_many(&rows)
}

fn transpose_then_hash_varlen_many_xfe(columns: &[&[XFieldElement]]) -> Vec<Digest> {
    let rows = transpose(columns);
    let rows = rows
        .chunks_exact(columns.len())
        .map(as_flat_slice)
        .collect::<Vec<_>>();
    Tip5::hash_varlen_many(&rows)
}

fn transpose<T: Copy>(columns: &[&[T]]) -> Vec<T> {
    let num_rows = columns[0].len();
    let mut rows = Vec::with_capacity(num_rows * columns.len());
    for row in 0..num_rows {
        rows.extend(columns.iter().map(|column| column[row]));
    }
    rows
}
