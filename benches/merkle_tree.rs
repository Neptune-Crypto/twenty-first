use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use twenty_first::util_types::merkle_tree::MerkleTree;
use twenty_first::util_types::merkle_tree::MerkleTree;

fn generate_input(length: usize) -> Vec<i128> {
    (0..length)
        .map(|_| rand::random::<i128>())
        .collect::<Vec<i128>>()
}

fn merkle_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("merkle_tree");
    // for size in [128usize, 256, 512, 1024].iter() {
    for size in [1024usize].iter() {
        let input = generate_input(*size);
        group.throughput(Throughput::Elements(*size as u64));
        group
            .bench_with_input(BenchmarkId::new("Library", size), &size, |b, _| {
                b.iter(|| MerkleTree::new_sha256_merkle_tree(input.clone()))
            })
            .sample_size(10);

        group
            .bench_with_input(BenchmarkId::new("Own", size), &size, |b, _| {
                b.iter(|| MerkleTree::from_vec(&input.clone()))
            })
            .sample_size(10);
    }
    group.finish();
}

criterion_group!(benches, merkle_benchmark);
criterion_main!(benches);
