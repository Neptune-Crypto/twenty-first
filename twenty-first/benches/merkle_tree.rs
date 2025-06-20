use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use twenty_first::math::other::random_elements;
use twenty_first::prelude::Digest;
use twenty_first::util_types::merkle_tree::MerkleTree;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = merkle_tree::<16>,
              merkle_tree::<20>,
);

fn merkle_tree<const TREE_HEIGHT: usize>(c: &mut Criterion) {
    let bench_id = BenchmarkId::new("height", TREE_HEIGHT);
    let leafs: Vec<Digest> = random_elements(1 << TREE_HEIGHT);

    c.benchmark_group("merkle_tree_parallel")
        .bench_function(bench_id.clone(), |bencher| {
            bencher.iter(|| MerkleTree::par_new(&leafs).unwrap());
        });

    c.benchmark_group("merkle_tree_sequential")
        .bench_function(bench_id.clone(), |bencher| {
            bencher.iter(|| MerkleTree::sequential_new(&leafs).unwrap());
        });

    c.benchmark_group("merkle_root_frugal_parallel")
        .bench_function(bench_id.clone(), |bencher| {
            bencher.iter(|| MerkleTree::par_frugal_root(&leafs).unwrap());
        });

    c.benchmark_group("merkle_root_frugal_sequential")
        .bench_function(bench_id, |bencher| {
            bencher.iter(|| MerkleTree::sequential_frugal_root(&leafs).unwrap());
        });
}
