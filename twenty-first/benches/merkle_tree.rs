use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use twenty_first::shared_math::other::random_elements;
use twenty_first::shared_math::rescue_prime_digest::Digest;
use twenty_first::shared_math::rescue_prime_regular::RescuePrimeRegular;
use twenty_first::util_types::merkle_tree::{CpuParallel, MerkleTree};
use twenty_first::util_types::merkle_tree_maker::MerkleTreeMaker;

fn merkle_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("merkle_tree");

    let exponent = 16;
    let size = usize::pow(2, exponent);
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(50));

    let elements: Vec<Digest> = random_elements(size);

    group.bench_function(BenchmarkId::new("merkle_tree", size), |bencher| {
        bencher.iter(|| -> MerkleTree<RescuePrimeRegular, CpuParallel> {
            CpuParallel::from_digests(&elements[..])
        });
    });
}

criterion_group!(benches, merkle_tree);
criterion_main!(benches);
