use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use twenty_first::math::digest::Digest;
use twenty_first::math::other::random_elements;
use twenty_first::util_types::merkle_tree::MerkleTree;

fn merkle_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("merkle_tree");

    let exponent = 16;
    let size = usize::pow(2, exponent);
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(50));

    let elements: Vec<Digest> = random_elements(size);

    group.bench_function(BenchmarkId::new("merkle_tree", size), |bencher| {
        bencher.iter(|| MerkleTree::par_new(&elements).unwrap());
    });
}

criterion_group!(benches, merkle_tree);
criterion_main!(benches);
