use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;
use itertools::Itertools;
use rand::rngs::StdRng;
use rand::Rng;
use rand::RngCore;
use rand::SeedableRng;

use twenty_first::shared_math::tip5::Tip5;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;
use twenty_first::util_types::merkle_tree::CpuParallel;
use twenty_first::util_types::merkle_tree::MerkleTree;
use twenty_first::util_types::merkle_tree_maker::MerkleTreeMaker;

fn merkle_tree_authenticate(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);

    let tree_height = 16;
    let num_leaves = 1 << tree_height;
    let leaves = (0..num_leaves).map(|_| rng.next_u64()).collect_vec();
    let leaf_digests = leaves.iter().map(Tip5::hash).collect_vec();
    let mt: MerkleTree<Tip5> = CpuParallel::from_digests(&leaf_digests);
    let mt_root = mt.get_root();

    let num_opened_indices = num_leaves / 4;
    let opened_indices = (0..num_opened_indices)
        .map(|_| rng.gen_range(0..num_leaves))
        .collect_vec();
    let authentication_structure = mt.get_authentication_structure(&opened_indices);
    let opened_leaves = opened_indices
        .iter()
        .map(|&i| leaf_digests[i])
        .collect_vec();

    let mut group = c.benchmark_group("merkle_tree_authenticate");
    group.bench_function(
        BenchmarkId::new("merkle_tree_authenticate", num_leaves),
        |bencher| {
            bencher.iter(|| {
                MerkleTree::<Tip5>::verify_authentication_structure_from_leaves(
                    mt_root,
                    tree_height,
                    &opened_indices,
                    &opened_leaves,
                    &authentication_structure,
                )
            });
        },
    );
}

criterion_group!(benches, merkle_tree_authenticate);
criterion_main!(benches);
