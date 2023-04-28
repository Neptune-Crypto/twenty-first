use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;
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

    let log_2_size = 16;
    let size = usize::pow(2, log_2_size);
    let leaves = (0..size).map(|_| rng.next_u64()).collect::<Vec<_>>();
    let leaf_digests = leaves.iter().map(|x| Tip5::hash(x)).collect::<Vec<_>>();
    let mt: MerkleTree<Tip5, _> = CpuParallel::from_digests(&leaf_digests);
    let mt_root = mt.get_root();

    let num_opened_indices = usize::pow(2, log_2_size - 2);
    let opened_indices = (0..num_opened_indices)
        .map(|_| rng.gen_range(0..size))
        .collect::<Vec<_>>();
    let authentication_structure = mt.get_authentication_structure(&opened_indices);
    let opened_leaves = opened_indices
        .iter()
        .map(|&i| leaf_digests[i])
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("merkle_tree_authenticate");
    group.bench_function(
        BenchmarkId::new("merkle_tree_authenticate", size),
        |bencher| {
            bencher.iter(|| {
                MerkleTree::<Tip5, CpuParallel>::verify_authentication_structure_from_leaves(
                    mt_root,
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
