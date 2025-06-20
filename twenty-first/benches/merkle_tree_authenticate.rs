use criterion::*;
use rand::rngs::StdRng;
use rand::*;
use twenty_first::prelude::Digest;
use twenty_first::prelude::Tip5;
use twenty_first::util_types::merkle_tree::*;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = auth_structure::<16>,
              auth_structure::<20>,
);

fn auth_structure<const TREE_HEIGHT: usize>(c: &mut Criterion) {
    let bench_id = BenchmarkId::new("height", TREE_HEIGHT);
    let mut sampler = MerkleTreeSampler::<TREE_HEIGHT>::default();
    let tree = sampler.tree();

    c.benchmark_group("gen_auth_structure")
        .bench_function(bench_id.clone(), |bencher| {
            bencher.iter_batched(
                || sampler.indices_to_open(),
                |indices| tree.authentication_structure(&indices),
                BatchSize::SmallInput,
            )
        });

    c.benchmark_group("verify_auth_structure")
        .bench_function(bench_id.clone(), |bencher| {
            bencher.iter_batched(
                || sampler.proof(&tree),
                |proof| proof.verify(tree.root()),
                BatchSize::SmallInput,
            );
        });

    let leafs = sampler.leaf_digests();
    c.benchmark_group("recompute_auth_structure_sequential")
        .sample_size(10)
        .bench_function(bench_id.clone(), |bencher| {
            bencher.iter_batched(
                || sampler.indices_to_open(),
                |idxs| MerkleTree::sequential_authentication_structure_from_leafs(&leafs, &idxs),
                BatchSize::SmallInput,
            )
        });

    c.benchmark_group("recompute_auth_structure_parallel")
        .bench_function(bench_id, |bencher| {
            bencher.iter_batched(
                || sampler.indices_to_open(),
                |idxs| MerkleTree::par_authentication_structure_from_leafs(&leafs, &idxs),
                BatchSize::SmallInput,
            )
        });
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MerkleTreeSampler<const HEIGHT: usize> {
    rng: StdRng,
    num_opened_indices: usize,
}

impl<const HEIGHT: usize> Default for MerkleTreeSampler<HEIGHT> {
    fn default() -> Self {
        Self {
            rng: StdRng::seed_from_u64(0),
            num_opened_indices: 40,
        }
    }
}

impl<const HEIGHT: usize> MerkleTreeSampler<HEIGHT> {
    const NUM_LEAFS: usize = 1 << HEIGHT;

    fn leaf_digests(&mut self) -> Vec<Digest> {
        (0..Self::NUM_LEAFS)
            .map(|_| self.rng.next_u64())
            .map(|leaf| Tip5::hash(&leaf))
            .collect()
    }

    fn tree(&mut self) -> MerkleTree {
        let leaf_digests = self.leaf_digests();
        MerkleTree::par_new(&leaf_digests).unwrap()
    }

    fn indices_to_open(&mut self) -> Vec<MerkleTreeLeafIndex> {
        (0..self.num_opened_indices)
            .map(|_| self.rng.random_range(0..Self::NUM_LEAFS))
            .collect()
    }

    fn proof(&mut self, tree: &MerkleTree) -> MerkleTreeInclusionProof {
        let leaf_indices = self.indices_to_open();
        tree.inclusion_proof_for_leaf_indices(&leaf_indices)
            .unwrap()
    }
}
