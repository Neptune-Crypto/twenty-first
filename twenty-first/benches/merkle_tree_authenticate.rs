use criterion::*;
use itertools::Itertools;
use rand::rngs::StdRng;
use rand::*;

use twenty_first::shared_math::digest::Digest;
use twenty_first::shared_math::tip5::Tip5;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;
use twenty_first::util_types::merkle_tree::*;
use twenty_first::util_types::merkle_tree_maker::MerkleTreeMaker;

criterion_main!(merkle_tree_authenticate);
criterion_group!(
    merkle_tree_authenticate,
    gen_auth_structure,
    verify_auth_structure
);

fn gen_auth_structure(c: &mut Criterion) {
    let mut sampler = MerkleTreeSampler::default();
    let tree = sampler.tree();

    c.bench_function("gen_auth_structure", |bencher| {
        bencher.iter_batched(
            || sampler.indices_to_open(),
            |indices| tree.authentication_structure(&indices),
            BatchSize::SmallInput,
        )
    });
}

fn verify_auth_structure(c: &mut Criterion) {
    let mut sampler = MerkleTreeSampler::default();
    let tree = sampler.tree();

    c.bench_function("verify_auth_structure", |bencher| {
        bencher.iter_batched(
            || sampler.proof(&tree),
            |proof| proof.verify(tree.root()),
            BatchSize::SmallInput,
        );
    });
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MerkleTreeSampler {
    rng: StdRng,
    tree_height: usize,
    num_opened_indices: usize,
}

impl Default for MerkleTreeSampler {
    fn default() -> Self {
        Self {
            rng: StdRng::seed_from_u64(0),
            tree_height: 22,
            num_opened_indices: 40,
        }
    }
}

impl MerkleTreeSampler {
    fn num_leaves(&self) -> usize {
        1 << self.tree_height
    }

    fn leaf_digests(&mut self) -> Vec<Digest> {
        let leaves = (0..self.num_leaves())
            .map(|_| self.rng.next_u64())
            .collect_vec();
        leaves.iter().map(Tip5::hash).collect_vec()
    }

    fn tree(&mut self) -> MerkleTree<Tip5> {
        let leaf_digests = self.leaf_digests();
        CpuParallel::from_digests(&leaf_digests).unwrap()
    }

    fn indices_to_open(&mut self) -> Vec<usize> {
        (0..self.num_opened_indices)
            .map(|_| self.rng.gen_range(0..self.num_leaves()))
            .collect()
    }

    fn proof(&mut self, tree: &MerkleTree<Tip5>) -> MerkleTreeInclusionProof<Tip5> {
        let leaf_indices = self.indices_to_open();
        tree.inclusion_proof_for_leaf_indices(&leaf_indices)
            .unwrap()
    }
}
