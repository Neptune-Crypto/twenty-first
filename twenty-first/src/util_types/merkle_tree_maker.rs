use crate::shared_math::digest::Digest;
use crate::util_types::algebraic_hasher::AlgebraicHasher;
use crate::util_types::merkle_tree::MerkleTree;

pub trait MerkleTreeMaker<H: AlgebraicHasher> {
    fn from_digests(digests: &[Digest]) -> MerkleTree<H>;
}
