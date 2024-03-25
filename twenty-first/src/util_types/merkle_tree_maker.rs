use crate::math::digest::Digest;
use crate::util_types::algebraic_hasher::AlgebraicHasher;
use crate::util_types::merkle_tree::*;

pub trait MerkleTreeMaker<H: AlgebraicHasher> {
    fn from_digests(digests: &[Digest]) -> Result<MerkleTree<H>, MerkleTreeError>;
}
