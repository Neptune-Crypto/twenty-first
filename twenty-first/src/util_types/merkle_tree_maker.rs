use crate::shared_math::rescue_prime_digest::Digest;
use crate::util_types::algebraic_hasher::AlgebraicHasher;
use crate::util_types::merkle_tree::MerkleTree;

/// A `MerkleTree<M, T>` is parameterised over both
///
/// - `H: AlgebraicHasher`
/// - `M: MerkleTreeMaker`
///
/// so that both the hash function and the method of assembling the Merkle tree may vary.
pub trait MerkleTreeMaker<H: AlgebraicHasher> {
    fn from_digests(digests: &[Digest]) -> MerkleTree<H, Self>;
}
