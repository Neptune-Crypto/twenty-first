use crate::shared_math::rescue_prime_digest::Digest;
use crate::util_types::algebraic_hasher::AlgebraicHasher;

use super::{mmr_accumulator::MmrAccumulator, mmr_membership_proof::MmrMembershipProof};

pub trait Mmr<H: AlgebraicHasher> {
    /// Create a new MMR instanc from a list of hash digests. The supplied digests
    /// are the leaves of the MMR.

    // constructors cannot be part of the interface sicne the archival version requires a
    // database which we want the caller to create, and the accumulator does not need a
    // constructor.
    // fn new(digests: Vec<Digest>) -> Self;

    /// Calculate a single hash digest committing to the entire MMR.
    fn bag_peaks(&mut self) -> Digest;

    /// Returns the peaks of the MMR, which are roots of the Merkle trees that constitute
    /// the MMR
    fn get_peaks(&mut self) -> Vec<Digest>;

    /// Returns `true` iff the MMR has no leaves
    fn is_empty(&mut self) -> bool;

    /// Returns the number of leaves in the MMR
    fn count_leaves(&mut self) -> u128;

    /// Append a hash digest to the MMR
    fn append(&mut self, new_leaf: Digest) -> MmrMembershipProof<H>;

    /// Mutate an existing leaf. It is the caller's responsibility that the
    /// membership proof is valid. If the membership proof is wrong, the MMR
    /// will end up in a broken state.
    fn mutate_leaf(&mut self, old_membership_proof: &MmrMembershipProof<H>, new_leaf: &Digest);

    /// Batch mutate an MMR while updating a list of membership proofs. Returns the indices of the
    /// membership proofs that have changed as a result of this operation.
    fn batch_mutate_leaf_and_update_mps(
        &mut self,
        membership_proofs: &mut [&mut MmrMembershipProof<H>],
        mutation_data: Vec<(MmrMembershipProof<H>, Digest)>,
    ) -> Vec<usize>;

    /// Returns true if a list of leaf mutations and a list of appends results in the expected
    /// `new_peaks`.
    fn verify_batch_update(
        &mut self,
        new_peaks: &[Digest],
        appended_leafs: &[Digest],
        leaf_mutations: &[(Digest, MmrMembershipProof<H>)],
    ) -> bool;

    /// Return an MMR accumulator containing only peaks and leaf count
    fn to_accumulator(&mut self) -> MmrAccumulator<H>;
}
