use super::{membership_proof::MembershipProof, mmr_accumulator::MmrAccumulator};
use crate::util_types::simple_hasher::Hasher;

pub trait Mmr<H>
where
    H: Hasher,
{
    /// Create a new MMR instanc from a list of hash digests. The supplied digests
    /// are the leaves of the MMR.
    fn new(digests: Vec<H::Digest>) -> Self;

    /// Calculate a single hash digest committing to the entire MMR.
    fn bag_peaks(&self) -> H::Digest;

    /// Returns the peaks of the MMR, which are roots of the Merkle trees that constitute
    /// the MMR
    fn get_peaks(&self) -> Vec<H::Digest>;

    /// Returns `true` iff the MMR has no leaves
    fn is_empty(&self) -> bool;

    /// Returns the number of leaves in the MMR
    fn count_leaves(&self) -> u128;

    /// Append a hash digest to the MMR
    fn append(&mut self, new_leaf: H::Digest) -> MembershipProof<H>;

    /// Mutate an existing leaf. It is the caller's responsibility that the
    /// membership proof is valid. If the membership proof is wrong, the MMR
    /// will end up in a broken state.
    fn mutate_leaf(&mut self, old_membership_proof: &MembershipProof<H>, new_leaf: &H::Digest);

    /// Batch mutate an MMR while updating a list of membership proofs. Returns the indices into
    /// the list of membership proofs that where given as argument.
    fn batch_mutate_leaf_and_update_mps(
        &mut self,
        membership_proofs: &mut Vec<MembershipProof<H>>,
        mutation_data: Vec<(MembershipProof<H>, H::Digest)>,
    ) -> Vec<usize>;

    /// Returns true if a list of leaf mutations and a list of appends results in the expected
    /// `new_peaks`.
    fn verify_batch_update(
        &self,
        new_peaks: &[H::Digest],
        appended_leafs: &[H::Digest],
        leaf_mutations: &[(H::Digest, MembershipProof<H>)],
    ) -> bool;

    /// Return an MMR accumulator containing only peaks and leaf count
    fn to_accumulator(&self) -> MmrAccumulator<H>;
}
