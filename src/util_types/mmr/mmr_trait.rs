use super::membership_proof::MembershipProof;
use crate::util_types::simple_hasher::Hasher;

pub trait Mmr<H>
where
    H: Hasher,
{
    fn new(digests: Vec<H::Digest>) -> Self;
    fn bag_peaks(&self) -> H::Digest;
    fn get_peaks(&self) -> Vec<H::Digest>;
    fn is_empty(&self) -> bool;
    fn count_leaves(&self) -> u128;
    fn append(&mut self, new_leaf: H::Digest) -> MembershipProof<H>;

    /// Mutate an existing leaf. It is the caller's responsibility that the
    /// membership proof is valid. If the membership proof is wrong, the MMR
    /// will end up in a broken state.
    fn mutate_leaf(&mut self, old_membership_proof: &MembershipProof<H>, new_leaf: &H::Digest);

    // Batch mutate an MMR while updating a list of membership proofs
    fn batch_mutate_leaf_and_update_mps(
        &mut self,
        membership_proofs: &mut Vec<MembershipProof<H>>,
        mutation_data: Vec<(MembershipProof<H>, H::Digest)>,
    ) -> Vec<u128>;

    // Returns true if a list of leaf mutations and a list of appends results in the expected
    // `new_peaks`.
    fn verify_batch_update(
        &self,
        new_peaks: &[H::Digest],
        appended_leafs: &[H::Digest],
        leaf_mutations: &[(H::Digest, MembershipProof<H>)],
    ) -> bool;
}
