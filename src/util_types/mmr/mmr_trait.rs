use super::membership_proof::MembershipProof;
use crate::util_types::simple_hasher::{Hasher, ToDigest};
use std::fmt::Debug;

pub trait Mmr<HashDigest, H>
where
    H: Hasher<Digest = HashDigest> + Clone,
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
{
    fn new(digests: Vec<HashDigest>) -> Self;
    fn bag_peaks(&self) -> HashDigest;
    fn get_peaks(&self) -> Vec<HashDigest>;
    fn is_empty(&self) -> bool;
    fn count_leaves(&self) -> u128;
    fn append(&mut self, new_leaf: HashDigest) -> MembershipProof<HashDigest, H>;

    /// Mutate an existing leaf. It is the caller's responsibility that the
    /// membership proof is valid. If the membership proof is wrong, the MMR
    /// will end up in a broken state.
    fn mutate_leaf(
        &mut self,
        old_membership_proof: &MembershipProof<HashDigest, H>,
        new_leaf: &HashDigest,
    );
    fn verify_batch_update(
        &self,
        new_peaks: &[HashDigest],
        appended_leafs: &[HashDigest],
        leaf_mutations: &[(HashDigest, MembershipProof<HashDigest, H>)],
    ) -> bool;
}
