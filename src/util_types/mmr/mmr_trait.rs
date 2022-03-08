use super::membership_proof::MembershipProof;
use crate::util_types::mmr::append_proof::AppendProof;
use crate::util_types::mmr::leaf_update_proof::LeafUpdateProof;
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
    fn prove_append(&self, new_leaf: HashDigest) -> AppendProof<HashDigest, H>;
    fn update_leaf(
        &mut self,
        old_membership_proof: &MembershipProof<HashDigest, H>,
        new_leaf: &HashDigest,
    );
    fn prove_update_leaf(
        &self,
        old_membership_proof: &MembershipProof<HashDigest, H>,
        new_leaf: &HashDigest,
    ) -> LeafUpdateProof<HashDigest, H>;
}
