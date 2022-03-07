use crate::util_types::simple_hasher::{Hasher, ToDigest};
use std::fmt::Debug;

use super::membership_proof::{verify_membership_proof, MembershipProof};

/// A proof of integral updating of a leaf. The membership_proof can be either before
/// or after the update, since it does not change up the update, only all other
/// membership proofs do.
#[derive(Debug, Clone)]
pub struct LeafUpdateProof<HashDigest, H>
where
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
    H: Hasher<Digest = HashDigest> + Clone,
{
    // The membership proof for a leaf does *not* change when that leaf is updated,
    // only all other membership proofs do. So we only include *one* membership proof
    // in this data structure. In other words: This membership proof is simulataneously
    // the old *and* the new membership proof of the updated leaf.
    pub membership_proof: MembershipProof<HashDigest, H>,
    pub old_peaks: Vec<HashDigest>,
    pub new_peaks: Vec<HashDigest>,
    // TODO: Add a verify method
}

impl<HashDigest, H> LeafUpdateProof<HashDigest, H>
where
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
    H: Hasher<Digest = HashDigest> + Clone,
{
    /// Verify a proof for the integral update of a leaf in the MMR
    pub fn verify(&self, new_leaf: &HashDigest, leaf_count: u128) -> bool
    where
        u128: ToDigest<HashDigest>,
    {
        // We need to verify that
        // 1: New authentication path is valid
        // 2: Only the targeted peak is changed, all other must remain unchanged

        // 1: New authentication path is valid
        let (new_valid, sub_tree_root_res) = verify_membership_proof(
            &self.membership_proof,
            &self.new_peaks,
            new_leaf,
            leaf_count,
        );
        if !new_valid {
            return false;
        }

        // 2: Only the targeted peak is changed, all other must remain unchanged
        let sub_tree_root = sub_tree_root_res.unwrap();
        let modified_peak_index_res = self
            .new_peaks
            .iter()
            .position(|peak| *peak == sub_tree_root);
        let modified_peak_index = match modified_peak_index_res {
            None => return false,
            Some(index) => index,
        };
        let mut calculated_new_peaks: Vec<HashDigest> = self.old_peaks.to_owned();
        calculated_new_peaks[modified_peak_index] = sub_tree_root;

        calculated_new_peaks == self.new_peaks
    }
}
