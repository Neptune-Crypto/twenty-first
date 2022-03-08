use crate::util_types::simple_hasher::{Hasher, ToDigest};
use std::fmt::Debug;

use super::membership_proof::MembershipProof;

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
}

impl<HashDigest: PartialEq, H> PartialEq for LeafUpdateProof<HashDigest, H>
where
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
    H: Hasher<Digest = HashDigest> + Clone,
{
    // Two leaf update proofs are considered equal if they contain the same membership
    // proofs, same new peaks, and same old peaks
    fn eq(&self, other: &Self) -> bool {
        self.membership_proof == other.membership_proof
            && self.old_peaks == other.old_peaks
            && self.new_peaks == other.new_peaks
    }
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
        let (new_valid, sub_tree_root_res) =
            self.membership_proof
                .verify(&self.new_peaks, new_leaf, leaf_count);
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

#[cfg(test)]
mod mmr_leaf_update_tests {
    use crate::util_types::mmr::{archive_mmr::MmrArchive, mmr_trait::Mmr};

    use super::*;

    #[test]
    fn equality_test() {
        let leaf_hashes: Vec<blake3::Hash> = vec![14u128, 15u128, 16u128]
            .iter()
            .map(|x| blake3::hash(bincode::serialize(x).expect("Encoding failed").as_slice()))
            .collect();
        let new_leaf = blake3::hash(
            bincode::serialize(&1000u128)
                .expect("Encoding failed")
                .as_slice(),
        );
        let mut archival_mmr = MmrArchive::<blake3::Hash, blake3::Hasher>::new(leaf_hashes.clone());
        let (mp0_old, peaks0_old) = archival_mmr.prove_membership(0);
        archival_mmr.append(new_leaf);
        let (mp0_new, peaks0_new) = archival_mmr.prove_membership(0);

        // Test equality
        // Note that these proofs do not have to be valid proofs
        let ulp0 = LeafUpdateProof {
            membership_proof: mp0_old.clone(),
            new_peaks: peaks0_old.clone(),
            old_peaks: peaks0_old.clone(),
        };
        let ulp1 = LeafUpdateProof {
            membership_proof: mp0_old.clone(),
            new_peaks: peaks0_new.clone(),
            old_peaks: peaks0_old.clone(),
        };
        let ulp2 = LeafUpdateProof {
            membership_proof: mp0_old.clone(),
            new_peaks: peaks0_old.clone(),
            old_peaks: peaks0_new.clone(),
        };
        let ulp3 = LeafUpdateProof {
            membership_proof: mp0_new.clone(),
            new_peaks: peaks0_old.clone(),
            old_peaks: peaks0_old.clone(),
        };
        let ulp4 = LeafUpdateProof {
            membership_proof: mp0_old.clone(),
            new_peaks: peaks0_old.clone(),
            old_peaks: peaks0_old.clone(),
        };

        assert_ne!(
            ulp0, ulp1,
            "leaf update proofs must be different when new peaks differ"
        );
        assert_ne!(
            ulp1, ulp2,
            "leaf update proofs must be different when old and new peaks differ"
        );
        assert_ne!(
            ulp2, ulp3,
            "leaf update proofs must be different when membership proof and old peaks differ"
        );
        assert_ne!(
            ulp3, ulp4,
            "leaf update proofs must be different when membership proof differ"
        );
        assert_eq!(
            ulp4, ulp0,
            "leaf update proofs must be equal when all three field agree"
        );

        assert_ne!(
            ulp0, ulp2,
            "leaf update proofs must be different when old peaks differ"
        );
    }
}
