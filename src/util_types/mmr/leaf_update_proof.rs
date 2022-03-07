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
    // TODO: Add a verify method
}
