use std::fmt::Debug;
use std::marker::PhantomData;

use crate::util_types::simple_hasher::{Hasher, ToDigest};

use super::{
    append_proof::AppendProof,
    archive_mmr::MmrArchive,
    membership_proof::MembershipProof,
    shared::{
        bag_peaks, calculate_new_peaks_and_membership_proof, data_index_to_node_index,
        get_peak_height, get_peak_heights_and_peak_node_indices, leaf_count_to_node_count, parent,
        right_child_and_height,
    },
};

#[derive(Debug, Clone)]
pub struct MmrAccumulator<HashDigest, H> {
    leaf_count: u128,
    peaks: Vec<HashDigest>,
    _hasher: PhantomData<H>,
}

// TODO: Write tests for the accumulator MMR functions
// 0. Create an (empty?) accumulator MMR
// 1. append a value to this
// 2. verify that the before state, after state,
//    and leaf hash constitute an append-proof
//    that can be verified with `verify_append`.
// 3. Repeat (2) n times.
// 4. Run prove/verify_membership with some values
//    But how do we get the authentication paths?
// 5. update hashes though `modify`
// 6. verify that this results in proofs that can
//    be verified with the verify_modify function.
impl<HashDigest, H> MmrAccumulator<HashDigest, H>
where
    H: Hasher<Digest = HashDigest> + Clone,
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
    u128: ToDigest<HashDigest>,
{
    /// Initialize a shallow MMR (only storing peaks) from a list of hash digests
    pub fn new(hashes: Vec<HashDigest>) -> Self {
        // If all the hash digests already exist in memory, we might as well
        // build the shallow MMR from an archival MMR, since it doesn't give
        // asymptotically higher RAM consumption than building it without storing
        // all digests. At least, I think that's the case.
        // Clearly, this function could use less RAM if we don't build the entire
        // archival MMR.
        let leaf_count = hashes.len() as u128;
        let archival = MmrArchive::new(hashes);
        let peaks_and_heights = archival.get_peaks_with_heights();
        Self {
            _hasher: archival._hasher,
            leaf_count,
            peaks: peaks_and_heights.iter().map(|x| x.0.clone()).collect(),
        }
    }

    pub fn bag_peaks(&self) -> HashDigest {
        bag_peaks::<HashDigest, H>(&self.peaks, leaf_count_to_node_count(self.leaf_count))
    }

    pub fn get_peaks(&self) -> Vec<HashDigest> {
        self.peaks.clone()
    }

    pub fn count_leaves(&self) -> u128 {
        self.leaf_count
    }

    pub fn is_empty(&self) -> bool {
        self.leaf_count == 0
    }

    /// Calculate the new accumulator MMR after inserting a new leaf and return the membership
    /// proof of this new leaf.
    /// The membership proof is returned here since the accumulater MMR has no other way of
    /// retrieving a membership proof for a leaf.
    pub fn append(&mut self, new_leaf: HashDigest) -> MembershipProof<HashDigest, H> {
        let (new_peaks, membership_proof) =
            calculate_new_peaks_and_membership_proof::<H, HashDigest>(
                self.leaf_count,
                self.peaks.clone(),
                new_leaf,
            )
            .unwrap();
        self.peaks = new_peaks;
        self.leaf_count += 1;

        membership_proof
    }

    /// Create a proof for honest appending. Verifiable by `AccumulatorMmr` implementation.
    /// Returns (old_peaks, old_leaf_count, new_peaks)
    pub fn prove_append(&self, new_leaf: HashDigest) -> AppendProof<HashDigest, H> {
        let old_peaks = self.peaks.clone();
        let old_leaf_count = self.leaf_count;
        let new_peaks = calculate_new_peaks_and_membership_proof::<H, HashDigest>(
            old_leaf_count,
            old_peaks.clone(),
            new_leaf,
        )
        .unwrap()
        .0;

        AppendProof {
            old_peaks,
            old_leaf_count,
            new_peaks,
            _hasher: PhantomData,
        }
    }

    /// Update a leaf hash and modify the peaks with this new hash
    pub fn update_leaf(
        &mut self,
        old_membership_proof: &MembershipProof<HashDigest, H>,
        new_leaf: &HashDigest,
    ) {
        let node_index = data_index_to_node_index(old_membership_proof.data_index);
        let mut hasher = H::new();
        let mut acc_hash: HashDigest = new_leaf.to_owned();
        let mut acc_index: u128 = node_index;
        for hash in old_membership_proof.authentication_path.iter() {
            let (acc_right, _acc_height) = right_child_and_height(acc_index);
            acc_hash = if acc_right {
                hasher.hash_two(hash, &acc_hash)
            } else {
                hasher.hash_two(&acc_hash, hash)
            };
            acc_index = parent(acc_index);
        }

        // This function is *not* secure when verified against *any* peak.
        // It **must** be compared against the correct peak.
        // Otherwise you could lie leaf_hash, data_index, authentication path
        let (peak_heights, _) = get_peak_heights_and_peak_node_indices(self.leaf_count);
        let expected_peak_height_res =
            get_peak_height(self.leaf_count, old_membership_proof.data_index);
        let expected_peak_height = match expected_peak_height_res {
            None => panic!("Did not find any peak height for (leaf_count, data_index) combination. Got: leaf_count = {}, data_index = {}", self.leaf_count, old_membership_proof.data_index),
            Some(eph) => eph,
        };

        let peak_height_index_res = peak_heights.iter().position(|x| *x == expected_peak_height);
        let peak_height_index = match peak_height_index_res {
            None => panic!("Did not find a matching peak"),
            Some(index) => index,
        };

        self.peaks[peak_height_index] = acc_hash;
    }

    /// Construct a proof of the integral update of a hash in an existing accumulator MMR
    /// New authentication path (membership proof) is unchanged by this operation, so
    /// it is not output. Outputs new_peaks.
    pub fn prove_update_leaf(
        &self,
        old_membership_proof: &MembershipProof<HashDigest, H>,
        new_leaf: &HashDigest,
    ) -> Vec<HashDigest> {
        let mut updated_self = self.clone();
        updated_self.update_leaf(old_membership_proof, new_leaf);

        updated_self.peaks
    }

    /// Prove that a specific leaf hash belongs in an MMR
    pub fn prove_membership(
        _membership_proof: &MembershipProof<HashDigest, H>,
        _leaf_hash: HashDigest,
    ) {
    }

    /// Verify a membership proof/leaf hash pair
    pub fn verify_membership_proof(
        &self,
        membership_proof: &MembershipProof<HashDigest, H>,
        leaf_hash: &HashDigest,
    ) -> (bool, Option<HashDigest>) {
        membership_proof.verify(&self.peaks, leaf_hash, self.leaf_count)
    }
}
