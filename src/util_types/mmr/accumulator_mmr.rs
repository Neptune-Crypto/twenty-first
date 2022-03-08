use std::fmt::Debug;
use std::marker::PhantomData;

use crate::util_types::simple_hasher::{Hasher, ToDigest};

use super::{
    append_proof::AppendProof,
    archive_mmr::MmrArchive,
    leaf_update_proof::LeafUpdateProof,
    membership_proof::MembershipProof,
    mmr_trait::Mmr,
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

impl<HashDigest, H> From<&MmrArchive<HashDigest, H>> for MmrAccumulator<HashDigest, H>
where
    H: Hasher<Digest = HashDigest> + Clone,
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
    u128: ToDigest<HashDigest>,
{
    fn from(archive: &MmrArchive<HashDigest, H>) -> Self {
        Self {
            leaf_count: archive.count_leaves(),
            peaks: archive.get_peaks(),
            _hasher: PhantomData,
        }
    }
}

impl<HashDigest, H> MmrAccumulator<HashDigest, H>
where
    H: Hasher<Digest = HashDigest> + Clone,
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
    u128: ToDigest<HashDigest>,
{
    pub fn init(peaks: Vec<HashDigest>, leaf_count: u128) -> Self {
        Self {
            leaf_count,
            peaks,
            _hasher: PhantomData,
        }
    }
}

impl<HashDigest, H> Mmr<HashDigest, H> for MmrAccumulator<HashDigest, H>
where
    H: Hasher<Digest = HashDigest> + Clone,
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
    u128: ToDigest<HashDigest>,
{
    fn new(digests: Vec<HashDigest>) -> Self {
        // If all the hash digests already exist in memory, we might as well
        // build the shallow MMR from an archival MMR, since it doesn't give
        // asymptotically higher RAM consumption than building it without storing
        // all digests. At least, I think that's the case.
        // Clearly, this function could use less RAM if we don't build the entire
        // archival MMR.
        let leaf_count = digests.len() as u128;
        let archival = MmrArchive::new(digests);
        let peaks_and_heights = archival.get_peaks_with_heights();
        Self {
            _hasher: archival._hasher,
            leaf_count,
            peaks: peaks_and_heights.iter().map(|x| x.0.clone()).collect(),
        }
    }

    fn bag_peaks(&self) -> HashDigest {
        bag_peaks::<HashDigest, H>(&self.peaks, leaf_count_to_node_count(self.leaf_count))
    }

    fn get_peaks(&self) -> Vec<HashDigest> {
        self.peaks.clone()
    }

    fn is_empty(&self) -> bool {
        self.leaf_count == 0
    }

    fn count_leaves(&self) -> u128 {
        self.leaf_count
    }

    fn append(&mut self, new_leaf: HashDigest) -> MembershipProof<HashDigest, H> {
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

    fn prove_append(&self, new_leaf: HashDigest) -> AppendProof<HashDigest, H> {
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

    fn update_leaf(
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

    fn prove_update_leaf(
        &self,
        old_membership_proof: &MembershipProof<HashDigest, H>,
        new_leaf: &HashDigest,
    ) -> LeafUpdateProof<HashDigest, H> {
        let mut updated_self = self.clone();
        updated_self.update_leaf(old_membership_proof, new_leaf);

        LeafUpdateProof {
            membership_proof: old_membership_proof.to_owned(),
            new_peaks: updated_self.peaks,
            old_peaks: self.get_peaks(),
        }
    }
}

#[cfg(test)]
mod accumulator_mmr_tests {
    use super::*;

    #[test]
    fn conversion_test() {
        let leaf_hashes: Vec<blake3::Hash> = vec![14u128, 15u128, 16u128]
            .iter()
            .map(|x| blake3::hash(bincode::serialize(x).expect("Encoding failed").as_slice()))
            .collect();
        let archival_mmr: MmrArchive<blake3::Hash, blake3::Hasher> =
            MmrArchive::new(leaf_hashes.clone());
        let accumulator_mmr: MmrAccumulator<blake3::Hash, blake3::Hasher> = (&archival_mmr).into();
        assert_eq!(archival_mmr.get_peaks(), accumulator_mmr.get_peaks());
        assert_eq!(archival_mmr.bag_peaks(), accumulator_mmr.bag_peaks());
        assert_eq!(archival_mmr.is_empty(), accumulator_mmr.is_empty());
        assert!(!archival_mmr.is_empty());
        assert_eq!(archival_mmr.count_leaves(), accumulator_mmr.count_leaves());
        assert_eq!(3, accumulator_mmr.count_leaves());
    }
}
