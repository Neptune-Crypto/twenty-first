use std::fmt::Debug;
use std::marker::PhantomData;

use crate::util_types::{
    mmr::membership_proof::MembershipProof,
    simple_hasher::{Hasher, ToDigest},
};

use super::{
    accumulator_mmr::AccumulatorMmr,
    mmr_trait::Mmr,
    shared::{
        bag_peaks, data_index_to_node_index, left_child, left_sibling, leftmost_ancestor,
        node_index_to_data_index, parent, right_child_and_height, right_sibling,
    },
};

/// A Merkle Mountain Range is a datastructure for storing a list of hashes.
///
/// Merkle Mountain Ranges only know about hashes. When values are to be associated with
/// MMRs, these values must be stored by the caller, or in a wrapper to this data structure.
#[derive(Debug, Clone)]
pub struct ArchiveMmr<HashDigest, H: Clone> {
    digests: Vec<HashDigest>,
    pub _hasher: PhantomData<H>,
}

impl<HashDigest, H> Mmr<HashDigest, H> for ArchiveMmr<HashDigest, H>
where
    H: Hasher<Digest = HashDigest> + Clone,
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
    u128: ToDigest<HashDigest>,
{
    fn new(digests: Vec<HashDigest>) -> Self {
        let dummy_digest = 0u128.to_digest();
        let mut new_mmr: Self = Self {
            digests: vec![dummy_digest],
            _hasher: PhantomData,
        };
        for digest in digests {
            new_mmr.append(digest);
        }

        new_mmr
    }

    /// Calculate the root for the entire MMR
    fn bag_peaks(&self) -> HashDigest {
        let peaks: Vec<HashDigest> = self.get_peaks();
        bag_peaks::<HashDigest, H>(&peaks, self.count_nodes() as u128)
    }

    /// Return the digests of the peaks of the MMR
    fn get_peaks(&self) -> Vec<HashDigest> {
        let peaks_and_heights = self.get_peaks_with_heights();
        peaks_and_heights.into_iter().map(|x| x.0).collect()
    }

    fn is_empty(&self) -> bool {
        self.digests.len() == 1
    }

    /// Return the number of leaves in the tree
    fn count_leaves(&self) -> u128 {
        let peaks_and_heights: Vec<(_, u128)> = self.get_peaks_with_heights();
        let mut acc = 0;
        for (_, height) in peaks_and_heights {
            acc += 1 << height
        }

        acc
    }

    /// Append an element to the archival MMR, return the membership proof of the newly added leaf.
    /// The membership proof is returned here since the accumulater MMR has no other way of
    /// retrieving a membership proof for a leaf. And the archival and accumulator MMR share
    /// this interface.
    fn append(&mut self, new_leaf: HashDigest) -> MembershipProof<HashDigest, H> {
        let node_index = self.digests.len() as u128;
        let data_index = node_index_to_data_index(node_index).unwrap();
        self.append_raw(new_leaf);
        self.prove_membership(data_index).0
    }

    /// Update a hash in the existing archival MMR
    fn mutate_leaf(
        &mut self,
        old_membership_proof: &MembershipProof<HashDigest, H>,
        new_leaf: &HashDigest,
    ) {
        // Sanity check
        let real_membership_proof: MembershipProof<HashDigest, H> =
            self.prove_membership(old_membership_proof.data_index).0;
        assert_eq!(
            real_membership_proof.authentication_path, old_membership_proof.authentication_path,
            "membership proof argument list must match internally calculated"
        );

        self.update_leaf_raw(real_membership_proof.data_index, new_leaf.to_owned())
    }

    fn verify_batch_update(
        &self,
        new_peaks: &[HashDigest],
        appended_leafs: &[HashDigest],
        leaf_mutations: &[(HashDigest, MembershipProof<HashDigest, H>)],
    ) -> bool {
        let accumulator: AccumulatorMmr<HashDigest, H> = self.into();
        accumulator.verify_batch_update(new_peaks, appended_leafs, leaf_mutations)
    }
}

impl<HashDigest, H> ArchiveMmr<HashDigest, H>
where
    H: Hasher<Digest = HashDigest> + Clone,
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
    u128: ToDigest<HashDigest>,
{
    /// Get a leaf from the MMR, will panic if index is out of range
    pub fn get_leaf(&self, data_index: u128) -> HashDigest {
        let node_index = data_index_to_node_index(data_index);
        self.digests[node_index as usize].clone()
    }

    /// Update a hash in the existing archival MMR
    pub fn update_leaf_raw(&mut self, data_index: u128, new_leaf: HashDigest) {
        // 1. change the leaf value
        let mut node_index = data_index_to_node_index(data_index);
        self.digests[node_index as usize] = new_leaf.clone();

        // 2. Calculate hash changes for all parents
        let mut parent_index = parent(node_index);
        let mut acc_hash = new_leaf;
        let mut hasher = H::new();

        // While parent exists in MMR, update parent
        while parent_index < self.digests.len() as u128 {
            let (is_right, height) = right_child_and_height(node_index);
            acc_hash = if is_right {
                hasher.hash_two(
                    &self.digests[left_sibling(node_index, height) as usize],
                    &acc_hash,
                )
            } else {
                hasher.hash_two(
                    &acc_hash,
                    &self.digests[right_sibling(node_index, height) as usize],
                )
            };
            self.digests[parent_index as usize] = acc_hash.clone();
            node_index = parent_index;
            parent_index = parent(parent_index);
        }
    }

    /// Return (membership_proof, peaks)
    pub fn prove_membership(
        &self,
        data_index: u128,
    ) -> (MembershipProof<HashDigest, H>, Vec<HashDigest>) {
        // A proof consists of an authentication path
        // and a list of peaks

        // Find out how long the authentication path is
        let node_index = data_index_to_node_index(data_index);
        let mut top_height: i32 = -1;
        let mut parent_index = node_index;
        while (parent_index as usize) < self.digests.len() {
            parent_index = parent(parent_index);
            top_height += 1;
        }

        // Build the authentication path
        let mut authentication_path: Vec<HashDigest> = vec![];
        let mut index = node_index;
        let (mut index_is_right_child, mut index_height): (bool, u128) =
            right_child_and_height(index);
        while index_height < top_height as u128 {
            if index_is_right_child {
                let left_sibling_index = left_sibling(index, index_height);
                authentication_path.push(self.digests[left_sibling_index as usize].clone());
            } else {
                let right_sibling_index = right_sibling(index, index_height);
                authentication_path.push(self.digests[right_sibling_index as usize].clone());
            }
            index = parent(index);
            let next_index_info = right_child_and_height(index);
            index_is_right_child = next_index_info.0;
            index_height = next_index_info.1;
        }

        let peaks: Vec<HashDigest> = self
            .get_peaks_with_heights()
            .iter()
            .map(|x| x.0.clone())
            .collect();

        let membership_proof = MembershipProof {
            authentication_path,
            data_index,
            _hasher: PhantomData,
        };

        (membership_proof, peaks)
    }

    /// Return a list of tuples (peaks, height)
    pub fn get_peaks_with_heights(&self) -> Vec<(HashDigest, u128)> {
        if self.is_empty() {
            return vec![];
        }

        // 1. Find top peak
        // 2. Jump to right sibling (will not be included)
        // 3. Take left child of sibling, continue until a node in tree is found
        // 4. Once new node is found, jump to right sibling (will not be included)
        // 5. Take left child of sibling, continue until a node in tree is found
        let mut peaks_and_heights: Vec<(HashDigest, u128)> = vec![];
        let (mut top_peak, mut top_height) = leftmost_ancestor(self.digests.len() as u128 - 1);
        if top_peak > self.digests.len() as u128 - 1 {
            top_peak = left_child(top_peak, top_height);
            top_height -= 1;
        }
        peaks_and_heights.push((self.digests[top_peak as usize].clone(), top_height)); // No clone needed bc array
        let mut height = top_height;
        let mut candidate = right_sibling(top_peak, height);
        'outer: while height > 0 {
            '_inner: while candidate > (self.digests.len() as u128) && height > 0 {
                candidate = left_child(candidate, height);
                height -= 1;
                if candidate < (self.digests.len() as u128) {
                    peaks_and_heights.push((self.digests[candidate as usize].clone(), height));
                    candidate = right_sibling(candidate, height);
                    continue 'outer;
                }
            }
        }

        peaks_and_heights
    }

    /// Return the number of nodes in all the trees in the MMR
    fn count_nodes(&self) -> u128 {
        self.digests.len() as u128 - 1
    }

    /// Append an element to the archival MMR
    pub fn append_raw(&mut self, new_leaf: HashDigest) {
        let node_index = self.digests.len() as u128;
        self.digests.push(new_leaf.clone());
        let (parent_needed, own_height) = right_child_and_height(node_index);
        if parent_needed {
            let left_sibling_hash =
                self.digests[left_sibling(node_index, own_height) as usize].clone();
            let mut hasher = H::new();
            let parent_hash: HashDigest = hasher.hash_two(&left_sibling_hash, &new_leaf);
            self.append_raw(parent_hash);
        }
    }
}

#[cfg(test)]
mod mmr_test {
    use itertools::izip;

    use super::*;
    use crate::{
        shared_math::{
            b_field_element::BFieldElement, rescue_prime::RescuePrime, rescue_prime_params,
        },
        util_types::{
            mmr::{
                accumulator_mmr::AccumulatorMmr, archive_mmr::ArchiveMmr,
                shared::get_peak_heights_and_peak_node_indices,
            },
            simple_hasher::RescuePrimeProduction,
        },
    };

    #[test]
    fn empty_mmr_behavior_test() {
        let mut archival_mmr: ArchiveMmr<blake3::Hash, blake3::Hasher> =
            ArchiveMmr::<blake3::Hash, blake3::Hasher>::new(vec![]);
        let mut accumulator_mmr: AccumulatorMmr<blake3::Hash, blake3::Hasher> =
            AccumulatorMmr::<blake3::Hash, blake3::Hasher>::new(vec![]);
        assert_eq!(0, archival_mmr.count_leaves());
        assert_eq!(0, accumulator_mmr.count_leaves());
        assert_eq!(archival_mmr.get_peaks(), accumulator_mmr.get_peaks());
        assert_eq!(Vec::<blake3::Hash>::new(), accumulator_mmr.get_peaks());
        assert_eq!(archival_mmr.bag_peaks(), accumulator_mmr.bag_peaks());
        assert_eq!(0, archival_mmr.count_nodes());
        assert!(accumulator_mmr.is_empty());
        assert!(archival_mmr.is_empty());

        // Test behavior of appending to an empty MMR
        let new_leaf = blake3::hash(
            bincode::serialize(&0xbeefu128)
                .expect("Encoding failed")
                .as_slice(),
        );
        let mut archival_mmr_appended = archival_mmr.clone();
        let archival_membership_proof = archival_mmr_appended.append(new_leaf);
        // let mut accumulator_mmr_appended = accumulator_mmr.clone();

        // Verify that the MMR update can be validated
        assert!(archival_mmr.verify_batch_update(
            &archival_mmr_appended.get_peaks(),
            &[new_leaf],
            &[]
        ));

        // Verify that failing MMR update for empty MMR fails gracefully
        assert!(!archival_mmr.verify_batch_update(
            &archival_mmr_appended.get_peaks(),
            &[],
            &[(new_leaf, archival_membership_proof)]
        ));

        // // Make the append and verify that the new peaks match the one from the proofs
        let archival_membership_proof = archival_mmr.append(new_leaf);
        let accumulator_membership_proof = accumulator_mmr.append(new_leaf);
        assert_eq!(archival_mmr.get_peaks(), archival_mmr_appended.get_peaks());
        assert_eq!(
            accumulator_mmr.get_peaks(),
            archival_mmr_appended.get_peaks()
        );

        // Verify that the appended value matches the one stored in the archival MMR
        assert_eq!(new_leaf, archival_mmr.get_leaf(0));

        // Verify that the membership proofs for the inserted leafs are valid and that they agree
        assert_eq!(
            archival_membership_proof, accumulator_membership_proof,
            "accumulator and archival membership proofs must agree"
        );
        assert!(
            archival_membership_proof
                .verify(
                    &archival_mmr.get_peaks(),
                    &new_leaf,
                    archival_mmr.count_leaves()
                )
                .0,
            "membership proof from arhival MMR must validate"
        );
    }

    #[test]
    fn verify_against_correct_peak_test() {
        // This test addresses a bug that was discovered late in the development process
        // where it was possible to fake a verification proof by providing a valid leaf
        // and authentication path but lying about the data index. This error occurred
        // because the derived hash was compared against all of the peaks to find a match
        // and it wasn't verified that the accumulated hash matched the *correct* peak.
        // This error was fixed and this test fails without that fix.
        let leaf_hashes: Vec<blake3::Hash> = vec![14u128, 15u128, 16u128]
            .iter()
            .map(|x| blake3::hash(bincode::serialize(x).expect("Encoding failed").as_slice()))
            .collect();
        let archival_mmr = ArchiveMmr::<blake3::Hash, blake3::Hasher>::new(leaf_hashes.clone());
        let (mut membership_proof, peaks): (
            MembershipProof<blake3::Hash, blake3::Hasher>,
            Vec<blake3::Hash>,
        ) = archival_mmr.prove_membership(0);

        // Verify that the accumulated hash in the verifier is compared against the **correct** hash,
        // not just **any** hash in the peaks list.
        assert!(membership_proof.verify(&peaks, &leaf_hashes[0], 3,).0);
        membership_proof.data_index = 2;
        assert!(!membership_proof.verify(&peaks, &leaf_hashes[0], 3,).0);
        membership_proof.data_index = 0;

        // verify the same behavior in the accumulator MMR
        let accumulator_mmr =
            AccumulatorMmr::<blake3::Hash, blake3::Hasher>::new(leaf_hashes.clone());
        assert!(
            membership_proof
                .verify(
                    &accumulator_mmr.get_peaks(),
                    &leaf_hashes[0],
                    accumulator_mmr.count_leaves()
                )
                .0
        );
        membership_proof.data_index = 2;
        assert!(
            !membership_proof
                .verify(
                    &accumulator_mmr.get_peaks(),
                    &leaf_hashes[0],
                    accumulator_mmr.count_leaves()
                )
                .0
        );
    }

    #[test]
    fn update_leaf_archival_test() {
        let mut rp = RescuePrimeProduction::new();
        let leaf_hashes: Vec<Vec<BFieldElement>> = (14..17)
            .map(|x| rp.hash_one(&vec![BFieldElement::new(x)]))
            .collect();
        let mut archival_mmr =
            ArchiveMmr::<Vec<BFieldElement>, RescuePrimeProduction>::new(leaf_hashes.clone());
        let (mp, old_peaks): (
            MembershipProof<Vec<BFieldElement>, RescuePrimeProduction>,
            Vec<Vec<BFieldElement>>,
        ) = archival_mmr.prove_membership(2);
        assert!(mp.verify(&old_peaks, &leaf_hashes[2], 3).0);
        let new_leaf = rp.hash_one(&vec![BFieldElement::new(10000)]);

        let mut archival_mmr_clone = archival_mmr.clone();
        archival_mmr_clone.mutate_leaf(&archival_mmr_clone.prove_membership(2).0, &new_leaf);
        let new_peaks_clone = archival_mmr_clone.get_peaks();
        archival_mmr.update_leaf_raw(2, new_leaf.clone());
        let new_peaks = archival_mmr.get_peaks();
        assert_eq!(
            new_peaks, new_peaks_clone,
            "peaks for two update leaf method calls must agree"
        );

        // Verify that peaks have changed as expected
        assert_ne!(old_peaks[1], new_peaks[1]);
        assert_eq!(old_peaks[0], new_peaks[0]);
        assert_eq!(2, new_peaks.len());
        assert_eq!(2, old_peaks.len());
        assert!(!mp.verify(&new_peaks, &leaf_hashes[2], 3).0);
        assert!(mp.verify(&new_peaks, &new_leaf, 3).0);

        // Create a new archival MMR with the same leaf hashes as in the
        // modified MMR, and verify that the two MMRs are equivalent
        let leaf_hashes_new = vec![
            rp.hash_one(&vec![BFieldElement::new(14)]),
            rp.hash_one(&vec![BFieldElement::new(15)]),
            rp.hash_one(&vec![BFieldElement::new(10000)]),
        ];
        let archival_mmr_new =
            ArchiveMmr::<Vec<BFieldElement>, RescuePrimeProduction>::new(leaf_hashes_new);
        assert_eq!(archival_mmr.digests, archival_mmr_new.digests);
    }

    #[test]
    fn bag_peaks_test() {
        // Verify that archival and accumulator MMR produce the same root
        // First with blake3
        let leaf_hashes_blake3: Vec<blake3::Hash> = vec![14u128, 15u128, 16u128]
            .iter()
            .map(|x| blake3::hash(bincode::serialize(x).expect("Encoding failed").as_slice()))
            .collect();
        let archival_mmr_small =
            ArchiveMmr::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_blake3.clone());
        let accumulator_mmr_small =
            AccumulatorMmr::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_blake3);
        assert_eq!(
            archival_mmr_small.bag_peaks(),
            accumulator_mmr_small.bag_peaks()
        );
        assert_eq!(
            archival_mmr_small.bag_peaks(),
            bag_peaks::<blake3::Hash, blake3::Hasher>(&accumulator_mmr_small.get_peaks(), 4)
        );
        assert!(!accumulator_mmr_small
            .get_peaks()
            .iter()
            .any(|peak| *peak == accumulator_mmr_small.bag_peaks()));

        // Then with Rescue Prime
        let leaf_hashes_rescue_prime: Vec<Vec<BFieldElement>> =
            (14..17).map(|x| vec![BFieldElement::new(x)]).collect();
        let archival_mmr_small_rp = ArchiveMmr::<Vec<BFieldElement>, RescuePrimeProduction>::new(
            leaf_hashes_rescue_prime.clone(),
        );
        let accumulator_mmr_small_rp =
            AccumulatorMmr::<Vec<BFieldElement>, RescuePrimeProduction>::new(
                leaf_hashes_rescue_prime,
            );
        assert_eq!(
            archival_mmr_small_rp.bag_peaks(),
            accumulator_mmr_small_rp.bag_peaks()
        );
        assert!(!accumulator_mmr_small_rp
            .get_peaks()
            .iter()
            .any(|peak| *peak == accumulator_mmr_small_rp.bag_peaks()));

        // Then with a bigger dataset
        let leaf_hashes_bigger_blake3: Vec<blake3::Hash> =
            vec![14u128, 15u128, 16u128, 206, 1232, 123, 9989]
                .iter()
                .map(|x| blake3::hash(bincode::serialize(x).expect("Encoding failed").as_slice()))
                .collect();
        let archival_mmr_bigger =
            ArchiveMmr::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_bigger_blake3.clone());
        let accumulator_mmr_bigger =
            AccumulatorMmr::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_bigger_blake3);
        assert_eq!(
            archival_mmr_bigger.bag_peaks(),
            accumulator_mmr_bigger.bag_peaks()
        );
        assert!(!accumulator_mmr_bigger
            .get_peaks()
            .iter()
            .any(|peak| *peak == accumulator_mmr_bigger.bag_peaks()));
    }

    #[test]
    fn accumulator_mmr_update_leaf_test() {
        // Verify that upating leafs in archival and in accumulator MMR results in the same peaks
        // and verify that updating all leafs in an MMR results in the expected MMR
        for size in 1..150 {
            let new_leaf = blake3::hash(
                bincode::serialize(&314159265358979u128)
                    .expect("Encoding failed")
                    .as_slice(),
            );
            let leaf_hashes_blake3: Vec<blake3::Hash> = (500u128..500 + size)
                .map(|x| blake3::hash(bincode::serialize(&x).expect("Encoding failed").as_slice()))
                .collect();
            let mut acc =
                AccumulatorMmr::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_blake3.clone());
            let mut archival =
                ArchiveMmr::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_blake3.clone());
            let archival_end_state =
                ArchiveMmr::<blake3::Hash, blake3::Hasher>::new(vec![new_leaf; size as usize]);
            for i in 0..size {
                let (mp, _archival_peaks) = archival.prove_membership(i);
                assert_eq!(i, mp.data_index);
                acc.mutate_leaf(&mp, &new_leaf);
                archival.update_leaf_raw(i, new_leaf);
                let new_archival_peaks = archival.get_peaks();
                assert_eq!(new_archival_peaks, acc.get_peaks());
            }

            assert_eq!(archival_end_state.get_peaks(), acc.get_peaks());
        }
    }

    #[test]
    fn mmr_prove_verify_leaf_update_test() {
        for size in 1..150 {
            let new_leaf = blake3::hash(
                bincode::serialize(&314159265358979u128)
                    .expect("Encoding failed")
                    .as_slice(),
            );
            let bad_leaf = blake3::hash(
                bincode::serialize(&27182818284590452353u128)
                    .expect("Encoding failed")
                    .as_slice(),
            );
            let leaf_hashes_blake3: Vec<blake3::Hash> = (500u128..500 + size)
                .map(|x| blake3::hash(bincode::serialize(&x).expect("Encoding failed").as_slice()))
                .collect();
            let mut acc =
                AccumulatorMmr::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_blake3.clone());
            let mut archival =
                ArchiveMmr::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_blake3.clone());
            let archival_end_state =
                ArchiveMmr::<blake3::Hash, blake3::Hasher>::new(vec![new_leaf; size as usize]);
            for i in 0..size {
                let (mp, peaks_before_update) = archival.prove_membership(i);
                assert_eq!(archival.get_peaks(), peaks_before_update);

                // Verify the update operation using the batch verifier
                archival.update_leaf_raw(i, new_leaf);
                assert!(
                    acc.verify_batch_update(&archival.get_peaks(), &[], &[(new_leaf, mp.clone())]),
                    "Valid batch update parameters must succeed"
                );
                assert!(
                    !acc.verify_batch_update(&archival.get_peaks(), &[], &[(bad_leaf, mp.clone())]),
                    "Inalid batch update parameters must fail"
                );

                acc.mutate_leaf(&mp, &new_leaf);
                let new_archival_peaks = archival.get_peaks();
                assert_eq!(new_archival_peaks, acc.get_peaks());
                assert_eq!(size, archival.count_leaves());
                assert_eq!(size, acc.count_leaves());
            }
            assert_eq!(archival_end_state.get_peaks(), acc.get_peaks());
        }
    }

    #[test]
    fn mmr_append_test() {
        // Verify that building an MMR iteratively or in *one* function call results in the same MMR
        for size in 1..260 {
            let leaf_hashes_blake3: Vec<blake3::Hash> = (500u128..500 + size)
                .map(|x| blake3::hash(bincode::serialize(&x).expect("Encoding failed").as_slice()))
                .collect();
            let mut archival_iterative = ArchiveMmr::<blake3::Hash, blake3::Hasher>::new(vec![]);
            let archival_batch =
                ArchiveMmr::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_blake3.clone());
            let mut accumulator_iterative =
                AccumulatorMmr::<blake3::Hash, blake3::Hasher>::new(vec![]);
            let accumulator_batch =
                AccumulatorMmr::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_blake3.clone());
            for (data_index, leaf_hash) in leaf_hashes_blake3.clone().into_iter().enumerate() {
                let archival_membership_proof: MembershipProof<blake3::Hash, blake3::Hasher> =
                    archival_iterative.append(leaf_hash);
                let accumulator_membership_proof = accumulator_iterative.append(leaf_hash);

                // Verify membership proofs returned from the append operation
                assert_eq!(
                    accumulator_membership_proof, archival_membership_proof,
                    "membership proofs from append operation must agree"
                );
                assert!(
                    archival_membership_proof
                        .verify(
                            &archival_iterative.get_peaks(),
                            &leaf_hash,
                            archival_iterative.count_leaves()
                        )
                        .0
                );

                // Verify that membership proofs are the same as generating them from an
                // archival MMR
                let archival_membership_proof_direct =
                    archival_iterative.prove_membership(data_index as u128).0;
                assert_eq!(archival_membership_proof_direct, archival_membership_proof);
            }

            // Verify that the MMRs built iteratively from `append` and
            // in *one* batch are the same
            assert_eq!(archival_iterative.digests, archival_batch.digests);
            assert_eq!(
                accumulator_batch.get_peaks(),
                accumulator_iterative.get_peaks()
            );
            assert_eq!(
                accumulator_batch.count_leaves(),
                accumulator_iterative.count_leaves()
            );
            assert_eq!(size, accumulator_iterative.count_leaves());
            assert_eq!(
                archival_iterative.get_peaks(),
                accumulator_iterative.get_peaks()
            );

            // Run a batch-append verification on the entire mutation of the MMR and verify that it succeeds
            let empty_accumulator = AccumulatorMmr::<blake3::Hash, blake3::Hasher>::new(vec![]);
            assert!(empty_accumulator.verify_batch_update(
                &archival_batch.get_peaks(),
                &leaf_hashes_blake3,
                &[],
            ));
        }
    }

    #[test]
    fn one_input_mmr_test() {
        let element = vec![BFieldElement::new(14)];
        let mut rp = RescuePrimeProduction::new();
        let input_hash = rp.hash_one(&element);
        let mut mmr =
            ArchiveMmr::<Vec<BFieldElement>, RescuePrimeProduction>::new(vec![input_hash.clone()]);
        assert_eq!(1, mmr.count_leaves());
        assert_eq!(1, mmr.count_nodes());
        let original_peaks_and_heights: Vec<(Vec<BFieldElement>, u128)> =
            mmr.get_peaks_with_heights();
        assert_eq!(1, original_peaks_and_heights.len());
        assert_eq!(0, original_peaks_and_heights[0].1);

        let data_index = 0;
        let (membership_proof, peaks) = mmr.prove_membership(data_index);
        let valid_res = membership_proof.verify(&peaks, &input_hash, 1);
        assert!(valid_res.0);
        assert!(valid_res.1.is_some());

        let new_input_hash = rp.hash_one(&vec![BFieldElement::new(201)]);
        let original_mmr = mmr.clone();
        mmr.append(new_input_hash.clone());
        assert_eq!(2, mmr.count_leaves());
        assert_eq!(3, mmr.count_nodes());
        let new_peaks_and_heights = mmr.get_peaks_with_heights();
        assert_eq!(1, new_peaks_and_heights.len());
        assert_eq!(1, new_peaks_and_heights[0].1);

        let new_peaks: Vec<Vec<BFieldElement>> =
            new_peaks_and_heights.iter().map(|x| x.0.to_vec()).collect();
        assert!(
            original_mmr.verify_batch_update(&new_peaks, &[new_input_hash.clone()], &[]),
            "verify batch update must succeed for a single append"
        );

        let mmr_after_append = mmr.clone();
        let new_leaf: Vec<BFieldElement> = rp.hash_one(&vec![BFieldElement::new(987223)]);

        // When verifying the batch update with two consequtive leaf mutations, we must get the
        // membership proofs prior to all mutations. This is because the `verify_batch_update` method
        // updates the membership proofs internally to account for the mutations.
        let leaf_mutations: Vec<(
            Vec<BFieldElement>,
            MembershipProof<Vec<BFieldElement>, RescuePrimeProduction>,
        )> = (0..2)
            .map(|i| (new_leaf.clone(), mmr_after_append.prove_membership(i).0))
            .collect();
        for &data_index in &[0u128, 1] {
            let mp = mmr.prove_membership(data_index).0;
            mmr.mutate_leaf(&mp, &new_leaf);
            assert_eq!(
                new_leaf,
                mmr.get_leaf(data_index),
                "fetched leaf must match what we put in"
            );
        }

        assert!(
            mmr_after_append.verify_batch_update(&mmr.get_peaks(), &[], &leaf_mutations),
            "The batch update of two leaf mutations must verify"
        );
    }

    #[test]
    fn two_input_mmr_test() {
        let values: Vec<Vec<BFieldElement>> = (0..2).map(|x| vec![BFieldElement::new(x)]).collect();
        let mut rp = RescuePrimeProduction::new();
        let input_hashes: Vec<Vec<BFieldElement>> = values.iter().map(|x| rp.hash_one(x)).collect();
        let mut mmr =
            ArchiveMmr::<Vec<BFieldElement>, RescuePrimeProduction>::new(input_hashes.clone());
        assert_eq!(2, mmr.count_leaves());
        assert_eq!(3, mmr.count_nodes());
        let original_peaks_and_heights: Vec<(Vec<BFieldElement>, u128)> =
            mmr.get_peaks_with_heights();
        assert_eq!(1, original_peaks_and_heights.len());

        let data_index: usize = 0;
        let (mut membership_proof, peaks) = mmr.prove_membership(data_index as u128);
        let valid_res = membership_proof.verify(&peaks, &input_hashes[data_index], 2);
        assert!(valid_res.0);
        assert!(valid_res.1.is_some());

        // Negative test for verify membership
        membership_proof.data_index += 1;
        assert!(
            !membership_proof
                .verify(&peaks, &input_hashes[data_index], 2)
                .0
        );

        let new_leaf_hash: Vec<BFieldElement> = rp.hash_one(&vec![BFieldElement::new(201)]);
        mmr.append(new_leaf_hash.clone());
        assert_eq!(3, mmr.count_leaves());
        assert_eq!(4, mmr.count_nodes());

        for &data_index in &[0u128, 1, 2] {
            let new_leaf: Vec<BFieldElement> = rp.hash_one(&vec![BFieldElement::new(987223)]);
            let mp = mmr.prove_membership(data_index).0;
            mmr.mutate_leaf(&mp, &new_leaf);
            assert_eq!(new_leaf, mmr.get_leaf(data_index));
        }
    }

    #[test]
    fn variable_size_rescue_prime_mmr_test() {
        let node_counts: Vec<u128> = vec![
            1, 3, 4, 7, 8, 10, 11, 15, 16, 18, 19, 22, 23, 25, 26, 31, 32, 34, 35, 38, 39, 41, 42,
            46, 47, 49, 50, 53, 54, 56, 57, 63, 64,
        ];
        let peak_counts: Vec<u128> = vec![
            1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4,
            4, 5, 1, 2,
        ];
        for (data_size, node_count, peak_count) in
            izip!((1u128..34).collect::<Vec<u128>>(), node_counts, peak_counts)
        {
            let input_prehashes: Vec<Vec<BFieldElement>> = (0..data_size)
                .map(|x| vec![BFieldElement::new(x as u128 + 14)])
                .collect();
            let rp: RescuePrime = rescue_prime_params::rescue_prime_params_bfield_0();
            let input_hashes: Vec<Vec<BFieldElement>> =
                input_prehashes.iter().map(|x| rp.hash(x)).collect();
            let mut mmr =
                ArchiveMmr::<Vec<BFieldElement>, RescuePrimeProduction>::new(input_hashes.clone());
            assert_eq!(data_size, mmr.count_leaves());
            assert_eq!(node_count, mmr.count_nodes());
            let original_peaks_and_heights = mmr.get_peaks_with_heights();
            let peak_heights_1: Vec<u128> =
                original_peaks_and_heights.iter().map(|x| x.1).collect();
            let (peak_heights_2, _) = get_peak_heights_and_peak_node_indices(data_size);
            assert_eq!(peak_heights_1, peak_heights_2);
            assert_eq!(peak_count, original_peaks_and_heights.len() as u128);

            // Get an authentication path for **all** values in MMR,
            // verify that it is valid
            for index in 0..data_size {
                let (membership_proof, peaks) = mmr.prove_membership(index as u128);
                let valid_res =
                    membership_proof.verify(&peaks, &input_hashes[index as usize], data_size);

                assert!(valid_res.0);
                assert!(valid_res.1.is_some());
            }

            // // Make a new MMR where we append with a value and run the verify_append
            let new_leaf_hash = rp.hash(&vec![BFieldElement::new(201)]);
            let orignal_peaks = mmr.get_peaks();
            let mp = mmr.append(new_leaf_hash.clone());
            assert!(
                mp.verify(&mmr.get_peaks(), &new_leaf_hash, data_size + 1).0,
                "Returned membership proof from append must verify"
            );
            assert_ne!(
                orignal_peaks,
                mmr.get_peaks(),
                "peaks must change when appending"
            );
        }
    }

    #[test]
    fn variable_size_blake3_mmr_test() {
        let node_counts: Vec<u128> = vec![
            1, 3, 4, 7, 8, 10, 11, 15, 16, 18, 19, 22, 23, 25, 26, 31, 32, 34, 35, 38, 39, 41, 42,
            46, 47, 49, 50, 53, 54, 56, 57, 63, 64,
        ];
        let peak_counts: Vec<u128> = vec![
            1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4,
            4, 5, 1, 2,
        ];
        for (data_size, node_count, peak_count) in
            izip!((1u128..34).collect::<Vec<u128>>(), node_counts, peak_counts)
        {
            let input_prehashes: Vec<Vec<BFieldElement>> = (0..data_size)
                .map(|x| vec![BFieldElement::new(x as u128 + 14)])
                .collect();

            let input_hashes: Vec<blake3::Hash> = input_prehashes
                .iter()
                .map(|x| blake3::hash(bincode::serialize(x).expect("Encoding failed").as_slice()))
                .collect();
            let mut mmr = ArchiveMmr::<blake3::Hash, blake3::Hasher>::new(input_hashes.clone());
            assert_eq!(data_size, mmr.count_leaves());
            assert_eq!(node_count, mmr.count_nodes());
            let original_peaks_and_heights: Vec<(blake3::Hash, u128)> =
                mmr.get_peaks_with_heights();
            let peak_heights_1: Vec<u128> =
                original_peaks_and_heights.iter().map(|x| x.1).collect();
            let (peak_heights_2, _) = get_peak_heights_and_peak_node_indices(data_size);
            assert_eq!(peak_heights_1, peak_heights_2);
            assert_eq!(peak_count, original_peaks_and_heights.len() as u128);

            // Get an authentication path for **all** values in MMR,
            // verify that it is valid
            for data_index in 0..data_size {
                let (mut membership_proof, peaks) = mmr.prove_membership(data_index);
                let valid_res =
                    membership_proof.verify(&peaks, &input_hashes[data_index as usize], data_size);
                assert!(valid_res.0);
                assert!(valid_res.1.is_some());

                let new_leaf: blake3::Hash = blake3::hash(
                    bincode::serialize(&98723u128)
                        .expect("Encoding failed")
                        .as_slice(),
                );
                // let mut leaf_update_proof = mmr.prove_update_leaf_raw(data_index, &new_leaf);
                // assert!(leaf_update_proof.verify(&new_leaf, data_size));

                // The below verify_modify tests should only fail if `wrong_data_index` is
                // different than `data_index`.
                let wrong_data_index = (data_index + 1) % mmr.count_leaves();
                membership_proof.data_index = wrong_data_index;
                assert!(
                    wrong_data_index == data_index
                        || !membership_proof.verify(&peaks, &new_leaf, data_size).0
                );
                membership_proof.data_index = data_index;

                // Modify an element in the MMR and run prove/verify for membership
                let old_leaf = input_hashes[data_index as usize];
                mmr.update_leaf_raw(data_index, new_leaf.clone());
                let (new_mp, new_peaks) = mmr.prove_membership(data_index);
                assert!(new_mp.verify(&new_peaks, &new_leaf, data_size).0);
                assert!(!new_mp.verify(&new_peaks, &old_leaf, data_size).0);

                // Return the element to its former value and run prove/verify for membership
                mmr.update_leaf_raw(data_index, old_leaf.clone());
                let (old_mp, old_peaks) = mmr.prove_membership(data_index);
                assert!(!old_mp.verify(&old_peaks, &new_leaf, data_size).0);
                assert!(old_mp.verify(&old_peaks, &old_leaf, data_size).0);
            }

            // Make a new MMR where we append with a value and run the verify_append
            let new_leaf_hash = blake3::hash(
                blake3::Hash::from_hex(format!("{:064x}", 519u128))
                    .unwrap()
                    .as_bytes(),
            );
            let mmr_original = mmr.clone();
            mmr.append(new_leaf_hash);
            assert!(mmr_original.verify_batch_update(&mmr.get_peaks(), &[new_leaf_hash], &[]));
        }
    }
}
