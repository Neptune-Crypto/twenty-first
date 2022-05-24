use std::fmt::Debug;
use std::marker::PhantomData;

use crate::{
    util_types::{
        mmr::membership_proof::MembershipProof,
        simple_hasher::{Hasher, ToDigest},
    },
    utils::has_unique_elements,
};

use super::{
    mmr_accumulator::MmrAccumulator,
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
pub struct ArchivalMmr<H: Hasher> {
    digests: Vec<H::Digest>,
    _hasher: PhantomData<H>,
}

impl<H> Mmr<H> for ArchivalMmr<H>
where
    H: Hasher,
    u128: ToDigest<H::Digest>,
{
    fn new(digests: Vec<H::Digest>) -> Self {
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
    fn bag_peaks(&self) -> H::Digest {
        let peaks: Vec<H::Digest> = self.get_peaks();
        bag_peaks::<H>(&peaks, self.count_nodes() as u128)
    }

    /// Return the digests of the peaks of the MMR
    fn get_peaks(&self) -> Vec<H::Digest> {
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
    fn append(&mut self, new_leaf: H::Digest) -> MembershipProof<H> {
        let node_index = self.digests.len() as u128;
        let data_index = node_index_to_data_index(node_index).unwrap();
        self.append_raw(new_leaf);
        self.prove_membership(data_index).0
    }

    /// Mutate an existing leaf. It is the caller's responsibility that the
    /// membership proof is valid. If the membership proof is wrong, the MMR
    /// will end up in a broken state.
    fn mutate_leaf(&mut self, old_membership_proof: &MembershipProof<H>, new_leaf: &H::Digest) {
        // Sanity check
        let real_membership_proof: MembershipProof<H> =
            self.prove_membership(old_membership_proof.data_index).0;
        assert_eq!(
            real_membership_proof.authentication_path, old_membership_proof.authentication_path,
            "membership proof argument list must match internally calculated"
        );

        self.mutate_leaf_raw(real_membership_proof.data_index, new_leaf.to_owned())
    }

    fn verify_batch_update(
        &self,
        new_peaks: &[H::Digest],
        appended_leafs: &[H::Digest],
        leaf_mutations: &[(H::Digest, MembershipProof<H>)],
    ) -> bool {
        let accumulator: MmrAccumulator<H> = self.into();
        accumulator.verify_batch_update(new_peaks, appended_leafs, leaf_mutations)
    }

    fn batch_mutate_leaf_and_update_mps(
        &mut self,
        membership_proofs: &mut Vec<MembershipProof<H>>,
        mutation_data: Vec<(MembershipProof<H>, <H as Hasher>::Digest)>,
    ) -> Vec<usize> {
        assert!(
            has_unique_elements(mutation_data.iter().map(|md| md.0.data_index)),
            "Duplicated leaves are not allowed in membership proof updater"
        );

        for (mp, digest) in mutation_data.iter() {
            self.mutate_leaf_raw(mp.data_index, digest.clone());
        }

        let mut modified_mps: Vec<usize> = vec![];
        for (i, mp) in membership_proofs.iter_mut().enumerate() {
            let new_mp = self.prove_membership(mp.data_index).0;
            if new_mp != *mp {
                modified_mps.push(i);
            }

            *mp = new_mp
        }

        modified_mps
    }

    fn to_accumulator(&self) -> MmrAccumulator<H> {
        MmrAccumulator::init(self.get_peaks(), self.count_leaves())
    }
}

impl<H> ArchivalMmr<H>
where
    H: Hasher,
    u128: ToDigest<H::Digest>,
{
    /// Get a leaf from the MMR, will panic if index is out of range
    pub fn get_leaf(&self, data_index: u128) -> H::Digest {
        let node_index = data_index_to_node_index(data_index);
        self.digests[node_index as usize].clone()
    }

    /// Update a hash in the existing archival MMR
    pub fn mutate_leaf_raw(&mut self, data_index: u128, new_leaf: H::Digest) {
        // 1. change the leaf value
        let mut node_index = data_index_to_node_index(data_index);
        self.digests[node_index as usize] = new_leaf.clone();

        // 2. Calculate hash changes for all parents
        let mut parent_index = parent(node_index);
        let mut acc_hash = new_leaf;
        let hasher = H::new();

        // While parent exists in MMR, update parent
        while parent_index < self.digests.len() as u128 {
            let (is_right, height) = right_child_and_height(node_index);
            acc_hash = if is_right {
                hasher.hash_pair(
                    &self.digests[left_sibling(node_index, height) as usize],
                    &acc_hash,
                )
            } else {
                hasher.hash_pair(
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
    pub fn prove_membership(&self, data_index: u128) -> (MembershipProof<H>, Vec<H::Digest>) {
        // A proof consists of an authentication path
        // and a list of peaks
        assert!(data_index < self.count_leaves());

        // Find out how long the authentication path is
        let node_index = data_index_to_node_index(data_index);
        let mut top_height: i32 = -1;
        let mut parent_index = node_index;
        while (parent_index as usize) < self.digests.len() {
            parent_index = parent(parent_index);
            top_height += 1;
        }

        // Build the authentication path
        let mut authentication_path: Vec<H::Digest> = vec![];
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

        let peaks: Vec<H::Digest> = self
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
    pub fn get_peaks_with_heights(&self) -> Vec<(H::Digest, u128)> {
        if self.is_empty() {
            return vec![];
        }

        // 1. Find top peak
        // 2. Jump to right sibling (will not be included)
        // 3. Take left child of sibling, continue until a node in tree is found
        // 4. Once new node is found, jump to right sibling (will not be included)
        // 5. Take left child of sibling, continue until a node in tree is found
        let mut peaks_and_heights: Vec<(H::Digest, u128)> = vec![];
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
    pub fn append_raw(&mut self, new_leaf: H::Digest) {
        let node_index = self.digests.len() as u128;
        self.digests.push(new_leaf.clone());
        let (parent_needed, own_height) = right_child_and_height(node_index);
        if parent_needed {
            let left_sibling_hash =
                self.digests[left_sibling(node_index, own_height) as usize].clone();
            let hasher = H::new();
            let parent_hash: H::Digest = hasher.hash_pair(&left_sibling_hash, &new_leaf);
            self.append_raw(parent_hash);
        }
    }
}

#[cfg(test)]
mod mmr_test {
    use super::*;
    use crate::util_types::blake3_wrapper::Blake3Hash;
    use crate::{
        shared_math::{
            b_field_element::BFieldElement, rescue_prime::RescuePrime, rescue_prime_params,
        },
        util_types::{
            mmr::{
                archival_mmr::ArchivalMmr, mmr_accumulator::MmrAccumulator,
                shared::get_peak_heights_and_peak_node_indices,
            },
            simple_hasher::RescuePrimeProduction,
        },
    };
    use itertools::izip;

    #[test]
    fn empty_mmr_behavior_test() {
        type Hasher = blake3::Hasher;

        let mut archival_mmr: ArchivalMmr<Hasher> = ArchivalMmr::<Hasher>::new(vec![]);
        let mut accumulator_mmr: MmrAccumulator<Hasher> = MmrAccumulator::<Hasher>::new(vec![]);

        assert_eq!(0, archival_mmr.count_leaves());
        assert_eq!(0, accumulator_mmr.count_leaves());
        assert_eq!(archival_mmr.get_peaks(), accumulator_mmr.get_peaks());
        assert_eq!(Vec::<Blake3Hash>::new(), accumulator_mmr.get_peaks());
        assert_eq!(archival_mmr.bag_peaks(), accumulator_mmr.bag_peaks());
        assert_eq!(0, archival_mmr.count_nodes());
        assert!(accumulator_mmr.is_empty());
        assert!(archival_mmr.is_empty());

        // Test behavior of appending to an empty MMR
        let new_leaf: Blake3Hash = 0xbeefu128.into();

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
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;

        // This test addresses a bug that was discovered late in the development process
        // where it was possible to fake a verification proof by providing a valid leaf
        // and authentication path but lying about the data index. This error occurred
        // because the derived hash was compared against all of the peaks to find a match
        // and it wasn't verified that the accumulated hash matched the *correct* peak.
        // This error was fixed and this test fails without that fix.
        let leaf_hashes: Vec<Blake3Hash> = vec![14u128, 15u128, 16u128]
            .iter()
            .map(|&x| x.into())
            .collect();

        let archival_mmr = ArchivalMmr::<Hasher>::new(leaf_hashes.clone());
        let (mut membership_proof, peaks): (MembershipProof<Hasher>, Vec<Digest>) =
            archival_mmr.prove_membership(0);

        // Verify that the accumulated hash in the verifier is compared against the **correct** hash,
        // not just **any** hash in the peaks list.
        assert!(membership_proof.verify(&peaks, &leaf_hashes[0], 3,).0);
        membership_proof.data_index = 2;
        assert!(!membership_proof.verify(&peaks, &leaf_hashes[0], 3,).0);
        membership_proof.data_index = 0;

        // verify the same behavior in the accumulator MMR
        let accumulator_mmr = MmrAccumulator::<Hasher>::new(leaf_hashes.clone());
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
    fn mutate_leaf_archival_test() {
        type Digest = Vec<BFieldElement>;
        type Hasher = RescuePrimeProduction;

        let rp = RescuePrimeProduction::new();
        let leaf_hashes: Vec<Vec<BFieldElement>> = (14..17)
            .map(|x| rp.hash(&vec![BFieldElement::new(x)]))
            .collect();
        let mut archival_mmr = ArchivalMmr::<Hasher>::new(leaf_hashes.clone());
        let (mp, old_peaks): (MembershipProof<Hasher>, Vec<Digest>) =
            archival_mmr.prove_membership(2);

        assert!(mp.verify(&old_peaks, &leaf_hashes[2], 3).0);
        let new_leaf = rp.hash(&vec![BFieldElement::new(10000)]);

        let mut archival_mmr_clone = archival_mmr.clone();
        archival_mmr_clone.mutate_leaf(&archival_mmr_clone.prove_membership(2).0, &new_leaf);
        let new_peaks_clone = archival_mmr_clone.get_peaks();
        archival_mmr.mutate_leaf_raw(2, new_leaf.clone());
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
            rp.hash(&vec![BFieldElement::new(14)]),
            rp.hash(&vec![BFieldElement::new(15)]),
            rp.hash(&vec![BFieldElement::new(10000)]),
        ];
        let archival_mmr_new = ArchivalMmr::<Hasher>::new(leaf_hashes_new);
        assert_eq!(archival_mmr.digests, archival_mmr_new.digests);
    }

    #[test]
    fn bag_peaks_test() {
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;

        // Verify that archival and accumulator MMR produce the same root
        // First with blake3
        let leaf_hashes_blake3: Vec<Digest> = vec![14u128, 15u128, 16u128]
            .iter()
            .map(|&x| x.into())
            .collect();
        let archival_mmr_small = ArchivalMmr::<Hasher>::new(leaf_hashes_blake3.clone());
        let accumulator_mmr_small = MmrAccumulator::<Hasher>::new(leaf_hashes_blake3);
        assert_eq!(
            archival_mmr_small.bag_peaks(),
            accumulator_mmr_small.bag_peaks()
        );
        assert_eq!(
            archival_mmr_small.bag_peaks(),
            bag_peaks::<Hasher>(&accumulator_mmr_small.get_peaks(), 4)
        );
        assert!(!accumulator_mmr_small
            .get_peaks()
            .iter()
            .any(|peak| *peak == accumulator_mmr_small.bag_peaks()));
    }

    #[test]
    fn bag_peaks_rp_test() {
        type Digest = Vec<BFieldElement>;
        type Hasher = RescuePrimeProduction;

        // Then with Rescue Prime
        let leaf_hashes_rescue_prime: Vec<Digest> =
            (14..17).map(|x| vec![BFieldElement::new(x)]).collect();
        let archival_mmr_small_rp = ArchivalMmr::<Hasher>::new(leaf_hashes_rescue_prime.clone());
        let accumulator_mmr_small_rp = MmrAccumulator::<Hasher>::new(leaf_hashes_rescue_prime);
        assert_eq!(
            archival_mmr_small_rp.bag_peaks(),
            accumulator_mmr_small_rp.bag_peaks()
        );
        assert!(!accumulator_mmr_small_rp
            .get_peaks()
            .iter()
            .any(|peak| *peak == accumulator_mmr_small_rp.bag_peaks()));
    }

    #[test]
    fn bag_peaks_blake3_test() {
        type Hasher = blake3::Hasher;

        // Then with a bigger dataset
        let leaf_hashes_bigger_blake3: Vec<Blake3Hash> =
            vec![14u128, 15u128, 16u128, 206, 1232, 123, 9989]
                .iter()
                .map(|&x| x.into())
                .collect();
        let archival_mmr_bigger = ArchivalMmr::<Hasher>::new(leaf_hashes_bigger_blake3.clone());
        let accumulator_mmr_bigger = MmrAccumulator::<Hasher>::new(leaf_hashes_bigger_blake3);
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
    fn accumulator_mmr_mutate_leaf_test() {
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;

        // Verify that upating leafs in archival and in accumulator MMR results in the same peaks
        // and verify that updating all leafs in an MMR results in the expected MMR
        for size in 1..150 {
            let new_leaf: Digest = 314159265358979u128.into();
            let leaf_hashes_blake3: Vec<Digest> = (500u128..500 + size).map(|x| x.into()).collect();

            let mut acc = MmrAccumulator::<Hasher>::new(leaf_hashes_blake3.clone());
            let mut archival = ArchivalMmr::<Hasher>::new(leaf_hashes_blake3.clone());
            let archival_end_state = ArchivalMmr::<Hasher>::new(vec![new_leaf; size as usize]);
            for i in 0..size {
                let (mp, _archival_peaks) = archival.prove_membership(i);
                assert_eq!(i, mp.data_index);
                acc.mutate_leaf(&mp, &new_leaf);
                archival.mutate_leaf_raw(i, new_leaf);
                let new_archival_peaks = archival.get_peaks();
                assert_eq!(new_archival_peaks, acc.get_peaks());
            }

            assert_eq!(archival_end_state.get_peaks(), acc.get_peaks());
        }
    }

    #[test]
    fn mmr_prove_verify_leaf_mutation_test() {
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;

        for size in 1..150 {
            let new_leaf: Digest = 314159265358979u128.into();
            let bad_leaf: Digest = 27182818284590452353u128.into();
            let leaf_hashes_blake3: Vec<Digest> = (500u128..500 + size).map(|x| x.into()).collect();
            let mut acc = MmrAccumulator::<Hasher>::new(leaf_hashes_blake3.clone());
            let mut archival = ArchivalMmr::<Hasher>::new(leaf_hashes_blake3.clone());
            let archival_end_state = ArchivalMmr::<Hasher>::new(vec![new_leaf; size as usize]);
            for i in 0..size {
                let (mp, peaks_before_update) = archival.prove_membership(i);
                assert_eq!(archival.get_peaks(), peaks_before_update);

                // Verify the update operation using the batch verifier
                archival.mutate_leaf_raw(i, new_leaf);
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
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;

        // Verify that building an MMR iteratively or in *one* function call results in the same MMR
        for size in 1..260 {
            let leaf_hashes_blake3: Vec<Digest> = (500u128..500 + size).map(|x| x.into()).collect();
            let mut archival_iterative = ArchivalMmr::<Hasher>::new(vec![]);
            let archival_batch = ArchivalMmr::<Hasher>::new(leaf_hashes_blake3.clone());
            let mut accumulator_iterative = MmrAccumulator::<Hasher>::new(vec![]);
            let accumulator_batch = MmrAccumulator::<Hasher>::new(leaf_hashes_blake3.clone());
            for (data_index, leaf_hash) in leaf_hashes_blake3.clone().into_iter().enumerate() {
                let archival_membership_proof: MembershipProof<Hasher> =
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
            let empty_accumulator = MmrAccumulator::<Hasher>::new(vec![]);
            assert!(empty_accumulator.verify_batch_update(
                &archival_batch.get_peaks(),
                &leaf_hashes_blake3,
                &[],
            ));
        }
    }

    #[test]
    fn one_input_mmr_test() {
        type Digest = Vec<BFieldElement>;
        type Hasher = RescuePrimeProduction;

        let element = vec![BFieldElement::new(14)];
        let rp = RescuePrimeProduction::new();
        let input_hash = rp.hash(&element);
        let mut mmr = ArchivalMmr::<Hasher>::new(vec![input_hash.clone()]);
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

        let new_input_hash = rp.hash(&vec![BFieldElement::new(201)]);
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
        let new_leaf: Vec<BFieldElement> = rp.hash(&vec![BFieldElement::new(987223)]);

        // When verifying the batch update with two consequtive leaf mutations, we must get the
        // membership proofs prior to all mutations. This is because the `verify_batch_update` method
        // updates the membership proofs internally to account for the mutations.
        let leaf_mutations: Vec<(Digest, MembershipProof<Hasher>)> = (0..2)
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
        type Hasher = RescuePrimeProduction;

        let values: Vec<Vec<BFieldElement>> = (0..2).map(|x| vec![BFieldElement::new(x)]).collect();
        let rp = RescuePrimeProduction::new();
        let input_hashes: Vec<Vec<BFieldElement>> = values.iter().map(|x| rp.hash(x)).collect();
        let mut mmr = ArchivalMmr::<Hasher>::new(input_hashes.clone());
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

        let new_leaf_hash: Vec<BFieldElement> = rp.hash(&vec![BFieldElement::new(201)]);
        mmr.append(new_leaf_hash.clone());
        assert_eq!(3, mmr.count_leaves());
        assert_eq!(4, mmr.count_nodes());

        for &data_index in &[0u128, 1, 2] {
            let new_leaf: Vec<BFieldElement> = rp.hash(&vec![BFieldElement::new(987223)]);
            let mp = mmr.prove_membership(data_index).0;
            mmr.mutate_leaf(&mp, &new_leaf);
            assert_eq!(new_leaf, mmr.get_leaf(data_index));
        }
    }

    #[test]
    fn variable_size_rescue_prime_mmr_test() {
        type Hasher = RescuePrimeProduction;

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
                .map(|x| vec![BFieldElement::new(x as u64 + 14)])
                .collect();
            let rp: RescuePrime = rescue_prime_params::rescue_prime_params_bfield_0();
            let input_hashes: Vec<Vec<BFieldElement>> =
                input_prehashes.iter().map(|x| rp.hash(x)).collect();
            let mut mmr = ArchivalMmr::<Hasher>::new(input_hashes.clone());
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
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;

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
            let input_prehashes: Vec<Digest> = (0..data_size).map(|x| (x + 14).into()).collect();

            let mut mmr = ArchivalMmr::<Hasher>::new(input_prehashes.clone());
            assert_eq!(data_size, mmr.count_leaves());
            assert_eq!(node_count, mmr.count_nodes());
            let original_peaks_and_heights: Vec<(Blake3Hash, u128)> = mmr.get_peaks_with_heights();
            let peak_heights_1: Vec<u128> =
                original_peaks_and_heights.iter().map(|x| x.1).collect();
            let (peak_heights_2, _) = get_peak_heights_and_peak_node_indices(data_size);
            assert_eq!(peak_heights_1, peak_heights_2);
            assert_eq!(peak_count, original_peaks_and_heights.len() as u128);

            // Get an authentication path for **all** values in MMR,
            // verify that it is valid
            for data_index in 0..data_size {
                let (mut membership_proof, peaks) = mmr.prove_membership(data_index);
                let valid_res = membership_proof.verify(
                    &peaks,
                    &input_prehashes[data_index as usize],
                    data_size,
                );
                assert!(valid_res.0);
                assert!(valid_res.1.is_some());

                let new_leaf: Blake3Hash = 98723u128.into();

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
                let old_leaf = input_prehashes[data_index as usize];
                mmr.mutate_leaf_raw(data_index, new_leaf.clone());

                let (new_mp, new_peaks) = mmr.prove_membership(data_index);
                assert!(new_mp.verify(&new_peaks, &new_leaf, data_size).0);
                assert!(!new_mp.verify(&new_peaks, &old_leaf, data_size).0);

                // Return the element to its former value and run prove/verify for membership
                mmr.mutate_leaf_raw(data_index, old_leaf.clone());
                let (old_mp, old_peaks) = mmr.prove_membership(data_index);
                assert!(!old_mp.verify(&old_peaks, &new_leaf, data_size).0);
                assert!(old_mp.verify(&old_peaks, &old_leaf, data_size).0);
            }

            // Make a new MMR where we append with a value and run the verify_append
            let new_leaf_hash: Digest = 519u128.into();
            let mmr_original: ArchivalMmr<Hasher> = mmr.clone();
            mmr.append(new_leaf_hash);
            assert!(mmr_original.verify_batch_update(&mmr.get_peaks(), &[new_leaf_hash], &[]));
        }
    }
}
