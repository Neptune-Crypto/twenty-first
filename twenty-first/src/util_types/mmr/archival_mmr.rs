//! Even though Archival Merkle Mountain Ranges are not part of the public API,
//! they exist as part of this repository (under the `test` flag) because
//! 1. they simplify testing [MmrAccumulator]s, and
//! 2. they used to be used, i.e., in order to simplify those tests, no new
//!    logic had to be written.
//!
//! Should you chose to include Archival MMRs in the public API once again,
//! please clean up this file a little. A few things come to mind immediately:
//! - many conversions that are currently `as` casts should not be
//! - a few functions are marked `pub` only in the knowledge that this module
//!   is under the `#[cfg(test)]` annotation; they shouldn't actually be `pub`
//! - this documentation comment is now out of date and has a different audience

use itertools::Itertools;

use crate::prelude::Digest;
use crate::prelude::Tip5;
use crate::util_types::mmr::mmr_accumulator::MmrAccumulator;
use crate::util_types::mmr::mmr_accumulator::bag_peaks;
use crate::util_types::mmr::mmr_membership_proof::MmrMembershipProof;
use crate::util_types::mmr::mmr_trait::LeafMutation;
use crate::util_types::mmr::mmr_trait::Mmr;
use crate::util_types::mmr::shared_advanced;
use crate::util_types::mmr::shared_basic;

/// In-memory Archival Merkle Mountain Range. For unit tests only.
///
/// An Archival-MMR stores all nodes, including leafs and internal nodes. In
/// contrast, an [`MmrAccumulator`] only stores the roots of the trees, no leafs
/// and no internal nodes.
///
/// Merkle Mountain Ranges only know about digests. When values are to be
/// associated with MMRs, these values must be stored by the caller, or in a
/// wrapper to this data structure.
#[derive(Debug, Clone)]
pub struct ArchivalMmr {
    nodes: Vec<Digest>,
}

impl Mmr for ArchivalMmr {
    fn bag_peaks(&self) -> Digest {
        bag_peaks(&self.peaks(), self.num_leafs())
    }

    fn peaks(&self) -> Vec<Digest> {
        let peaks_and_heights = self.get_peaks_with_heights();
        peaks_and_heights.into_iter().map(|x| x.0).collect()
    }

    // Note that since indexing starts at 1, the `digests` contain must always
    // contain at least one element: a dummy digest.
    fn is_empty(&self) -> bool {
        self.nodes.len() == 1
    }

    fn num_leafs(&self) -> u64 {
        let peaks_and_heights: Vec<(_, u32)> = self.get_peaks_with_heights();
        let mut acc = 0;
        for (_, height) in peaks_and_heights {
            acc += 1 << height
        }

        acc
    }

    // The membership proof is returned here since
    // 1. the archival MMR and MMR accumulator share this interface, and
    // 2. the MMR accumulator has no other way of retrieving a membership proof
    //    for a leaf.
    fn append(&mut self, new_leaf: Digest) -> MmrMembershipProof {
        let node_index = self.nodes.len() as u64;
        let leaf_index = shared_advanced::node_index_to_leaf_index(node_index).unwrap();
        self.append_internal(new_leaf);
        self.prove_membership(leaf_index)
    }

    fn mutate_leaf(&mut self, leaf_mutation: LeafMutation) {
        // Sanity check
        let real_membership_proof = self.prove_membership(leaf_mutation.leaf_index);
        assert_eq!(
            real_membership_proof.authentication_path,
            leaf_mutation.membership_proof.authentication_path,
            "membership proof argument list must match internally calculated"
        );

        self.mutate_leaf_unchecked(leaf_mutation.leaf_index, leaf_mutation.new_leaf)
    }

    fn batch_mutate_leaf_and_update_mps(
        &mut self,
        membership_proofs: &mut [&mut MmrMembershipProof],
        membership_proof_leaf_indices: &[u64],
        leaf_mutations: Vec<LeafMutation>,
    ) -> Vec<usize> {
        assert!(
            leaf_mutations
                .iter()
                .map(|leaf_mutation| leaf_mutation.leaf_index)
                .all_unique(),
            "Duplicated leafs are not allowed in membership proof updater"
        );

        for LeafMutation {
            leaf_index,
            new_leaf,
            ..
        } in leaf_mutations.iter()
        {
            self.mutate_leaf_unchecked(*leaf_index, *new_leaf);
        }

        let mut modified_mps: Vec<usize> = vec![];
        for (i, (mp, mp_leaf_index)) in membership_proofs
            .iter_mut()
            .zip_eq(membership_proof_leaf_indices)
            .enumerate()
        {
            let new_mp = self.prove_membership(*mp_leaf_index);
            if new_mp != **mp {
                modified_mps.push(i);
            }

            **mp = new_mp
        }

        modified_mps
    }

    fn verify_batch_update(
        &self,
        new_peaks: &[Digest],
        appended_leafs: &[Digest],
        leaf_mutations: Vec<LeafMutation>,
    ) -> bool {
        self.to_accumulator()
            .verify_batch_update(new_peaks, appended_leafs, leaf_mutations)
    }

    fn to_accumulator(&self) -> MmrAccumulator {
        MmrAccumulator::init(self.peaks(), self.num_leafs())
    }
}

impl Default for ArchivalMmr {
    fn default() -> Self {
        Self::new_from_leafs(Vec::new())
    }
}

impl ArchivalMmr {
    /// Create a new
    pub fn new_from_leafs(leafs: Vec<Digest>) -> Self {
        // Insert a dummy digest. Due to 1-indexation, the nodes must always
        // contain at least one element, even if it is never used.
        let nodes = vec![Digest::default()];
        let mut mmr = Self { nodes };
        for leaf in leafs {
            mmr.append_internal(leaf);
        }

        mmr
    }

    /// The number of nodes in the MMR.
    fn num_nodes(&self) -> u64 {
        self.nodes.len() as u64 - 1
    }

    /// Get a leaf from the MMR.
    ///
    /// Panic if index is out of range.
    fn get_leaf(&self, leaf_index: u64) -> Digest {
        let node_index = shared_advanced::leaf_index_to_node_index(leaf_index);
        self.nodes[node_index as usize]
    }

    /// Update a hash in the existing MMR.
    ///
    /// Unlike [mutate_leaf](Mmr::mutate_leaf), does not require the leaf's
    /// corresponding authentication path.
    pub fn mutate_leaf_unchecked(&mut self, leaf_index: u64, new_leaf: Digest) {
        // 1. change the leaf value
        let mut node_index = shared_advanced::leaf_index_to_node_index(leaf_index);
        self.nodes[node_index as usize] = new_leaf;

        // 2. Calculate hash changes for all parents
        let mut parent_index = shared_advanced::parent(node_index);
        let mut acc_hash = new_leaf;

        // While parent exists in MMR, update parent
        while parent_index < self.nodes.len() as u64 {
            let (right_lineage_count, height) =
                shared_advanced::right_lineage_length_and_own_height(node_index);
            acc_hash = if right_lineage_count != 0 {
                // node is right child
                let left_sibling =
                    self.nodes[shared_advanced::left_sibling(node_index, height) as usize];
                Tip5::hash_pair(left_sibling, acc_hash)
            } else {
                // node is left child
                let right_sibling =
                    self.nodes[shared_advanced::right_sibling(node_index, height) as usize];
                Tip5::hash_pair(acc_hash, right_sibling)
            };
            self.nodes[parent_index as usize] = acc_hash;
            node_index = parent_index;
            parent_index = shared_advanced::parent(parent_index);
        }
    }

    /// Return the MMR membership proof for a leaf with a given index.
    pub fn prove_membership(&self, leaf_index: u64) -> MmrMembershipProof {
        // A proof consists of an authentication path and a list of peaks
        assert!(
            leaf_index < self.num_leafs(),
            "Cannot prove membership of leaf outside of range. \
            Leaf index: {leaf_index}. Number of leafs: {}",
            self.num_leafs()
        );

        // Find out how long the authentication path is
        let node_index = shared_advanced::leaf_index_to_node_index(leaf_index);
        let mut top_height: i32 = -1;
        let mut parent_index = node_index;
        while parent_index < self.nodes.len() as u64 {
            parent_index = shared_advanced::parent(parent_index);
            top_height += 1;
        }

        // Build the authentication path
        let mut authentication_path: Vec<Digest> = vec![];
        let mut index = node_index;
        let (mut right_ancestor_count, mut index_height): (u32, u32) =
            shared_advanced::right_lineage_length_and_own_height(index);
        while index_height < top_height as u32 {
            if right_ancestor_count != 0 {
                // index is right child
                let left_sibling_index = shared_advanced::left_sibling(index, index_height);
                authentication_path.push(self.nodes[left_sibling_index as usize]);

                // parent of right child is index + 1
                index += 1;
            } else {
                // index is left child
                let right_sibling_index = shared_advanced::right_sibling(index, index_height);
                authentication_path.push(self.nodes[right_sibling_index as usize]);

                // parent of left child:
                index += 1 << (index_height + 1);
            }
            let next_index_info = shared_advanced::right_lineage_length_and_own_height(index);
            right_ancestor_count = next_index_info.0;
            index_height = next_index_info.1;
        }

        MmrMembershipProof::new(authentication_path)
    }

    /// Return a list of tuples (peaks, height)
    fn get_peaks_with_heights(&self) -> Vec<(Digest, u32)> {
        if self.is_empty() {
            return vec![];
        }

        // 1. Find top peak
        // 2. Jump to right sibling (will not be included)
        // 3. Take left child of sibling, continue until a node in tree is found
        // 4. Once new node is found, jump to right sibling (will not be
        //    included)
        // 5. Take left child of sibling, continue until a node in tree is found
        let mut peaks_and_heights: Vec<(Digest, u32)> = vec![];
        let (mut top_peak, mut top_height) = shared_advanced::leftmost_ancestor(self.num_nodes());
        if top_peak > self.num_nodes() {
            top_peak = shared_basic::left_child(top_peak, top_height);
            top_height -= 1;
        }

        peaks_and_heights.push((self.nodes[top_peak as usize], top_height));
        let mut height = top_height;
        let mut candidate = shared_advanced::right_sibling(top_peak, height);
        'outer: while height > 0 {
            while candidate > self.nodes.len() as u64 && height > 0 {
                candidate = shared_basic::left_child(candidate, height);
                height -= 1;
                if candidate < self.nodes.len() as u64 {
                    peaks_and_heights.push((self.nodes[candidate as usize], height));
                    candidate = shared_advanced::right_sibling(candidate, height);
                    continue 'outer;
                }
            }
        }

        peaks_and_heights
    }

    /// Append an element to the MMR.
    fn append_internal(&mut self, new_leaf: Digest) {
        let node_index = self.nodes.len();
        self.nodes.push(new_leaf);
        let (right_parent_count, own_height) =
            shared_advanced::right_lineage_length_and_own_height(node_index as u64);

        // This function could be rewritten with a while-loop instead of being
        // recursive.
        if right_parent_count != 0 {
            let left_sibling_hash =
                self.nodes[shared_advanced::left_sibling(node_index as u64, own_height) as usize];
            let parent_hash: Digest = Tip5::hash_pair(left_sibling_hash, new_leaf);
            self.append_internal(parent_hash);
        }
    }
}

mod tests {
    use std::collections::HashSet;

    use itertools::Itertools;
    use itertools::izip;
    use rand::Rng;
    use rand::random;

    use super::*;
    use crate::math::b_field_element::BFieldElement;
    use crate::math::other::random_elements;
    use crate::util_types::mmr::mmr_accumulator::util::mmra_with_mps;
    use crate::util_types::mmr::shared_advanced::get_peak_heights_and_peak_node_indices;

    #[test]
    fn empty_mmr_behavior_test() {
        let mut archival_mmr = ArchivalMmr::default();
        let mut accumulator_mmr = MmrAccumulator::new_from_leafs(vec![]);

        assert_eq!(0, archival_mmr.num_leafs());
        assert_eq!(0, accumulator_mmr.num_leafs());
        assert_eq!(archival_mmr.peaks(), accumulator_mmr.peaks());
        assert_eq!(Vec::<Digest>::new(), accumulator_mmr.peaks());
        assert_eq!(archival_mmr.bag_peaks(), accumulator_mmr.bag_peaks());
        assert_eq!(
            archival_mmr.bag_peaks(),
            MmrAccumulator::new_from_leafs(vec![]).bag_peaks(),
        );
        assert_eq!(0, archival_mmr.num_nodes());
        assert!(accumulator_mmr.is_empty());
        assert!(archival_mmr.is_empty());

        // Test behavior of appending to an empty MMR
        let new_leaf = random();

        let mut archival_mmr_appended = ArchivalMmr::default();
        {
            let archival_membership_proof = archival_mmr_appended.append(new_leaf);

            // Verify that the MMR update can be validated
            assert!(archival_mmr.verify_batch_update(
                &archival_mmr_appended.peaks(),
                &[new_leaf],
                vec![],
            ));

            // Verify that failing MMR update for empty MMR fails gracefully
            let leaf_mutations = vec![LeafMutation::new(0, new_leaf, archival_membership_proof)];
            assert!(!archival_mmr.verify_batch_update(
                &archival_mmr_appended.peaks(),
                &[],
                leaf_mutations,
            ));
        }

        // Make the append and verify that the new peaks match the one from the
        // proofs
        let archival_membership_proof = archival_mmr.append(new_leaf);
        let accumulator_membership_proof = accumulator_mmr.append(new_leaf);
        assert_eq!(archival_mmr.peaks(), archival_mmr_appended.peaks());
        assert_eq!(accumulator_mmr.peaks(), archival_mmr_appended.peaks());

        // Verify that the appended value matches the one stored in the archival
        // MMR
        assert_eq!(new_leaf, archival_mmr.get_leaf(0));

        // Verify that the membership proofs for the inserted leafs are valid
        // and that they agree
        assert_eq!(
            archival_membership_proof, accumulator_membership_proof,
            "accumulator and archival membership proofs must agree"
        );
        assert!(
            archival_membership_proof.verify(
                0,
                new_leaf,
                &archival_mmr.peaks(),
                archival_mmr.num_leafs(),
            ),
            "membership proof from arhival MMR must validate"
        );
    }

    #[test]
    fn verify_against_correct_peak_test() {
        // This test addresses a bug that was discovered late in the development
        // process where it was possible to fake a verification proof by
        // providing a valid leaf and authentication path but lying about the
        // data index. This error occurred because the derived hash was compared
        // against all the peaks to find a match, and it wasn't verified that
        // the accumulated hash matched the *correct* peak. This error was fixed
        // and this test fails without that fix.
        let leaf_hashes: Vec<Digest> = random_elements(3);

        let archival_mmr = ArchivalMmr::new_from_leafs(leaf_hashes.clone());
        let mp_leaf_index = 0;
        let membership_proof = archival_mmr.prove_membership(mp_leaf_index);
        let peaks = archival_mmr.peaks();

        // Verify that the accumulated hash in the verifier is compared against
        // the **correct** hash, not just **any** hash in the peaks list.
        assert!(membership_proof.verify(mp_leaf_index, leaf_hashes[0], &peaks, 3));
        assert!(!membership_proof.verify(2, leaf_hashes[0], &peaks, 3));

        // verify the same behavior in the accumulator MMR
        let accumulator_mmr = MmrAccumulator::new_from_leafs(leaf_hashes.clone());
        assert!(membership_proof.verify(
            mp_leaf_index,
            leaf_hashes[0],
            &accumulator_mmr.peaks(),
            accumulator_mmr.num_leafs(),
        ));
        assert!(!membership_proof.verify(
            2,
            leaf_hashes[0],
            &accumulator_mmr.peaks(),
            accumulator_mmr.num_leafs(),
        ));
    }

    #[test]
    fn mutate_leaf_archival_test() {
        let leaf_count: u64 = 3;
        let leaf_hashes: Vec<Digest> = random_elements(leaf_count as usize);
        let mut archival_mmr = ArchivalMmr::new_from_leafs(leaf_hashes.clone());

        let leaf_index: u64 = 2;
        let mp1: MmrMembershipProof = archival_mmr.prove_membership(leaf_index);
        let old_peaks = archival_mmr.peaks();

        // Verify single leaf
        let mp1_verifies = mp1.verify(
            leaf_index,
            leaf_hashes[leaf_index as usize],
            &old_peaks,
            leaf_count,
        );
        assert!(mp1_verifies);

        // Create copy of archival MMR, recreate membership proof
        let mut other_archival_mmr = ArchivalMmr::new_from_leafs(leaf_hashes.clone());
        let mp2 = other_archival_mmr.prove_membership(leaf_index);

        // Mutate leaf + mutate leaf raw, assert that they're equivalent
        let mutated_leaf = Tip5::hash(&BFieldElement::new(10000));
        let leaf_mutation = LeafMutation::new(leaf_index, mutated_leaf, mp2.clone());
        other_archival_mmr.mutate_leaf(leaf_mutation);

        let new_peaks_one = other_archival_mmr.peaks();
        archival_mmr.mutate_leaf_unchecked(leaf_index, mutated_leaf);

        let new_peaks_two = archival_mmr.peaks();
        assert_eq!(
            new_peaks_two, new_peaks_one,
            "peaks for two update leaf method calls must agree"
        );

        // Verify that peaks have changed as expected

        let expected_num_peaks = 2;
        assert_ne!(old_peaks[1], new_peaks_two[1]);
        assert_eq!(old_peaks[0], new_peaks_two[0]);
        assert_eq!(expected_num_peaks, new_peaks_two.len());
        assert_eq!(expected_num_peaks, old_peaks.len());

        let mp2_verifies_non_mutated_leaf = mp2.verify(
            leaf_index,
            leaf_hashes[leaf_index as usize],
            &new_peaks_two,
            leaf_count,
        );
        assert!(!mp2_verifies_non_mutated_leaf);

        let mp2_verifies_mutated_leaf =
            mp2.verify(leaf_index, mutated_leaf, &new_peaks_two, leaf_count);
        assert!(mp2_verifies_mutated_leaf);

        // Create a new archival MMR with the same leaf hashes as in the
        // modified MMR accumulator, and verify that the two MMRs are equivalent

        let archival_mmr_new = ArchivalMmr::new_from_leafs(leaf_hashes);
        assert_eq!(archival_mmr.nodes.len(), archival_mmr_new.nodes.len());

        for i in 0..leaf_count as usize {
            assert_eq!(archival_mmr.nodes[i], archival_mmr_new.nodes[i]);
        }
    }

    #[test]
    fn bagging_peaks_is_equivalent_for_archival_and_accumulator_mmrs() {
        let leaf_digests: Vec<Digest> = random_elements(3);
        let leafs = leaf_digests.clone();
        let archival_mmr_small = ArchivalMmr::new_from_leafs(leafs);
        let accumulator_mmr_small = MmrAccumulator::new_from_leafs(leaf_digests);
        assert_eq!(
            archival_mmr_small.bag_peaks(),
            accumulator_mmr_small.bag_peaks()
        );

        // bagged peaks must never correspond to any peak
        let bag = accumulator_mmr_small.bag_peaks();
        let no_peak_is_bag = accumulator_mmr_small
            .peaks()
            .into_iter()
            .all(|peak| peak != bag);
        assert!(no_peak_is_bag);
    }

    #[test]
    fn accumulator_mmr_mutate_leaf_test() {
        // Verify that updating leafs in archival and in accumulator MMR results
        // in the same peaks and verify that updating all leafs in an MMR
        // results in the expected MMR
        for size in 1..150 {
            let new_leaf: Digest = random();
            let leaf_digests: Vec<Digest> = random_elements(size);

            let mut acc = MmrAccumulator::new_from_leafs(leaf_digests.clone());
            let mut archival = ArchivalMmr::new_from_leafs(leaf_digests.clone());
            let leafs = vec![new_leaf; size];
            let archival_end_state = ArchivalMmr::new_from_leafs(leafs);
            for i in 0..size {
                let leaf_index = i as u64;
                let mp = archival.prove_membership(leaf_index);
                let leaf_mutation = LeafMutation::new(leaf_index, new_leaf, mp);
                acc.mutate_leaf(leaf_mutation);
                archival.mutate_leaf_unchecked(leaf_index, new_leaf);
                let new_archival_peaks = archival.peaks();
                assert_eq!(new_archival_peaks, acc.peaks());
            }

            assert_eq!(archival_end_state.peaks(), acc.peaks());
        }
    }

    #[test]
    fn mmr_prove_verify_leaf_mutation_test() {
        for size in 1..150 {
            let new_leaf: Digest = random();
            let bad_leaf: Digest = random();
            let leaf_digests: Vec<Digest> = random_elements(size);
            let mut acc = MmrAccumulator::new_from_leafs(leaf_digests.clone());
            let mut archival = ArchivalMmr::new_from_leafs(leaf_digests.clone());
            let archival_end_state = ArchivalMmr::new_from_leafs(vec![new_leaf; size]);
            for i in 0..size {
                let leaf_index = i as u64;
                let mp = archival.prove_membership(leaf_index);

                // Verify the update operation using the batch verifier
                archival.mutate_leaf_unchecked(leaf_index, new_leaf);
                let leaf_mutation = LeafMutation::new(leaf_index, new_leaf, mp.clone());
                assert!(
                    acc.verify_batch_update(&archival.peaks(), &[], vec![leaf_mutation.clone()]),
                    "Valid batch update parameters must succeed"
                );

                let bad_leaf_mutation = LeafMutation::new(leaf_index, bad_leaf, mp);
                assert!(
                    !acc.verify_batch_update(&archival.peaks(), &[], vec![bad_leaf_mutation]),
                    "Invalid batch update parameters must fail"
                );

                acc.mutate_leaf(leaf_mutation);
                let new_archival_peaks = archival.peaks();
                assert_eq!(new_archival_peaks, acc.peaks());
                assert_eq!(size as u64, archival.num_leafs());
                assert_eq!(size as u64, acc.num_leafs());
            }
            assert_eq!(archival_end_state.peaks(), acc.peaks());
        }
    }

    #[test]
    fn mmr_append_test() {
        // Verify that building an MMR iteratively or in *one* function call
        // results in the same MMR
        for size in 1..260 {
            let leaf_digests: Vec<Digest> = random_elements(size);
            let leafs = vec![];
            let mut archival_iterative = ArchivalMmr::new_from_leafs(leafs);
            let archival_batch = ArchivalMmr::new_from_leafs(leaf_digests.clone());
            let mut accumulator_iterative = MmrAccumulator::new_from_leafs(vec![]);
            let accumulator_batch = MmrAccumulator::new_from_leafs(leaf_digests.clone());
            for (leaf_index, leaf_hash) in leaf_digests.clone().into_iter().enumerate() {
                let leaf_index = leaf_index as u64;
                let archival_membership_proof: MmrMembershipProof =
                    archival_iterative.append(leaf_hash);
                let accumulator_membership_proof = accumulator_iterative.append(leaf_hash);

                // Verify membership proofs returned from the append operation
                assert_eq!(
                    accumulator_membership_proof, archival_membership_proof,
                    "membership proofs from append operation must agree"
                );
                assert!(archival_membership_proof.verify(
                    leaf_index,
                    leaf_hash,
                    &archival_iterative.peaks(),
                    archival_iterative.num_leafs(),
                ));

                // Verify that membership proofs are the same as generating them
                // from an archival MMR
                let archival_membership_proof_direct =
                    archival_iterative.prove_membership(leaf_index);
                assert_eq!(archival_membership_proof_direct, archival_membership_proof);
            }

            // Verify that the MMRs built iteratively from `append` and in *one*
            // batch are the same
            assert_eq!(accumulator_batch.peaks(), accumulator_iterative.peaks());
            assert_eq!(
                accumulator_batch.num_leafs(),
                accumulator_iterative.num_leafs()
            );
            assert_eq!(size as u64, accumulator_iterative.num_leafs());
            assert_eq!(archival_iterative.peaks(), accumulator_iterative.peaks());

            // Run a batch-append verification on the entire mutation of the MMR
            // and verify that it succeeds
            let empty_accumulator = MmrAccumulator::new_from_leafs(vec![]);
            assert!(empty_accumulator.verify_batch_update(
                &archival_batch.peaks(),
                &leaf_digests,
                vec![],
            ));
        }
    }

    #[test]
    fn one_input_mmr_test() {
        let input_hash = Tip5::hash(&BFieldElement::new(14));
        let new_input_hash = Tip5::hash(&BFieldElement::new(201));
        let mut mmr = ArchivalMmr::new_from_leafs(vec![input_hash]);
        let original_mmr = ArchivalMmr::new_from_leafs(vec![input_hash]);
        let leafs = vec![input_hash, new_input_hash];
        let mmr_after_append = ArchivalMmr::new_from_leafs(leafs);
        assert_eq!(1, mmr.num_leafs());
        assert_eq!(1, mmr.num_nodes());

        let original_peaks_and_heights: Vec<(Digest, u32)> = mmr.get_peaks_with_heights();
        assert_eq!(1, original_peaks_and_heights.len());
        assert_eq!(0, original_peaks_and_heights[0].1);

        {
            let leaf_index = 0;
            let leaf_count = 1;
            let membership_proof = mmr.prove_membership(leaf_index);
            assert!(membership_proof.verify(leaf_index, input_hash, &mmr.peaks(), leaf_count));
        }

        mmr.append(new_input_hash);
        assert_eq!(2, mmr.num_leafs());
        assert_eq!(3, mmr.num_nodes());

        let new_peaks_and_heights = mmr.get_peaks_with_heights();
        assert_eq!(1, new_peaks_and_heights.len());
        assert_eq!(1, new_peaks_and_heights[0].1);

        let new_peaks: Vec<Digest> = new_peaks_and_heights.iter().map(|x| x.0).collect();
        assert!(
            original_mmr.verify_batch_update(&new_peaks, &[new_input_hash], vec![]),
            "verify batch update must succeed for a single append"
        );

        // let mmr_after_append = mmr.clone();
        let new_leaf: Digest = Tip5::hash(&BFieldElement::new(987223));

        // When verifying the batch update with two consecutive leaf mutations,
        // we must get the membership proofs prior to all mutations. This is
        // because the `verify_batch_update` method updates the membership
        // proofs internally to account for the mutations.
        for &leaf_index in &[0u64, 1] {
            let mp = mmr.prove_membership(leaf_index);
            let leaf_mutation = LeafMutation::new(leaf_index, new_leaf, mp);
            mmr.mutate_leaf(leaf_mutation);
            assert_eq!(
                new_leaf,
                mmr.get_leaf(leaf_index),
                "fetched leaf must match what we put in"
            );
        }

        let mps = (0..2)
            .map(|i| mmr_after_append.prove_membership(i))
            .collect_vec();
        let leaf_mutations = (0..2)
            .map(|i| LeafMutation::new(i, new_leaf, mps[i as usize].clone()))
            .collect_vec();
        assert!(
            mmr_after_append.verify_batch_update(&mmr.peaks(), &[], leaf_mutations),
            "The batch update of two leaf mutations must verify"
        );
    }

    #[test]
    fn two_input_mmr_test() {
        let num_leafs: u64 = 3;
        let input_digests: Vec<Digest> = random_elements(num_leafs as usize);

        let leafs = input_digests.clone();
        let mut mmr = ArchivalMmr::new_from_leafs(leafs);
        assert_eq!(num_leafs, mmr.num_leafs());
        assert_eq!(1 + num_leafs, mmr.num_nodes());

        let original_peaks_and_heights: Vec<(Digest, u32)> = mmr.get_peaks_with_heights();
        let expected_peaks = 2;
        assert_eq!(expected_peaks, original_peaks_and_heights.len());

        {
            let leaf_index: u64 = 0;
            let input_digest = input_digests[leaf_index as usize];
            let membership_proof = mmr.prove_membership(leaf_index);
            let peaks = mmr.peaks();

            assert!(membership_proof.verify(leaf_index, input_digest, &peaks, num_leafs));

            // Negative test for verify membership
            let leaf_index = leaf_index + 1;

            assert!(!membership_proof.verify(leaf_index, input_digest, &peaks, num_leafs));
        }

        let new_leaf_hash: Digest = Tip5::hash(&BFieldElement::new(201));
        mmr.append(new_leaf_hash);

        let expected_num_leafs = 1 + num_leafs;
        assert_eq!(expected_num_leafs, mmr.num_leafs());

        let expected_node_count = 3 + expected_num_leafs;
        assert_eq!(expected_node_count, mmr.num_nodes());

        for leaf_index in 0..num_leafs {
            let new_leaf: Digest = Tip5::hash(&BFieldElement::new(987223));
            let mp = mmr.prove_membership(leaf_index);
            let leaf_mutation = LeafMutation::new(leaf_index, new_leaf, mp);
            mmr.mutate_leaf(leaf_mutation);
            assert_eq!(new_leaf, mmr.get_leaf(leaf_index));
        }
    }

    #[test]
    fn variable_size_tip5_mmr_test() {
        let leaf_counts = (1..34).collect_vec();
        let node_counts = [
            1, 3, 4, 7, 8, 10, 11, 15, 16, 18, 19, 22, 23, 25, 26, 31, 32, 34, 35, 38, 39, 41, 42,
            46, 47, 49, 50, 53, 54, 56, 57, 63, 64,
        ];
        let peak_counts = [
            1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4,
            4, 5, 1, 2,
        ];

        for (leaf_count, node_count, peak_count) in izip!(leaf_counts, node_counts, peak_counts) {
            let input_hashes: Vec<Digest> = random_elements(leaf_count as usize);
            let leafs = input_hashes.clone();
            let mut ammr = ArchivalMmr::new_from_leafs(leafs);

            assert_eq!(leaf_count, ammr.num_leafs());
            assert_eq!(node_count, ammr.num_nodes());

            let original_peaks_and_heights = ammr.get_peaks_with_heights();
            let peak_heights_1: Vec<u32> = original_peaks_and_heights.iter().map(|x| x.1).collect();

            let (peak_heights_2, _) = get_peak_heights_and_peak_node_indices(leaf_count);
            assert_eq!(peak_heights_1, peak_heights_2);

            let actual_peak_count = original_peaks_and_heights.len() as u64;
            assert_eq!(peak_count, actual_peak_count);

            // Verify that MMR root from odd number of digests and MMR bagged
            // peaks agree
            let ammr_root = ammr.bag_peaks();
            let mmra_root = MmrAccumulator::new_from_leafs(input_hashes.clone()).bag_peaks();
            assert_eq!(ammr_root, mmra_root);

            // Get an authentication path for **all** values in MMR, verify that
            // it is valid
            for leaf_index in 0..leaf_count {
                let membership_proof = ammr.prove_membership(leaf_index);
                assert!(membership_proof.verify(
                    leaf_index,
                    input_hashes[leaf_index as usize],
                    &ammr.peaks(),
                    leaf_count
                ));
            }

            // Make a new MMR where we append with a value and run the
            // verify_append
            let new_leaf_hash = Tip5::hash(&BFieldElement::new(201));
            let original_peaks = ammr.peaks();
            let mp = ammr.append(new_leaf_hash);
            let leaf_index = leaf_count;
            assert!(
                mp.verify(leaf_index, new_leaf_hash, &ammr.peaks(), leaf_count + 1),
                "Returned membership proof from append must verify"
            );
            assert_ne!(
                original_peaks,
                ammr.peaks(),
                "peaks must change when appending"
            );
        }
    }

    #[test]
    fn variable_size_mmr_test() {
        let node_counts = [
            1, 3, 4, 7, 8, 10, 11, 15, 16, 18, 19, 22, 23, 25, 26, 31, 32, 34, 35, 38, 39, 41, 42,
            46, 47, 49, 50, 53, 54, 56, 57, 63, 64,
        ];
        let peak_counts = [
            1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4,
            4, 5, 1, 2,
        ];
        let leaf_counts: Vec<usize> = (1..34).collect();
        for (leaf_count, node_count, peak_count) in izip!(leaf_counts, node_counts, peak_counts) {
            let size = leaf_count as u64;
            let input_digests: Vec<Digest> = random_elements(leaf_count);
            let mut mmr = ArchivalMmr::new_from_leafs(input_digests.clone());
            let mmr_original = ArchivalMmr::new_from_leafs(input_digests.clone());
            assert_eq!(size, mmr.num_leafs());
            assert_eq!(node_count, mmr.num_nodes());
            let original_peaks_and_heights: Vec<(Digest, u32)> = mmr.get_peaks_with_heights();
            let peak_heights_1: Vec<u32> = original_peaks_and_heights.iter().map(|x| x.1).collect();
            let (peak_heights_2, _) = get_peak_heights_and_peak_node_indices(size);
            assert_eq!(peak_heights_1, peak_heights_2);
            assert_eq!(peak_count, original_peaks_and_heights.len() as u64);

            // Verify that MMR root from odd number of digests and MMR bagged
            //peaks agree
            let ammr_root = mmr.bag_peaks();
            let mmra_root = MmrAccumulator::new_from_leafs(input_digests.clone()).bag_peaks();
            assert_eq!(ammr_root, mmra_root);

            // Get an authentication path for **all** values in MMR, verify that
            // it is valid
            for leaf_index in 0..size {
                let membership_proof = mmr.prove_membership(leaf_index);
                let peaks = mmr.peaks();
                assert!(membership_proof.verify(
                    leaf_index,
                    input_digests[leaf_index as usize],
                    &peaks,
                    size
                ));

                let new_leaf: Digest = random();

                // The below verify_modify tests should only fail if
                // `wrong_leaf_index` is different than `leaf_index`.
                let wrong_leaf_index = (leaf_index + 1) % mmr.num_leafs();
                assert!(
                    wrong_leaf_index == leaf_index
                        || !membership_proof.verify(wrong_leaf_index, new_leaf, &peaks, size)
                );

                // Modify an element in the MMR and run prove/verify for
                // membership
                let old_leaf = input_digests[leaf_index as usize];
                mmr.mutate_leaf_unchecked(leaf_index, new_leaf);

                let new_mp = mmr.prove_membership(leaf_index);
                let new_peaks = mmr.peaks();
                assert!(new_mp.verify(leaf_index, new_leaf, &new_peaks, size));
                assert!(!new_mp.verify(leaf_index, old_leaf, &new_peaks, size));

                // Return the element to its former value and run prove/verify
                // for membership
                mmr.mutate_leaf_unchecked(leaf_index, old_leaf);
                let old_mp = mmr.prove_membership(leaf_index);
                let old_peaks = mmr.peaks();
                assert!(!old_mp.verify(leaf_index, new_leaf, &old_peaks, size));
                assert!(old_mp.verify(leaf_index, old_leaf, &old_peaks, size));
            }

            // Make a new MMR where we append with a value and run the
            // verify_append
            let new_leaf_hash: Digest = random();
            mmr.append(new_leaf_hash);
            assert!(mmr_original.verify_batch_update(&mmr.peaks(), &[new_leaf_hash], vec![]));
        }
    }

    #[test]
    #[should_panic]
    fn disallow_repeated_leaf_indices_in_construction() {
        mmra_with_mps(14, vec![(0, random()), (0, random())]);
    }

    #[test]
    fn mmra_and_mps_construct_test_cornercases() {
        let mut rng = rand::rng();
        for leaf_count in 0..5 {
            let (_mmra, _mps) = mmra_with_mps(leaf_count, vec![]);
        }
        let some: Digest = rng.random();
        for leaf_count in 1..10 {
            for leaf_index in 0..leaf_count {
                let (mmra, mps) = mmra_with_mps(leaf_count, vec![(leaf_index, some)]);
                assert!(mps[0].verify(leaf_index, some, &mmra.peaks(), leaf_count));
            }
        }

        let other: Digest = rng.random();
        for leaf_count in 2..10 {
            for some_index in 0..leaf_count {
                for other_index in 0..leaf_count {
                    if some_index == other_index {
                        continue;
                    }
                    let (mmra, mps) =
                        mmra_with_mps(leaf_count, vec![(some_index, some), (other_index, other)]);
                    assert!(mps[0].verify(some_index, some, &mmra.peaks(), leaf_count));
                    assert!(mps[1].verify(other_index, other, &mmra.peaks(), leaf_count));
                }
            }
        }

        // Full specification, set *all* leafs in MMR explicitly.
        for leaf_count in 3..10 {
            let specifications = (0..leaf_count).map(|i| (i, random())).collect_vec();
            let (mmra, mps) = mmra_with_mps(leaf_count, specifications.clone());
            for (mp, (leaf_index, leaf)) in mps.iter().zip(specifications) {
                assert!(mp.verify(leaf_index, leaf, &mmra.peaks(), leaf_count));
            }
        }
    }

    #[test]
    fn mmra_and_mps_construct_test_small() {
        let mut rng = rand::rng();
        let digest_leaf_idx12: Digest = rng.random();
        let digest_leaf_idx14: Digest = rng.random();

        let (mmra, mps) = mmra_with_mps(32, vec![(12, digest_leaf_idx12), (14, digest_leaf_idx14)]);
        assert!(mps[0].verify(12, digest_leaf_idx12, &mmra.peaks(), mmra.num_leafs()));
        assert!(mps[1].verify(14, digest_leaf_idx14, &mmra.peaks(), mmra.num_leafs()));
    }

    #[test]
    fn mmra_and_mps_construct_test_pbt() {
        let mut rng = rand::rng();

        for leaf_count in 2..25 {
            for specified_count in 0..leaf_count {
                let mut specified_indices: HashSet<u64> = HashSet::default();
                for _ in 0..specified_count {
                    specified_indices.insert(rng.random_range(0..leaf_count));
                }

                let collected_values = specified_indices.len();
                let specified_leafs: Vec<(u64, Digest)> = specified_indices
                    .into_iter()
                    .zip_eq(random_elements(collected_values))
                    .collect_vec();
                let (mmra, mps) = mmra_with_mps(leaf_count, specified_leafs.clone());

                for (mp, (leaf_idx, leaf)) in mps.iter().zip_eq(specified_leafs) {
                    assert!(mp.verify(leaf_idx, leaf, &mmra.peaks(), leaf_count));
                }
            }
        }
    }

    #[test]
    fn mmra_and_mps_construct_test_big() {
        let mut rng = rand::rng();
        let leaf_count = (1 << 59) + (1 << 44) + 1234567890;
        let specified_count = 40;
        let mut specified_indices: HashSet<u64> = HashSet::default();
        for _ in 0..specified_count {
            specified_indices.insert(rng.random_range(0..leaf_count));
        }

        let collected_values = specified_indices.len();
        let specified_leafs: Vec<(u64, Digest)> = specified_indices
            .into_iter()
            .zip_eq(random_elements(collected_values))
            .collect_vec();
        let (mmra, mps) = mmra_with_mps(leaf_count, specified_leafs.clone());

        for (mp, (leaf_idx, leaf)) in mps.iter().zip_eq(specified_leafs) {
            assert!(mp.verify(leaf_idx, leaf, &mmra.peaks(), leaf_count));
        }
    }
}
