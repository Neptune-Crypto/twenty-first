use itertools::Itertools;

use crate::math::digest::Digest;
use crate::prelude::Tip5;
use crate::util_types::algebraic_hasher::AlgebraicHasher;
use crate::util_types::mmr::mmr_trait::LeafMutation;
use crate::util_types::shared::bag_peaks;

use crate::util_types::mmr::mmr_accumulator::MmrAccumulator;
use crate::util_types::mmr::mmr_membership_proof::MmrMembershipProof;
use crate::util_types::mmr::mmr_trait::Mmr;
use crate::util_types::mmr::shared_advanced;
use crate::util_types::mmr::shared_basic;

/// MockMmr is available for feature `mock` and for unit tests.
///
/// It implements an in-memory Archival-Mmr using a Vec<Digest> as the data
/// store.
///
/// Archival-Mmr vs Accumulator-Mmr:
///
///   An Archival-MMR stores all nodes in multiple trees whereas an
///   Accumulator-MMR only stores the roots of the binary trees.
///
/// MockMmr is not intended or tested for any kind of production usage.
///
/// A Merkle Mountain Range is a datastructure for storing a list of hashes.
///
/// Merkle Mountain Ranges only know about hashes. When values are to be
/// associated with MMRs, these values must be stored by the caller, or in a
/// wrapper to this data structure.
#[derive(Debug, Clone)]
pub struct MockMmr {
    digests: MyVec<Digest>,
}

impl Mmr for MockMmr {
    /// Calculate the root for the entire MMR
    fn bag_peaks(&self) -> Digest {
        let peaks: Vec<Digest> = self.peaks();
        bag_peaks(&peaks)
    }

    /// Return the digests of the peaks of the MMR
    fn peaks(&self) -> Vec<Digest> {
        let peaks_and_heights = self.get_peaks_with_heights();
        peaks_and_heights.into_iter().map(|x| x.0).collect()
    }

    /// Whether the MMR is empty. Note that since indexing starts at
    /// 1, the `digests` contain must always contain at least one
    /// element: a dummy digest.
    fn is_empty(&self) -> bool {
        self.digests.len() == 1
    }

    /// Return the number of leafs in the tree
    fn num_leafs(&self) -> u64 {
        let peaks_and_heights: Vec<(_, u32)> = self.get_peaks_with_heights();
        let mut acc = 0;
        for (_, height) in peaks_and_heights {
            acc += 1 << height
        }

        acc
    }

    /// Append an element to the MockMmr, return the membership proof of the newly added leaf.
    /// The membership proof is returned here since the accumulater MMR has no other way of
    /// retrieving a membership proof for a leaf. And the archival and accumulator MMR share
    /// this interface.
    fn append(&mut self, new_leaf: Digest) -> MmrMembershipProof {
        let node_index = self.digests.len();
        let leaf_index = shared_advanced::node_index_to_leaf_index(node_index).unwrap();
        self.append_raw(new_leaf);
        self.prove_membership(leaf_index)
    }

    /// Mutate an existing leaf. It is the caller's responsibility that the
    /// membership proof is valid. If the membership proof is wrong, the MMR
    /// will end up in a broken state.
    fn mutate_leaf(&mut self, leaf_mutation: LeafMutation) {
        // Sanity check
        let real_membership_proof: MmrMembershipProof =
            self.prove_membership(leaf_mutation.leaf_index);
        assert_eq!(
            real_membership_proof.authentication_path,
            leaf_mutation.membership_proof.authentication_path,
            "membership proof argument list must match internally calculated"
        );

        self.mutate_leaf_raw(leaf_mutation.leaf_index, leaf_mutation.new_leaf)
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
            self.mutate_leaf_raw(*leaf_index, *new_leaf);
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
        let accumulator: MmrAccumulator = self.to_accumulator();
        accumulator.verify_batch_update(new_peaks, appended_leafs, leaf_mutations)
    }

    fn to_accumulator(&self) -> MmrAccumulator {
        MmrAccumulator::init(self.peaks(), self.num_leafs())
    }
}

impl MockMmr {
    /// Create a new MockMmr
    pub fn new(pv: Vec<Digest>) -> Self {
        let mut ret = Self {
            digests: MyVec::from(pv),
        };
        ret.fix_dummy();
        ret
    }

    /// Inserts a dummy digest into the `digests` container. Due to
    /// 1-indexation, this structure must always contain one element
    /// (even if it is never used). Due to the persistence layer,
    /// this data structure can be set to the default vector, which
    /// is the empty vector. This method fixes that.
    pub fn fix_dummy(&mut self) {
        if self.digests.len() == 0 {
            self.digests.push(Digest::default());
        }
    }

    /// Get a leaf from the MMR, will panic if index is out of range
    pub fn get_leaf(&self, leaf_index: u64) -> Digest {
        let node_index = shared_advanced::leaf_index_to_node_index(leaf_index);
        self.digests.get(node_index)
    }

    /// Update a hash in the existing MockMmr
    pub fn mutate_leaf_raw(&mut self, leaf_index: u64, new_leaf: Digest) {
        // 1. change the leaf value
        let mut node_index = shared_advanced::leaf_index_to_node_index(leaf_index);
        self.digests.set(node_index, new_leaf);

        // 2. Calculate hash changes for all parents
        let mut parent_index = shared_advanced::parent(node_index);
        let mut acc_hash = new_leaf;

        // While parent exists in MMR, update parent
        while parent_index < self.digests.len() {
            let (right_lineage_count, height) =
                shared_advanced::right_lineage_length_and_own_height(node_index);
            acc_hash = if right_lineage_count != 0 {
                // node is right child
                Tip5::hash_pair(
                    self.digests
                        .get(shared_advanced::left_sibling(node_index, height)),
                    acc_hash,
                )
            } else {
                // node is left child
                Tip5::hash_pair(
                    acc_hash,
                    self.digests
                        .get(shared_advanced::right_sibling(node_index, height)),
                )
            };
            self.digests.set(parent_index, acc_hash);
            node_index = parent_index;
            parent_index = shared_advanced::parent(parent_index);
        }
    }

    /// Return the MMR membership proof for a leaf with a given index.
    pub fn prove_membership(&self, leaf_index: u64) -> MmrMembershipProof {
        // A proof consists of an authentication path
        // and a list of peaks
        assert!(
            leaf_index < self.num_leafs(),
            "Cannot prove membership of leaf outside of range. Got leaf_index {leaf_index}. Leaf count is {}", self.num_leafs()
        );

        // Find out how long the authentication path is
        let node_index = shared_advanced::leaf_index_to_node_index(leaf_index);
        let mut top_height: i32 = -1;
        let mut parent_index = node_index;
        while parent_index < self.digests.len() {
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
                authentication_path.push(self.digests.get(left_sibling_index));

                // parent of right child is index + 1
                index += 1;
            } else {
                // index is left child
                let right_sibling_index = shared_advanced::right_sibling(index, index_height);
                authentication_path.push(self.digests.get(right_sibling_index));

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
    pub fn get_peaks_with_heights(&self) -> Vec<(Digest, u32)> {
        if self.is_empty() {
            return vec![];
        }

        // 1. Find top peak
        // 2. Jump to right sibling (will not be included)
        // 3. Take left child of sibling, continue until a node in tree is found
        // 4. Once new node is found, jump to right sibling (will not be included)
        // 5. Take left child of sibling, continue until a node in tree is found
        let mut peaks_and_heights: Vec<(Digest, u32)> = vec![];
        let (mut top_peak, mut top_height) =
            shared_advanced::leftmost_ancestor(self.digests.len() - 1);
        if top_peak > self.digests.len() - 1 {
            top_peak = shared_basic::left_child(top_peak, top_height);
            top_height -= 1;
        }

        peaks_and_heights.push((self.digests.get(top_peak), top_height));
        let mut height = top_height;
        let mut candidate = shared_advanced::right_sibling(top_peak, height);
        'outer: while height > 0 {
            '_inner: while candidate > self.digests.len() && height > 0 {
                candidate = shared_basic::left_child(candidate, height);
                height -= 1;
                if candidate < self.digests.len() {
                    peaks_and_heights.push((self.digests.get(candidate), height));
                    candidate = shared_advanced::right_sibling(candidate, height);
                    continue 'outer;
                }
            }
        }

        peaks_and_heights
    }

    /// Append an element to the MockMmr
    pub fn append_raw(&mut self, new_leaf: Digest) {
        let node_index = self.digests.len();
        self.digests.push(new_leaf);
        let (right_parent_count, own_height) =
            shared_advanced::right_lineage_length_and_own_height(node_index);

        // This function could be rewritten with a while-loop instead of being recursive.
        if right_parent_count != 0 {
            let left_sibling_hash = self
                .digests
                .get(shared_advanced::left_sibling(node_index, own_height));
            let parent_hash: Digest = Tip5::hash_pair(left_sibling_hash, new_leaf);
            self.append_raw(parent_hash);
        }
    }

    /// Remove the last leaf from the MockMmr
    pub fn remove_last_leaf(&mut self) -> Option<Digest> {
        if self.is_empty() {
            return None;
        }

        let node_index = self.digests.len() - 1;
        let mut ret = self.digests.pop().unwrap();
        let (_, mut height) = shared_advanced::right_lineage_length_and_own_height(node_index);
        while height > 0 {
            ret = self.digests.pop().unwrap();
            height -= 1;
        }

        Some(ret)
    }
}

// a tiny wrapper around Vec<T> to make it compatible with
// StorageVec trait, as used by MockMmr.
//
// note: we do NOT impl StorageVec trait, as that is going
// to become async and we want to remain sync here.
#[derive(Debug, Clone, Default)]
struct MyVec<T>(Vec<T>);

impl<T> From<Vec<T>> for MyVec<T> {
    fn from(vec: Vec<T>) -> Self {
        Self(vec)
    }
}

impl<T: Clone> MyVec<T> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn get(&self, index: u64) -> T {
        self.0.get(index as usize).unwrap().clone()
    }

    fn set(&mut self, index: u64, value: T) {
        self.0[index as usize] = value;
    }

    fn push(&mut self, value: T) {
        self.0.push(value)
    }

    fn pop(&mut self) -> Option<T> {
        self.0.pop()
    }
}

#[cfg(test)]
mod mmr_test {
    use itertools::*;

    use rand::random;
    use test_strategy::proptest;

    use crate::math::b_field_element::BFieldElement;
    use crate::math::other::*;
    use crate::math::tip5::Tip5;

    use crate::mock::mmr::*;
    use crate::util_types::merkle_tree::merkle_tree_test::MerkleTreeToTest;
    use crate::util_types::merkle_tree::*;
    use crate::util_types::mmr::mmr_accumulator::MmrAccumulator;
    use crate::util_types::mmr::shared_advanced::get_peak_heights;
    use crate::util_types::mmr::shared_advanced::get_peak_heights_and_peak_node_indices;

    use super::*;

    impl MockMmr {
        /// Return the number of nodes in all the trees in the MMR
        fn count_nodes(&mut self) -> u64 {
            self.digests.len() - 1
        }
    }

    /// Calculate a Merkle root from a list of digests of arbitrary length.
    pub fn root_from_arbitrary_number_of_digests(digests: &[Digest]) -> Digest {
        let mut trees = vec![];
        let mut num_processed_digests = 0;
        for tree_height in get_peak_heights(digests.len() as u64) {
            let num_leafs_in_tree = 1 << tree_height;
            let leaf_digests =
                &digests[num_processed_digests..num_processed_digests + num_leafs_in_tree];
            let tree = MerkleTree::new::<CpuParallel>(leaf_digests).unwrap();
            num_processed_digests += num_leafs_in_tree;
            trees.push(tree);
        }
        let roots = trees.iter().map(|t| t.root()).collect_vec();
        bag_peaks(&roots)
    }

    #[test]
    fn computing_mmr_root_for_no_leafs_produces_some_digest() {
        root_from_arbitrary_number_of_digests(&[]);
    }

    #[proptest(cases = 30)]
    fn mmr_root_of_arbitrary_number_of_leafs_is_merkle_root_when_number_of_leafs_is_a_power_of_two(
        test_tree: MerkleTreeToTest,
    ) {
        let root = root_from_arbitrary_number_of_digests(test_tree.tree.leafs());
        assert_eq!(test_tree.tree.root(), root);
    }

    #[test]
    fn empty_mmr_behavior_test() {
        let mut archival_mmr: MockMmr = get_empty_mock_ammr();
        let mut accumulator_mmr: MmrAccumulator = MmrAccumulator::new_from_leafs(vec![]);

        assert_eq!(0, archival_mmr.num_leafs());
        assert_eq!(0, accumulator_mmr.num_leafs());
        assert_eq!(archival_mmr.peaks(), accumulator_mmr.peaks());
        assert_eq!(Vec::<Digest>::new(), accumulator_mmr.peaks());
        assert_eq!(archival_mmr.bag_peaks(), accumulator_mmr.bag_peaks());
        assert_eq!(
            archival_mmr.bag_peaks(),
            root_from_arbitrary_number_of_digests(&[]),
            "Bagged peaks for empty MMR must agree with MT root finder"
        );
        assert_eq!(0, archival_mmr.count_nodes());
        assert!(accumulator_mmr.is_empty());
        assert!(archival_mmr.is_empty());

        // Test behavior of appending to an empty MMR
        let new_leaf = random();

        let mut archival_mmr_appended = get_empty_mock_ammr();
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

        // Make the append and verify that the new peaks match the one from the proofs
        let archival_membership_proof = archival_mmr.append(new_leaf);
        let accumulator_membership_proof = accumulator_mmr.append(new_leaf);
        assert_eq!(archival_mmr.peaks(), archival_mmr_appended.peaks());
        assert_eq!(accumulator_mmr.peaks(), archival_mmr_appended.peaks());

        // Verify that the appended value matches the one stored in the MockMmr
        assert_eq!(new_leaf, archival_mmr.get_leaf(0));

        // Verify that the membership proofs for the inserted leafs are valid and that they agree
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
        // This test addresses a bug that was discovered late in the development process
        // where it was possible to fake a verification proof by providing a valid leaf
        // and authentication path but lying about the data index. This error occurred
        // because the derived hash was compared against all of the peaks to find a match
        // and it wasn't verified that the accumulated hash matched the *correct* peak.
        // This error was fixed and this test fails without that fix.
        let leaf_hashes: Vec<Digest> = random_elements(3);

        let archival_mmr: MockMmr = get_mock_ammr_from_digests(leaf_hashes.clone());
        let mp_leaf_index = 0;
        let membership_proof = archival_mmr.prove_membership(mp_leaf_index);
        let peaks = archival_mmr.peaks();

        // Verify that the accumulated hash in the verifier is compared against the **correct** hash,
        // not just **any** hash in the peaks list.
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
        type H = Tip5;

        // Create MockMmr

        let leaf_count: u64 = 3;
        let leaf_hashes: Vec<Digest> = random_elements(leaf_count as usize);
        let mut archival_mmr = get_mock_ammr_from_digests(leaf_hashes.clone());

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

        // Create copy of MockMmr, recreate membership proof

        let mut other_archival_mmr: MockMmr = get_mock_ammr_from_digests(leaf_hashes.clone());

        let mp2 = other_archival_mmr.prove_membership(leaf_index);

        // Mutate leaf + mutate leaf raw, assert that they're equivalent

        let mutated_leaf = H::hash(&BFieldElement::new(10000));
        let leaf_mutation = LeafMutation::new(leaf_index, mutated_leaf, mp2.clone());
        other_archival_mmr.mutate_leaf(leaf_mutation);

        let new_peaks_one = other_archival_mmr.peaks();
        archival_mmr.mutate_leaf_raw(leaf_index, mutated_leaf);

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

        // Create a new MockMmr with the same leaf hashes as in the
        // modified MMR, and verify that the two MMRs are equivalent

        let archival_mmr_new: MockMmr = get_mock_ammr_from_digests(leaf_hashes);
        assert_eq!(archival_mmr.digests.len(), archival_mmr_new.digests.len());

        for i in 0..leaf_count {
            assert_eq!(archival_mmr.digests.get(i), archival_mmr_new.digests.get(i));
        }
    }

    #[test]
    fn bagging_peaks_is_equivalent_for_archival_and_accumulator_mmrs() {
        let leaf_digests: Vec<Digest> = random_elements(3);
        let archival_mmr_small: MockMmr = get_mock_ammr_from_digests(leaf_digests.clone());
        let accumulator_mmr_small = MmrAccumulator::new_from_leafs(leaf_digests);
        assert_eq!(
            archival_mmr_small.bag_peaks(),
            accumulator_mmr_small.bag_peaks()
        );
        assert_eq!(
            archival_mmr_small.bag_peaks(),
            bag_peaks(&accumulator_mmr_small.peaks())
        );
        assert!(!accumulator_mmr_small
            .peaks()
            .iter()
            .any(|peak| *peak == accumulator_mmr_small.bag_peaks()));
    }

    #[test]
    fn accumulator_mmr_mutate_leaf_test() {
        // Verify that upating leafs in archival and in accumulator MMR results in the same peaks
        // and verify that updating all leafs in an MMR results in the expected MMR
        for size in 1..150 {
            let new_leaf: Digest = random();
            let leaf_digests: Vec<Digest> = random_elements(size);

            let mut acc = MmrAccumulator::new_from_leafs(leaf_digests.clone());
            let mut archival: MockMmr = get_mock_ammr_from_digests(leaf_digests.clone());
            let archival_end_state: MockMmr = get_mock_ammr_from_digests(vec![new_leaf; size]);
            for i in 0..size {
                let leaf_index = i as u64;
                let mp = archival.prove_membership(leaf_index);
                let leaf_mutation = LeafMutation::new(leaf_index, new_leaf, mp);
                acc.mutate_leaf(leaf_mutation);
                archival.mutate_leaf_raw(leaf_index, new_leaf);
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
            let mut archival: MockMmr = get_mock_ammr_from_digests(leaf_digests.clone());
            let archival_end_state: MockMmr = get_mock_ammr_from_digests(vec![new_leaf; size]);
            for i in 0..size {
                let leaf_index = i as u64;
                let mp = archival.prove_membership(leaf_index);

                // Verify the update operation using the batch verifier
                archival.mutate_leaf_raw(leaf_index, new_leaf);
                let leaf_mutation = LeafMutation::new(leaf_index, new_leaf, mp.clone());
                assert!(
                    acc.verify_batch_update(&archival.peaks(), &[], vec![leaf_mutation.clone()]),
                    "Valid batch update parameters must succeed"
                );

                let bad_leaf_mutation = LeafMutation::new(leaf_index, bad_leaf, mp);
                assert!(
                    !acc.verify_batch_update(&archival.peaks(), &[], vec![bad_leaf_mutation]),
                    "Inalid batch update parameters must fail"
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
        // Verify that building an MMR iteratively or in *one* function call results in the same MMR
        for size in 1..260 {
            let leaf_digests: Vec<Digest> = random_elements(size);
            let mut archival_iterative: MockMmr = get_mock_ammr_from_digests(vec![]);
            let archival_batch: MockMmr = get_mock_ammr_from_digests(leaf_digests.clone());
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

                // Verify that membership proofs are the same as generating them from a MockMmr
                let archival_membership_proof_direct =
                    archival_iterative.prove_membership(leaf_index);
                assert_eq!(archival_membership_proof_direct, archival_membership_proof);
            }

            // Verify that the MMRs built iteratively from `append` and in *one* batch are the same
            assert_eq!(accumulator_batch.peaks(), accumulator_iterative.peaks());
            assert_eq!(
                accumulator_batch.num_leafs(),
                accumulator_iterative.num_leafs()
            );
            assert_eq!(size as u64, accumulator_iterative.num_leafs());
            assert_eq!(archival_iterative.peaks(), accumulator_iterative.peaks());

            // Run a batch-append verification on the entire mutation of the MMR and verify that it succeeds
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
        type H = Tip5;

        let input_hash = H::hash(&BFieldElement::new(14));
        let new_input_hash = H::hash(&BFieldElement::new(201));
        let mut mmr: MockMmr = get_mock_ammr_from_digests(vec![input_hash]);
        let original_mmr: MockMmr = get_mock_ammr_from_digests(vec![input_hash]);
        let mmr_after_append: MockMmr =
            get_mock_ammr_from_digests(vec![input_hash, new_input_hash]);
        assert_eq!(1, mmr.num_leafs());
        assert_eq!(1, mmr.count_nodes());

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
        assert_eq!(3, mmr.count_nodes());

        let new_peaks_and_heights = mmr.get_peaks_with_heights();
        assert_eq!(1, new_peaks_and_heights.len());
        assert_eq!(1, new_peaks_and_heights[0].1);

        let new_peaks: Vec<Digest> = new_peaks_and_heights.iter().map(|x| x.0).collect();
        assert!(
            original_mmr.verify_batch_update(&new_peaks, &[new_input_hash], vec![]),
            "verify batch update must succeed for a single append"
        );

        // let mmr_after_append = mmr.clone();
        let new_leaf: Digest = H::hash(&BFieldElement::new(987223));

        // When verifying the batch update with two consequtive leaf mutations, we must get the
        // membership proofs prior to all mutations. This is because the `verify_batch_update` method
        // updates the membership proofs internally to account for the mutations.
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
        type H = Tip5;

        let num_leafs: u64 = 3;
        let input_digests: Vec<Digest> = random_elements(num_leafs as usize);

        let mut mmr: MockMmr = get_mock_ammr_from_digests(input_digests.clone());
        assert_eq!(num_leafs, mmr.num_leafs());
        assert_eq!(1 + num_leafs, mmr.count_nodes());

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

        let new_leaf_hash: Digest = H::hash(&BFieldElement::new(201));
        mmr.append(new_leaf_hash);

        let expected_num_leafs = 1 + num_leafs;
        assert_eq!(expected_num_leafs, mmr.num_leafs());

        let expected_node_count = 3 + expected_num_leafs;
        assert_eq!(expected_node_count, mmr.count_nodes());

        for leaf_index in 0..num_leafs {
            let new_leaf: Digest = H::hash(&BFieldElement::new(987223));
            let mp = mmr.prove_membership(leaf_index);
            let leaf_mutation = LeafMutation::new(leaf_index, new_leaf, mp);
            mmr.mutate_leaf(leaf_mutation);
            assert_eq!(new_leaf, mmr.get_leaf(leaf_index));
        }
    }

    #[test]
    fn variable_size_tip5_mmr_test() {
        type H = Tip5;

        let leaf_counts: Vec<u64> = (1..34).collect();
        let node_counts: Vec<u64> = vec![
            1, 3, 4, 7, 8, 10, 11, 15, 16, 18, 19, 22, 23, 25, 26, 31, 32, 34, 35, 38, 39, 41, 42,
            46, 47, 49, 50, 53, 54, 56, 57, 63, 64,
        ];
        let peak_counts: Vec<u64> = vec![
            1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4,
            4, 5, 1, 2,
        ];

        for (leaf_count, node_count, peak_count) in izip!(leaf_counts, node_counts, peak_counts) {
            let input_hashes: Vec<Digest> = random_elements(leaf_count as usize);
            let mut mmr: MockMmr = get_mock_ammr_from_digests(input_hashes.clone());

            assert_eq!(leaf_count, mmr.num_leafs());
            assert_eq!(node_count, mmr.count_nodes());

            let original_peaks_and_heights = mmr.get_peaks_with_heights();
            let peak_heights_1: Vec<u32> = original_peaks_and_heights.iter().map(|x| x.1).collect();

            let (peak_heights_2, _) = get_peak_heights_and_peak_node_indices(leaf_count);
            assert_eq!(peak_heights_1, peak_heights_2);

            let actual_peak_count = original_peaks_and_heights.len() as u64;
            assert_eq!(peak_count, actual_peak_count);

            // Verify that MMR root from odd number of digests and MMR bagged peaks agree
            let mmra_root = mmr.bag_peaks();
            let mt_root = root_from_arbitrary_number_of_digests(&input_hashes);

            assert_eq!(
                mmra_root, mt_root,
                "MMRA bagged peaks and MT root must agree"
            );

            // Get an authentication path for **all** values in MMR,
            // verify that it is valid
            for leaf_index in 0..leaf_count {
                let membership_proof = mmr.prove_membership(leaf_index);
                assert!(membership_proof.verify(
                    leaf_index,
                    input_hashes[leaf_index as usize],
                    &mmr.peaks(),
                    leaf_count
                ));
            }

            // // Make a new MMR where we append with a value and run the verify_append
            let new_leaf_hash = H::hash(&BFieldElement::new(201));
            let orignal_peaks = mmr.peaks();
            let mp = mmr.append(new_leaf_hash);
            let leaf_index = leaf_count;
            assert!(
                mp.verify(leaf_index, new_leaf_hash, &mmr.peaks(), leaf_count + 1),
                "Returned membership proof from append must verify"
            );
            assert_ne!(
                orignal_peaks,
                mmr.peaks(),
                "peaks must change when appending"
            );
        }
    }

    #[test]
    fn remove_last_leaf_test() {
        let input_digests: Vec<Digest> = random_elements(12);
        let mut mmr: MockMmr = get_mock_ammr_from_digests(input_digests.clone());
        assert_eq!(22, mmr.count_nodes());
        assert_eq!(Some(input_digests[11]), mmr.remove_last_leaf());
        assert_eq!(19, mmr.count_nodes());
        assert_eq!(Some(input_digests[10]), mmr.remove_last_leaf());
        assert_eq!(18, mmr.count_nodes());
        assert_eq!(Some(input_digests[9]), mmr.remove_last_leaf());
        assert_eq!(16, mmr.count_nodes());
        assert_eq!(Some(input_digests[8]), mmr.remove_last_leaf());
        assert_eq!(15, mmr.count_nodes());
        assert_eq!(Some(input_digests[7]), mmr.remove_last_leaf());
        assert_eq!(11, mmr.count_nodes());
        assert_eq!(Some(input_digests[6]), mmr.remove_last_leaf());
        assert_eq!(10, mmr.count_nodes());
        assert_eq!(Some(input_digests[5]), mmr.remove_last_leaf());
        assert_eq!(8, mmr.count_nodes());
        assert_eq!(Some(input_digests[4]), mmr.remove_last_leaf());
        assert_eq!(7, mmr.count_nodes());
        assert_eq!(Some(input_digests[3]), mmr.remove_last_leaf());
        assert_eq!(4, mmr.count_nodes());
        assert_eq!(Some(input_digests[2]), mmr.remove_last_leaf());
        assert_eq!(3, mmr.count_nodes());
        assert_eq!(Some(input_digests[1]), mmr.remove_last_leaf());
        assert_eq!(1, mmr.count_nodes());
        assert_eq!(Some(input_digests[0]), mmr.remove_last_leaf());
        assert_eq!(0, mmr.count_nodes());
        assert!(mmr.is_empty());
        assert!(mmr.remove_last_leaf().is_none());
    }

    #[test]
    fn remove_last_leaf_pbt() {
        let small_size: usize = 100;
        let big_size: usize = 350;
        let input_digests_big: Vec<Digest> = random_elements(big_size);
        let input_digests_small: Vec<Digest> = input_digests_big[0..small_size].to_vec();

        let mut mmr_small: MockMmr = get_mock_ammr_from_digests(input_digests_small);
        let mut mmr_big: MockMmr = get_mock_ammr_from_digests(input_digests_big);

        for _ in 0..(big_size - small_size) {
            mmr_big.remove_last_leaf();
        }

        assert_eq!(mmr_big.peaks(), mmr_small.peaks());
        assert_eq!(mmr_big.bag_peaks(), mmr_small.bag_peaks());
        assert_eq!(mmr_big.num_leafs(), mmr_small.num_leafs());
        assert_eq!(mmr_big.count_nodes(), mmr_small.count_nodes());
    }

    #[test]
    fn variable_size_mmr_test() {
        let node_counts: Vec<u64> = vec![
            1, 3, 4, 7, 8, 10, 11, 15, 16, 18, 19, 22, 23, 25, 26, 31, 32, 34, 35, 38, 39, 41, 42,
            46, 47, 49, 50, 53, 54, 56, 57, 63, 64,
        ];
        let peak_counts: Vec<u64> = vec![
            1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4,
            4, 5, 1, 2,
        ];
        let leaf_counts: Vec<usize> = (1..34).collect();
        for (leaf_count, node_count, peak_count) in izip!(leaf_counts, node_counts, peak_counts) {
            let size = leaf_count as u64;
            let input_digests: Vec<Digest> = random_elements(leaf_count);
            let mut mmr: MockMmr = get_mock_ammr_from_digests(input_digests.clone());
            let mmr_original: MockMmr = get_mock_ammr_from_digests(input_digests.clone());
            assert_eq!(size, mmr.num_leafs());
            assert_eq!(node_count, mmr.count_nodes());
            let original_peaks_and_heights: Vec<(Digest, u32)> = mmr.get_peaks_with_heights();
            let peak_heights_1: Vec<u32> = original_peaks_and_heights.iter().map(|x| x.1).collect();
            let (peak_heights_2, _) = get_peak_heights_and_peak_node_indices(size);
            assert_eq!(peak_heights_1, peak_heights_2);
            assert_eq!(peak_count, original_peaks_and_heights.len() as u64);

            // Verify that MMR root from odd number of digests and MMR bagged peaks agree
            let mmra_root = mmr.bag_peaks();
            let mt_root = root_from_arbitrary_number_of_digests(&input_digests);
            assert_eq!(
                mmra_root, mt_root,
                "MMRA bagged peaks and MT root must agree"
            );

            // Get an authentication path for **all** values in MMR,
            // verify that it is valid
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

                // The below verify_modify tests should only fail if `wrong_leaf_index` is
                // different than `leaf_index`.
                let wrong_leaf_index = (leaf_index + 1) % mmr.num_leafs();
                assert!(
                    wrong_leaf_index == leaf_index
                        || !membership_proof.verify(wrong_leaf_index, new_leaf, &peaks, size)
                );

                // Modify an element in the MMR and run prove/verify for membership
                let old_leaf = input_digests[leaf_index as usize];
                mmr.mutate_leaf_raw(leaf_index, new_leaf);

                let new_mp = mmr.prove_membership(leaf_index);
                let new_peaks = mmr.peaks();
                assert!(new_mp.verify(leaf_index, new_leaf, &new_peaks, size));
                assert!(!new_mp.verify(leaf_index, old_leaf, &new_peaks, size));

                // Return the element to its former value and run prove/verify for membership
                mmr.mutate_leaf_raw(leaf_index, old_leaf);
                let old_mp = mmr.prove_membership(leaf_index);
                let old_peaks = mmr.peaks();
                assert!(!old_mp.verify(leaf_index, new_leaf, &old_peaks, size));
                assert!(old_mp.verify(leaf_index, old_leaf, &old_peaks, size));
            }

            // Make a new MMR where we append with a value and run the verify_append
            let new_leaf_hash: Digest = random();
            mmr.append(new_leaf_hash);
            assert!(mmr_original.verify_batch_update(&mmr.peaks(), &[new_leaf_hash], vec![]));
        }
    }
}
