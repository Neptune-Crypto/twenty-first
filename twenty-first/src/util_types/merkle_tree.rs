use std::collections::hash_map::Entry::*;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::result;

use arbitrary::*;
use itertools::Itertools;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use thiserror::Error;

use crate::shared_math::digest::Digest;
use crate::util_types::algebraic_hasher::AlgebraicHasher;
use crate::util_types::merkle_tree_maker::MerkleTreeMaker;

/// Chosen from a very small number of benchmark runs, optimized for a slow hash function (the original Rescue Prime
/// implementation). It should probably be a higher number than 16 when using a faster hash function.
const PARALLELIZATION_THRESHOLD: usize = 16;

type Result<T> = result::Result<T, MerkleTreeError>;

/// # Design
///
/// Static methods are called from the verifier, who does not have the original `MerkleTree` object, but only partial
/// information from it, in the form of the quadruples: `(root_hash, index, digest, auth_path)`. These are exactly the
/// arguments for the `verify_*` family of static methods.
#[derive(Debug, Clone)]
pub struct MerkleTree<H>
where
    H: AlgebraicHasher,
{
    nodes: Vec<Digest>,
    _hasher: PhantomData<H>,
}

impl<H> MerkleTree<H>
where
    H: AlgebraicHasher,
{
    const MAX_NUM_NODES: usize = 1 << 32;
    const MAX_NUM_LEAVES: usize = Self::MAX_NUM_NODES / 2;
    pub const MAX_TREE_HEIGHT: usize = Self::MAX_NUM_LEAVES.ilog2() as usize;

    fn num_leaves(tree_height: usize) -> Result<usize> {
        let max_tree_height = Self::MAX_TREE_HEIGHT;
        if tree_height > max_tree_height {
            return Err(MerkleTreeError::TreeTooHigh { max_tree_height });
        }
        Ok(1 << tree_height)
    }

    /// Given a list of leaf indices, return the indices of exactly those nodes that are needed to prove (or verify)
    /// that the indicated leaves are in the Merkle tree.
    // This function is not defined as a method (taking self as argument) since it's needed by the verifier who does not
    // have access to the Merkle tree.
    fn indices_of_nodes_in_authentication_structure(
        num_nodes: usize,
        leaf_indices: &[usize],
    ) -> Result<Vec<usize>> {
        let num_leaves = num_nodes / 2;
        let root_index = 1;

        let some_index_is_invalid = leaf_indices.iter().any(|&i| i >= num_leaves);
        if some_index_is_invalid {
            return Err(MerkleTreeError::LeafIndexInvalid { num_leaves });
        }

        // The set of indices of nodes that need to be included in the authentications structure.
        // In principle, every node of every authentication path is needed. The root is never
        // needed. Hence, it is not considered in the computation below.
        let mut node_is_needed = HashSet::new();

        // The set of indices of nodes that can be computed from other nodes in the authentication
        // structure or the leafs that are explicitly supplied during verification.
        // Every node on the direct path from the leaf to the root can be computed by the very
        // nature of “authentication path”.
        let mut node_can_be_computed = HashSet::new();

        for leaf_index in leaf_indices {
            let mut node_index = leaf_index + num_leaves;
            while node_index > root_index {
                let sibling_index = node_index ^ 1;
                node_can_be_computed.insert(node_index);
                node_is_needed.insert(sibling_index);
                node_index /= 2;
            }
        }

        let set_difference = node_is_needed.difference(&node_can_be_computed).cloned();
        Ok(set_difference.sorted_unstable().rev().collect())
    }

    /// Generate a de-duplicated authentication structure for the given leaf indices.
    /// If a single index is supplied, the authentication structure is the authentication path
    /// for the indicated leaf.
    ///
    /// For example, consider the following Merkle tree.
    ///
    /// ```markdown
    ///         ──── 1 ────          ╮
    ///        ╱           ╲         │
    ///       2             3        │
    ///      ╱  ╲          ╱  ╲      ├╴ node indices
    ///     ╱    ╲        ╱    ╲     │
    ///    4      5      6      7    │
    ///   ╱ ╲    ╱ ╲    ╱ ╲    ╱ ╲   │
    ///  8   9  10 11  12 13  14 15  ╯
    ///
    ///  0   1  2   3  4   5  6   7  ←── leaf indices
    /// ```
    ///
    /// The authentication path for leaf 2, _i.e._, node 10, is nodes [11, 4, 3].
    ///
    /// The authentication structure for leaves 0 and 2, _i.e._, nodes 8 and 10 respectively, is nodes [11, 9, 3].
    /// Note how:
    /// - Node 3 is included only once, even though the individual authentication paths for leaves 0 and 2 both include
    /// node 3. This is one part of the de-duplication.
    /// - Node 4 is not included at all, even though the authentication path for leaf 2 requires the node. Instead,
    /// node 4 can be computed from nodes 8 and 9;
    /// the former is supplied explicitly for during [verification][verify],
    /// the latter is included in the authentication structure.
    /// This is the other part of the de-duplication.
    ///
    /// [verify]: Self::verify_authentication_structure
    pub fn authentication_structure(&self, leaf_indices: &[usize]) -> Result<Vec<Digest>> {
        let num_nodes = self.nodes.len();
        let indices = Self::indices_of_nodes_in_authentication_structure(num_nodes, leaf_indices)?;
        let auth_structure = indices.into_iter().map(|idx| self.nodes[idx]).collect();
        Ok(auth_structure)
    }

    /// Verify a list of indicated digests and corresponding authentication structure against a Merkle root.
    /// See also [`get_authentication_structure`][Self::get_authentication_structure].
    pub fn verify_authentication_structure(
        expected_root: Digest,
        tree_height: usize,
        leaf_indices: &[usize],
        leaf_digests: &[Digest],
        authentication_structure: &[Digest],
    ) -> bool {
        if leaf_indices.is_empty() && leaf_digests.is_empty() && authentication_structure.is_empty()
        {
            return true;
        }
        let Ok(mut partial_tree) = Self::partial_tree_from_authentication_structure(
            tree_height,
            leaf_indices,
            leaf_digests,
            authentication_structure,
        ) else {
            return false;
        };
        let Ok(()) = Self::fill_partial_tree(&mut partial_tree, tree_height, leaf_indices) else {
            return false;
        };
        let computed_root = partial_tree[&1];
        computed_root == expected_root
    }

    /// Given a list of leaf indices and corresponding digests as well as an authentication structure for a tree of
    /// indicated height, build a partial Merkle tree.
    ///
    /// Continuing the example from [`get_authentication_structure`][Self::get_authentication_structure], the partial
    /// tree for leaves 0 and 2, _i.e._, nodes 8 and 10 respectively, with nodes [11, 9, 3] from the authentication
    /// structure is:
    ///
    /// ```markdown
    ///         ──── _ ────
    ///        ╱           ╲
    ///       _             3
    ///      ╱  ╲
    ///     ╱    ╲
    ///    _      _
    ///   ╱ ╲    ╱ ╲
    ///  8   9  10 11
    /// ```
    fn partial_tree_from_authentication_structure(
        tree_height: usize,
        leaf_indices: &[usize],
        leaf_digests: &[Digest],
        authentication_structure: &[Digest],
    ) -> Result<HashMap<usize, Digest>> {
        let num_leaves = Self::num_leaves(tree_height)?;
        let num_nodes = num_leaves * 2;

        if leaf_indices.len() != leaf_digests.len() {
            return Err(MerkleTreeError::NumLeafIndicesAndDigestsMismatch);
        }
        if leaf_indices.is_empty() && authentication_structure.is_empty() {
            return Ok(HashMap::new());
        }
        if leaf_indices.iter().any(|&i| i >= num_leaves) {
            return Err(MerkleTreeError::LeafIndexInvalid { num_leaves });
        }

        let indices_of_nodes_in_authentication_structure =
            Self::indices_of_nodes_in_authentication_structure(num_nodes, leaf_indices)?;
        if authentication_structure.len() != indices_of_nodes_in_authentication_structure.len() {
            return Err(MerkleTreeError::AuthenticationPathLengthMismatch);
        }

        let mut partial_merkle_tree: HashMap<_, _> = indices_of_nodes_in_authentication_structure
            .into_iter()
            .zip(authentication_structure.iter().copied())
            .collect();

        for (leaf_index, &leaf_digest) in leaf_indices.iter().zip(leaf_digests.iter()) {
            let node_index = leaf_index + num_leaves;
            if let Vacant(entry) = partial_merkle_tree.entry(node_index) {
                entry.insert(leaf_digest);
            } else if partial_merkle_tree[&node_index] != leaf_digest {
                return Err(MerkleTreeError::RepeatedLeafDigestMismatch);
            }
        }
        Ok(partial_merkle_tree)
    }

    /// Compute all computable digests of the partial Merkle tree, modifying the given partial tree.
    /// Returns an error if the given tree is either
    /// - incomplete, _i.e._, does not contain all the nodes required to compute the root, or
    /// - not minimal, _i.e._, if it contains nodes that can be computed from other nodes.
    ///
    /// On success, the given partial tree is guaranteed to contain the root digest at index 1.
    fn fill_partial_tree(
        partial_tree: &mut HashMap<usize, Digest>,
        tree_height: usize,
        leaf_indices: &[usize],
    ) -> Result<()> {
        let num_leaves = Self::num_leaves(tree_height)?;

        // Deduplicate parent node indices to avoid hashing the same nodes twice,
        // which happens when two leaves are siblings.
        let mut parent_node_indices = leaf_indices
            .iter()
            .map(|&leaf_index| (leaf_index + num_leaves) / 2)
            .collect_vec();
        parent_node_indices.sort_unstable();
        parent_node_indices.dedup();

        // hash the partial tree from the bottom up
        for _ in 0..tree_height {
            for &parent_node_index in parent_node_indices.iter() {
                let left_node_index = parent_node_index * 2;
                let right_node_index = left_node_index ^ 1;

                if partial_tree.contains_key(&parent_node_index) {
                    return Err(MerkleTreeError::SpuriousNodeIndex(parent_node_index));
                }

                let &left_node = partial_tree
                    .get(&left_node_index)
                    .ok_or(MerkleTreeError::MissingNodeIndex(left_node_index))?;
                let &right_node = partial_tree
                    .get(&right_node_index)
                    .ok_or(MerkleTreeError::MissingNodeIndex(right_node_index))?;

                let parent_digest = H::hash_pair(left_node, right_node);
                partial_tree.insert(parent_node_index, parent_digest);
            }

            // Move parent nodes indices one layer up,
            // deduplicate to guarantee minimal number of hash operations.
            parent_node_indices.iter_mut().for_each(|i| *i /= 2);
            parent_node_indices.dedup();
        }

        if !partial_tree.contains_key(&1) {
            return Err(MerkleTreeError::RootNotFound);
        }

        Ok(())
    }

    /// Transform an authentication structure into a list of authentication paths.
    /// This corresponds to a decompression of the authentication structure.
    /// In some contexts, it is easier to deal with individual authentication paths than with the de-duplicated
    /// authentication structure.
    ///
    /// Continuing the example from [`get_authentication_structure`][Self::get_authentication_structure],
    /// the authentication structure for leaves 0 and 2, _i.e._, nodes 8 and 10 respectively, is nodes [11, 9, 3].
    ///
    /// The authentication path
    /// - for leaf 0 is [9, 5, 3], and
    /// - for leaf 2 is [11, 4, 3].
    ///
    /// ```markdown
    ///         ──── 1 ────
    ///        ╱           ╲
    ///       2             3
    ///      ╱  ╲          ╱  ╲
    ///     ╱    ╲        ╱    ╲
    ///    4      5      6      7
    ///   ╱ ╲    ╱ ╲    ╱ ╲    ╱ ╲
    ///  8   9  10 11  12 13  14 15
    /// ```
    pub fn authentication_paths_from_authentication_structure(
        tree_height: usize,
        leaf_indices: &[usize],
        leaf_digests: &[Digest],
        authentication_structure: &[Digest],
    ) -> Result<Vec<Vec<Digest>>> {
        let mut partial_tree = Self::partial_tree_from_authentication_structure(
            tree_height,
            leaf_indices,
            leaf_digests,
            authentication_structure,
        )?;
        Self::fill_partial_tree(&mut partial_tree, tree_height, leaf_indices)?;
        Self::authentication_paths_from_partial_tree(&partial_tree, tree_height, leaf_indices)
    }

    /// Given a partial Merkle tree, collect the authentication paths for the indicated leaves.
    fn authentication_paths_from_partial_tree(
        partial_tree: &HashMap<usize, Digest>,
        tree_height: usize,
        leaf_indices: &[usize],
    ) -> Result<Vec<Vec<Digest>>> {
        let mut authentication_paths = vec![];
        for &leaf_index in leaf_indices {
            let authentication_path = Self::single_authentication_path_from_partial_tree(
                partial_tree,
                tree_height,
                leaf_index,
            )?;
            authentication_paths.push(authentication_path);
        }
        Ok(authentication_paths)
    }

    /// Given a single leaf index and a partial Merkle tree, collect the authentication path for the indicated leaf.
    ///
    /// Fails if the partial Merkle tree does not contain the entire authentication path.
    fn single_authentication_path_from_partial_tree(
        partial_tree: &HashMap<usize, Digest>,
        tree_height: usize,
        leaf_index: usize,
    ) -> Result<Vec<Digest>> {
        let root_index = 1;
        let num_leaves = Self::num_leaves(tree_height)?;
        let mut authentication_path = vec![];
        let mut node_index = leaf_index + num_leaves;
        while node_index > root_index {
            let sibling_index = node_index ^ 1;
            let &sibling = partial_tree
                .get(&sibling_index)
                .ok_or(MerkleTreeError::MissingNodeIndex(sibling_index))?;
            authentication_path.push(sibling);
            node_index /= 2;
        }
        Ok(authentication_path)
    }

    pub fn root(&self) -> Digest {
        self.nodes[1]
    }

    pub fn num_leafs(&self) -> usize {
        let node_count = self.nodes.len();
        debug_assert!(node_count.is_power_of_two());
        node_count / 2
    }

    pub fn height(&self) -> usize {
        let leaf_count = self.num_leafs();
        debug_assert!(leaf_count.is_power_of_two());
        leaf_count.ilog2() as usize
    }

    /// All nodes of the Merkle tree.
    pub fn nodes(&self) -> &[Digest] {
        &self.nodes
    }

    /// The node at the given node index, if it exists.
    pub fn node(&self, index: usize) -> Option<Digest> {
        self.nodes.get(index).copied()
    }

    /// All leaves of the Merkle tree.
    pub fn leaves(&self) -> &[Digest] {
        let first_leaf = self.nodes.len() / 2;
        &self.nodes[first_leaf..]
    }

    /// The leaf at the given index, if it exists.
    pub fn leaf(&self, index: usize) -> Option<Digest> {
        let first_leaf_index = self.nodes.len() / 2;
        self.nodes.get(first_leaf_index + index).copied()
    }
}

impl<'a, H> Arbitrary<'a> for MerkleTree<H>
where
    H: AlgebraicHasher,
{
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let height = u.int_in_range(0..=13)?;
        let num_leaves = 1 << height;
        let mut leaves = Vec::with_capacity(num_leaves);
        for _ in 0..num_leaves {
            leaves.push(u.arbitrary()?);
        }

        Ok(CpuParallel::from_digests(&leaves))
    }
}

#[derive(Debug)]
pub struct CpuParallel;

impl<H: AlgebraicHasher> MerkleTreeMaker<H> for CpuParallel {
    /// Takes an array of digests and builds a MerkleTree over them.
    /// The digests are used copied over as the leaves of the tree.
    fn from_digests(digests: &[Digest]) -> MerkleTree<H> {
        let leaves_count = digests.len();

        assert!(
            leaves_count.is_power_of_two(),
            "Size of input for Merkle tree must be a power of 2"
        );

        let filler = digests[0];

        // nodes[0] is never used for anything.
        let mut nodes = vec![filler; 2 * leaves_count];
        nodes[leaves_count..(leaves_count + leaves_count)]
            .clone_from_slice(&digests[..leaves_count]);

        // Parallel digest calculations
        let mut node_count_on_this_level: usize = digests.len() / 2;
        let mut count_acc: usize = 0;
        while node_count_on_this_level >= PARALLELIZATION_THRESHOLD {
            let mut local_digests: Vec<Digest> = Vec::with_capacity(node_count_on_this_level);
            (0..node_count_on_this_level)
                .into_par_iter()
                .map(|i| {
                    let j = node_count_on_this_level + i;
                    let left_child = nodes[j * 2];
                    let right_child = nodes[j * 2 + 1];
                    H::hash_pair(left_child, right_child)
                })
                .collect_into_vec(&mut local_digests);
            nodes[node_count_on_this_level..(node_count_on_this_level + node_count_on_this_level)]
                .clone_from_slice(&local_digests[..node_count_on_this_level]);
            count_acc += node_count_on_this_level;
            node_count_on_this_level /= 2;
        }

        // Sequential digest calculations
        for i in (1..(digests.len() - count_acc)).rev() {
            nodes[i] = H::hash_pair(nodes[i * 2], nodes[i * 2 + 1]);
        }

        MerkleTree {
            nodes,
            _hasher: PhantomData,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MerkleTreeError {
    #[error("Number of leaf indices must match number of leaf digests.")]
    NumLeafIndicesAndDigestsMismatch,

    #[error("All leaf indices must be valid, i.e., less than {num_leaves}.")]
    LeafIndexInvalid { num_leaves: usize },

    #[error("The length of the supplied authentication must match the expected length.")]
    AuthenticationPathLengthMismatch,

    #[error("Leaf digests of repeated indices must be identical.")]
    RepeatedLeafDigestMismatch,

    #[error("The partial tree must be minimal. Node {0} was supplied but can be computed.")]
    SpuriousNodeIndex(usize),

    #[error("The partial tree must contain all necessary information. Node {0} is missing.")]
    MissingNodeIndex(usize),

    #[error("Could not compute the root. Maybe no leaf indices were supplied?")]
    RootNotFound,

    #[error("Tree height must not exceed {max_tree_height}.")]
    TreeTooHigh { max_tree_height: usize },
}

#[cfg(test)]
pub mod merkle_tree_test {
    use itertools::Itertools;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use rand::thread_rng;
    use rand::Rng;
    use test_strategy::proptest;

    use crate::shared_math::b_field_element::BFieldElement;
    use crate::shared_math::digest::digest_tests::DigestCorruptor;
    use crate::shared_math::other::{indices_of_set_bits, random_elements};
    use crate::shared_math::tip5::Tip5;
    use crate::shared_math::x_field_element::XFieldElement;
    use crate::util_types::shared::bag_peaks;

    use super::*;

    impl<H> MerkleTree<H>
    where
        H: AlgebraicHasher,
    {
        fn leaves_by_indices(&self, leaf_indices: &[usize]) -> Vec<Digest> {
            // test helper function, so unwrap is fine
            leaf_indices
                .iter()
                .map(|&i| self.leaf(i).unwrap())
                .collect()
        }
    }

    /// Calculate a Merkle root from a list of digests of arbitrary length.
    pub fn root_from_arbitrary_number_of_digests<H: AlgebraicHasher>(digests: &[Digest]) -> Digest {
        let mut trees = vec![];
        let mut num_processed_digests = 0;
        for tree_height in indices_of_set_bits(digests.len() as u64) {
            let num_leaves_in_tree = 1 << tree_height;
            let leaf_digests =
                &digests[num_processed_digests..num_processed_digests + num_leaves_in_tree];
            let tree: MerkleTree<H> = CpuParallel::from_digests(leaf_digests);
            num_processed_digests += num_leaves_in_tree;
            trees.push(tree);
        }
        let roots = trees.iter().map(|t| t.root()).collect_vec();
        bag_peaks::<H>(&roots)
    }

    /// Test helper to deduplicate generation of Merkle trees.
    #[derive(Debug, Clone, test_strategy::Arbitrary)]
    struct MerkleTreeToTest {
        #[strategy(arb())]
        tree: MerkleTree<Tip5>,

        #[strategy(vec(0..#tree.num_leafs(), 0..#tree.num_leafs()))]
        selected_indices: Vec<usize>,

        #[strategy(Just(#tree.authentication_structure(&#selected_indices).unwrap()))]
        auth_structure: Vec<Digest>,

        #[strategy(Just(#tree.leaves_by_indices(&#selected_indices)))]
        leaves: Vec<Digest>,
    }

    impl MerkleTreeToTest {
        fn has_non_trivial_proof(&self) -> bool {
            !self.selected_indices.is_empty()
        }
    }

    #[proptest]
    fn accessing_number_of_leaves_and_height_never_panics(
        #[strategy(arb())] merkle_tree: MerkleTree<Tip5>,
    ) {
        let _ = merkle_tree.num_leafs();
        let _ = merkle_tree.height();
    }

    #[proptest]
    fn trivial_proof_can_be_verified(#[strategy(arb())] merkle_tree: MerkleTree<Tip5>) {
        let leaf_indices = [];
        let leaf_digests = [];

        let proof = merkle_tree.authentication_structure(&leaf_indices).unwrap();
        prop_assert!(proof.is_empty());

        let verdict = MerkleTree::<Tip5>::verify_authentication_structure(
            merkle_tree.root(),
            merkle_tree.height(),
            &leaf_indices,
            &leaf_digests,
            &proof,
        );
        prop_assert!(verdict);
    }

    #[proptest]
    fn honestly_generated_authentication_structure_can_be_verified(test_tree: MerkleTreeToTest) {
        let verified = MerkleTree::<Tip5>::verify_authentication_structure(
            test_tree.tree.root(),
            test_tree.tree.height(),
            &test_tree.selected_indices,
            &test_tree.leaves,
            &test_tree.auth_structure,
        );
        prop_assert!(verified);
    }

    #[proptest]
    fn corrupt_root_cannot_be_verified(
        #[filter(#test_tree.has_non_trivial_proof())] test_tree: MerkleTreeToTest,
        corruptor: DigestCorruptor,
    ) {
        let bad_root = corruptor.corrupt_digest(test_tree.tree.root())?;
        let verified = MerkleTree::<Tip5>::verify_authentication_structure(
            bad_root,
            test_tree.tree.height(),
            &test_tree.selected_indices,
            &test_tree.leaves,
            &test_tree.auth_structure,
        );
        prop_assert!(!verified);
    }

    #[proptest]
    fn supplying_too_many_indices_leads_to_verification_failure(
        test_tree: MerkleTreeToTest,
        #[strategy(vec(0..#test_tree.tree.num_leafs(), 1..100))] spurious_indices: Vec<usize>,
    ) {
        let mut all_indices = test_tree.selected_indices.clone();
        all_indices.extend(spurious_indices);
        let verified = MerkleTree::<Tip5>::verify_authentication_structure(
            test_tree.tree.root(),
            test_tree.tree.height(),
            &all_indices,
            &test_tree.leaves,
            &test_tree.auth_structure,
        );
        prop_assert!(!verified);
    }

    #[proptest]
    fn supplying_too_few_indices_leads_to_verification_failure(
        #[filter(#test_tree.has_non_trivial_proof())] test_tree: MerkleTreeToTest,
        #[strategy(vec(0..#test_tree.selected_indices.len(), 0..#test_tree.selected_indices.len()))]
        indices_to_remove: Vec<usize>,
    ) {
        let mut all_indices = test_tree.selected_indices.clone();
        for index_to_remove in indices_to_remove {
            if all_indices.len() > index_to_remove {
                all_indices.remove(index_to_remove);
            }
        }
        if all_indices == test_tree.selected_indices {
            let reject_reason = "index manipulation unsuccessful".into();
            return Err(TestCaseError::Reject(reject_reason));
        }

        let verified = MerkleTree::<Tip5>::verify_authentication_structure(
            test_tree.tree.root(),
            test_tree.tree.height(),
            &all_indices,
            &test_tree.leaves,
            &test_tree.auth_structure,
        );
        prop_assert!(!verified);
    }

    #[proptest]
    fn corrupt_authentication_structure_leads_to_verification_failure(
        #[filter(!#test_tree.auth_structure.is_empty())] test_tree: MerkleTreeToTest,
        #[strategy(vec(0..#test_tree.auth_structure.len(), 1..=#test_tree.auth_structure.len()))]
        indices_to_corrupt: Vec<usize>,
        #[strategy(vec(any::<DigestCorruptor>(),  #indices_to_corrupt.len()))]
        digest_corruptors: Vec<DigestCorruptor>,
    ) {
        let mut corrupt_structure = test_tree.auth_structure.clone();
        for (i, digest_corruptor) in indices_to_corrupt.into_iter().zip(digest_corruptors) {
            corrupt_structure[i] = digest_corruptor.corrupt_digest(corrupt_structure[i])?;
        }
        if corrupt_structure == test_tree.auth_structure {
            let reject_reason = "corruption must change authentication structure".into();
            return Err(TestCaseError::Reject(reject_reason));
        }
        let verified = MerkleTree::<Tip5>::verify_authentication_structure(
            test_tree.tree.root(),
            test_tree.tree.height(),
            &test_tree.selected_indices,
            &test_tree.leaves,
            &corrupt_structure,
        );
        prop_assert!(!verified);
    }

    #[proptest]
    fn incorrect_tree_height_leads_to_verification_failure(
        #[filter(#test_tree.has_non_trivial_proof())] test_tree: MerkleTreeToTest,
        #[strategy(0..=MerkleTree::<Tip5>::MAX_TREE_HEIGHT)]
        #[filter(#test_tree.tree.height() != #incorrect_height)]
        incorrect_height: usize,
    ) {
        let verified = MerkleTree::<Tip5>::verify_authentication_structure(
            test_tree.tree.root(),
            incorrect_height,
            &test_tree.selected_indices,
            &test_tree.leaves,
            &test_tree.auth_structure,
        );
        prop_assert!(!verified);
    }

    #[test]
    fn verify_merkle_tree_with_duplicated_indices() {
        type H = blake3::Hasher;
        type M = CpuParallel;
        type MT = MerkleTree<H>;
        let tree_height = 5;
        let num_leaves = 1 << tree_height;
        let leaf_digests: Vec<Digest> = random_elements(num_leaves);
        let tree: MT = M::from_digests(&leaf_digests);

        let leaf_indices = [0, 5, 3, 5];
        let opened_leaves = leaf_indices.iter().map(|&i| leaf_digests[i]).collect_vec();
        let authentication_structure = tree.authentication_structure(&leaf_indices).unwrap();
        let verdict = MT::verify_authentication_structure(
            tree.root(),
            tree_height,
            &leaf_indices,
            &opened_leaves,
            &authentication_structure,
        );
        assert!(verdict, "Repeated indices must be tolerated.");

        let incorrectly_opened_leaves = [
            opened_leaves[0],
            opened_leaves[1],
            opened_leaves[2],
            opened_leaves[0],
        ];
        let verdict_for_incorrect_statement = MT::verify_authentication_structure(
            tree.root(),
            tree_height,
            &leaf_indices,
            &incorrectly_opened_leaves,
            &authentication_structure,
        );
        assert!(
            !verdict_for_incorrect_statement,
            "Repeated indices with different leaves must be rejected."
        );
    }

    #[test]
    fn verify_merkle_tree_where_every_leaf_is_revealed() {
        type H = blake3::Hasher;
        type M = CpuParallel;
        type MT = MerkleTree<H>;

        let tree_height = 5;
        let num_leaves = 1 << tree_height;
        let leaf_digests: Vec<Digest> = random_elements(num_leaves);
        let tree: MT = M::from_digests(&leaf_digests);

        let leaf_indices = (0..num_leaves).collect_vec();
        let opened_leaves = leaf_indices.iter().map(|&i| leaf_digests[i]).collect_vec();
        let authentication_structure = tree.authentication_structure(&leaf_indices).unwrap();
        assert!(authentication_structure.is_empty());

        let verdict = MT::verify_authentication_structure(
            tree.root(),
            tree_height,
            &leaf_indices,
            &opened_leaves,
            &authentication_structure,
        );
        assert!(verdict, "Revealing all leaves must be tolerated.");

        let leaf_indices_x2 = leaf_indices
            .iter()
            .chain(leaf_indices.iter())
            .copied()
            .collect_vec();
        let opened_leaves_x2 = leaf_indices_x2
            .iter()
            .map(|&i| leaf_digests[i])
            .collect_vec();
        let authentication_structure_x2 = tree.authentication_structure(&leaf_indices_x2).unwrap();
        assert!(authentication_structure_x2.is_empty());

        let verdict_x2 = MT::verify_authentication_structure(
            tree.root(),
            tree_height,
            &leaf_indices_x2,
            &opened_leaves_x2,
            &authentication_structure_x2,
        );
        assert!(verdict_x2, "Revealing all leaves twice must be tolerated.");
    }

    #[test]
    fn merkle_tree_get_authentication_path_test() {
        type H = blake3::Hasher;
        type M = CpuParallel;
        type MT = MerkleTree<H>;

        // 1: Create Merkle tree

        //     _ 1_
        //    /    \
        //   2      3
        //  / \    / \
        // 4   5  6   7
        // 0   1  2   3 <- leaf indices
        let num_leaves_a = 4;
        let leaves_a: Vec<Digest> = random_elements(num_leaves_a);
        let tree_a: MT = M::from_digests(&leaves_a);

        // 2: Get the path for some index
        let leaf_index_a = 2;
        let auth_path_a = tree_a.authentication_structure(&[leaf_index_a]).unwrap();

        let auth_path_a_len = 2;
        assert_eq!(auth_path_a_len, auth_path_a.len());
        assert_eq!(tree_a.nodes[7], auth_path_a[0]);
        assert_eq!(tree_a.nodes[2], auth_path_a[1]);

        // Also test this small Merkle tree with compressed auth paths. To get the node index
        // in the tree assign 1 to the root, 2/3 to its left/right child, and so on. To convert
        // from a leaf index to a node index, add the number of leaves. So leaf number 3 above
        // is node index 7. `x` is node index 2.
        let needed_nodes = MerkleTree::<Tip5>::indices_of_nodes_in_authentication_structure(
            tree_a.num_leafs() * 2,
            &[leaf_index_a],
        )
        .unwrap();
        assert_eq!(vec![7, 2], needed_nodes);

        // 1: Create Merkle tree
        //
        //         ──── 1 ────
        //        ╱           ╲
        //       2             3
        //      ╱  ╲          ╱  ╲
        //     ╱    ╲        ╱    ╲
        //    4      5      6      7
        //   ╱ ╲    ╱ ╲    ╱ ╲    ╱ ╲
        //  8   9  10 11  12 13  14 15
        //
        //  0   1  2   3  4   5  6   7  <- leaf indices
        let num_leaves_b = 8;
        let leaves_b: Vec<Digest> = random_elements(num_leaves_b);
        let tree_b: MT = M::from_digests(&leaves_b);

        // 2: Get the path for some index
        let leaf_index_b = 5;
        let auth_path_b = tree_b.authentication_structure(&[leaf_index_b]).unwrap();

        let auth_path_b_len = 3;
        assert_eq!(auth_path_b_len, auth_path_b.len());
        assert_eq!(tree_b.nodes[12], auth_path_b[0]);
        assert_eq!(tree_b.nodes[7], auth_path_b[1]);
        assert_eq!(tree_b.nodes[2], auth_path_b[2]);
    }

    #[test]
    fn verify_all_leaves_individually() {
        /*
        Essentially this:

        ```
        from_digests

        for each leaf:
            get ap
            verify(leaf, ap)
        ```
        */

        type H = Tip5;
        type M = CpuParallel;
        type MT = MerkleTree<H>;

        let exponent = 6;
        let num_leaves = usize::pow(2, exponent);
        assert!(
            num_leaves.is_power_of_two(),
            "Size of input for Merkle tree must be a power of 2"
        );

        let offset = 17;

        let values: Vec<[BFieldElement; 1]> = (offset..num_leaves + offset)
            .map(|i| [BFieldElement::new(i as u64)])
            .collect_vec();

        let leafs = values.iter().map(|leaf| H::hash_varlen(leaf)).collect_vec();

        let tree: MT = M::from_digests(&leafs);

        assert_eq!(
            tree.num_leafs(),
            num_leaves,
            "All leaves should have been added to the Merkle tree."
        );

        let root_hash = tree.root().to_owned();

        for (leaf_idx, leaf) in leafs.iter().enumerate() {
            let ap = tree.authentication_structure(&[leaf_idx]).unwrap();
            let verdict = MT::verify_authentication_structure(
                root_hash,
                tree.height(),
                &[leaf_idx],
                &[*leaf],
                &ap,
            );
            assert!(
                verdict,
                "Rejected: `leaf: {:?}` at `leaf_idx: {:?}` failed to verify.",
                { leaf },
                { leaf_idx }
            );
        }
    }

    #[test]
    fn verify_some_payload() {
        /// This tests that we do not confuse indices and payloads in the test `verify_all_leaves_individually`.

        type H = Tip5;
        type M = CpuParallel;
        type MT = MerkleTree<H>;

        let exponent = 6;
        let num_leaves = usize::pow(2, exponent);
        assert!(
            num_leaves.is_power_of_two(),
            "Size of input for Merkle tree must be a power of 2"
        );

        let offset = 17 * 17;

        let values: Vec<[BFieldElement; 1]> = (offset..num_leaves as u64 + offset)
            .map(|i| [BFieldElement::new(i); 1])
            .collect_vec();

        let mut leafs: Vec<Digest> = values.iter().map(|leaf| H::hash_varlen(leaf)).collect_vec();

        // A payload integrity test
        let test_leaf_idx = 42;
        let payload_offset = 317;
        let payload_leaf = vec![BFieldElement::new((test_leaf_idx + payload_offset) as u64)];

        // Embed
        leafs[test_leaf_idx] = H::hash_varlen(&payload_leaf);

        let tree: MT = M::from_digests(&leafs[..]);

        assert_eq!(
            tree.num_leafs(),
            num_leaves,
            "All leaves should have been added to the Merkle tree."
        );

        let root_hash = tree.root().to_owned();

        let ap = tree.authentication_structure(&[test_leaf_idx]).unwrap();
        let verdict = MT::verify_authentication_structure(
            root_hash,
            tree.height(),
            &[test_leaf_idx],
            &[H::hash_varlen(&payload_leaf)],
            &ap,
        );
        assert_eq!(
            tree.leaf(test_leaf_idx).unwrap(),
            H::hash_varlen(&payload_leaf)
        );
        assert!(
            verdict,
            "Rejected: `leaf: {payload_leaf:?}` at `leaf_idx: {test_leaf_idx:?}` failed to verify."
        );
    }

    #[test]
    fn root_from_odd_number_of_digests_test() {
        type H = Tip5;
        type M = CpuParallel;
        type MT = MerkleTree<H>;

        let leafs: Vec<Digest> = random_elements(128);
        let mt: MT = M::from_digests(&leafs);

        println!("Merkle root (RP 1): {:?}", mt.root());

        assert_eq!(
            mt.root(),
            root_from_arbitrary_number_of_digests::<H>(&leafs)
        );
    }

    #[test]
    fn root_from_arbitrary_number_of_digests_empty_test() {
        // Ensure that we can calculate a Merkle root from an empty list of digests.
        // This is needed since a block can contain an empty list of addition or
        // removal records.

        type H = Tip5;
        root_from_arbitrary_number_of_digests::<H>(&[]);
    }

    #[test]
    fn merkle_tree_with_xfes_as_leafs() {
        type MT = MerkleTree<Tip5>;

        let num_leaves = 128;
        let leafs: Vec<XFieldElement> = random_elements(num_leaves);
        let mt: MT = CpuParallel::from_digests(&leafs.iter().map(|&x| x.into()).collect_vec());

        let leaf_index: usize = thread_rng().gen_range(0..num_leaves);
        let path = mt.authentication_structure(&[leaf_index]).unwrap();
        let sibling = leafs[leaf_index ^ 1];
        assert_eq!(path[0], sibling.into());
    }

    #[test]
    fn build_partial_tree_from_authentication_structure_test() {
        type H = Tip5;
        type M = CpuParallel;
        type MT = MerkleTree<H>;

        let tree_height = 3;
        let num_leaves = 1 << tree_height;
        let leafs: Vec<Digest> = random_elements(num_leaves);
        let merkle_tree: MT = M::from_digests(&leafs);

        let opened_leaf_indices = [0, 2];
        let opened_leaves = opened_leaf_indices.iter().map(|&i| leafs[i]).collect_vec();
        let authentication_structure = merkle_tree
            .authentication_structure(&opened_leaf_indices)
            .unwrap();

        let partial_mt = MT::partial_tree_from_authentication_structure(
            merkle_tree.height(),
            &opened_leaf_indices,
            &opened_leaves,
            &authentication_structure,
        )
        .unwrap();

        //         ──── _ ────
        //        ╱           ╲
        //       _             3
        //      ╱  ╲
        //     ╱    ╲
        //    _      _
        //   ╱ ╲    ╱ ╲
        //  8   9  10 11
        //
        //  0      2   <-- opened_leaf_indices
        assert_eq!(5, partial_mt.len());
        assert!(partial_mt.contains_key(&3));
        assert!(partial_mt.contains_key(&8));
        assert!(partial_mt.contains_key(&9));
        assert!(partial_mt.contains_key(&10));
        assert!(partial_mt.contains_key(&11));
    }

    #[test]
    fn compute_root_of_partial_tree_test() {
        type H = Tip5;
        type MT = MerkleTree<H>;

        //         ──── _ ────
        //        ╱           ╲
        //       _             3
        //      ╱  ╲
        //     ╱    ╲
        //    _      _
        //   ╱ ╲    ╱ ╲
        //  8   9  10 11
        let tree_height = 3;
        let mut partial_tree = HashMap::new();
        partial_tree.insert(3, rand::random());
        partial_tree.insert(8, rand::random());
        partial_tree.insert(9, rand::random());
        partial_tree.insert(10, rand::random());
        partial_tree.insert(11, rand::random());
        MT::fill_partial_tree(&mut partial_tree, tree_height, &[0, 2]).unwrap();
    }

    #[test]
    fn compute_root_of_incomplete_partial_tree_test() {
        type H = Tip5;
        type MT = MerkleTree<H>;

        //         ──── _ ────
        //        ╱           ╲
        //       _             X
        //      ╱  ╲
        //     ╱    ╲
        //    _      _
        //   ╱ ╲    ╱ ╲
        //  8   9  10 11
        let tree_height = 3;
        let mut partial_tree = HashMap::new();
        partial_tree.insert(8, rand::random());
        partial_tree.insert(9, rand::random());
        partial_tree.insert(10, rand::random());
        partial_tree.insert(11, rand::random());
        let err = MT::fill_partial_tree(&mut partial_tree, tree_height, &[0, 2]).unwrap_err();
        assert_eq!(MerkleTreeError::MissingNodeIndex(3), err);
    }

    #[test]
    fn compute_root_of_non_minimal_partial_tree_test() {
        type H = Tip5;
        type MT = MerkleTree<H>;

        //         ──── _ ────
        //        ╱           ╲
        //       2 (!)         3
        //      ╱  ╲
        //     ╱    ╲
        //    _      _
        //   ╱ ╲    ╱ ╲
        //  8   9  10 11
        let tree_height = 3;
        let mut partial_tree = HashMap::new();
        partial_tree.insert(2, rand::random());
        partial_tree.insert(3, rand::random());
        partial_tree.insert(8, rand::random());
        partial_tree.insert(9, rand::random());
        partial_tree.insert(10, rand::random());
        partial_tree.insert(11, rand::random());
        let err = MT::fill_partial_tree(&mut partial_tree, tree_height, &[0, 2]).unwrap_err();
        assert_eq!(MerkleTreeError::SpuriousNodeIndex(2), err);
    }

    #[test]
    fn authentication_paths_from_authentication_structure_test() {
        type H = Tip5;
        type M = CpuParallel;
        type MT = MerkleTree<H>;

        let tree_height = 3;
        let num_leaves = 1 << tree_height;
        let leafs: Vec<Digest> = random_elements(num_leaves);
        let merkle_tree: MT = M::from_digests(&leafs);

        let opened_leaf_indices = [0, 2];
        let opened_leaves = opened_leaf_indices.iter().map(|&i| leafs[i]).collect_vec();
        let authentication_structure = merkle_tree
            .authentication_structure(&opened_leaf_indices)
            .unwrap();

        let authentication_paths = MT::authentication_paths_from_authentication_structure(
            tree_height,
            &opened_leaf_indices,
            &opened_leaves,
            &authentication_structure,
        )
        .unwrap();

        assert_eq!(2, authentication_paths.len());
        assert_eq!(tree_height, authentication_paths[0].len());
        assert_eq!(tree_height, authentication_paths[1].len());

        let nodes = merkle_tree.nodes;
        let expected_path_0 = vec![nodes[9], nodes[5], nodes[3]];
        assert_eq!(expected_path_0, authentication_paths[0]);

        let expected_path_1 = vec![nodes[11], nodes[4], nodes[3]];
        assert_eq!(expected_path_1, authentication_paths[1]);
    }
}
