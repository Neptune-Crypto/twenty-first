use std::collections::hash_map::Entry::*;
use std::collections::*;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::result;

use arbitrary::*;
use itertools::Itertools;
use rayon::prelude::*;
use thiserror::Error;

use crate::shared_math::digest::Digest;
use crate::util_types::algebraic_hasher::AlgebraicHasher;
use crate::util_types::merkle_tree_maker::MerkleTreeMaker;

/// Chosen from a very small number of benchmark runs, optimized for a slow hash function (the original Rescue Prime
/// implementation). It should probably be a higher number than 16 when using a faster hash function.
const PARALLELIZATION_THRESHOLD: usize = 16;

const ROOT_INDEX: usize = 1;

type Result<T> = result::Result<T, MerkleTreeError>;

/// # Design
///
/// Static methods are called from the verifier, who does not have the original `MerkleTree` object, but only partial
/// information from it, in the form of the quadruples: `(root_hash, index, digest, auth_path)`. These are exactly the
/// arguments for the `verify_*` family of static methods.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    ) -> Result<impl ExactSizeIterator<Item = usize>> {
        let num_leaves = num_nodes / 2;

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
            while node_index > ROOT_INDEX {
                let sibling_index = node_index ^ 1;
                node_can_be_computed.insert(node_index);
                node_is_needed.insert(sibling_index);
                node_index /= 2;
            }
        }

        let set_difference = node_is_needed.difference(&node_can_be_computed).cloned();
        Ok(set_difference.sorted_unstable().rev())
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
        let auth_structure = indices.map(|idx| self.nodes[idx]).collect();
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
        let computed_root = partial_tree[&ROOT_INDEX];
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

        if !partial_tree.contains_key(&ROOT_INDEX) {
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
        let num_leaves = Self::num_leaves(tree_height)?;
        let mut authentication_path = vec![];
        let mut node_index = leaf_index + num_leaves;
        while node_index > ROOT_INDEX {
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
        self.nodes[ROOT_INDEX]
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

        let tree = CpuParallel::from_digests(&leaves).unwrap();
        Ok(tree)
    }
}

#[derive(Debug)]
pub struct CpuParallel;

impl<H: AlgebraicHasher> MerkleTreeMaker<H> for CpuParallel {
    /// Takes an array of digests and builds a MerkleTree over them. The digests are copied as the leaves of the tree.
    fn from_digests(digests: &[Digest]) -> Result<MerkleTree<H>> {
        if digests.is_empty() {
            return Err(MerkleTreeError::TooFewLeaves);
        }

        let leaves_count = digests.len();
        if !leaves_count.is_power_of_two() {
            return Err(MerkleTreeError::IncorrectNumberOfLeaves);
        }

        // nodes[0] is never used for anything.
        let filler = Digest::default();
        let mut nodes = vec![filler; 2 * leaves_count];
        nodes[leaves_count..(leaves_count + leaves_count)]
            .clone_from_slice(&digests[..leaves_count]);

        // Parallel digest calculations
        let mut node_count_on_this_level: usize = leaves_count / 2;
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

        let tree = MerkleTree {
            nodes,
            _hasher: PhantomData,
        };
        Ok(tree)
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

    #[error("Too few leaves to build a Merkle tree.")]
    TooFewLeaves,

    #[error("The number of leaves must be a power of two.")]
    IncorrectNumberOfLeaves,

    #[error("Tree height must not exceed {max_tree_height}.")]
    TreeTooHigh { max_tree_height: usize },
}

#[cfg(test)]
pub mod merkle_tree_test {
    use itertools::Itertools;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use crate::shared_math::b_field_element::BFieldElement;
    use crate::shared_math::digest::digest_tests::DigestCorruptor;
    use crate::shared_math::tip5::Tip5;

    use super::*;

    impl<H> MerkleTree<H>
    where
        H: AlgebraicHasher,
    {
        fn test_tree_of_height(tree_height: usize) -> Self {
            let num_leaves = 1 << tree_height;
            let leaves = (0..num_leaves).map(BFieldElement::new);
            let leaf_digests = leaves.map(|bfe| H::hash_varlen(&[bfe])).collect_vec();
            let tree: MerkleTree<H> = CpuParallel::from_digests(&leaf_digests).unwrap();
            assert!(leaf_digests.iter().all_unique());
            tree
        }

        fn partial_test_tree_for_node_indices(node_indices: &[usize]) -> HashMap<usize, Digest> {
            node_indices
                .iter()
                .map(|&i| (i, BFieldElement::new(i as u64)))
                .map(|(i, leaf)| (i, Tip5::hash_varlen(&[leaf])))
                .collect()
        }

        fn leaves_by_indices(&self, leaf_indices: &[usize]) -> Vec<Digest> {
            // test helper: `.unwrap()` is fine
            leaf_indices
                .iter()
                .map(|&i| self.leaf(i).unwrap())
                .collect()
        }

        fn authentication_info_for_leaf_indices(&self, indices: Vec<usize>) -> AuthenticationInfo {
            let opened_leaves = self.leaves_by_indices(&indices);
            let auth_structure = self.authentication_structure(&indices).unwrap();
            AuthenticationInfo {
                opened_indices: indices,
                opened_leaves,
                auth_structure,
            }
        }
    }

    #[derive(Debug, Clone)]
    struct AuthenticationInfo {
        opened_indices: Vec<usize>,
        opened_leaves: Vec<Digest>,
        auth_structure: Vec<Digest>,
    }

    /// Test helper to deduplicate generation of Merkle trees.
    #[derive(Debug, Clone, test_strategy::Arbitrary)]
    pub(crate) struct MerkleTreeToTest {
        #[strategy(arb())]
        pub tree: MerkleTree<Tip5>,

        #[strategy(vec(0..#tree.num_leafs(), 0..#tree.num_leafs()))]
        pub selected_indices: Vec<usize>,

        #[strategy(Just(#tree.authentication_structure(&#selected_indices).unwrap()))]
        pub auth_structure: Vec<Digest>,

        #[strategy(Just(#tree.leaves_by_indices(&#selected_indices)))]
        pub leaves: Vec<Digest>,
    }

    impl MerkleTreeToTest {
        fn has_non_trivial_proof(&self) -> bool {
            !self.selected_indices.is_empty()
        }
    }

    #[test]
    fn building_merkle_tree_from_empty_list_of_digests_fails_with_expected_error() {
        let maybe_tree: Result<MerkleTree<Tip5>> = CpuParallel::from_digests(&[]);
        let err = maybe_tree.unwrap_err();
        assert_eq!(MerkleTreeError::TooFewLeaves, err);
    }

    #[proptest]
    fn building_merkle_tree_from_one_digest_makes_that_digest_the_root(
        #[strategy(arb())] digest: Digest,
    ) {
        let tree: MerkleTree<Tip5> = CpuParallel::from_digests(&[digest]).unwrap();
        assert_eq!(digest, tree.root());
    }

    #[proptest]
    fn building_merkle_tree_from_list_of_digests_with_incorrect_number_of_leaves_fails_with_expected_error(
        #[filter(!#num_leaves.is_power_of_two())]
        #[strategy(1_usize..1 << 13)]
        num_leaves: usize,
    ) {
        let digest = Digest::default();
        let digests = vec![digest; num_leaves];
        let maybe_tree: Result<MerkleTree<Tip5>> = CpuParallel::from_digests(&digests);
        let err = maybe_tree.unwrap_err();
        assert_eq!(MerkleTreeError::IncorrectNumberOfLeaves, err);
    }

    #[proptest(cases = 100)]
    fn accessing_number_of_leaves_and_height_never_panics(
        #[strategy(arb())] merkle_tree: MerkleTree<Tip5>,
    ) {
        let _ = merkle_tree.num_leafs();
        let _ = merkle_tree.height();
    }

    #[proptest(cases = 50)]
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

    #[proptest(cases = 40)]
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

    #[proptest(cases = 30)]
    fn corrupt_root_leads_to_verification_failure(
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

    #[proptest(cases = 50)]
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

    #[proptest(cases = 50)]
    fn supplying_too_few_indices_leads_to_verification_failure(
        #[filter(#test_tree.has_non_trivial_proof())] test_tree: MerkleTreeToTest,
        #[strategy(vec(0..#test_tree.selected_indices.len(), 1..=#test_tree.selected_indices.len()))]
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

    #[proptest(cases = 20)]
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

    #[proptest(cases = 30)]
    fn corrupt_leaf_digests_lead_to_verification_failure(
        #[filter(#test_tree.has_non_trivial_proof())] test_tree: MerkleTreeToTest,
        #[strategy(vec(0..#test_tree.leaves.len(), 1..=#test_tree.leaves.len()))]
        indices_to_corrupt: Vec<usize>,
        #[strategy(vec(any::<DigestCorruptor>(), #indices_to_corrupt.len()))]
        digest_corruptors: Vec<DigestCorruptor>,
    ) {
        let mut corrupt_leaves = test_tree.leaves.clone();
        for (&i, digest_corruptor) in indices_to_corrupt.iter().zip(&digest_corruptors) {
            corrupt_leaves[i] = digest_corruptor.corrupt_digest(corrupt_leaves[i])?;
        }
        if corrupt_leaves == test_tree.leaves {
            let reject_reason = "corruption must change leaf digest".into();
            return Err(TestCaseError::Reject(reject_reason));
        }

        let verified = MerkleTree::<Tip5>::verify_authentication_structure(
            test_tree.tree.root(),
            test_tree.tree.height(),
            &test_tree.selected_indices,
            &corrupt_leaves,
            &test_tree.auth_structure,
        );
        prop_assert!(!verified);
    }

    #[proptest(cases = 40)]
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

    /// The property-test framework can already select the same leaves multiple times. However, this
    /// a. ensures that property, making the test explicit instead of implicit, and
    /// b. ensures the property holds even for an already-generated proof.
    #[proptest(cases = 30)]
    fn honestly_generated_proof_with_duplicate_leaves_can_be_verified(
        #[filter(#test_tree.has_non_trivial_proof())] test_tree: MerkleTreeToTest,
        #[strategy(vec(0..#test_tree.selected_indices.len(), 1..=#test_tree.selected_indices.len()))]
        indices_to_duplicate: Vec<usize>,
    ) {
        let additional_indices = indices_to_duplicate
            .iter()
            .map(|&i| test_tree.selected_indices[i])
            .collect_vec();
        let all_indices = [test_tree.selected_indices.clone(), additional_indices].concat();

        let additional_leaves = indices_to_duplicate
            .into_iter()
            .map(|i| test_tree.leaves[i])
            .collect_vec();
        let all_leaves = [test_tree.leaves.clone(), additional_leaves].concat();

        let verified = MerkleTree::<Tip5>::verify_authentication_structure(
            test_tree.tree.root(),
            test_tree.tree.height(),
            &all_indices,
            &all_leaves,
            &test_tree.auth_structure,
        );
        prop_assert!(verified);
    }

    #[proptest(cases = 20)]
    fn honestly_generated_proof_with_all_leaves_revealed_can_be_verified(
        #[strategy(arb())] tree: MerkleTree<Tip5>,
    ) {
        let leaf_indices = (0..tree.num_leafs()).collect_vec();
        let proof = tree.authentication_structure(&leaf_indices).unwrap();
        let verified = MerkleTree::<Tip5>::verify_authentication_structure(
            tree.root(),
            tree.height(),
            &leaf_indices,
            tree.leaves(),
            &proof,
        );
        prop_assert!(verified);
    }

    #[test]
    fn authentication_paths_of_extremely_small_tree_use_expected_digests() {
        //     _ 1_
        //    /    \
        //   2      3
        //  / \    / \
        // 4   5  6   7
        //
        // 0   1  2   3 <- leaf indices

        let tree = MerkleTree::<Tip5>::test_tree_of_height(2);
        let auth_path_with_nodes = |indices: [usize; 2]| indices.map(|i| tree.nodes[i]).to_vec();
        let auth_path_for_leaf = |index| tree.authentication_structure(&[index]).unwrap();

        assert_eq!(auth_path_with_nodes([5, 3]), auth_path_for_leaf(0));
        assert_eq!(auth_path_with_nodes([4, 3]), auth_path_for_leaf(1));
        assert_eq!(auth_path_with_nodes([7, 2]), auth_path_for_leaf(2));
        assert_eq!(auth_path_with_nodes([6, 2]), auth_path_for_leaf(3));
    }

    #[test]
    fn authentication_paths_of_very_small_tree_use_expected_digests() {
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

        let tree = MerkleTree::<Tip5>::test_tree_of_height(3);
        let auth_path_with_nodes = |indices: [usize; 3]| indices.map(|i| tree.nodes[i]).to_vec();
        let auth_path_for_leaf = |index| tree.authentication_structure(&[index]).unwrap();

        assert_eq!(auth_path_with_nodes([9, 5, 3]), auth_path_for_leaf(0));
        assert_eq!(auth_path_with_nodes([8, 5, 3]), auth_path_for_leaf(1));
        assert_eq!(auth_path_with_nodes([11, 4, 3]), auth_path_for_leaf(2));
        assert_eq!(auth_path_with_nodes([10, 4, 3]), auth_path_for_leaf(3));
        assert_eq!(auth_path_with_nodes([13, 7, 2]), auth_path_for_leaf(4));
        assert_eq!(auth_path_with_nodes([12, 7, 2]), auth_path_for_leaf(5));
        assert_eq!(auth_path_with_nodes([15, 6, 2]), auth_path_for_leaf(6));
        assert_eq!(auth_path_with_nodes([14, 6, 2]), auth_path_for_leaf(7));
    }

    #[proptest(cases = 10)]
    fn each_leaf_can_be_verified_individually(test_tree: MerkleTreeToTest) {
        let tree = test_tree.tree;
        for (leaf_index, &leaf) in tree.leaves().iter().enumerate() {
            let authentication_path = tree.authentication_structure(&[leaf_index]).unwrap();
            let verdict = MerkleTree::<Tip5>::verify_authentication_structure(
                tree.root(),
                tree.height(),
                &[leaf_index],
                &[leaf],
                &authentication_path,
            );
            prop_assert!(verdict);
        }
    }

    #[test]
    fn partial_merkle_tree_built_from_authentication_structure_contains_expected_nodes() {
        let merkle_tree = MerkleTree::<Tip5>::test_tree_of_height(3);
        let AuthenticationInfo {
            opened_indices,
            opened_leaves,
            auth_structure,
        } = merkle_tree.authentication_info_for_leaf_indices(vec![0, 2]);

        let partial_tree = MerkleTree::<Tip5>::partial_tree_from_authentication_structure(
            merkle_tree.height(),
            &opened_indices,
            &opened_leaves,
            &auth_structure,
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

        let expected_node_indices = vec![3, 8, 9, 10, 11];
        let node_indices = partial_tree.keys().copied().sorted().collect_vec();
        assert_eq!(expected_node_indices, node_indices);
    }

    #[test]
    fn manually_constructed_partial_tree_can_be_filled() {
        //         ──── _ ───
        //        ╱           ╲
        //       _             3
        //      ╱  ╲
        //     ╱    ╲
        //    _      _
        //   ╱ ╲    ╱ ╲
        //  8   9  10 11
        //
        //  0      2   <-- opened_leaf_indices

        let tree_height = 3;
        let opened_leaf_indices = [0, 2];
        let mut partial_tree =
            MerkleTree::<Tip5>::partial_test_tree_for_node_indices(&[3, 8, 9, 10, 11]);
        MerkleTree::<Tip5>::fill_partial_tree(&mut partial_tree, tree_height, &opened_leaf_indices)
            .unwrap();
    }

    #[test]
    fn trying_to_compute_root_of_partial_tree_with_necessary_node_missing_gives_expected_error() {
        //         ──── _ ────
        //        ╱           ╲
        //       _             _ (!)
        //      ╱  ╲
        //     ╱    ╲
        //    _      _
        //   ╱ ╲    ╱ ╲
        //  8   9  10 11
        //
        //  0      2   <-- opened_leaf_indices

        let tree_height = 3;
        let opened_leaf_indices = [0, 2];
        let mut partial_tree =
            MerkleTree::<Tip5>::partial_test_tree_for_node_indices(&[8, 9, 10, 11]);

        let err = MerkleTree::<Tip5>::fill_partial_tree(
            &mut partial_tree,
            tree_height,
            &opened_leaf_indices,
        )
        .unwrap_err();
        assert_eq!(MerkleTreeError::MissingNodeIndex(3), err);
    }

    #[test]
    fn trying_to_compute_root_of_partial_tree_with_redundant_node_gives_expected_error() {
        //         ──── _ ────
        //        ╱           ╲
        //       2 (!)         3
        //      ╱  ╲
        //     ╱    ╲
        //    _      _
        //   ╱ ╲    ╱ ╲
        //  8   9  10 11
        //
        //  0      2   <-- opened_leaf_indices

        let tree_height = 3;
        let opened_leaf_indices = [0, 2];
        let mut partial_tree =
            MerkleTree::<Tip5>::partial_test_tree_for_node_indices(&[2, 3, 8, 9, 10, 11]);

        let err = MerkleTree::<Tip5>::fill_partial_tree(
            &mut partial_tree,
            tree_height,
            &opened_leaf_indices,
        )
        .unwrap_err();
        assert_eq!(MerkleTreeError::SpuriousNodeIndex(2), err);
    }

    #[test]
    fn converting_authentication_structure_to_authentication_paths_results_in_expected_paths() {
        const TREE_HEIGHT: usize = 3;
        let merkle_tree = MerkleTree::<Tip5>::test_tree_of_height(TREE_HEIGHT);
        let AuthenticationInfo {
            opened_indices,
            opened_leaves,
            auth_structure,
        } = merkle_tree.authentication_info_for_leaf_indices(vec![0, 2]);

        let auth_paths = MerkleTree::<Tip5>::authentication_paths_from_authentication_structure(
            TREE_HEIGHT,
            &opened_indices,
            &opened_leaves,
            &auth_structure,
        )
        .unwrap();

        let auth_path_with_nodes =
            |indices: [usize; TREE_HEIGHT]| indices.map(|i| merkle_tree.nodes[i]).to_vec();
        let expected_path_0 = auth_path_with_nodes([9, 5, 3]);
        let expected_path_1 = auth_path_with_nodes([11, 4, 3]);
        let expected_paths = vec![expected_path_0, expected_path_1];

        assert_eq!(expected_paths, auth_paths);
    }
}
