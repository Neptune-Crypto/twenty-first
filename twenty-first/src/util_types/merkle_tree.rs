use std::collections::hash_map::Entry::*;
use std::collections::*;
use std::env;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::result;

use arbitrary::*;
use itertools::Itertools;
use lazy_static::lazy_static;
use rayon::prelude::*;
use thiserror::Error;

use crate::shared_math::digest::Digest;
use crate::util_types::algebraic_hasher::AlgebraicHasher;
use crate::util_types::merkle_tree_maker::MerkleTreeMaker;

const DEFAULT_PARALLELIZATION_CUTOFF: usize = 256;

lazy_static! {
    static ref PARALLELIZATION_CUTOFF: usize = env::var("MERKLE_TREE_PARALLELIZATION_CUTOFF")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_PARALLELIZATION_CUTOFF);
}

const MAX_NUM_NODES: usize = 1 << 32;
const MAX_NUM_LEAVES: usize = MAX_NUM_NODES / 2;
pub const MAX_TREE_HEIGHT: usize = MAX_NUM_LEAVES.ilog2() as usize;

const ROOT_INDEX: usize = 1;

type Result<T> = result::Result<T, MerkleTreeError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MerkleTree<H>
where
    H: AlgebraicHasher,
{
    nodes: Vec<Digest>,
    _hasher: PhantomData<H>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MerkleTreeInclusionProof<H>
where
    H: AlgebraicHasher,
{
    pub tree_height: usize,

    // Purposefully not a HashMap to preserve order of the keys.
    pub indexed_leaves: Vec<(usize, Digest)>,
    pub authentication_structure: Vec<Digest>,
    pub _hasher: PhantomData<H>,
}

/// Helper struct for verifying inclusion of items in a Merkle tree.
///
/// Continuing the example from [`authentication_structure`][authentication_structure], the partial
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
///
/// [authentication_structure]: MerkleTree::authentication_structure
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct PartialMerkleTree<H>
where
    H: AlgebraicHasher,
{
    tree_height: usize,
    leaf_indices: Vec<usize>,
    nodes: HashMap<usize, Digest>,
    _hasher: PhantomData<H>,
}

impl<H> MerkleTree<H>
where
    H: AlgebraicHasher,
{
    /// Given a list of leaf indices, return the indices of exactly those nodes that are needed to prove (or verify)
    /// that the indicated leaves are in the Merkle tree.
    // This function is not defined as a method (taking self as argument) since it's needed by the verifier, who does
    // not have access to the Merkle tree.
    fn authentication_structure_node_indices(
        num_leaves: usize,
        leaf_indices: &[usize],
    ) -> Result<impl ExactSizeIterator<Item = usize>> {
        // The set of indices of nodes that need to be included in the authentications structure.
        // In principle, every node of every authentication path is needed. The root is never
        // needed. Hence, it is not considered in the computation below.
        let mut node_is_needed = HashSet::new();

        // The set of indices of nodes that can be computed from other nodes in the authentication
        // structure or the leafs that are explicitly supplied during verification.
        // Every node on the direct path from the leaf to the root can be computed by the very
        // nature of “authentication path”.
        let mut node_can_be_computed = HashSet::new();

        for &leaf_index in leaf_indices {
            if leaf_index >= num_leaves {
                return Err(MerkleTreeError::LeafIndexInvalid { num_leaves });
            }

            let mut node_index = leaf_index + num_leaves;
            while node_index > ROOT_INDEX {
                let sibling_index = node_index ^ 1;
                node_can_be_computed.insert(node_index);
                node_is_needed.insert(sibling_index);
                node_index /= 2;
            }
        }

        let set_difference = node_is_needed.difference(&node_can_be_computed).copied();
        Ok(set_difference.sorted_unstable().rev())
    }

    /// Generate a de-duplicated authentication structure for the given leaf indices.
    /// If a single index is supplied, the authentication structure is the authentication path for the indicated leaf.
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
    /// the former is supplied explicitly during [verification][verify],
    /// the latter is included in the authentication structure.
    /// This is the other part of the de-duplication.
    ///
    /// [verify]: MerkleTreeInclusionProof::verify
    pub fn authentication_structure(&self, leaf_indices: &[usize]) -> Result<Vec<Digest>> {
        let num_leafs = self.num_leafs();
        let indices = Self::authentication_structure_node_indices(num_leafs, leaf_indices)?;
        let auth_structure = indices.map(|idx| self.nodes[idx]).collect();
        Ok(auth_structure)
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

    pub fn indexed_leaves(&self, indices: &[usize]) -> Result<Vec<(usize, Digest)>> {
        let num_leaves = self.num_leafs();
        if indices.iter().any(|&i| i >= num_leaves) {
            return Err(MerkleTreeError::LeafIndexInvalid { num_leaves });
        }
        let indexed_leaves = indices.iter().copied().map(|i| (i, self.leaf(i).unwrap()));
        Ok(indexed_leaves.collect())
    }

    /// A full inclusion proof for the leaves at the supplied indices, including the leaves. Generally, using
    /// [`authentication_structure`](Self::authentication_structure) is preferable. Use this method only if the
    /// verifier needs explicit access to the leaves, _i.e._, cannot compute them from other information.
    pub fn inclusion_proof_for_leaf_indices(
        &self,
        indices: &[usize],
    ) -> MerkleTreeInclusionProof<H> {
        MerkleTreeInclusionProof {
            tree_height: self.height(),
            indexed_leaves: self.indexed_leaves(indices).unwrap(),
            authentication_structure: self.authentication_structure(indices).unwrap(),
            _hasher: PhantomData,
        }
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

impl<H> MerkleTreeInclusionProof<H>
where
    H: AlgebraicHasher,
{
    fn leaf_indices(&self) -> impl Iterator<Item = &usize> {
        self.indexed_leaves.iter().map(|(index, _)| index)
    }

    fn is_trivial(&self) -> bool {
        self.indexed_leaves.is_empty() && self.authentication_structure.is_empty()
    }

    /// Verify that the given root digest is the root of a Merkle tree that contains the indicated leaves.
    pub fn verify(self, expected_root: Digest) -> bool {
        if self.is_trivial() {
            return true;
        }
        let Ok(partial_tree) = PartialMerkleTree::try_from(self) else {
            return false;
        };
        let Ok(computed_root) = partial_tree.root() else {
            return false;
        };
        computed_root == expected_root
    }

    /// Transform the inclusion proof into a list of authentication paths.
    ///
    /// This corresponds to a decompression of the authentication structure.
    /// In some contexts, it is easier to deal with individual authentication paths than with the de-duplicated
    /// authentication structure.
    ///
    /// Continuing the example from [`authentication_structure`][authentication_structure],
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
    ///
    /// [authentication_structure]: MerkleTree::authentication_structure
    pub fn into_authentication_paths(self) -> Result<Vec<Vec<Digest>>> {
        let partial_tree = PartialMerkleTree::try_from(self)?;
        partial_tree.into_authentication_paths()
    }
}

impl<H> PartialMerkleTree<H>
where
    H: AlgebraicHasher,
{
    pub fn root(&self) -> Result<Digest> {
        self.nodes
            .get(&ROOT_INDEX)
            .copied()
            .ok_or(MerkleTreeError::RootNotFound)
    }

    fn num_leaves(&self) -> Result<usize> {
        if self.tree_height > MAX_TREE_HEIGHT {
            return Err(MerkleTreeError::TreeTooHigh);
        }
        Ok(1 << self.tree_height)
    }

    /// Compute all computable digests of the partial Merkle tree, modifying self. Returns an error if self is either
    /// - incomplete, _i.e._, does not contain all the nodes required to compute the root, or
    /// - not minimal, _i.e._, if it contains nodes that can be computed from other nodes.
    ///
    /// On success, [`root()`](Self::root) is guaranteed to return `Ok(…)`.
    pub fn fill(&mut self) -> Result<()> {
        let num_leaves = self.num_leaves()?;

        // De-duplicate parent node indices to avoid hashing the same nodes twice,
        // which happens when two leaves are siblings.
        let mut parent_node_indices = self
            .leaf_indices
            .iter()
            .map(|&leaf_index| (leaf_index + num_leaves) / 2)
            .collect_vec();
        parent_node_indices.sort_unstable();
        parent_node_indices.dedup();

        // hash the partial tree from the bottom up
        for _ in 0..self.tree_height {
            for &parent_node_index in parent_node_indices.iter() {
                let left_node_index = parent_node_index * 2;
                let right_node_index = left_node_index ^ 1;

                if self.nodes.contains_key(&parent_node_index) {
                    return Err(MerkleTreeError::SpuriousNodeIndex(parent_node_index));
                }

                let &left_node = self
                    .nodes
                    .get(&left_node_index)
                    .ok_or(MerkleTreeError::MissingNodeIndex(left_node_index))?;
                let &right_node = self
                    .nodes
                    .get(&right_node_index)
                    .ok_or(MerkleTreeError::MissingNodeIndex(right_node_index))?;

                let parent_digest = H::hash_pair(left_node, right_node);
                self.nodes.insert(parent_node_index, parent_digest);
            }

            // Move parent nodes indices one layer up,
            // deduplicate to guarantee minimal number of hash operations.
            parent_node_indices.iter_mut().for_each(|i| *i /= 2);
            parent_node_indices.dedup();
        }

        if !self.nodes.contains_key(&ROOT_INDEX) {
            return Err(MerkleTreeError::RootNotFound);
        }

        Ok(())
    }

    /// Collect all individual authentication paths for the indicated leaves.
    fn into_authentication_paths(self) -> Result<Vec<Vec<Digest>>> {
        self.leaf_indices
            .iter()
            .map(|&i| self.authentication_path_for_index(i))
            .collect()
    }

    /// Given a single leaf index and a partial Merkle tree, collect the authentication path for the indicated leaf.
    ///
    /// Fails if the partial Merkle tree does not contain the entire authentication path.
    fn authentication_path_for_index(&self, leaf_index: usize) -> Result<Vec<Digest>> {
        let num_leaves = self.num_leaves()?;
        let mut authentication_path = vec![];
        let mut node_index = leaf_index + num_leaves;
        while node_index > ROOT_INDEX {
            let sibling_index = node_index ^ 1;
            let &sibling = self
                .nodes
                .get(&sibling_index)
                .ok_or(MerkleTreeError::MissingNodeIndex(sibling_index))?;
            authentication_path.push(sibling);
            node_index /= 2;
        }
        Ok(authentication_path)
    }
}

impl<H> TryFrom<MerkleTreeInclusionProof<H>> for PartialMerkleTree<H>
where
    H: AlgebraicHasher,
{
    type Error = MerkleTreeError;

    fn try_from(proof: MerkleTreeInclusionProof<H>) -> Result<Self> {
        let leaf_indices = proof.leaf_indices().copied().collect();
        let mut partial_tree = PartialMerkleTree {
            tree_height: proof.tree_height,
            leaf_indices,
            nodes: HashMap::new(),
            _hasher: PhantomData,
        };

        let num_leaves = partial_tree.num_leaves()?;
        if proof.leaf_indices().any(|&i| i >= num_leaves) {
            return Err(MerkleTreeError::LeafIndexInvalid { num_leaves });
        }

        let node_indices = MerkleTree::<H>::authentication_structure_node_indices(
            num_leaves,
            &partial_tree.leaf_indices,
        )?;
        if proof.authentication_structure.len() != node_indices.len() {
            return Err(MerkleTreeError::AuthenticationStructureLengthMismatch);
        }

        let mut nodes: HashMap<_, _> = node_indices
            .zip_eq(proof.authentication_structure)
            .collect();

        for (leaf_index, leaf_digest) in proof.indexed_leaves {
            let node_index = leaf_index + num_leaves;
            if let Vacant(entry) = nodes.entry(node_index) {
                entry.insert(leaf_digest);
            } else if nodes[&node_index] != leaf_digest {
                return Err(MerkleTreeError::RepeatedLeafDigestMismatch);
            }
        }

        partial_tree.nodes = nodes;
        partial_tree.fill()?;
        Ok(partial_tree)
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
        while node_count_on_this_level >= *PARALLELIZATION_CUTOFF {
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
    #[error("All leaf indices must be valid, i.e., less than {num_leaves}.")]
    LeafIndexInvalid { num_leaves: usize },

    #[error("The length of the supplied authentication structure must match the expected length.")]
    AuthenticationStructureLengthMismatch,

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

    #[error("Tree height must not exceed {MAX_TREE_HEIGHT}.")]
    TreeTooHigh,
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

    impl MerkleTree<Tip5> {
        fn test_tree_of_height(tree_height: usize) -> Self {
            let num_leaves = 1 << tree_height;
            let leaves = (0..num_leaves).map(BFieldElement::new);
            let leaf_digests = leaves.map(|bfe| Tip5::hash_varlen(&[bfe])).collect_vec();
            let tree = CpuParallel::from_digests(&leaf_digests).unwrap();
            assert!(leaf_digests.iter().all_unique());
            tree
        }
    }

    impl PartialMerkleTree<Tip5> {
        fn dummy_nodes_for_indices(node_indices: &[usize]) -> HashMap<usize, Digest> {
            node_indices
                .iter()
                .map(|&i| (i, BFieldElement::new(i as u64)))
                .map(|(i, leaf)| (i, Tip5::hash_varlen(&[leaf])))
                .collect()
        }
    }

    /// Test helper to deduplicate generation of Merkle trees.
    #[derive(Debug, Clone, test_strategy::Arbitrary)]
    pub(crate) struct MerkleTreeToTest {
        #[strategy(arb())]
        pub tree: MerkleTree<Tip5>,

        #[strategy(vec(0..#tree.num_leafs(), 0..#tree.num_leafs()))]
        pub selected_indices: Vec<usize>,
    }

    impl MerkleTreeToTest {
        fn has_non_trivial_proof(&self) -> bool {
            !self.selected_indices.is_empty()
        }

        fn proof(&self) -> MerkleTreeInclusionProof<Tip5> {
            self.tree
                .inclusion_proof_for_leaf_indices(&self.selected_indices)
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
        let proof = merkle_tree.inclusion_proof_for_leaf_indices(&[]);
        prop_assert!(proof.authentication_structure.is_empty());
        let verdict = proof.verify(merkle_tree.root());
        prop_assert!(verdict);
    }

    #[proptest(cases = 40)]
    fn honestly_generated_authentication_structure_can_be_verified(test_tree: MerkleTreeToTest) {
        let proof = test_tree.proof();
        let verdict = proof.verify(test_tree.tree.root());
        prop_assert!(verdict);
    }

    #[proptest(cases = 30)]
    fn corrupt_root_leads_to_verification_failure(
        #[filter(#test_tree.has_non_trivial_proof())] test_tree: MerkleTreeToTest,
        corruptor: DigestCorruptor,
    ) {
        let bad_root = corruptor.corrupt_digest(test_tree.tree.root())?;
        let proof = test_tree.proof();
        let verdict = proof.verify(bad_root);
        prop_assert!(!verdict);
    }

    #[proptest(cases = 20)]
    fn corrupt_authentication_structure_leads_to_verification_failure(
        #[filter(!#test_tree.proof().authentication_structure.is_empty())]
        test_tree: MerkleTreeToTest,
        #[strategy(Just(#test_tree.proof().authentication_structure.len()))]
        _num_auth_structure_entries: usize,
        #[strategy(vec(0..#_num_auth_structure_entries, 1..=#_num_auth_structure_entries))]
        indices_to_corrupt: Vec<usize>,
        #[strategy(vec(any::<DigestCorruptor>(),  #indices_to_corrupt.len()))]
        digest_corruptors: Vec<DigestCorruptor>,
    ) {
        let mut proof = test_tree.proof();
        for (i, digest_corruptor) in indices_to_corrupt.into_iter().zip_eq(digest_corruptors) {
            proof.authentication_structure[i] =
                digest_corruptor.corrupt_digest(proof.authentication_structure[i])?;
        }
        if proof == test_tree.proof() {
            let reject_reason = "corruption must change authentication structure".into();
            return Err(TestCaseError::Reject(reject_reason));
        }

        let verdict = proof.verify(test_tree.tree.root());
        prop_assert!(!verdict);
    }

    #[proptest(cases = 30)]
    fn corrupt_leaf_digests_lead_to_verification_failure(
        #[filter(#test_tree.has_non_trivial_proof())] test_tree: MerkleTreeToTest,
        #[strategy(vec(0..#test_tree.proof().indexed_leaves.len(), 1..=#test_tree.proof().indexed_leaves.len()))]
        leaves_to_corrupt: Vec<usize>,
        #[strategy(vec(any::<DigestCorruptor>(), #leaves_to_corrupt.len()))] digest_corruptors: Vec<
            DigestCorruptor,
        >,
    ) {
        let mut proof = test_tree.proof();
        for (&i, digest_corruptor) in leaves_to_corrupt.iter().zip_eq(&digest_corruptors) {
            let (leaf_index, leaf_digest) = proof.indexed_leaves[i];
            let corrupt_digest = digest_corruptor.corrupt_digest(leaf_digest)?;
            proof.indexed_leaves[i] = (leaf_index, corrupt_digest);
        }
        if proof == test_tree.proof() {
            let reject_reason = "corruption must change leaf digests".into();
            return Err(TestCaseError::Reject(reject_reason));
        }

        let verdict = proof.verify(test_tree.tree.root());
        prop_assert!(!verdict);
    }

    #[proptest(cases = 30)]
    fn removing_leaves_from_proof_leads_to_verification_failure(
        #[filter(#test_tree.has_non_trivial_proof())] test_tree: MerkleTreeToTest,
        #[strategy(vec(0..#test_tree.proof().indexed_leaves.len(), 1..=#test_tree.proof().indexed_leaves.len()))]
        leaf_indices_to_remove: Vec<usize>,
    ) {
        let mut proof = test_tree.proof();
        let leaves_to_keep = proof
            .indexed_leaves
            .iter()
            .filter(|(i, _)| !leaf_indices_to_remove.contains(i));
        proof.indexed_leaves = leaves_to_keep.copied().collect();
        if proof == test_tree.proof() {
            let reject_reason = "removing leaves must change proof".into();
            return Err(TestCaseError::Reject(reject_reason));
        }

        let verdict = proof.verify(test_tree.tree.root());
        prop_assert!(!verdict);
    }

    #[proptest(cases = 30)]
    fn checking_set_inclusion_of_items_not_in_set_leads_to_verification_failure(
        #[filter(#test_tree.has_non_trivial_proof())] test_tree: MerkleTreeToTest,
        #[strategy(vec(0..#test_tree.tree.num_leafs(), 1..=#test_tree.tree.num_leafs()))]
        spurious_indices: Vec<usize>,
        #[strategy(vec(arb(), #spurious_indices.len()))] spurious_digests: Vec<Digest>,
    ) {
        let spurious_leaves = spurious_indices
            .into_iter()
            .zip_eq(spurious_digests)
            .collect_vec();
        let mut proof = test_tree.proof();
        proof.indexed_leaves.extend(spurious_leaves);

        let verdict = proof.verify(test_tree.tree.root());
        prop_assert!(!verdict);
    }

    #[proptest(cases = 40)]
    fn incorrect_tree_height_leads_to_verification_failure(
        #[filter(#test_tree.has_non_trivial_proof())] test_tree: MerkleTreeToTest,
        #[strategy(0..=MAX_TREE_HEIGHT)]
        #[filter(#test_tree.tree.height() != #incorrect_height)]
        incorrect_height: usize,
    ) {
        let mut proof = test_tree.proof();
        proof.tree_height = incorrect_height;
        let verdict = proof.verify(test_tree.tree.root());
        prop_assert!(!verdict);
    }

    #[proptest(cases = 20)]
    fn honestly_generated_proof_with_all_leaves_revealed_can_be_verified(
        #[strategy(arb())] tree: MerkleTree<Tip5>,
    ) {
        let leaf_indices = (0..tree.num_leafs()).collect_vec();
        let proof = tree.inclusion_proof_for_leaf_indices(&leaf_indices);
        let verdict = proof.verify(tree.root());
        prop_assert!(verdict);
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
            let proof = MerkleTreeInclusionProof::<Tip5> {
                tree_height: tree.height(),
                indexed_leaves: [(leaf_index, leaf)].into(),
                authentication_structure: authentication_path,
                _hasher: PhantomData,
            };
            let verdict = proof.verify(tree.root());
            prop_assert!(verdict);
        }
    }

    #[test]
    fn partial_merkle_tree_built_from_authentication_structure_contains_expected_nodes() {
        let merkle_tree = MerkleTree::<Tip5>::test_tree_of_height(3);
        let proof = merkle_tree.inclusion_proof_for_leaf_indices(&[0, 2]);
        let partial_tree = PartialMerkleTree::try_from(proof).unwrap();

        //         ──── 1 ────
        //        ╱           ╲
        //       2             3
        //      ╱  ╲
        //     ╱    ╲
        //    4      5
        //   ╱ ╲    ╱ ╲
        //  8   9  10 11
        //
        //  0      2   <-- opened_leaf_indices

        let expected_node_indices = vec![1, 2, 3, 4, 5, 8, 9, 10, 11];
        let node_indices = partial_tree.nodes.keys().copied().sorted().collect_vec();
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

        let node_indices = [3, 8, 9, 10, 11];
        let mut partial_tree = PartialMerkleTree::<Tip5> {
            tree_height: 3,
            leaf_indices: vec![0, 2],
            nodes: PartialMerkleTree::<Tip5>::dummy_nodes_for_indices(&node_indices),
            _hasher: PhantomData,
        };
        partial_tree.fill().unwrap();
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

        let node_indices = [8, 9, 10, 11];
        let mut partial_tree = PartialMerkleTree::<Tip5> {
            tree_height: 3,
            leaf_indices: vec![0, 2],
            nodes: PartialMerkleTree::<Tip5>::dummy_nodes_for_indices(&node_indices),
            _hasher: PhantomData,
        };

        let err = partial_tree.fill().unwrap_err();
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

        let node_indices = [2, 3, 8, 9, 10, 11];
        let mut partial_tree = PartialMerkleTree::<Tip5> {
            tree_height: 3,
            leaf_indices: vec![0, 2],
            nodes: PartialMerkleTree::<Tip5>::dummy_nodes_for_indices(&node_indices),
            _hasher: PhantomData,
        };

        let err = partial_tree.fill().unwrap_err();
        assert_eq!(MerkleTreeError::SpuriousNodeIndex(2), err);
    }

    #[test]
    fn converting_authentication_structure_to_authentication_paths_results_in_expected_paths() {
        const TREE_HEIGHT: usize = 3;
        let merkle_tree = MerkleTree::<Tip5>::test_tree_of_height(TREE_HEIGHT);
        let proof = merkle_tree.inclusion_proof_for_leaf_indices(&[0, 2]);
        let auth_paths = proof.into_authentication_paths().unwrap();

        let auth_path_with_nodes =
            |indices: [usize; TREE_HEIGHT]| indices.map(|i| merkle_tree.nodes[i]).to_vec();
        let expected_path_0 = auth_path_with_nodes([9, 5, 3]);
        let expected_path_1 = auth_path_with_nodes([11, 4, 3]);
        let expected_paths = vec![expected_path_0, expected_path_1];

        assert_eq!(expected_paths, auth_paths);
    }
}
