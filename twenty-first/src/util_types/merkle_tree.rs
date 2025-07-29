use std::borrow::Cow;
use std::collections::hash_map::Entry::*;
use std::collections::*;
use std::fmt::Debug;
use std::hash::Hash;
use std::mem::MaybeUninit;
use std::ops::Add;
use std::ops::BitAnd;
use std::ops::BitXor;
use std::ops::Div;
use std::ops::Sub;
use std::result;

use arbitrary::*;
use get_size2::GetSize;
use itertools::Itertools;
use num_traits::ConstOne;
use num_traits::ConstZero;
use rayon::prelude::*;
use thiserror::Error;

use crate::error::U32_TO_USIZE_ERR;
use crate::prelude::*;

/// Indexes internal nodes of a [`MerkleTree`].
///
/// The following convention is used.
///  - Nothing lives at index 0.
///  - Index 1 points to the root.
///  - Indices 2 and 3 contain the two children of the root.
///  - Indices 4 and 5 contain the two children of node 2.
///  - Indices 6 and 7 contain the two children of node 3.
///  - And so on. In general, the position (starting at 0) of the top bit
///    indicates the number of layers of separation between this node and the
///    root.
///  - The node indices corresponding to leafs range from (1<<tree_height) to
///    (2<<tree_height)-1.
///
/// For example:
/// ```markdown
///         ──── 1 ────          ╮
///        ╱           ╲         │
///       2             3        │
///      ╱  ╲          ╱  ╲      ├╴ node indices
///     ╱    ╲        ╱    ╲     │
///    4      5      6      7    │
///   ╱ ╲    ╱ ╲    ╱ ╲    ╱ ╲   │
///  8   9  10 11  12 13  14 15  ╯
/// ```
///
/// Type alias for [usize].
pub type MerkleTreeNodeIndex = usize;

/// Indexes the leafs of a Merkle tree, left to right, starting with zero and
/// ending with one less than a power of two. The exponent of that power of two
/// coincides with the tree's height.
///
/// Type alias for [usize].
pub type MerkleTreeLeafIndex = usize;

/// Counts the number of layers in the Merkle tree, not including the root.
/// Equivalently, counts the number of nodes on a path from a leaf to the root,
/// including the leaf but not the root.
///
/// Type alias for [u32].
pub type MerkleTreeHeight = u32;

/// The index of the root node.
pub(crate) const ROOT_INDEX: MerkleTreeNodeIndex = 1;

type Result<T> = result::Result<T, MerkleTreeError>;

/// A [Merkle tree][1] is a binary tree of [digests](Digest) that is
/// used to efficiently prove the inclusion of items in a set. Set inclusion can
/// be verified through an [inclusion proof](MerkleTreeInclusionProof). This
/// struct can hold at most 2^25 digests[^2], limiting the height of the tree to
/// 2^24. However, the associated functions (*i.e.*, the ones that don't take
/// `self`) make abstraction of this limitation and work for Merkle trees of up
/// to 2^63 nodes, 2^62 leafs, or height up to 62.
///
/// The used hash function is [`Tip5`].
///
/// [1]: <https://en.wikipedia.org/wiki/Merkle_tree>
/// [^2]: <https://github.com/Neptune-Crypto/twenty-first/pull/250#issuecomment-2782490889>
#[derive(Debug, Clone, PartialEq, Eq, GetSize)]
pub struct MerkleTree {
    nodes: Vec<Digest>,
}

/// A full inclusion proof for the leafs at the supplied indices, including the
/// leafs themselves. The proof is relative to some [Merkle tree](MerkleTree),
/// which is not necessarily (and generally cannot be) known in its entirety by
/// the verifier.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MerkleTreeInclusionProof {
    /// The stated height of the Merkle tree this proof is relative to.
    pub tree_height: MerkleTreeHeight,

    /// The leafs the proof is about, _i.e._, the revealed leafs.
    ///
    /// Purposefully not a [`HashMap`] to preserve order of the keys, which is
    /// relevant for [`into_authentication_paths`][paths].
    ///
    /// [paths]: MerkleTreeInclusionProof::into_authentication_paths
    pub indexed_leafs: Vec<(MerkleTreeLeafIndex, Digest)>,

    /// The proof's witness: de-duplicated authentication structure for the
    /// leafs this proof is about. See [`authentication_structure`][auth_structure]
    /// for details.
    ///
    /// [auth_structure]: MerkleTree::authentication_structure
    pub authentication_structure: Vec<Digest>,
}

/// Helper struct for verifying inclusion of items in a Merkle tree.
///
/// Continuing the example from [`authentication_structure`][auth_structure],
/// the partial tree for leafs 0 and 2, _i.e._, nodes 8 and 10 respectively,
/// with nodes [11, 9, 3] from the authentication structure is:
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
/// [auth_structure]: MerkleTree::authentication_structure
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct PartialMerkleTree {
    tree_height: MerkleTreeHeight,
    leaf_indices: Vec<MerkleTreeLeafIndex>,
    nodes: HashMap<MerkleTreeNodeIndex, Digest>,
}

impl MerkleTree {
    /// Build a MerkleTree with the given leafs.
    ///
    /// [`MerkleTree::par_new`] is equivalent and usually faster.
    ///
    /// # Errors
    ///
    /// - If the number of leafs is zero.
    /// - If the number of leafs is not a power of two.
    pub fn sequential_new(leafs: &[Digest]) -> Result<Self> {
        let nodes = Self::initialize_merkle_tree_nodes(leafs)?;
        let num_remaining_nodes = leafs.len();
        Self::sequentially_fill_tree(nodes, num_remaining_nodes)
    }

    /// Build a MerkleTree with the given leafs.
    ///
    /// Uses [`rayon`] to parallelize Merkle tree construction. If the use of
    /// [`rayon`] is not an option in your context, use
    /// [`MerkleTree::sequential_new`], which is equivalent but usually slower.
    ///
    /// # Errors
    ///
    /// - If the number of leafs is zero.
    /// - If the number of leafs is not a power of two.
    pub fn par_new(leafs: &[Digest]) -> Result<Self> {
        let mut nodes = Self::initialize_merkle_tree_nodes(leafs)?;

        // parallel
        let mut num_remaining_nodes = leafs.len();
        let mut num_threads = Self::num_threads();
        while num_remaining_nodes >= crate::config::merkle_tree_parallelization_cutoff() {
            // If the number of threads is so large that the chunk size is 1
            // (or even 0), each individual thread performs no work anymore.
            // In such a case, the most reasonable course of action is to reduce
            // the effective number of worker threads, which increases the chunk
            // size.
            //
            // Since parallelization_cutoff >= 2, it follows that
            // num_nodes_missing / 2 >= 1. Hence, the loop terminates at latest
            // once num_threads equals 1.
            while num_threads > num_remaining_nodes / 2 {
                num_threads /= 2;
            }

            // re-slice to only include
            // 1. the nodes that need to be computed and
            // 2. exactly those nodes required to compute them.
            let nodes = &mut nodes[..2 * num_remaining_nodes];
            let subtrees = Self::subtrees_mut(nodes, num_threads);
            subtrees.into_par_iter().for_each(|mut tree_layers| {
                debug_assert!(tree_layers.len() > 1, "internal error: infinite iteration");
                let mut previous_layer = tree_layers.pop().unwrap();
                for next_layer in tree_layers.into_iter().rev() {
                    for (node, (&left, &right)) in
                        next_layer.iter_mut().zip(previous_layer.iter().tuples())
                    {
                        *node = Tip5::hash_pair(left, right);
                    }
                    previous_layer = next_layer;
                }
            });

            // Update the number of remaining nodes by subtracting the number of
            // freshly computed nodes. Equivalently, record that the tree has
            // grown by subtree_height many layers.
            let current_tree_height = num_remaining_nodes.ilog2();
            let subtree_height = current_tree_height - num_threads.ilog2();
            num_remaining_nodes >>= subtree_height;
        }

        Self::sequentially_fill_tree(nodes, num_remaining_nodes)
    }

    /// Internal helper function to de-duplicate code between
    /// [`Self::sequential_new`] and [`Self::par_new`].
    fn sequentially_fill_tree(mut nodes: Vec<Digest>, num_remaining_nodes: usize) -> Result<Self> {
        for i in (ROOT_INDEX..num_remaining_nodes).rev() {
            nodes[i] = Tip5::hash_pair(nodes[i * 2], nodes[i * 2 + 1]);
        }

        Ok(MerkleTree { nodes })
    }

    /// Divides the given, contiguous slice into `num_trees` subtrees.
    ///
    /// The passed-in slice must represent a complete Merkle tree. Each subtree
    /// is returned as a number of mutable slices, where each such slice
    /// represents one layer in the subtree. The first slice contains only one
    /// element, the subtree's root, the next slice contains two elements, the
    /// root's children, and so on.
    ///
    /// In general, the top-most nodes of the complete tree will not be covered
    /// by the returned subtrees. For example, when requesting 2 subtrees, the
    /// complete tree's root will not be covered; when requesting 4 subtrees,
    /// the root and its direct children will not be covered; and so on.
    ///
    /// The number of subtrees must be a power of two, and must not exceed the
    /// number of leafs in the complete Merkle tree represented by the given
    /// slice.
    ///
    /// Because this is an internal helper function (and only for that reason),
    /// it's the caller's responsibility to ensure that the arguments are
    /// integral. To recap:
    /// - the number of `nodes` must be a power of 2
    /// - the number of `num_trees` must be a power of 2
    /// - the number of `nodes` must be at least `2 * num_trees`
    fn subtrees_mut<T>(nodes: &mut [T], num_trees: usize) -> Vec<Vec<&mut [T]>> {
        let num_leafs = nodes.len() / 2;
        let total_tree_height = num_leafs.ilog2();
        let sub_tree_height =
            usize::try_from(total_tree_height - num_trees.ilog2()).expect(U32_TO_USIZE_ERR);

        // a tree's “height” is the number of layers excluding the root,
        // but we want to include the root
        let num_layers = sub_tree_height + 1;
        let mut subtrees = (0..num_trees)
            .map(|_| Vec::with_capacity(num_layers))
            .collect_vec();

        // the number of nodes to skip includes the dummy node at index 0
        let nodes_to_skip = num_trees;
        let (_, mut nodes) = nodes.split_at_mut(nodes_to_skip);

        for layer_idx in 0..num_layers {
            let nodes_at_this_layer = 1 << layer_idx;
            for tree in &mut subtrees {
                let (layer, rest) = nodes.split_at_mut(nodes_at_this_layer);
                tree.push(layer);
                nodes = rest;
            }
        }
        debug_assert!(nodes.is_empty());

        subtrees
    }

    /// Compute the Merkle root from the given leafs without recording any
    /// internal nodes.
    ///
    /// This is equivalent to
    /// [`MerkleTree::sequential_new`]`(leafs).map(|t| t.`[`root()`][root]`)`
    /// but requires considerably less RAM without impacting runtime performance
    /// (it neither improves nor worsens). RAM consumption is reduced because
    /// the tree's internal nodes are discarded as soon as possible. If you
    /// later want to [prove](MerkleTreeInclusionProof) set-membership of any
    /// leaf(s), the corresponding internal nodes that make up the
    /// [authentication structure][auth_struct] will have to be recomputed,
    /// whereas they can simply be copied if you have access to a full
    /// [`MerkleTree`].
    ///
    /// See [`MerkleTree::sequential_authentication_structure_from_leafs`] for a
    /// function that computes the authentication structure from the leafs.
    ///
    /// See also [`MerkleTree::par_frugal_root`] for a parallel version of this
    /// function.
    ///
    /// [root]: Self::root
    /// [auth_struct]: MerkleTreeInclusionProof::authentication_structure
    pub fn sequential_frugal_root(leafs: &[Digest]) -> Result<Digest> {
        if leafs.is_empty() {
            return Err(MerkleTreeError::TooFewLeafs);
        };
        let peaks = super::mmr::mmr_accumulator::MmrAccumulator::peaks_from_leafs(leafs);
        let [root] = peaks[..] else {
            return Err(MerkleTreeError::IncorrectNumberOfLeafs);
        };

        Ok(root)
    }

    /// Compute the Merkle root from the given leafs in parallel without
    /// recording any internal nodes.
    ///
    /// This is equivalent to
    /// [`MerkleTree::par_new`]`(leafs).map(|t| t.`[`root()`][root]`)`
    /// but requires considerably less RAM. Runtime performance is similar to
    /// [`MerkleTree::par_new`] and might be faster, depending on your hardware.
    ///
    /// See [`MerkleTree::par_authentication_structure_from_leafs`] for a
    /// function that computes the authentication structure from the leafs.
    ///
    /// See also [`MerkleTree::sequential_frugal_root`] for a sequential version
    /// of this function. It also lists additional benefits and drawbacks that
    /// are applicable to both the sequential and the parallel version.
    ///
    /// Note that the RAM usage of this parallel version depends on
    /// [`RAYON_NUM_THREADS`](rayon), since each thread requires some RAM. If
    /// you require the absolute minimum amount of RAM usage, use the
    /// [sequential version](Self::sequential_frugal_root) instead.
    ///
    /// [root]: Self::root
    pub fn par_frugal_root(leafs: &[Digest]) -> Result<Digest> {
        if !leafs.len().is_power_of_two() {
            return Err(MerkleTreeError::IncorrectNumberOfLeafs);
        }

        // parallel
        let mut num_threads = Self::num_threads();
        let mut leafs = Cow::Borrowed(leafs);
        while leafs.len() >= crate::config::merkle_tree_parallelization_cutoff() {
            // If the number of threads is so large that the chunk size is 1
            // (or even 0), each individual thread performs no work anymore.
            // In such a case, the most reasonable course of action is to reduce
            // the effective number of worker threads, which increases the chunk
            // size.
            //
            // Since parallelization_cutoff >= 2, it follows that
            // leafs.len() / 2 >= 1. Hence, the loop terminates at latest once
            // num_threads equals 1.
            while num_threads > leafs.len() / 2 {
                num_threads /= 2;
            }

            let chunk_size = leafs.len() / num_threads;
            let next_layer = (0..num_threads)
                .into_par_iter()
                .map(|i| Self::sequential_frugal_root(&leafs[i * chunk_size..(i + 1) * chunk_size]))
                .collect::<Result<_>>()?;
            leafs = Cow::Owned(next_layer);
        }

        // sequential
        Self::sequential_frugal_root(&leafs)
    }

    /// Internal helper function to determine the number of threads to use for
    /// parallel Merkle tree construction.
    ///
    /// Can be used to figure out the number of chunks to split the work into,
    /// but take care that each chunk contains at least 2 nodes, else no
    /// meaningful work will be done and your iteration might run forever.
    ///
    /// Guaranteed to be a power of two.
    ///
    /// Respects the [`RAYON_NUM_THREADS`][rayon] environment variable, if set.
    fn num_threads() -> usize {
        // To guarantee that all chunks correspond to trees of the same height,
        // the number of threads must divide the number of leafs cleanly.
        let num_threads = rayon::current_num_threads();
        let num_threads = if num_threads.is_power_of_two() {
            num_threads
        } else {
            num_threads.next_power_of_two() / 2 // previous power of 2
        };

        // avoid division by 0
        num_threads.max(1)
    }

    /// Helps to kick off Merkle tree construction. Sets up the Merkle tree's
    /// internal nodes if (and only if) it is possible to construct a Merkle
    /// tree with the given leafs.
    fn initialize_merkle_tree_nodes(leafs: &[Digest]) -> Result<Vec<Digest>> {
        if leafs.is_empty() {
            return Err(MerkleTreeError::TooFewLeafs);
        }

        let num_leafs = leafs.len();
        if !num_leafs.is_power_of_two() {
            return Err(MerkleTreeError::IncorrectNumberOfLeafs);
        }

        let num_nodes = 2 * num_leafs;
        let mut nodes = Vec::new();

        // Use `try_reserve_exact` because we want to get an error not a panic
        // if allocation fails. The error can be bubbled up.
        nodes
            .try_reserve_exact(num_nodes)
            .map_err(|_| MerkleTreeError::TreeTooHigh)?;

        // Parallel initialization is slower for small trees, but faster for
        // tall trees. If the slowdown is deemed too big for small trees, this
        // is the place to change it.
        nodes
            .spare_capacity_mut()
            .par_iter_mut()
            .take(num_nodes)
            .for_each(|n| *n = const { MaybeUninit::new(Digest::ALL_ZERO) });

        // SAFETY:
        // - the requested capacity is num_nodes, and so is the new length
        // - the first num_nodes elements are initialized to Digest::ALL_ZERO
        unsafe { nodes.set_len(num_nodes) };

        nodes[num_leafs..].copy_from_slice(leafs);

        Ok(nodes)
    }

    /// Compute the [`MerkleTreeNodeIndex`]es for an authentication structure.
    ///
    /// Given a list of [`MerkleTreeLeafIndex`]es, return the (node) indices of
    /// exactly those nodes that are needed to prove (or verify) that the
    /// indicated leafs are in the Merkle tree.
    ///
    /// For an explanation of the term “authentication structure”, please refer
    /// to [`authentication_structure`][Self::authentication_structure].
    ///
    /// Returns an error if any of the leaf indices is bigger than or equal to
    /// the total number of leafs in the tree, or if the total number of leafs
    /// is not a power of two.
    //
    // The implementation is this generic to allow using it with type `usize` in
    // this crate as well as type `u64` in a downstream dependency.
    //
    // Reducing the number of trait bounds (without sacrificing performance) is
    // a desirable goal.
    pub fn authentication_structure_node_indices<I>(
        num_leafs: I,
        leaf_indices: &[I],
    ) -> Result<impl ExactSizeIterator<Item = I> + use<I>>
    where
        I: Copy
            + Hash
            + Ord
            + Add<Output = I>
            + Sub<Output = I>
            + Div<Output = I>
            + BitAnd<Output = I>
            + BitXor<Output = I>
            + ConstZero
            + ConstOne,
    {
        // The number of leafs must be a power of 2. Because the method
        // `is_power_of_two()` is not part of any trait, rely on some
        // bit twiddling hacks instead.
        // <https://graphics.stanford.edu/~seander/bithacks.html>
        if num_leafs == I::ZERO || ((num_leafs - I::ONE) & num_leafs) != I::ZERO {
            return Err(MerkleTreeError::IncorrectNumberOfLeafs);
        }

        // The set of indices of nodes that need to be included in the
        // authentication structure. In principle, every node of every
        // authentication path is needed. The root is never needed. Hence, it is
        // not considered below.
        let mut node_is_needed = HashSet::new();

        // The set of indices of nodes that can be computed from other nodes in
        // the authentication structure or the leafs that are explicitly
        // supplied during verification. Every node on the direct path from the
        // leaf to the root can be computed by the very nature of
        // “authentication path”.
        let mut node_can_be_computed = HashSet::new();

        let two = I::ONE + I::ONE; // cannot be `const` because of `+`
        for &leaf_index in leaf_indices {
            if leaf_index >= num_leafs {
                return Err(MerkleTreeError::LeafIndexInvalid);
            }

            let root_index = const { I::ONE };
            let mut node_index = leaf_index + num_leafs;
            while node_index > root_index {
                let sibling_index = node_index ^ I::ONE;
                node_can_be_computed.insert(node_index);
                node_is_needed.insert(sibling_index);
                node_index = node_index / two;
            }
        }

        let set_difference = node_is_needed.difference(&node_can_be_computed).copied();
        Ok(set_difference.sorted_unstable().rev())
    }

    /// Construct an [authentication structure][Self::authentication_structure]
    /// without access to a full [`MerkleTree`], only the leafs. Particularly
    /// useful in combination with [RAM-frugal root][frugal] computation.
    ///
    /// See also [`MerkleTree::par_authentication_structure_from_leafs`] for a
    /// parallel version of this function.
    ///
    /// [frugal]: Self::sequential_frugal_root
    pub fn sequential_authentication_structure_from_leafs(
        leafs: &[Digest],
        leaf_indices: &[MerkleTreeLeafIndex],
    ) -> Result<Vec<Digest>> {
        Self::authentication_structure_node_indices(leafs.len(), leaf_indices)?
            .map(|node_index| Self::subtree_leafs(leafs, node_index))
            .map(Self::sequential_frugal_root)
            .collect()
    }

    /// Construct an [authentication structure][Self::authentication_structure]
    /// without access to a full [`MerkleTree`], only the leafs. Particularly
    /// useful in combination with [RAM-frugal root][frugal] computation.
    ///
    /// See also [`MerkleTree::sequential_authentication_structure_from_leafs`]
    /// for a sequential version of this function.
    ///
    /// [frugal]: Self::par_frugal_root
    pub fn par_authentication_structure_from_leafs(
        leafs: &[Digest],
        leaf_indices: &[MerkleTreeLeafIndex],
    ) -> Result<Vec<Digest>> {
        Self::authentication_structure_node_indices(leafs.len(), leaf_indices)?
            .collect_vec()
            .into_par_iter()
            .map(|node_index| Self::subtree_leafs(leafs, node_index))
            .map(Self::par_frugal_root)
            .collect()
    }

    /// Given a list of leafs and a [`MerkleTreeNodeIndex`] within the tree
    /// defined by those leafs, return the leafs that are part of the subtree
    /// rooted at the given node index.
    ///
    /// For example, given 4 leafs and node index 3, returns the leafs with
    /// node indices 6 and 7:
    ///
    /// ```markdown
    ///         1
    ///       ╱   ╲
    ///      2      3
    ///     ╱ ╲    ╱ ╲
    ///    4   5  6   7
    /// ```
    ///
    /// Because this is an internal helper function (and only for that reason),
    /// it's the caller's responsibility to ensure that the arguments are
    /// integral:
    /// - the number of `leafs` must be a power of 2
    /// - the `node_index` must be valid with respect to the tree induced by the
    ///   `leafs`
    fn subtree_leafs(leafs: &[Digest], node_index: MerkleTreeNodeIndex) -> &[Digest] {
        let total_num_leafs = leafs.len();
        let total_tree_height = total_num_leafs.ilog2();

        let tree_height = total_tree_height - node_index.ilog2();
        let left_leaf_node_index = node_index * (1 << tree_height);
        let left_leaf_index = left_leaf_node_index - total_num_leafs;
        let num_leafs = 1 << tree_height;

        &leafs[left_leaf_index..left_leaf_index + num_leafs]
    }

    /// Generate a de-duplicated authentication structure for the given
    /// [`MerkleTreeLeafIndex`]es.
    ///
    /// If a single index is supplied, the authentication structure is the
    /// authentication path for the indicated leaf.
    ///
    /// For example, consider the following Merkle tree, and note the difference
    /// between [`MerkleTreeLeafIndex`] and [`MerkleTreeNodeIndex`].
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
    /// The authentication structure for leafs 0 and 2, _i.e._, nodes 8 and 10
    /// respectively, is nodes [11, 9, 3].
    /// Note how:
    /// - Node 3 is included only once, even though the individual authentication
    ///   paths for leafs 0 and 2 both include node 3. This is one part of the
    ///   de-duplication.
    /// - Node 4 is not included at all, even though the authentication path for
    ///   leaf 2 requires the node: node 4 can be computed from nodes 8 and 9;
    ///   the former is supplied explicitly during [verification][verify],
    ///   the latter is included in the authentication structure.
    ///   This is the other part of the de-duplication.
    ///
    /// [verify]: MerkleTreeInclusionProof::verify
    pub fn authentication_structure(
        &self,
        leaf_indices: &[MerkleTreeLeafIndex],
    ) -> Result<Vec<Digest>> {
        let num_leafs = self.num_leafs();
        let indices = Self::authentication_structure_node_indices(num_leafs, leaf_indices)?;
        let auth_structure = indices.map(|idx| self.node(idx).unwrap()).collect();
        Ok(auth_structure)
    }

    pub fn root(&self) -> Digest {
        self.nodes[ROOT_INDEX]
    }

    pub fn num_leafs(&self) -> MerkleTreeLeafIndex {
        let node_count = self.nodes.len();
        debug_assert!(node_count.is_power_of_two());
        node_count / 2
    }

    pub fn height(&self) -> MerkleTreeHeight {
        let leaf_count = self.num_leafs();
        debug_assert!(leaf_count.is_power_of_two());
        leaf_count.ilog2()
    }

    /// The node at the given [`MerkleTreeNodeIndex`], if it exists.
    ///
    /// Note that nodes are 1-indexed, meaning that the root lives at index 1
    /// and all the other nodes have larger indices.
    pub fn node(&self, index: MerkleTreeNodeIndex) -> Option<Digest> {
        if index == 0 {
            None
        } else {
            self.nodes.get(index).copied()
        }
    }

    /// All leafs of the Merkle tree.
    pub fn leafs(&self) -> impl Iterator<Item = &Digest> {
        self.nodes.iter().skip(self.num_leafs())
    }

    /// The leaf at the given [`MerkleTreeLeafIndex`], if it exists.
    pub fn leaf(&self, index: MerkleTreeLeafIndex) -> Option<Digest> {
        let first_leaf_index = self.num_leafs();
        self.node(first_leaf_index + index)
    }

    /// Produce a [`Vec`] of ([`MerkleTreeLeafIndex`], [`Digest`]) covering all
    /// leafs.
    pub fn indexed_leafs(
        &self,
        indices: &[MerkleTreeLeafIndex],
    ) -> Result<Vec<(MerkleTreeLeafIndex, Digest)>> {
        let invalid_index = MerkleTreeError::LeafIndexInvalid;
        let maybe_indexed_leaf = |&i| self.leaf(i).ok_or(invalid_index).map(|leaf| (i, leaf));

        indices.iter().map(maybe_indexed_leaf).collect()
    }

    /// A full inclusion proof for the leafs at the supplied
    /// [`MerkleTreeLeafIndex`]es, *including* the leafs
    ///
    /// Generally, using [`authentication_structure`][auth_structure] is
    /// preferable. Use this method only if the verifier needs explicit access
    /// to the leafs, _i.e._, cannot compute them from other information.
    ///
    /// [auth_structure]: Self::authentication_structure
    pub fn inclusion_proof_for_leaf_indices(
        &self,
        indices: &[MerkleTreeLeafIndex],
    ) -> Result<MerkleTreeInclusionProof> {
        let proof = MerkleTreeInclusionProof {
            tree_height: self.height(),
            indexed_leafs: self.indexed_leafs(indices)?,
            authentication_structure: self.authentication_structure(indices)?,
        };
        Ok(proof)
    }
}

impl<'a> Arbitrary<'a> for MerkleTree {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let height = u.int_in_range(0..=13)?;
        let num_leafs = 1 << height;
        let leaf_digests: arbitrary::Result<Vec<_>> =
            (0..num_leafs).map(|_| u.arbitrary()).collect();

        let tree = Self::par_new(&leaf_digests?).unwrap();
        Ok(tree)
    }
}

impl MerkleTreeInclusionProof {
    fn leaf_indices(&self) -> impl Iterator<Item = &MerkleTreeLeafIndex> {
        self.indexed_leafs.iter().map(|(index, _)| index)
    }

    fn is_trivial(&self) -> bool {
        self.indexed_leafs.is_empty() && self.authentication_structure.is_empty()
    }

    /// Verify that the given root digest is the root of a Merkle tree that contains
    /// the indicated leafs.
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
    /// In some contexts, it is easier to deal with individual authentication paths
    /// than with the de-duplicated authentication structure.
    ///
    /// Continuing the example from [`authentication_structure`][auth_structure],
    /// the authentication structure for leafs 0 and 2, _i.e._, nodes 8 and 10
    /// respectively, is nodes [11, 9, 3].
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
    /// [auth_structure]: MerkleTree::authentication_structure
    pub fn into_authentication_paths(self) -> Result<Vec<Vec<Digest>>> {
        let partial_tree = PartialMerkleTree::try_from(self)?;
        partial_tree.into_authentication_paths()
    }
}

impl PartialMerkleTree {
    pub fn root(&self) -> Result<Digest> {
        self.nodes
            .get(&ROOT_INDEX)
            .copied()
            .ok_or(MerkleTreeError::RootNotFound)
    }

    fn node(&self, index: MerkleTreeNodeIndex) -> Result<Digest> {
        self.nodes
            .get(&index)
            .copied()
            .ok_or(MerkleTreeError::MissingNodeIndex(index))
    }

    fn num_leafs(&self) -> Result<MerkleTreeLeafIndex> {
        1_usize
            .checked_shl(self.tree_height)
            .ok_or(MerkleTreeError::TreeTooHigh)
    }

    /// Compute all computable digests of the partial Merkle tree, modifying self.
    /// Returns an error if self is either
    /// - incomplete, _i.e._, does not contain all the nodes required to compute
    ///   the root, or
    /// - not minimal, _i.e._, if it contains nodes that can be computed from other
    ///   nodes.
    pub fn fill(&mut self) -> Result<()> {
        let mut parent_node_indices = self.first_layer_parent_node_indices()?;

        for _ in 0..self.tree_height {
            for &parent_node_index in &parent_node_indices {
                self.insert_digest_for_index(parent_node_index)?;
            }
            parent_node_indices = Self::move_indices_one_layer_up(parent_node_indices);
        }

        Ok(())
    }

    /// Any parent node index is included only once. This guarantees that the number
    /// of hash operations is minimal.
    fn first_layer_parent_node_indices(&self) -> Result<Vec<MerkleTreeNodeIndex>> {
        let num_leafs = self.num_leafs()?;
        let leaf_to_parent_node_index = |&leaf_index| (leaf_index + num_leafs) / 2;

        let parent_node_indices = self.leaf_indices.iter().map(leaf_to_parent_node_index);
        let mut parent_node_indices = parent_node_indices.collect_vec();
        parent_node_indices.sort_unstable();
        parent_node_indices.dedup();
        Ok(parent_node_indices)
    }

    fn insert_digest_for_index(&mut self, parent_index: MerkleTreeNodeIndex) -> Result<()> {
        let (left_child, right_child) = self.children_of_node(parent_index)?;
        let parent_digest = Tip5::hash_pair(left_child, right_child);

        match self.nodes.insert(parent_index, parent_digest) {
            Some(_) => Err(MerkleTreeError::SpuriousNodeIndex(parent_index)),
            None => Ok(()),
        }
    }

    fn children_of_node(&self, parent_index: MerkleTreeNodeIndex) -> Result<(Digest, Digest)> {
        let left_child_index = parent_index * 2;
        let right_child_index = left_child_index ^ 1;

        let left_child = self.node(left_child_index)?;
        let right_child = self.node(right_child_index)?;
        Ok((left_child, right_child))
    }

    /// Indices are deduplicated to guarantee minimal number of hash operations.
    fn move_indices_one_layer_up(
        mut indices: Vec<MerkleTreeNodeIndex>,
    ) -> Vec<MerkleTreeNodeIndex> {
        indices.iter_mut().for_each(|i| *i /= 2);
        indices.dedup();
        indices
    }

    /// Collect all individual authentication paths for the indicated leafs.
    fn into_authentication_paths(self) -> Result<Vec<Vec<Digest>>> {
        self.leaf_indices
            .iter()
            .map(|&i| self.authentication_path_for_index(i))
            .collect()
    }

    /// Given a single leaf index and a partial Merkle tree, collect the
    /// authentication path for the indicated leaf.
    ///
    /// Fails if the partial Merkle tree does not contain the entire
    /// authentication path.
    fn authentication_path_for_index(
        &self,
        leaf_index: MerkleTreeLeafIndex,
    ) -> Result<Vec<Digest>> {
        let num_leafs = self.num_leafs()?;
        let mut authentication_path = vec![];
        let mut node_index = leaf_index + num_leafs;
        while node_index > ROOT_INDEX {
            let sibling_index = node_index ^ 1;
            let sibling = self.node(sibling_index)?;
            authentication_path.push(sibling);
            node_index /= 2;
        }
        Ok(authentication_path)
    }
}

impl TryFrom<MerkleTreeInclusionProof> for PartialMerkleTree {
    type Error = MerkleTreeError;

    fn try_from(proof: MerkleTreeInclusionProof) -> Result<Self> {
        let leaf_indices = proof.leaf_indices().copied().collect();
        let mut partial_tree = PartialMerkleTree {
            tree_height: proof.tree_height,
            leaf_indices,
            nodes: HashMap::new(),
        };

        let num_leafs = partial_tree.num_leafs()?;
        if partial_tree.leaf_indices.iter().any(|&i| i >= num_leafs) {
            return Err(MerkleTreeError::LeafIndexInvalid);
        }

        let node_indices = MerkleTree::authentication_structure_node_indices(
            num_leafs,
            &partial_tree.leaf_indices,
        )?;
        if proof.authentication_structure.len() != node_indices.len() {
            return Err(MerkleTreeError::AuthenticationStructureLengthMismatch);
        }

        let mut nodes: HashMap<_, _> = node_indices
            .zip_eq(proof.authentication_structure)
            .collect();

        for (leaf_index, leaf_digest) in proof.indexed_leafs {
            let node_index = leaf_index + num_leafs;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MerkleTreeError {
    #[error("All leaf indices must be valid, i.e., less than the number of leafs.")]
    LeafIndexInvalid,

    #[error("The length of the supplied authentication structure must match the expected length.")]
    AuthenticationStructureLengthMismatch,

    #[error("Leaf digests of repeated indices must be identical.")]
    RepeatedLeafDigestMismatch,

    #[error("The partial tree must be minimal. Node {0} was supplied but can be computed.")]
    SpuriousNodeIndex(MerkleTreeNodeIndex),

    #[error("The partial tree must contain all necessary information. Node {0} is missing.")]
    MissingNodeIndex(MerkleTreeNodeIndex),

    #[error("Could not compute the root. Maybe no leaf indices were supplied?")]
    RootNotFound,

    #[error("Too few leafs to build a Merkle tree.")]
    TooFewLeafs,

    #[error("The number of leafs must be a power of two.")]
    IncorrectNumberOfLeafs,

    #[error("Tree height implies a tree that does not fit in RAM")]
    TreeTooHigh,
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(crate) mod tests {
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use super::*;
    use crate::tip5::digest::tests::DigestCorruptor;

    impl MerkleTree {
        fn test_tree_of_height(tree_height: MerkleTreeHeight) -> Self {
            let num_leafs = 1 << tree_height;
            let leafs = (0..num_leafs).map(BFieldElement::new);
            let leaf_digests = leafs.map(|bfe| Tip5::hash_varlen(&[bfe])).collect_vec();
            let tree = Self::par_new(&leaf_digests).unwrap();
            assert!(leaf_digests.iter().all_unique());
            tree
        }
    }

    impl PartialMerkleTree {
        fn dummy_nodes_for_indices(
            node_indices: &[MerkleTreeNodeIndex],
        ) -> HashMap<MerkleTreeNodeIndex, Digest> {
            node_indices
                .iter()
                .map(|&i| (i, bfe!(i)))
                .map(|(i, leaf)| (i, Tip5::hash_varlen(&[leaf])))
                .collect()
        }
    }

    /// Test helper to deduplicate generation of Merkle trees.
    #[derive(Debug, Clone, test_strategy::Arbitrary)]
    pub(crate) struct MerkleTreeToTest {
        #[strategy(arb())]
        pub tree: MerkleTree,

        #[strategy(vec(0..#tree.num_leafs(), 0..(#tree.num_leafs())))]
        pub selected_indices: Vec<MerkleTreeLeafIndex>,
    }

    impl MerkleTreeToTest {
        fn has_non_trivial_proof(&self) -> bool {
            !self.selected_indices.is_empty()
        }

        fn proof(&self) -> MerkleTreeInclusionProof {
            // test helper – unwrap is fine
            self.tree
                .inclusion_proof_for_leaf_indices(&self.selected_indices)
                .unwrap()
        }
    }

    #[test]
    fn building_merkle_tree_from_empty_list_of_digests_fails_with_expected_error() {
        let maybe_tree = MerkleTree::par_new(&[]);
        let err = maybe_tree.unwrap_err();
        assert_eq!(MerkleTreeError::TooFewLeafs, err);
    }

    #[test]
    fn merkle_tree_with_one_leaf_has_expected_height_and_number_of_leafs() {
        let digest = Digest::default();
        let tree = MerkleTree::par_new(&[digest]).unwrap();
        assert_eq!(1, tree.num_leafs());
        assert_eq!(0, tree.height());
    }

    #[proptest]
    fn building_merkle_tree_from_one_digest_makes_that_digest_the_root(digest: Digest) {
        let tree = MerkleTree::par_new(&[digest]).unwrap();
        assert_eq!(digest, tree.root());
    }

    #[proptest]
    fn building_merkle_tree_from_list_of_digests_with_incorrect_number_of_leafs_fails(
        #[filter(!#num_leafs.is_power_of_two())]
        #[strategy(1_usize..1 << 13)]
        num_leafs: usize,
    ) {
        let digest = Digest::default();
        let digests = vec![digest; num_leafs];
        let maybe_tree = MerkleTree::par_new(&digests);
        let err = maybe_tree.unwrap_err();
        assert_eq!(MerkleTreeError::IncorrectNumberOfLeafs, err);
    }

    #[proptest]
    fn merkle_tree_construction_strategies_behave_identically_on_random_input(leafs: Vec<Digest>) {
        let sequential = MerkleTree::sequential_new(&leafs);
        let parallel = MerkleTree::par_new(&leafs);
        prop_assert_eq!(sequential, parallel);
    }

    #[proptest]
    fn merkle_tree_construction_strategies_produce_identical_trees(
        #[strategy(0_usize..10)] _tree_height: usize,
        #[strategy(vec(arb(), 1 << #_tree_height))] leafs: Vec<Digest>,
    ) {
        let sequential = MerkleTree::sequential_new(&leafs)?;
        let parallel = MerkleTree::par_new(&leafs)?;
        prop_assert_eq!(sequential, parallel);
    }

    #[proptest]
    fn merkle_tree_construction_strategies_are_independent_of_parallelization_cutoff(
        #[strategy(0_usize..10)] _tree_height: usize,
        #[strategy(vec(arb(), 1 << #_tree_height))] leafs: Vec<Digest>,
        cutoff: usize,
    ) {
        crate::config::set_merkle_tree_parallelization_cutoff(cutoff);

        let sequential = MerkleTree::sequential_new(&leafs)?;
        let parallel = MerkleTree::par_new(&leafs)?;
        prop_assert_eq!(sequential, parallel);
    }

    #[proptest]
    fn ram_frugal_merkle_root_is_identical_to_full_tree_root(
        #[strategy(0_usize..10)] _tree_height: usize,
        #[strategy(vec(arb(), 1 << #_tree_height))] leafs: Vec<Digest>,
    ) {
        let hungry = MerkleTree::par_new(&leafs)?.root();
        let seq_frugal = MerkleTree::sequential_frugal_root(&leafs)?;
        prop_assert_eq!(seq_frugal, hungry);

        let par_frugal = MerkleTree::par_frugal_root(&leafs)?;
        prop_assert_eq!(par_frugal, hungry);
    }

    #[proptest]
    fn ram_frugal_merkle_root_is_independent_of_parallelization_cutoff(
        #[strategy(0_usize..10)] _tree_height: usize,
        #[strategy(vec(arb(), 1 << #_tree_height))] leafs: Vec<Digest>,
        cutoff: usize,
    ) {
        crate::config::set_merkle_tree_parallelization_cutoff(cutoff);

        let hungry = MerkleTree::par_new(&leafs)?.root();
        let seq_frugal = MerkleTree::sequential_frugal_root(&leafs)?;
        prop_assert_eq!(seq_frugal, hungry);

        let par_frugal = MerkleTree::par_frugal_root(&leafs)?;
        prop_assert_eq!(par_frugal, hungry);
    }

    #[proptest(cases = 100)]
    fn various_small_parallelization_cutoffs_dont_cause_infinite_iterations(
        #[strategy(0_usize..10)] _tree_height: usize,
        #[strategy(vec(arb(), 1 << #_tree_height))] leafs: Vec<Digest>,
    ) {
        for cutoff in 0..=16 {
            crate::config::set_merkle_tree_parallelization_cutoff(cutoff);
            let _tree = MerkleTree::par_new(&leafs);
            let _root = MerkleTree::par_frugal_root(&leafs);
        }
    }

    #[proptest(cases = 100)]
    fn accessing_number_of_leafs_and_height_never_panics(
        #[strategy(arb())] merkle_tree: MerkleTree,
    ) {
        let _ = merkle_tree.num_leafs();
        let _ = merkle_tree.height();
    }

    #[proptest(cases = 50)]
    fn trivial_proof_can_be_verified(#[strategy(arb())] merkle_tree: MerkleTree) {
        let proof = merkle_tree.inclusion_proof_for_leaf_indices(&[]).unwrap();
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
        #[strategy(Just(#test_tree.proof().indexed_leafs.len()))] _n_leafs: usize,
        #[strategy(vec(0..#_n_leafs, 1..=#_n_leafs))] leafs_to_corrupt: Vec<usize>,
        #[strategy(vec(any::<DigestCorruptor>(), #leafs_to_corrupt.len()))] digest_corruptors: Vec<
            DigestCorruptor,
        >,
    ) {
        let mut proof = test_tree.proof();
        for (&i, digest_corruptor) in leafs_to_corrupt.iter().zip_eq(&digest_corruptors) {
            let (leaf_index, leaf_digest) = proof.indexed_leafs[i];
            let corrupt_digest = digest_corruptor.corrupt_digest(leaf_digest)?;
            proof.indexed_leafs[i] = (leaf_index, corrupt_digest);
        }
        if proof == test_tree.proof() {
            let reject_reason = "corruption must change leaf digests".into();
            return Err(TestCaseError::Reject(reject_reason));
        }

        let verdict = proof.verify(test_tree.tree.root());
        prop_assert!(!verdict);
    }

    #[proptest(cases = 30)]
    fn removing_leafs_from_proof_leads_to_verification_failure(
        #[filter(#test_tree.has_non_trivial_proof())] test_tree: MerkleTreeToTest,
        #[strategy(Just(#test_tree.proof().indexed_leafs.len()))] _n_leafs: usize,
        #[strategy(vec(0..(#_n_leafs), 1..=#_n_leafs))] leaf_indices_to_remove: Vec<
            MerkleTreeLeafIndex,
        >,
    ) {
        let mut proof = test_tree.proof();
        let leafs_to_keep = proof
            .indexed_leafs
            .iter()
            .filter(|(i, _)| !leaf_indices_to_remove.contains(i));
        proof.indexed_leafs = leafs_to_keep.copied().collect();
        if proof == test_tree.proof() {
            let reject_reason = "removing leafs must change proof".into();
            return Err(TestCaseError::Reject(reject_reason));
        }

        let verdict = proof.verify(test_tree.tree.root());
        prop_assert!(!verdict);
    }

    #[proptest(cases = 30)]
    fn checking_set_inclusion_of_items_not_in_set_leads_to_verification_failure(
        #[filter(#test_tree.has_non_trivial_proof())] test_tree: MerkleTreeToTest,
        #[strategy(vec(0..#test_tree.tree.num_leafs(), 1..=(#test_tree.tree.num_leafs())))]
        spurious_indices: Vec<MerkleTreeLeafIndex>,
        #[strategy(vec(any::<Digest>(), #spurious_indices.len()))] spurious_digests: Vec<Digest>,
    ) {
        let spurious_leafs = spurious_indices
            .into_iter()
            .zip_eq(spurious_digests)
            .collect_vec();
        let mut proof = test_tree.proof();
        proof.indexed_leafs.extend(spurious_leafs);

        let verdict = proof.verify(test_tree.tree.root());
        prop_assert!(!verdict);
    }

    #[proptest(cases = 30)]
    fn honestly_generated_proof_with_duplicate_leafs_can_be_verified(
        #[filter(#test_tree.has_non_trivial_proof())] test_tree: MerkleTreeToTest,
        #[strategy(Just(#test_tree.proof().indexed_leafs.len()))] _n_leafs: usize,
        #[strategy(vec(0..#_n_leafs, 1..=#_n_leafs))] indices_to_duplicate: Vec<usize>,
    ) {
        let mut proof = test_tree.proof();
        let duplicate_leafs = indices_to_duplicate
            .into_iter()
            .map(|i| proof.indexed_leafs[i])
            .collect_vec();
        proof.indexed_leafs.extend(duplicate_leafs);
        let verdict = proof.verify(test_tree.tree.root());
        prop_assert!(verdict);
    }

    #[proptest(cases = 40)]
    fn incorrect_tree_height_leads_to_verification_failure(
        #[filter(#test_tree.has_non_trivial_proof())] test_tree: MerkleTreeToTest,
        #[strategy(0_u32..64)]
        #[filter(#test_tree.tree.height() != #incorrect_height)]
        incorrect_height: MerkleTreeHeight,
    ) {
        let mut proof = test_tree.proof();
        proof.tree_height = incorrect_height;
        let verdict = proof.verify(test_tree.tree.root());
        prop_assert!(!verdict);
    }

    #[proptest(cases = 20)]
    fn honestly_generated_proof_with_all_leafs_revealed_can_be_verified(
        #[strategy(arb())] tree: MerkleTree,
    ) {
        let leaf_indices = (0..tree.num_leafs()).collect_vec();
        let proof = tree
            .inclusion_proof_for_leaf_indices(&leaf_indices)
            .unwrap();
        let verdict = proof.verify(tree.root());
        prop_assert!(verdict);
    }

    #[proptest(cases = 30)]
    fn requesting_inclusion_proof_for_nonexistent_leaf_fails_with_expected_error(
        #[strategy(arb())] tree: MerkleTree,
        #[filter(#leaf_indices.iter().any(|&i| i > #tree.num_leafs()))] leaf_indices: Vec<
            MerkleTreeLeafIndex,
        >,
    ) {
        let maybe_proof = tree.inclusion_proof_for_leaf_indices(&leaf_indices);
        let err = maybe_proof.unwrap_err();

        assert_eq!(MerkleTreeError::LeafIndexInvalid, err);
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

        let tree = MerkleTree::test_tree_of_height(2);
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

        let tree = MerkleTree::test_tree_of_height(3);
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

    #[test]
    fn authentication_paths_of_very_small_tree_are_identical_when_using_tree_or_only_leafs() {
        let tree = MerkleTree::test_tree_of_height(3);
        let leafs = tree.leafs().copied().collect_vec();

        let tree_path = |i| tree.authentication_structure(&[i]).unwrap();
        let seq_leaf_path =
            |i| MerkleTree::sequential_authentication_structure_from_leafs(&leafs, &[i]).unwrap();
        let par_leaf_path =
            |i| MerkleTree::par_authentication_structure_from_leafs(&leafs, &[i]).unwrap();

        for leaf_idx in 0..tree.num_leafs() {
            dbg!(leaf_idx);
            assert_eq!(tree_path(leaf_idx), seq_leaf_path(leaf_idx));
            assert_eq!(tree_path(leaf_idx), par_leaf_path(leaf_idx));
        }
    }

    #[proptest(cases = 100)]
    fn auth_structure_is_independent_of_compute_method(test_tree: MerkleTreeToTest) {
        let tree = test_tree.tree;
        let selected_indices = test_tree.selected_indices;
        let leafs = tree.leafs().copied().collect_vec();

        let cached_auth_structure = tree.authentication_structure(&selected_indices)?;
        let seq_auth_structure =
            MerkleTree::sequential_authentication_structure_from_leafs(&leafs, &selected_indices)?;
        let par_auth_structure =
            MerkleTree::par_authentication_structure_from_leafs(&leafs, &selected_indices)?;

        prop_assert_eq!(&cached_auth_structure, &seq_auth_structure);
        prop_assert_eq!(&cached_auth_structure, &par_auth_structure);
    }

    #[test]
    fn subtree_leafs_are_actually_sub_tree_leafs() {
        let tree = MerkleTree::test_tree_of_height(5);
        let leafs = tree.leafs().copied().collect_vec();
        let subtree = |node_idx| MerkleTree::subtree_leafs(&leafs, node_idx);

        // Check all node indices, one level at a time. Going level by level
        // ensures that the expected subtrees (per level) have the same number
        // of leafs.
        for node_indices in [1..2, 2..4, 4..8, 8..16, 16..32, 32..64] {
            // the expected number of leafs in any subtree at the current level
            let num_leafs = tree.num_leafs() / node_indices.len();
            for (i, node_index) in node_indices.enumerate() {
                let expected_leafs = &leafs[i * num_leafs..(i + 1) * num_leafs];
                assert_eq!(expected_leafs, subtree(node_index));
            }
        }
    }

    #[proptest(cases = 10)]
    fn each_leaf_can_be_verified_individually(test_tree: MerkleTreeToTest) {
        let tree = test_tree.tree;
        for (leaf_index, &leaf) in tree.leafs().enumerate() {
            let authentication_path = tree.authentication_structure(&[leaf_index]).unwrap();
            let proof = MerkleTreeInclusionProof {
                tree_height: tree.height(),
                indexed_leafs: [(leaf_index, leaf)].into(),
                authentication_structure: authentication_path,
            };
            let verdict = proof.verify(tree.root());
            prop_assert!(verdict);
        }
    }

    #[test]
    fn partial_merkle_tree_built_from_authentication_structure_contains_expected_nodes() {
        let merkle_tree = MerkleTree::test_tree_of_height(3);
        let proof = merkle_tree
            .inclusion_proof_for_leaf_indices(&[0, 2])
            .unwrap();
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
        let mut partial_tree = PartialMerkleTree {
            tree_height: 3,
            leaf_indices: vec![0, 2],
            nodes: PartialMerkleTree::dummy_nodes_for_indices(&node_indices),
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
        let mut partial_tree = PartialMerkleTree {
            tree_height: 3,
            leaf_indices: vec![0, 2],
            nodes: PartialMerkleTree::dummy_nodes_for_indices(&node_indices),
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
        let mut partial_tree = PartialMerkleTree {
            tree_height: 3,
            leaf_indices: vec![0, 2],
            nodes: PartialMerkleTree::dummy_nodes_for_indices(&node_indices),
        };

        let err = partial_tree.fill().unwrap_err();
        assert_eq!(MerkleTreeError::SpuriousNodeIndex(2), err);
    }

    #[test]
    fn converting_authentication_structure_to_authentication_paths_results_in_expected_paths() {
        const TREE_HEIGHT: MerkleTreeHeight = 3;
        let merkle_tree = MerkleTree::test_tree_of_height(TREE_HEIGHT);
        let proof = merkle_tree
            .inclusion_proof_for_leaf_indices(&[0, 2])
            .unwrap();
        let auth_paths = proof.into_authentication_paths().unwrap();

        let auth_path_with_nodes = |indices: [MerkleTreeNodeIndex; TREE_HEIGHT as usize]| {
            indices.map(|i| merkle_tree.node(i).unwrap()).to_vec()
        };
        let expected_path_0 = auth_path_with_nodes([9, 5, 3]);
        let expected_path_1 = auth_path_with_nodes([11, 4, 3]);
        let expected_paths = vec![expected_path_0, expected_path_1];

        assert_eq!(expected_paths, auth_paths);
    }

    #[test]
    fn merkle_subtrees_are_sliced_correctly() {
        const TREE_HEIGHT: usize = 5;
        const NUM_LEAFS: u32 = 1 << TREE_HEIGHT;
        const NUM_NODES: u32 = 2 * NUM_LEAFS;
        debug_assert_eq!(64, NUM_NODES);

        let mut nodes = (0..NUM_NODES).collect_vec();

        let all_nodes = MerkleTree::subtrees_mut(&mut nodes, 1);
        assert_eq!(1, all_nodes.len());
        let subtree = &all_nodes[0];
        assert_eq!([1], subtree[0]);
        assert_eq!([2, 3], subtree[1]);
        assert_eq!((4..8).collect_vec().as_slice(), subtree[2]);
        assert_eq!((8..16).collect_vec().as_slice(), subtree[3]);
        assert_eq!((16..32).collect_vec().as_slice(), subtree[4]);
        assert_eq!((32..64).collect_vec().as_slice(), subtree[5]);

        let two_trees = MerkleTree::subtrees_mut(&mut nodes, 2);
        assert_eq!(2, two_trees.len());
        let left_tree = &two_trees[0];
        assert_eq!([2], left_tree[0]);
        assert_eq!([4, 5], left_tree[1]);
        assert_eq!((8..12).collect_vec().as_slice(), left_tree[2]);
        assert_eq!((16..24).collect_vec().as_slice(), left_tree[3]);
        assert_eq!((32..48).collect_vec().as_slice(), left_tree[4]);
        let right_tree = &two_trees[1];
        assert_eq!([3], right_tree[0]);
        assert_eq!([6, 7], right_tree[1]);
        assert_eq!((12..16).collect_vec().as_slice(), right_tree[2]);
        assert_eq!((24..32).collect_vec().as_slice(), right_tree[3]);
        assert_eq!((48..64).collect_vec().as_slice(), right_tree[4]);

        let four_trees = MerkleTree::subtrees_mut(&mut nodes, 4);
        assert_eq!(4, four_trees.len());
        let left_left_tree = &four_trees[0];
        assert_eq!([4], left_left_tree[0]);
        assert_eq!([8, 9], left_left_tree[1]);
        assert_eq!((16..20).collect_vec().as_slice(), left_left_tree[2]);
        assert_eq!((32..40).collect_vec().as_slice(), left_left_tree[3]);
        let left_right_tree = &four_trees[1];
        assert_eq!([5], left_right_tree[0]);
        assert_eq!([10, 11], left_right_tree[1]);
        assert_eq!((20..24).collect_vec().as_slice(), left_right_tree[2]);
        assert_eq!((40..48).collect_vec().as_slice(), left_right_tree[3]);
        let right_left_tree = &four_trees[2];
        assert_eq!([6], right_left_tree[0]);
        assert_eq!([12, 13], right_left_tree[1]);
        assert_eq!((24..28).collect_vec().as_slice(), right_left_tree[2]);
        assert_eq!((48..56).collect_vec().as_slice(), right_left_tree[3]);
        let right_right_tree = &four_trees[3];
        assert_eq!([7], right_right_tree[0]);
        assert_eq!([14, 15], right_right_tree[1]);
        assert_eq!((28..32).collect_vec().as_slice(), right_right_tree[2]);
        assert_eq!((56..64).collect_vec().as_slice(), right_right_tree[3]);
    }

    #[test]
    fn auth_structure_node_indices_can_be_computed_with_different_types() -> Result<()> {
        let u32_indices: Vec<u32> =
            MerkleTree::authentication_structure_node_indices(8, &[0_u32, 1])?.collect();
        assert_eq!(vec![5, 3], u32_indices);

        let u64_indices: Vec<u64> =
            MerkleTree::authentication_structure_node_indices(8, &[0_u64, 2])?.collect();
        assert_eq!(vec![11, 9, 3], u64_indices);

        let usize_indices: Vec<usize> =
            MerkleTree::authentication_structure_node_indices(8, &[4_usize, 5, 6, 7])?.collect();
        assert_eq!(vec![2], usize_indices);

        Ok(())
    }
}
