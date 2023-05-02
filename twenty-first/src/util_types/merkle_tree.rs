use itertools::izip;
use itertools::Itertools;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ops::DerefMut;

use crate::shared_math::other::{bit_representation, is_power_of_two, log_2_floor};
use crate::shared_math::rescue_prime_digest::Digest;
use crate::util_types::algebraic_hasher::AlgebraicHasher;
use crate::util_types::merkle_tree_maker::MerkleTreeMaker;
use crate::util_types::shared::bag_peaks;

// Chosen from a very small number of benchmark runs, optimized for a slow
// hash function (the original Rescue Prime implementation). It should probably
// be a higher number than 16 when using a faster hash function.
const PARALLELLIZATION_THRESHOLD: usize = 16;

#[derive(Debug)]
pub struct MerkleTree<H>
where
    H: AlgebraicHasher,
{
    pub nodes: Vec<Digest>,
    pub _hasher: PhantomData<H>,
}

impl<H> Clone for MerkleTree<H>
where
    H: AlgebraicHasher,
{
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            _hasher: PhantomData,
        }
    }
}

/// A partial authentication path is like a full authentication path, but with some nodes missing.
/// A single partial authentication path probably does not make a lot of sense. However, if you
/// have multiple authentication paths that overlap, using multiple partial authentication paths
/// is more space efficient.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct PartialAuthenticationPath<Digest>(pub Vec<Option<Digest>>);

impl Deref for PartialAuthenticationPath<Digest> {
    type Target = Vec<Option<Digest>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for PartialAuthenticationPath<Digest> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// # Design
/// Static methods are called from the verifier, who does not have
/// the original `MerkleTree` object, but only partial information from it,
/// in the form of the quadruples: `(root_hash, index, digest, auth_path)`.
/// These are exactly the arguments for the `verify_*` family of static methods.
impl<H> MerkleTree<H>
where
    H: AlgebraicHasher,
{
    /// Calculate a Merkle root from a list of digests that is not necessarily a power of two.
    pub fn root_from_arbitrary_number_of_digests(digests: &[Digest]) -> Digest {
        // This function should preferably construct a whole Merkle tree data structure and not just the root,
        // but I couldn't figure out how to do that as the indexing for this problem seems hard to me. Perhaps, the
        // data structure would need to be changed, since some of the nodes will be `None`/null.

        // The main reason this function exists is that I wanted to be able to calculate a Merkle
        // root from an odd (non-2^k) number of digests in parallel. This will be used when calculating the digest
        // of a block, where one of the components is a list of MS addition/removal records.

        // Note that this function *does* allow the calculation of a MT root from an empty list of digests
        // since the number of removal records in a block can be zero.

        let heights = bit_representation(digests.len() as u64);
        let mut trees: Vec<MerkleTree<H>> = vec![];
        let mut acc_counter = 0;
        for height in heights {
            let sub_tree =
                CpuParallel::from_digests(&digests[acc_counter..acc_counter + (1 << height)]);
            acc_counter += 1 << height;
            trees.push(sub_tree);
        }

        // Calculate the root from a list of Merkle trees
        let roots: Vec<Digest> = trees.iter().map(|t| t.get_root()).collect();

        bag_peaks::<H>(&roots)
    }

    /// Similar to `get_proof', but instead of returning a `Vec<Node<T>>`, we only
    /// return the hashes, not the tree nodes (and potential leaf values), and instead
    /// of referring to this as a `proof', we call it the `authentication path'.
    ///
    /// ```markdown
    ///              root
    ///             /    \
    /// H(H(a)+H(b))      H(H(c)+H(d))
    ///   /      \        /      \
    /// H(a)    H(b)    H(c)    H(d)
    /// ```
    ///
    /// The authentication path for `c` (index: 2) would be `vec![ H(d), H(H(a)+H(b)) ]`, i.e.,
    /// a criss-cross of siblings upwards.
    pub fn get_authentication_path(&self, leaf_index: usize) -> Vec<Digest> {
        let auth_path_len = self.get_height();
        let mut auth_path: Vec<Digest> = Vec::with_capacity(auth_path_len);

        let num_leaves = self.nodes.len() / 2;
        let mut node_index = leaf_index + num_leaves;
        while node_index > 1 {
            // We get the sibling node by XOR'ing with 1.
            let sibling_i = node_index ^ 1;
            auth_path.push(self.nodes[sibling_i]);
            node_index /= 2;
        }

        // We don't include the root hash in the authentication path
        // because it's implied in the context of use.
        debug_assert_eq!(auth_path_len, auth_path.len());
        auth_path
    }

    // Consider renaming this `verify_leaf_with_authentication_path()`.
    pub fn verify_authentication_path_from_leaf_hash(
        root_hash: Digest,
        leaf_index: u32,
        leaf_hash: Digest,
        auth_path: Vec<Digest>,
    ) -> bool {
        let path_length = auth_path.len() as u32;

        // Initialize `acc_hash' as leaf_hash
        let mut acc_hash = leaf_hash;
        let mut i = leaf_index + 2u32.pow(path_length);
        for path_hash in auth_path.iter() {
            // Use Merkle tree index parity (odd/even) to determine which
            // order to concatenate the hashes before hashing them.
            match i % 2 {
                0 => acc_hash = H::hash_pair(&acc_hash, path_hash),
                1 => acc_hash = H::hash_pair(path_hash, &acc_hash),
                _ => unreachable!(),
            }
            i /= 2;
        }

        acc_hash == root_hash
    }

    /// Compact Merkle Authentication Structure Generation
    pub fn get_authentication_structure(
        &self,
        leaf_indices: &[usize],
    ) -> Vec<PartialAuthenticationPath<Digest>> {
        let num_leaves = self.nodes.len() / 2;
        let mut calculable_indices: HashSet<usize> = HashSet::new();
        let mut partial_authentication_paths: Vec<PartialAuthenticationPath<Digest>> =
            Vec::with_capacity(leaf_indices.len());
        for leaf_index in leaf_indices.iter() {
            let authentication_path = PartialAuthenticationPath(
                self.get_authentication_path(*leaf_index)
                    .into_iter()
                    .map(Some)
                    .collect(),
            );
            let mut node_index = num_leaves + leaf_index;
            calculable_indices.insert(node_index);
            for _ in 1..authentication_path.len() {
                let sibling_node_index = node_index ^ 1;
                calculable_indices.insert(sibling_node_index);
                node_index /= 2;
            }
            partial_authentication_paths.push(authentication_path);
        }

        let mut complete = false;
        while !complete {
            complete = true;
            let parent_node_indices: Vec<usize> =
                calculable_indices.iter().map(|&x| x / 2).collect();

            for parent_node_index in parent_node_indices.into_iter() {
                let left_child_node_index = parent_node_index * 2;
                let right_child_node_index = left_child_node_index + 1;
                if calculable_indices.contains(&left_child_node_index)
                    && calculable_indices.contains(&right_child_node_index)
                    && !calculable_indices.contains(&parent_node_index)
                {
                    calculable_indices.insert(parent_node_index);
                    complete = false;
                }
            }
        }

        let mut scanned: HashSet<usize> = HashSet::new();
        for (leaf_index, partial_authentication_path) in leaf_indices
            .iter()
            .zip(partial_authentication_paths.iter_mut())
        {
            let mut node_index: usize = num_leaves + leaf_index;
            scanned.insert(node_index);
            for authentication_path_element in partial_authentication_path.iter_mut() {
                // Note that the authentication path contains the siblings to the nodes given by
                // the list (node_index, node_index / 2, node_index / 4, …). Hence, all the tests
                // performed here exclusively deal with the current node's sibling.
                let sibling_node_index = node_index ^ 1;
                let sibling_already_covered = scanned.contains(&sibling_node_index);

                let siblings_left_child_node_index = sibling_node_index * 2;
                let siblings_right_child_node_index = siblings_left_child_node_index + 1;
                let both_sibling_children_can_be_calculated = calculable_indices
                    .contains(&siblings_left_child_node_index)
                    && calculable_indices.contains(&siblings_right_child_node_index);

                let sibling_is_in_leaf_layer = sibling_node_index >= num_leaves;
                let sibling_is_explicitly_requested = if sibling_is_in_leaf_layer {
                    let sibling_leaf_index = sibling_node_index - num_leaves;
                    leaf_indices.contains(&sibling_leaf_index)
                } else {
                    // Only leaves can be explicitly requested.
                    false
                };

                let authentication_path_element_is_redundant = sibling_already_covered
                    || both_sibling_children_can_be_calculated
                    || sibling_is_explicitly_requested;
                if authentication_path_element_is_redundant {
                    *authentication_path_element = None;
                }
                scanned.insert(sibling_node_index);
                node_index /= 2;
            }
        }

        partial_authentication_paths
    }

    /// Verifies a list of indicated digests and corresponding authentication paths against a
    /// Merkle root.
    ///
    /// # Arguments
    ///
    /// * `root` - The Merkle root
    /// * `tree_height` - The height of the Merkle tree
    /// * `leaf_indices` - List of identifiers of the leaves to verify
    /// * `leaf_digests` - List of the leaves' values (i.e. digests) to verify
    /// * `auth_paths` - List of paths corresponding to the leaves.
    pub fn verify_authentication_structure_from_leaves(
        root: Digest,
        tree_height: usize,
        leaf_indices: &[usize],
        leaf_digests: &[Digest],
        partial_auth_paths: &[PartialAuthenticationPath<Digest>],
    ) -> bool {
        let num_authentication_paths = partial_auth_paths.len();
        if leaf_indices.len() != num_authentication_paths {
            return false;
        }
        if leaf_digests.len() != num_authentication_paths {
            return false;
        }
        if leaf_indices.is_empty() {
            return true;
        }

        let num_leaves = 1 << tree_height;
        let num_nodes = num_leaves * 2;

        // All leaf indices must be valid. Uniqueness is not required.
        if leaf_indices.iter().any(|&i| i >= num_leaves) {
            return false;
        }

        // The partial merkle tree is represented as a vector of Option<Digest> where the
        // Option is None if the node is not in the partial tree.
        // The indexing works identical as in the general Merkle tree.
        let mut partial_merkle_tree = vec![None; num_nodes];

        // Translate partial authentication paths into partial merkle tree.
        for (&leaf_index, &leaf_digest, partial_authentication_path) in
            izip!(leaf_indices, leaf_digests, partial_auth_paths)
        {
            let mut current_node_index = leaf_index + num_leaves;

            if partial_merkle_tree[current_node_index].is_none() {
                partial_merkle_tree[current_node_index] = Some(leaf_digest);
            } else if partial_merkle_tree[current_node_index] != Some(leaf_digest) {
                // In case of repeated leaf indices, the leaf digests must be identical.
                return false;
            }

            for &sibling in partial_authentication_path.iter() {
                if let Some(sibling) = sibling {
                    // Check that the partial tree does not already have an entry at the current
                    // sibling node index. This would indicate that the authentication paths are
                    // not fully de-duplicated. Whether due to benign or malicious reasons, this
                    // is not allowed.
                    let sibling_node_index = current_node_index ^ 1;
                    if partial_merkle_tree[sibling_node_index].is_some() {
                        return false;
                    }
                    partial_merkle_tree[sibling_node_index] = Some(sibling);
                }
                current_node_index /= 2;
            }
            // Assert the length of the partial authentication path is consistent with the height
            // of the partial tree. A different length might indicate a maliciously constructed
            // partial authentication path.
            if current_node_index != 1 {
                return false;
            }
        }

        // In order to perform the minimal number of hash operations, we only hash the nodes that
        // are required to calculate the root. This is done by starting at the leaves and
        // calculating the parent nodes. The parent nodes are then used to calculate their parent
        // nodes, and so on, until the root is reached.
        // The parent nodes' indices are deduplicated to avoid hashing the same nodes twice,
        // which would happen whenever two leaves are siblings.
        let mut parent_node_indices = leaf_indices
            .iter()
            .map(|&leaf_index| (leaf_index + num_leaves) / 2)
            .collect_vec();
        parent_node_indices.sort();
        parent_node_indices.dedup();

        // Hash the partial tree from the bottom up.
        for _ in 0..tree_height {
            for &parent_node_index in parent_node_indices.iter() {
                let left_node_index = parent_node_index * 2;
                let right_node_index = left_node_index ^ 1;

                // Check that the parent node does not already exist. This would indicate that the
                // partial authentication paths are not fully de-duplicated.
                // This, in turn, might point to inconsistency or maliciousness, both of which
                // should be rejected.
                if partial_merkle_tree[parent_node_index].is_some() {
                    return false;
                }

                // Similarly, check that the children nodes do exist. If they don't, the partial
                // authentication paths are incomplete, making verification impossible.
                if partial_merkle_tree[left_node_index].is_none()
                    || partial_merkle_tree[right_node_index].is_none()
                {
                    return false;
                }
                let parent_digest = H::hash_pair(
                    &partial_merkle_tree[left_node_index].unwrap(),
                    &partial_merkle_tree[right_node_index].unwrap(),
                );
                partial_merkle_tree[parent_node_index] = Some(parent_digest);
            }

            // Move the indices for the parent nodes one layer up, deduplicate to guarantee the
            // minimal number of hash operations.
            parent_node_indices.iter_mut().for_each(|i| *i /= 2);
            parent_node_indices.dedup();
        }

        debug_assert_eq!(1, parent_node_indices.len());
        debug_assert_eq!(0, parent_node_indices[0]);
        debug_assert!(partial_merkle_tree[1].is_some());

        // Finally, check that the root of the partial tree matches the expected root.
        partial_merkle_tree[1] == Some(root)
    }

    /// Verifies a list of `leaf_indices` and corresponding authentication pairs `auth_pairs`,
    /// consisting of authentication paths `auth_path` and leaf digests `leaf_digest`,
    /// against a Merkle root `root` for a Merkle tree of expected height `tree_height`.
    pub fn verify_authentication_structure(
        root: Digest,
        tree_height: usize,
        leaf_indices: &[usize],
        auth_pairs: &[(PartialAuthenticationPath<Digest>, Digest)],
    ) -> bool {
        if leaf_indices.len() != auth_pairs.len() {
            return false;
        }
        if leaf_indices.is_empty() {
            return true;
        }

        let (auth_paths, leaves): (Vec<_>, Vec<_>) = auth_pairs.iter().cloned().unzip();
        Self::verify_authentication_structure_from_leaves(
            root,
            tree_height,
            leaf_indices,
            &leaves,
            &auth_paths,
        )
    }

    pub fn get_root(&self) -> Digest {
        self.nodes[1]
    }

    pub fn get_leaf_count(&self) -> usize {
        let node_count = self.nodes.len();
        assert!(is_power_of_two(node_count));
        node_count / 2
    }

    pub fn get_height(&self) -> usize {
        let leaf_count = self.get_leaf_count() as u128;
        assert!(is_power_of_two(leaf_count));
        log_2_floor(leaf_count) as usize
    }

    pub fn get_all_leaves(&self) -> Vec<Digest> {
        let first_leaf = self.nodes.len() / 2;
        self.nodes[first_leaf..].to_vec()
    }

    pub fn get_leaf_by_index(&self, index: usize) -> Digest {
        let first_leaf_index = self.nodes.len() / 2;
        let beyond_last_leaf_index = self.nodes.len();
        assert!(
            index < first_leaf_index || beyond_last_leaf_index <= index,
            "Out of bounds index requested"
        );
        self.nodes[first_leaf_index + index]
    }

    pub fn get_leaves_by_indices(&self, leaf_indices: &[usize]) -> Vec<Digest> {
        let leaf_count = leaf_indices.len();

        let mut result = Vec::with_capacity(leaf_count);

        for index in leaf_indices {
            result.push(self.get_leaf_by_index(*index));
        }
        result
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
            is_power_of_two(leaves_count),
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
        while node_count_on_this_level >= PARALLELLIZATION_THRESHOLD {
            let mut local_digests: Vec<Digest> = Vec::with_capacity(node_count_on_this_level);
            (0..node_count_on_this_level)
                .into_par_iter()
                .map(|i| {
                    let j = node_count_on_this_level + i;
                    let left_child = &nodes[j * 2];
                    let right_child = &nodes[j * 2 + 1];
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
            nodes[i] = H::hash_pair(&nodes[i * 2], &nodes[i * 2 + 1]);
        }

        MerkleTree {
            nodes,
            _hasher: PhantomData,
        }
    }
}

#[cfg(test)]
mod merkle_tree_test {
    use super::*;
    use crate::shared_math::b_field_element::BFieldElement;
    use crate::shared_math::other::{
        random_elements, random_elements_distinct_range, random_elements_range,
    };
    use crate::shared_math::rescue_prime_regular::RescuePrimeRegular;
    use crate::shared_math::tip5::Tip5;
    use crate::shared_math::x_field_element::XFieldElement;
    use crate::test_shared::corrupt_digest;
    use itertools::Itertools;
    use rand::thread_rng;
    use rand::Rng;
    use rand::RngCore;
    use std::iter::zip;

    #[test]
    fn merkle_tree_test_32() {
        type H = blake3::Hasher;
        type M = CpuParallel;
        type MT = MerkleTree<H>;

        let tree_height = 5;
        let num_leaves = 1 << tree_height;
        let leaves: Vec<Digest> = random_elements(num_leaves);
        let tree: MT = M::from_digests(&leaves);

        for test_size in 0..20 {
            // Create a vector of distinct, uniform random indices `random_indices`
            // Separate one of these distinct indices `random_index` for negative testing.
            let num_indices = test_size + 10;
            let (bad_index, random_indices): (usize, Vec<usize>) = {
                let mut tmp = random_elements_distinct_range(num_indices, 0..num_leaves);
                (tmp.remove(0), tmp)
            };

            // Get a vector of digests for each of those indices
            let selected_leaves: Vec<Digest> = tree.get_leaves_by_indices(&random_indices);

            // Get the partial authentication paths for those indices
            let partial_auth_paths = tree.get_authentication_structure(&random_indices);

            // Get a membership proof for those indices
            let proof: Vec<(PartialAuthenticationPath<Digest>, Digest)> =
                zip(partial_auth_paths, selected_leaves.clone()).collect();

            // Assert membership of randomly chosen leaves
            let random_leaves_are_members = MT::verify_authentication_structure(
                tree.get_root(),
                tree_height,
                &random_indices,
                &proof,
            );
            assert!(random_leaves_are_members);

            // Assert completeness of proof
            let all_randomly_chosen_leaves_occur_in_proof = proof
                .iter()
                .enumerate()
                .all(|(i, (_auth_path, digest))| *digest == leaves[random_indices[i]]);
            assert!(all_randomly_chosen_leaves_occur_in_proof);

            // Negative: Verify bad Merkle root
            let bad_root_digest = corrupt_digest(&tree.get_root());
            let bad_root_verifies = MT::verify_authentication_structure(
                bad_root_digest,
                tree_height,
                &random_indices,
                &proof,
            );
            assert!(!bad_root_verifies);

            // Negative: Make random indices not match proof length (too long)
            let bad_random_indices_1 = {
                let mut tmp = random_indices.clone();
                tmp.push(tmp[0]);
                tmp
            };
            let too_many_indices_verifies = MT::verify_authentication_structure(
                tree.get_root(),
                tree_height,
                &bad_random_indices_1,
                &proof,
            );
            assert!(!too_many_indices_verifies);

            // Negative: Make random indices not match proof length (too short)
            let bad_random_indices_2 = {
                let mut tmp = random_indices.clone();
                tmp.remove(0);
                tmp
            };
            let too_few_indices_verifies = MT::verify_authentication_structure(
                tree.get_root(),
                tree_height,
                &bad_random_indices_2,
                &proof,
            );
            assert!(!too_few_indices_verifies);

            // Negative: Request non-existent index
            let bad_random_indices_3 = {
                let mut tmp = random_indices.clone();
                tmp[0] = bad_index;
                tmp
            };
            let non_existent_index_verifies = MT::verify_authentication_structure(
                tree.get_root(),
                tree_height,
                &bad_random_indices_3,
                &proof,
            );
            assert!(!non_existent_index_verifies);
        }
    }

    #[test]
    fn merkle_tree_verify_authentication_structure_degenerate_test() {
        type H = blake3::Hasher;
        type M = CpuParallel;
        type MT = MerkleTree<H>;

        let tree_height = 5;
        let num_leaves = 1 << tree_height;
        let leaves: Vec<Digest> = random_elements(num_leaves);
        let tree: MT = M::from_digests(&leaves);

        let empty_proof = tree.get_authentication_structure(&[]);
        let auth_pairs = zip(empty_proof, leaves).collect_vec();
        let empty_proof_verifies =
            MT::verify_authentication_structure(tree.get_root(), tree_height, &[], &auth_pairs);
        assert!(empty_proof_verifies);
    }

    #[test]
    fn merkle_tree_verify_authentication_structure_test() {
        type H = blake3::Hasher;
        type M = CpuParallel;
        type MT = MerkleTree<H>;
        let mut rng = thread_rng();

        for tree_height in 2..=13 {
            let num_leaves = 1 << tree_height;
            let leaves: Vec<Digest> = random_elements(num_leaves);
            let tree: MT = M::from_digests(&leaves);

            for _ in 0..3 {
                // Generate an arbitrary, positive amount of indices less than the total
                let num_indices = (rng.next_u64() % num_leaves as u64 / 2) as usize + 1;

                // Generate that amount of indices in the valid index range [0,num_leaves)
                let selected_indices: Vec<usize> =
                    random_elements_range(num_indices, 0..num_leaves)
                        .iter()
                        .copied()
                        .unique()
                        .collect();

                let selected_leaves = tree.get_leaves_by_indices(&selected_indices);
                let selected_auth_paths = tree.get_authentication_structure(&selected_indices);

                for auth_path in selected_auth_paths.iter() {
                    assert_eq!(tree_height, auth_path.len());
                }

                let auth_pairs: Vec<(PartialAuthenticationPath<Digest>, Digest)> =
                    zip(selected_auth_paths, selected_leaves.clone()).collect();

                let good_tree = MT::verify_authentication_structure(
                    tree.get_root(),
                    tree_height,
                    &selected_indices,
                    &auth_pairs,
                );
                assert!(
                    good_tree,
                    "An uncorrupted tree and an uncorrupted proof should verify."
                );

                // Negative: Corrupt the root and thereby the tree
                let bad_root_hash = corrupt_digest(&tree.get_root());

                let verified = MT::verify_authentication_structure(
                    bad_root_hash,
                    tree_height,
                    &selected_indices,
                    &auth_pairs,
                );
                assert!(!verified, "Should not verify against bad root hash.");

                // Negative: Corrupt proof at random index
                let bad_proof = {
                    let mut tmp = auth_pairs.clone();
                    let random_index =
                        ((rng.next_u64() % num_indices as u64 / 2) as usize) % tmp.len();
                    tmp[random_index].1 = corrupt_digest(&tmp[random_index].1);
                    tmp
                };

                let corrupted_proof_verifies = MT::verify_authentication_structure(
                    tree.get_root(),
                    tree_height,
                    &selected_indices,
                    &bad_proof,
                );
                assert!(!corrupted_proof_verifies);
            }
        }
    }

    #[test]
    fn fail_on_bad_specified_length_test() {
        type H = blake3::Hasher;
        type M = CpuParallel;
        type MT = MerkleTree<H>;
        let tree_height = 5;
        let num_leaves = 1 << tree_height;
        let leaf_digests: Vec<Digest> = random_elements(num_leaves);
        let tree: MT = M::from_digests(&leaf_digests);

        let leaf_indices = [0, 3, 5];
        let opened_leaves = leaf_indices.iter().map(|&i| leaf_digests[i]).collect_vec();
        let mut authentication_structure = tree.get_authentication_structure(&leaf_indices);
        assert!(
            !MT::verify_authentication_structure_from_leaves(
                tree.get_root(),
                tree_height - 1,
                &leaf_indices,
                &opened_leaves,
                &authentication_structure
            ),
            "Must return false when called with wrong height, minus one"
        );

        assert!(
            !MT::verify_authentication_structure_from_leaves(
                tree.get_root(),
                tree_height + 1,
                &leaf_indices,
                &opened_leaves,
                &authentication_structure
            ),
            "Must return false when called with wrong height, plus one"
        );

        assert!(
            MT::verify_authentication_structure_from_leaves(
                tree.get_root(),
                tree_height,
                &leaf_indices,
                &opened_leaves,
                &authentication_structure
            ),
            "Must return true when called with correct height"
        );

        // Modify length of *one* authentication path. Verify failure.
        authentication_structure[1].pop();
        assert!(
            !MT::verify_authentication_structure_from_leaves(
                tree.get_root(),
                tree_height,
                &leaf_indices,
                &opened_leaves,
                &authentication_structure
            ),
            "Must return false when called with too short an auth path"
        );
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
        let authentication_structure = tree.get_authentication_structure(&leaf_indices);
        let verdict = MT::verify_authentication_structure_from_leaves(
            tree.get_root(),
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
        let verdict_for_incorrect_statement = MT::verify_authentication_structure_from_leaves(
            tree.get_root(),
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
    fn merkle_tree_get_authentication_path_test() {
        type H = blake3::Hasher;
        type M = CpuParallel;
        type MT = MerkleTree<H>;

        // 1: Create Merkle tree
        //
        //     root
        //    /    \
        //   x      y
        //  / \    / \
        // 0   1  2   3
        let num_leaves_a = 4;
        let leaves_a: Vec<Digest> = random_elements(num_leaves_a);
        let tree_a: MT = M::from_digests(&leaves_a);

        // 2: Get the path for some index
        let leaf_index_a = 2;
        let auth_path_a = tree_a.get_authentication_path(leaf_index_a);

        let auth_path_a_len = 2;
        assert_eq!(auth_path_a_len, auth_path_a.len());
        assert_eq!(tree_a.nodes[2], auth_path_a[1]);
        assert_eq!(tree_a.nodes[7], auth_path_a[0]);

        // 1: Create Merkle tree
        //
        //        ___root___
        //       /          \
        //      e            f
        //    /   \        /   \
        //   a     b      c     d
        //  / \   / \    / \   / \
        // 0   1 2   3  4   5 6   7
        let num_leaves_b = 8;
        let leaves_b: Vec<Digest> = random_elements(num_leaves_b);
        let tree_b: MT = M::from_digests(&leaves_b);

        // 2: Get the path for some index
        let leaf_index_b = 5;
        let auth_path_b = tree_b.get_authentication_path(leaf_index_b);

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

        type H = RescuePrimeRegular;
        type M = CpuParallel;
        type MT = MerkleTree<H>;

        let exponent = 6;
        let num_leaves = usize::pow(2, exponent);
        assert!(
            is_power_of_two(num_leaves),
            "Size of input for Merkle tree must be a power of 2"
        );

        let offset = 17;

        let values: Vec<[BFieldElement; 1]> = (offset..num_leaves + offset)
            .map(|i| [BFieldElement::new(i as u64)])
            .collect_vec();

        let leafs = values.iter().map(|leaf| H::hash_varlen(leaf)).collect_vec();

        let tree: MT = M::from_digests(&leafs);

        assert_eq!(
            tree.get_leaf_count(),
            num_leaves,
            "All leaves should have been added to the Merkle tree."
        );

        let root_hash = tree.get_root().to_owned();

        for (leaf_idx, leaf) in leafs.iter().enumerate() {
            let ap = tree.get_authentication_path(leaf_idx);
            let verdict = MT::verify_authentication_path_from_leaf_hash(
                root_hash,
                leaf_idx as u32,
                *leaf,
                ap,
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

        type H = RescuePrimeRegular;
        type M = CpuParallel;
        type MT = MerkleTree<H>;

        let exponent = 6;
        let num_leaves = usize::pow(2, exponent);
        assert!(
            is_power_of_two(num_leaves),
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
            tree.get_leaf_count(),
            num_leaves,
            "All leaves should have been added to the Merkle tree."
        );

        let root_hash = tree.get_root().to_owned();

        let ap = tree.get_authentication_path(test_leaf_idx);
        let verdict = MT::verify_authentication_path_from_leaf_hash(
            root_hash,
            test_leaf_idx as u32,
            H::hash_varlen(&payload_leaf),
            ap,
        );
        assert_eq!(
            tree.get_leaf_by_index(test_leaf_idx),
            H::hash_varlen(&payload_leaf)
        );
        assert!(
            verdict,
            "Rejected: `leaf: {payload_leaf:?}` at `leaf_idx: {test_leaf_idx:?}` failed to verify."
        );
    }

    #[test]
    fn root_from_odd_number_of_digests_test() {
        type H = RescuePrimeRegular;
        type M = CpuParallel;
        type MT = MerkleTree<H>;

        let leafs: Vec<Digest> = random_elements(128);
        let mt: MT = M::from_digests(&leafs);

        println!("Merkle root (RP 1): {:?}", mt.get_root());

        assert_eq!(
            mt.get_root(),
            MT::root_from_arbitrary_number_of_digests(&leafs)
        );
    }

    #[test]
    fn root_from_arbitrary_number_of_digests_empty_test() {
        // Ensure that we can calculate a Merkle root from an empty list of digests.
        // This is needed since a block can contain an empty list of addition or
        // removal records.

        type H = RescuePrimeRegular;
        type MT = MerkleTree<H>;

        MT::root_from_arbitrary_number_of_digests(&[]);
    }

    #[test]
    fn merkle_tree_with_xfes_as_leafs() {
        type MT = MerkleTree<Tip5>;

        let num_leaves = 128;
        let leafs: Vec<XFieldElement> = random_elements(num_leaves);
        let mt: MT = CpuParallel::from_digests(&leafs.iter().map(|&x| x.into()).collect_vec());

        let leaf_index: usize = thread_rng().gen_range(0..num_leaves);
        let path = mt.get_authentication_path(leaf_index);
        let sibling = leafs[leaf_index ^ 1];
        assert_eq!(path[0], sibling.into());
    }
}
