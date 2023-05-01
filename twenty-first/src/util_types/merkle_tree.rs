use itertools::izip;
use itertools::Itertools;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt::Debug;
use std::marker::{PhantomData, Send, Sync};
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
                // performed here exclusively deal with the current node_index's sibling.
                let sibling_node_index = node_index ^ 1;
                let sibling_already_covered = scanned.contains(&sibling_node_index);

                let siblings_left_child_node_index = sibling_node_index * 2;
                let siblings_right_child_node_index = siblings_left_child_node_index + 1;
                let both_sibling_children_can_be_calculated = calculable_indices
                    .contains(&siblings_left_child_node_index)
                    && calculable_indices.contains(&siblings_right_child_node_index);

                // In the leaf layer, check if sibling is explicitly provided
                let potential_sibling_leaf_index = sibling_node_index as i128 - num_leaves as i128;
                let sibling_is_in_leaf_layer = potential_sibling_leaf_index >= 0;
                let sibling_is_explicitly_requested = sibling_is_in_leaf_layer
                    && leaf_indices.contains(&(potential_sibling_leaf_index as usize));

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

        // The partial merkle tree is represented as a vector of Option<Digest> where the
        // Option is None if the node is not in the partial tree.
        // The indexing works identical as in the general Merkle tree.
        let mut partial_merkle_tree = vec![None; num_nodes];

        // All indices must be valid, unique leaf indices. Uniqueness is checked below, when
        // inserting the leaf digests into the partial tree.
        if leaf_indices.iter().any(|&i| i >= num_leaves) {
            return false;
        }

        // Translate partial authentication paths into partial merkle tree.
        for ((&leaf_index, &leaf_digest), partial_authentication_path) in leaf_indices
            .iter()
            .zip_eq(leaf_digests.iter())
            .zip_eq(partial_auth_paths.iter())
        {
            let mut current_node_index = leaf_index + num_leaves;

            // Check that the partial tree does not already have an entry at the current leaf,
            // i.e., that the leaf indices are unique.
            if partial_merkle_tree[current_node_index].is_some() {
                return false;
            }
            partial_merkle_tree[current_node_index] = Some(leaf_digest);

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
        // The parent nodes indices are deduplicated to avoid hashing the same nodes twice.
        // This happens when the set of leaf indices contains neighboring leaves.
        let mut parent_node_indices = leaf_indices
            .iter()
            .map(|&i| (i + num_leaves) / 2)
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

pub type SaltedAuthenticationStructure<Digest> = Vec<(PartialAuthenticationPath<Digest>, Digest)>;

#[derive(Clone, Debug)]
pub struct SaltedMerkleTree<H>
where
    H: AlgebraicHasher,
{
    internal_merkle_tree: MerkleTree<H>,
    salts: Vec<Digest>,
}

impl<H: AlgebraicHasher> SaltedMerkleTree<H>
where
    H: Sync + Send, // FIXME: Remove these.
{
    pub fn from_digests(leaves: &[Digest], salts: &[Digest]) -> Self {
        assert!(
            is_power_of_two(leaves.len()),
            "Size of input for Merkle tree must be a power of 2"
        );

        // nodes[0] is never used for anything.
        let mut nodes: Vec<Digest> = vec![Digest::default(); 2 * leaves.len()];

        for i in 0..leaves.len() {
            let value = leaves[i];
            let leaf_digest = H::hash_pair(&value, &salts[i]);

            nodes[leaves.len() + i] = leaf_digest;
        }

        // loop from `len(L) - 1` to 1
        for i in (1..(nodes.len() / 2)).rev() {
            let left = nodes[i * 2];
            let right = nodes[i * 2 + 1];
            nodes[i] = H::hash_pair(&left, &right);
        }

        let internal_merkle_tree = MerkleTree::<H> {
            nodes,
            _hasher: PhantomData,
        };

        Self {
            internal_merkle_tree,
            salts: salts.to_vec(),
        }
    }

    pub fn get_authentication_path_and_salt(&self, index: usize) -> (Vec<Digest>, Digest) {
        let authentication_path = self.internal_merkle_tree.get_authentication_path(index);
        let salt = self.salts[index];

        (authentication_path, salt)
    }

    pub fn verify_authentication_path(
        root_hash: Digest,
        index: u32,
        leaf: Digest,
        auth_path: Vec<Digest>,
        leaf_salt: Digest,
    ) -> bool {
        let leaf_hash = H::hash_pair(&leaf, &leaf_salt);

        // Set `leaf_hash` to H(value + salts[0..])
        MerkleTree::<H>::verify_authentication_path_from_leaf_hash(
            root_hash, index, leaf_hash, auth_path,
        )
    }

    pub fn get_authentication_structure_and_salt(
        &self,
        indices: &[usize],
    ) -> SaltedAuthenticationStructure<Digest> {
        // Get the partial authentication paths without salts
        let partial_authentication_paths: Vec<PartialAuthenticationPath<Digest>> = self
            .internal_merkle_tree
            .get_authentication_structure(indices);

        // Get the salts associated with the leafs
        // salts are random data, so they cannot be compressed
        let mut ret: SaltedAuthenticationStructure<Digest> = Vec::with_capacity(indices.len());
        for (i, index) in indices.iter().enumerate() {
            let salt = self.salts[*index];
            ret.push((partial_authentication_paths[i].clone(), salt));
        }

        ret
    }

    /// To use this function the user must provide the corresponding *UNSALTED* `leaves`.
    pub fn verify_authentication_structure(
        root: Digest,
        tree_height: usize,
        indices: &[usize],
        unsalted_leaves: &[Digest],
        proof: &SaltedAuthenticationStructure<Digest>,
    ) -> bool {
        if indices.len() != proof.len() || indices.len() != unsalted_leaves.len() {
            debug_assert!(indices.len() == proof.len());
            debug_assert!(indices.len() == unsalted_leaves.len());

            return false;
        }

        if indices.is_empty() {
            return true;
        }

        let mut leaf_hashes: Vec<Digest> = Vec::with_capacity(indices.len());
        for (value, (_, salt)) in izip!(unsalted_leaves, proof) {
            let leaf_hash = H::hash_pair(value, salt);
            leaf_hashes.push(leaf_hash);
        }

        let saltless_proof: Vec<PartialAuthenticationPath<Digest>> =
            proof.iter().map(|x| x.0.clone()).collect();

        MerkleTree::<H>::verify_authentication_structure_from_leaves(
            root,
            tree_height,
            indices,
            &leaf_hashes,
            &saltless_proof,
        )
    }

    pub fn get_root(&self) -> Digest {
        self.internal_merkle_tree.get_root()
    }

    pub fn get_leaf_count(&self) -> usize {
        self.internal_merkle_tree.get_leaf_count()
    }

    #[allow(dead_code)]
    pub fn get_height(&self) -> usize {
        self.internal_merkle_tree.get_height()
    }

    pub fn get_all_salted_leaves(&self) -> Vec<Digest> {
        let first_leaf = self.internal_merkle_tree.nodes.len() / 2;
        self.internal_merkle_tree.nodes[first_leaf..].to_vec()
    }

    pub fn get_salted_leaf_by_index(&self, index: usize) -> Digest {
        let first_leaf_index = self.internal_merkle_tree.nodes.len() / 2;
        let beyond_last_leaf_index = self.internal_merkle_tree.nodes.len();
        assert!(
            index < first_leaf_index || beyond_last_leaf_index <= index,
            "Out of bounds index requested"
        );
        self.internal_merkle_tree.nodes[first_leaf_index + index]
    }

    pub fn get_salted_leaves_by_indices(&self, leaf_indices: &[usize]) -> Vec<Digest> {
        let leaf_count = leaf_indices.len();

        let mut result = Vec::with_capacity(leaf_count);

        for index in leaf_indices {
            result.push(self.get_salted_leaf_by_index(*index));
        }
        result
    }

    pub fn get_salts(&self) -> &[Digest] {
        &self.salts
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

    /// Count the number of hashes present in all partial authentication paths
    fn count_hashes<Digest: Clone>(proof: &SaltedAuthenticationStructure<Digest>) -> usize {
        proof
            .iter()
            .map(|(partial_auth_path, _)| partial_auth_path.0.iter().flatten().count())
            .sum()
    }

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
    fn merkle_tree_verify_authentication_structure_equivalence_test() {
        type H = blake3::Hasher;
        type M = CpuParallel;
        type MT = MerkleTree<H>;

        // Check that regular merkle trees and salted merkle trees with 0 salts work equivalently.

        let tree_height = 4;
        let num_leaves = 1 << tree_height;
        let leaves: Vec<Digest> = random_elements(num_leaves);
        let regular_tree: MT = M::from_digests(&leaves);

        let selected_indices: Vec<usize> = vec![0, 1];
        let selected_leaves = regular_tree.get_leaves_by_indices(&selected_indices);
        let selected_auth_paths = regular_tree.get_authentication_structure(&selected_indices);
        let auth_pairs = zip(selected_auth_paths, selected_leaves.clone()).collect_vec();

        for (partial_auth_path, _digest) in auth_pairs.clone() {
            assert_eq!(
                tree_height,
                partial_auth_path.len(),
                "The length of an authentication path must be equal to the height of the tree."
            );
        }

        let regular_verify = MT::verify_authentication_structure(
            regular_tree.get_root(),
            tree_height,
            &selected_indices,
            &auth_pairs,
        );

        let salts: Vec<Digest> = random_elements(num_leaves);
        let unsalted_salted_tree: SaltedMerkleTree<H> =
            SaltedMerkleTree::<H>::from_digests(&selected_leaves, &salts);

        let salted_proof =
            unsalted_salted_tree.get_authentication_structure_and_salt(&selected_indices);

        let unsalted_salted_verify = SaltedMerkleTree::<H>::verify_authentication_structure(
            unsalted_salted_tree.get_root(),
            unsalted_salted_tree.get_height(),
            &selected_indices,
            &selected_leaves,
            &salted_proof,
        );

        assert_eq!(regular_verify, unsalted_salted_verify);
    }

    #[test]
    fn merkle_tree_verify_authentication_structure_test() {
        type H = blake3::Hasher;
        type M = CpuParallel;
        type MT = MerkleTree<H>;
        let mut rng = rand::thread_rng();

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
        type MT = MerkleTree<H, M>;
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

    fn make_salted_merkle_tree<H: AlgebraicHasher>(
        num_leaves: usize,
        salts_per_leaf: usize,
    ) -> (Vec<Digest>, Vec<Digest>, SaltedMerkleTree<H>) {
        let leaves: Vec<Digest> = random_elements(num_leaves);
        let salts: Vec<Digest> = random_elements(num_leaves * salts_per_leaf);
        let tree = SaltedMerkleTree::<H>::from_digests(&leaves, &salts);
        (leaves, salts, tree)
    }

    // Test of salted Merkle trees
    #[test]
    fn salted_merkle_tree_get_authentication_path_small_test() {
        type H = blake3::Hasher;

        // 1: Create Merkle tree
        //
        //     root
        //    /    \
        //   x      y
        //  / \    / \
        // 0   1  2   3
        let num_leaves = 4;
        let salts_per_leaf = 3;

        let (leaves_a, _salts_a, tree_a) = make_salted_merkle_tree::<H>(num_leaves, salts_per_leaf);
        assert_eq!(tree_a.get_leaf_count() * salts_per_leaf, tree_a.salts.len());

        // 2: Get the path for digest at index
        let leaf_a_index = 2;
        let leaf_a_digest = leaves_a[leaf_a_index];
        let auth_path_a_and_salt = tree_a.get_authentication_path_and_salt(leaf_a_index);

        // 3: Verify that the proof, along with the salt, works
        assert!(SaltedMerkleTree::<H>::verify_authentication_path(
            tree_a.get_root(),
            leaf_a_index as u32,
            leaf_a_digest,
            auth_path_a_and_salt.0.clone(),
            auth_path_a_and_salt.1,
        ));

        assert_eq!(
            leaf_a_index,
            auth_path_a_and_salt.0.len(),
            "authentication path a has right length"
        );
        assert_eq!(
            tree_a.internal_merkle_tree.nodes[2],
            auth_path_a_and_salt.0[1],
        );
        assert_eq!(
            tree_a.internal_merkle_tree.nodes[7],
            auth_path_a_and_salt.0[0],
        );

        // 4: Negative: Change salt and verify that the proof does not work
        let auth_path_a_and_bad_salt = {
            let mut tmp = auth_path_a_and_salt.clone();
            tmp.1 = corrupt_digest(&auth_path_a_and_salt.1);
            tmp
        };
        let bad_salt_verifies = SaltedMerkleTree::<H>::verify_authentication_path(
            tree_a.get_root(),
            leaf_a_index as u32,
            leaf_a_digest,
            auth_path_a_and_bad_salt.0.clone(),
            auth_path_a_and_bad_salt.1,
        );
        assert!(!bad_salt_verifies);

        // 5: Change the value and verify that the proof does not work
        let corrupt_leaf_a_digest = corrupt_digest(&leaf_a_digest);
        let corrupt_leaf_verifies = SaltedMerkleTree::<H>::verify_authentication_path(
            tree_a.get_root(),
            leaf_a_index as u32,
            corrupt_leaf_a_digest,
            auth_path_a_and_bad_salt.0.clone(),
            auth_path_a_and_bad_salt.1,
        );
        assert!(!corrupt_leaf_verifies);
    }

    #[test]
    fn salted_merkle_tree_get_authentication_path_medium_test() {
        type H = blake3::Hasher;

        //        ___root___
        //       /          \
        //      e            f
        //    /   \        /   \
        //   a     b      c     d
        //  / \   / \    / \   / \
        // 0   1 2   3  4   5 6   7
        let num_leaves = 8;
        let salts_per_leaf = 4;

        let (leaves_b, _salts_b, tree_b) = make_salted_merkle_tree::<H>(num_leaves, salts_per_leaf);

        let leaf_b_index = 6;
        let auth_path_and_salts_b = tree_b.get_authentication_path_and_salt(leaf_b_index);

        let auth_path_b_len = 3;
        assert_eq!(auth_path_b_len, auth_path_and_salts_b.0.len());
        assert_eq!(num_leaves * salts_per_leaf, tree_b.get_salts().len());

        // 6: Ensure that all salts are most probably unique
        let unique_salts = tree_b.get_salts().iter().unique();
        assert_eq!(
            tree_b.get_salts().len(),
            unique_salts.count(),
            "Salts are most probably unique"
        );
        assert_eq!(
            tree_b.internal_merkle_tree.nodes[15],
            auth_path_and_salts_b.0[0],
        );
        assert_eq!(
            tree_b.internal_merkle_tree.nodes[6],
            auth_path_and_salts_b.0[1],
        );
        assert_eq!(
            tree_b.internal_merkle_tree.nodes[2],
            auth_path_and_salts_b.0[2],
        );

        let test_value_b = leaves_b[leaf_b_index];
        let root_hash_b = &tree_b.get_root();
        let leaf_b_verifies = SaltedMerkleTree::<H>::verify_authentication_path(
            *root_hash_b,
            leaf_b_index as u32,
            test_value_b,
            auth_path_and_salts_b.0.clone(),
            auth_path_and_salts_b.1,
        );
        assert!(leaf_b_verifies);

        // 7: Negative: Change the value and verify that it fails
        let corrupt_leaf = corrupt_digest(&test_value_b);
        let corrupt_leaf_verifies = SaltedMerkleTree::<H>::verify_authentication_path(
            *root_hash_b,
            6,
            corrupt_leaf,
            auth_path_and_salts_b.0.clone(),
            auth_path_and_salts_b.1,
        );
        assert!(!corrupt_leaf_verifies);

        // 8: Negative: Change salt and verify that verification fails
        let auth_path_and_bad_salts = {
            let mut tmp = auth_path_and_salts_b;
            tmp.1 = corrupt_digest(&tmp.1);
            tmp
        };
        let bad_salts_verify = SaltedMerkleTree::<H>::verify_authentication_path(
            *root_hash_b,
            6,
            test_value_b,
            auth_path_and_bad_salts.0,
            auth_path_and_bad_salts.1,
        );
        assert!(!bad_salts_verify);
    }

    fn make_salted_merkle_tree_test<H: AlgebraicHasher>(
        leaf_indices: &[usize],
        leaves_c: &[Digest],
        tree: &SaltedMerkleTree<H>,
        tree_root: Digest,
    ) -> (bool, Vec<(PartialAuthenticationPath<Digest>, Digest)>) {
        let auth_path_and_salts: SaltedAuthenticationStructure<Digest> =
            tree.get_authentication_structure_and_salt(leaf_indices);

        let selected_leaves: Vec<Digest> = leaf_indices
            .iter()
            .map(|leaf_index| leaves_c[*leaf_index])
            .collect();

        let salted_merkle_tree_verifies = SaltedMerkleTree::<H>::verify_authentication_structure(
            tree_root,
            tree.get_height(),
            leaf_indices,
            &selected_leaves,
            &auth_path_and_salts,
        );

        (salted_merkle_tree_verifies, auth_path_and_salts)
    }

    #[test]
    fn salted_merkle_tree_multipath_test() {
        type H = blake3::Hasher;

        let num_leaves = 8;
        let salts_per_leaf = 3;

        let (leaves_c, _salts_c, tree_c) = make_salted_merkle_tree::<H>(num_leaves, salts_per_leaf);
        let root_hash_c = tree_c.get_root();
        let tree_height_c = tree_c.get_height();

        // 9: Verify that simple multipath authentication paths work
        {
            let leaf_indices = [0, 1];
            let (salted_merkle_tree_verifies, auth_path_and_salts) =
                make_salted_merkle_tree_test(&leaf_indices, &leaves_c, &tree_c, tree_c.get_root());
            assert!(salted_merkle_tree_verifies);

            let expected_hashes = 2;
            assert_eq!(expected_hashes, count_hashes(&auth_path_and_salts));
        }

        // Assert that bad root hash does not verify
        {
            let leaf_indices = [0, 1];
            let bad_root_hash_c = corrupt_digest(&root_hash_c);
            let (salted_merkle_tree_verifies, _auth_path_and_salts) =
                make_salted_merkle_tree_test(&leaf_indices, &leaves_c, &tree_c, bad_root_hash_c);

            assert!(!salted_merkle_tree_verifies);
        }

        {
            let leaf_indices = [1];
            let (salted_merkle_tree_verifies, auth_path_and_salts) =
                make_salted_merkle_tree_test(&leaf_indices, &leaves_c, &tree_c, tree_c.get_root());
            assert!(salted_merkle_tree_verifies);

            let expected_hashes = 3;
            assert_eq!(expected_hashes, count_hashes(&auth_path_and_salts));
        }

        {
            let leaf_indices = [1, 0];
            let (salted_merkle_tree_verifies, auth_path_and_salts) =
                make_salted_merkle_tree_test(&leaf_indices, &leaves_c, &tree_c, tree_c.get_root());
            assert!(salted_merkle_tree_verifies);

            let expected_hashes = 2;
            assert_eq!(expected_hashes, count_hashes(&auth_path_and_salts));
        }

        {
            let leaf_indices = [0, 1, 2, 4, 7];
            let (salted_merkle_tree_verifies, mut auth_path_and_salts) =
                make_salted_merkle_tree_test(&leaf_indices, &leaves_c, &tree_c, tree_c.get_root());
            assert!(salted_merkle_tree_verifies);

            let expected_hashes = 3;
            assert_eq!(
                expected_hashes,
                count_hashes(&auth_path_and_salts),
                "paths [0, 1, 2, 4, 7] need three hashes"
            );

            let selected_leaves = leaf_indices
                .into_iter()
                .map(|leaf_index| leaves_c[leaf_index])
                .collect_vec();

            // 10: change a hash, verify failure
            let orig_digest = auth_path_and_salts[1].1;
            auth_path_and_salts[1].1 = corrupt_digest(&orig_digest);
            assert!(!SaltedMerkleTree::<H>::verify_authentication_structure(
                root_hash_c,
                tree_height_c,
                &leaf_indices,
                &selected_leaves,
                &auth_path_and_salts
            ));

            auth_path_and_salts[1].1 = orig_digest;
            assert!(SaltedMerkleTree::<H>::verify_authentication_structure(
                root_hash_c,
                tree_height_c,
                &leaf_indices,
                &selected_leaves,
                &auth_path_and_salts
            ));

            // Change root hash again, verify failue
            let another_bad_root_hash_c = corrupt_digest(&root_hash_c);
            assert!(!SaltedMerkleTree::<H>::verify_authentication_structure(
                another_bad_root_hash_c,
                tree_height_c,
                &leaf_indices,
                &selected_leaves,
                &auth_path_and_salts,
            ));
        }
    }

    #[test]
    fn salted_merkle_tree_regression_test_0() {
        type H = blake3::Hasher;

        // This test was used to catch a bug in the implementation of
        // `SaltedMerkleTree::get_leafless_authentication_structure_with_salts_and_values`
        // The bug that this test caught was *fixed* in 5ad285bd867bf8c6c4be380d8539ba37f4a7409a
        // and introduced in 89cfb194f02903534b1621b03a047c128af7d6c2.

        // Build a representation of the SaltedMerkleTree::<H> made from values
        // `451282252958277131` and `3796554602848593414` as this is where the error was
        // first caught.

        let values_reg0 = vec![
            BFieldElement::new(451282252958277131),
            BFieldElement::new(3796554602848593414),
        ];
        let leaves_reg0: Vec<Digest> = values_reg0.iter().map(H::hash).collect();
        let salts_per_leaf = 3;
        let salts_reg0: Vec<Digest> = random_elements(values_reg0.len() * salts_per_leaf);
        let tree_reg0 = SaltedMerkleTree::<H>::from_digests(&leaves_reg0, &salts_reg0);

        let selected_leaf_indices_reg0 = [0];
        let selected_leaves_reg0 = leaves_reg0[0];
        let proof_reg0 =
            tree_reg0.get_authentication_structure_and_salt(&selected_leaf_indices_reg0);

        let proof_verifies_1 = SaltedMerkleTree::<H>::verify_authentication_structure(
            tree_reg0.get_root(),
            tree_reg0.get_height(),
            &selected_leaf_indices_reg0,
            &[selected_leaves_reg0],
            &proof_reg0,
        );
        assert!(proof_verifies_1);

        let selected_leaf_indices_reg1 = vec![1];
        let selected_leaves_reg1 = leaves_reg0[1];
        let proof_reg1 =
            tree_reg0.get_authentication_structure_and_salt(&selected_leaf_indices_reg1);

        let proof_verifies_2 = SaltedMerkleTree::<H>::verify_authentication_structure(
            tree_reg0.get_root(),
            tree_reg0.get_height(),
            &selected_leaf_indices_reg1,
            &[selected_leaves_reg1],
            &proof_reg1,
        );
        assert!(proof_verifies_2);
    }

    #[test]
    fn salted_merkle_tree_verify_authentication_structure_test() {
        type H = blake3::Hasher;

        let mut rng = rand::thread_rng();

        // Number of Merkle tree leaves
        let n_valuess = &[2, 4, 8, 16, 128, 256, 512, 1024, 2048, 4096, 8192];
        let expected_path_lengths = &[1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13]; // log2(128), root node not included

        for (n_values, expected_path_length) in izip!(n_valuess, expected_path_lengths) {
            let values: Vec<BFieldElement> = random_elements(*n_values);
            let leaves: Vec<Digest> = values.iter().map(H::hash).collect();
            let salts_per_leaf = 3;
            let salts_preimage: Vec<BFieldElement> = random_elements(values.len() * salts_per_leaf);
            let salts: Vec<_> = salts_preimage.iter().map(H::hash).collect();
            let tree = SaltedMerkleTree::<H>::from_digests(&leaves, &salts);

            for _ in 0..3 {
                // Ask for an arbitrary amount of indices less than the total
                let max_indices: usize = rng.gen_range(1..=n_values / 2);

                // Generate that amount of distinct, uniform random indices in the valid index range
                let indices: Vec<usize> = random_elements_range(max_indices, 0..*n_values)
                    .into_iter()
                    .unique()
                    .collect();
                let actual_number_of_indices = indices.len();

                let selected_leaves: Vec<_> = indices.iter().map(|i| leaves[*i]).collect();
                let mut proof: SaltedAuthenticationStructure<Digest> =
                    tree.get_authentication_structure_and_salt(&indices);

                for path in proof.iter() {
                    assert_eq!(*expected_path_length, path.0.len());
                }

                assert!(SaltedMerkleTree::<H>::verify_authentication_structure(
                    tree.get_root(),
                    tree.get_height(),
                    &indices,
                    &selected_leaves,
                    &proof,
                ));
                let bad_root_hash = corrupt_digest(&tree.get_root());

                assert!(!SaltedMerkleTree::<H>::verify_authentication_structure(
                    bad_root_hash,
                    tree.get_height(),
                    &indices,
                    &selected_leaves,
                    &proof,
                ));

                // Verify that an invalid leaf fails verification

                let pick: usize = rng.gen_range(0..actual_number_of_indices);
                let rnd_leaf_idx = rng.gen_range(0..selected_leaves.len());
                let mut corrupted_leaves = selected_leaves.clone();
                let rnd_leaf = corrupted_leaves[rnd_leaf_idx];
                corrupted_leaves[rnd_leaf_idx] = corrupt_digest(&rnd_leaf);
                assert!(!SaltedMerkleTree::<H>::verify_authentication_structure(
                    tree.get_root(),
                    tree.get_height(),
                    &indices,
                    &corrupted_leaves,
                    &proof,
                ));

                corrupted_leaves[rnd_leaf_idx] = rnd_leaf;
                assert!(SaltedMerkleTree::<H>::verify_authentication_structure(
                    tree.get_root(),
                    tree.get_height(),
                    &indices,
                    &corrupted_leaves,
                    &proof,
                ));

                // Verify that an invalid salt fails verification
                let invalid_salt = proof[(pick + 1) % actual_number_of_indices].1;
                proof[(pick + 1) % actual_number_of_indices].1 = corrupt_digest(&invalid_salt);
                let mut more_corrupted_leaves = selected_leaves.clone();

                more_corrupted_leaves[rnd_leaf_idx] = corrupt_digest(&rnd_leaf);
                assert!(!SaltedMerkleTree::<H>::verify_authentication_structure(
                    tree.get_root(),
                    tree.get_height(),
                    &indices,
                    &selected_leaves,
                    &proof,
                ));
            }
        }
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
