use itertools::izip;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::marker::{PhantomData, Send, Sync};

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
pub struct MerkleTree<H, M>
where
    H: AlgebraicHasher,
    M: MerkleTreeMaker<H> + ?Sized,
{
    pub nodes: Vec<Digest>,
    pub _hasher: PhantomData<H>,
    pub _maker: PhantomData<M>,
}

impl<H, M> Clone for MerkleTree<H, M>
where
    H: AlgebraicHasher,
    M: MerkleTreeMaker<H>,
{
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            _hasher: PhantomData,
            _maker: PhantomData,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct PartialAuthenticationPath<Digest>(pub Vec<Option<Digest>>);

/// # Design
/// The following are implemented as static methods:
///
/// - `verify_authentication_path`
/// - `verify_authentication_path_from_leaf_hash`
/// - `convert_pat`
/// - `verify_authentication_structure`
/// - `verify_authentication_structure_from_leaf_hashes`
/// - `unwrap_partial_authentication_path`
///
/// The reason being that they are called from the verifier, who does not have
/// the original `MerkleTree` object, but only partial information from it,
/// in the form of the quadrupples: `(root_hash, index, digest, auth_path)`.
/// These are exactly the arguments for the `verify_*` family of static methods.
impl<H, M> MerkleTree<H, M>
where
    H: AlgebraicHasher,
    M: MerkleTreeMaker<H>,
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

        let heights = bit_representation(digests.len() as u128);
        let mut trees: Vec<MerkleTree<H, M>> = vec![];
        let mut acc_counter = 0;
        for height in heights {
            let sub_tree = M::from_digests(&digests[acc_counter..acc_counter + (1 << height)]);
            acc_counter += 1 << height;
            trees.push(sub_tree);
        }

        // Calculate the root from a list of Merkle trees
        let roots: Vec<Digest> = trees.iter().map(|t| t.get_root()).collect();

        bag_peaks::<H>(&roots)
    }

    // Similar to `get_proof', but instead of returning a `Vec<Node<T>>`, we only
    // return the hashes, not the tree nodes (and potential leaf values), and instead
    // of referring to this as a `proof', we call it the `authentication path'.
    //
    //              root
    //             /    \
    // H(H(a)+H(b))      H(H(c)+H(d))
    //   /      \        /      \
    // H(a)    H(b)    H(c)    H(d)
    //
    // The authentication path for `c' (index: 2) would be
    //
    //   vec![ H(d), H(H(a)+H(b)) ]
    //
    // ... so a criss-cross of siblings upwards.
    pub fn get_authentication_path(&self, leaf_index: usize) -> Vec<Digest> {
        let height = self.get_height();
        let mut auth_path: Vec<Digest> = Vec::with_capacity(height);

        let mut node_index = leaf_index + self.nodes.len() / 2;
        while node_index > 1 {
            // We get the sibling node by XOR'ing with 1.
            let sibling_i = node_index ^ 1;
            auth_path.push(self.nodes[sibling_i]);
            node_index /= 2;
        }

        // We don't include the root hash in the authentication path
        // because it's implied in the context of use.
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
            if i % 2 == 0 {
                acc_hash = H::hash_pair(&acc_hash, path_hash);
            } else {
                acc_hash = H::hash_pair(path_hash, &acc_hash);
            }
            i /= 2;
        }

        acc_hash == root_hash
    }

    /// Given a hash map of precalculated digests in the Merkle tree, indexed
    /// by node index, verify an authentication path. The hash map *must*
    /// contain the leaf node that we are verifying for, otherwise this
    /// function will panic.
    fn verify_authentication_path_from_leaf_hash_with_memoization(
        root_hash: &Digest,
        leaf_index: u32,
        auth_path: &[Digest],
        partial_tree: &HashMap<u64, Digest>,
    ) -> bool {
        let path_length = auth_path.len() as u32;

        // Get the last (highest) digest in the authentication path that is contained
        // in `partial_tree`.
        let node_index = leaf_index + 2u32.pow(path_length);
        let mut level_in_tree = 0;
        while partial_tree.contains_key(&(node_index as u64 >> (level_in_tree + 1))) {
            level_in_tree += 1;
        }

        let mut i = node_index >> level_in_tree;
        let mut acc_hash = partial_tree[&(i as u64)];
        while i / 2 >= 1 {
            if i % 2 == 0 {
                acc_hash = H::hash_pair(&acc_hash, &auth_path[level_in_tree]);
            } else {
                acc_hash = H::hash_pair(&auth_path[level_in_tree], &acc_hash);
            }
            i /= 2;
            level_in_tree += 1;
        }

        acc_hash == *root_hash
    }

    // Compact Merkle Authentication Structure Generation
    pub fn get_authentication_structure(
        &self,
        indices: &[usize],
    ) -> Vec<PartialAuthenticationPath<Digest>> {
        let mut calculable_indices: HashSet<usize> = HashSet::new();
        let mut output: Vec<PartialAuthenticationPath<Digest>> = Vec::with_capacity(indices.len());
        for i in indices.iter() {
            let new_branch = PartialAuthenticationPath(
                self.get_authentication_path(*i)
                    .into_iter()
                    .map(Some)
                    .collect(),
            );
            let mut index = self.nodes.len() / 2 + i;
            calculable_indices.insert(index);
            for _ in 1..new_branch.0.len() {
                calculable_indices.insert(index ^ 1);
                index /= 2;
            }
            output.push(new_branch);
        }

        let mut complete = false;
        while !complete {
            complete = true;
            let mut keys: Vec<usize> = calculable_indices.iter().copied().map(|x| x / 2).collect();
            // reverse sort, from big to small, This should be the fastest way to reverse sort.
            // cf. https://stackoverflow.com/a/60916195/2574407
            keys.sort_by_key(|w| Reverse(*w));
            for key in keys.iter() {
                if calculable_indices.contains(&(key * 2))
                    && calculable_indices.contains(&(key * 2 + 1))
                    && !calculable_indices.contains(key)
                {
                    calculable_indices.insert(*key);
                    complete = false;
                }
            }
        }

        let mut scanned: HashSet<usize> = HashSet::new();
        for (i, b) in indices.iter().zip(output.iter_mut()) {
            let mut index: usize = self.nodes.len() / 2 + i;
            scanned.insert(index);
            for elem in b.0.iter_mut() {
                if calculable_indices.contains(&((index ^ 1) * 2))
                    && calculable_indices.contains(&((index ^ 1) * 2 + 1))
                    || (index ^ 1) as i64 - self.nodes.len() as i64 / 2 > 0 // TODO: Maybe > 1 here?
                        && indices.contains(&((index ^ 1) - self.nodes.len() / 2))
                    || scanned.contains(&(index ^ 1))
                {
                    *elem = None;
                }
                scanned.insert(index ^ 1);
                index /= 2;
            }
        }

        output
    }

    /// Verifies a list of leaf_indices and corresponding
    /// auth_pairs
    ///     ///  against a Merkle root.
    ///
    /// # Arguments
    ///
    /// * `root_hash` - The Merkle root
    /// * `leaf_indices` - List of identifiers of the leaves to verify
    /// * `leaf_digests` - List of the leaves' values (i.e. digests) to verify
    /// * `auth_paths` - List of paths corresponding to the leaves.
    pub fn verify_authentication_structure_from_leaves(
        root_hash: Digest,
        leaf_indices: &[usize],
        leaf_digests: &[Digest],
        partial_auth_paths: &[PartialAuthenticationPath<Digest>],
    ) -> bool {
        if leaf_indices.len() != partial_auth_paths.len()
            || leaf_indices.len() != leaf_digests.len()
        {
            return false;
        }

        if leaf_indices.is_empty() {
            return true;
        }
        debug_assert_eq!(leaf_indices.len(), leaf_digests.len());
        debug_assert_eq!(leaf_digests.len(), partial_auth_paths.len());
        debug_assert_eq!(partial_auth_paths.len(), leaf_indices.len());

        let mut partial_auth_paths: Vec<PartialAuthenticationPath<Digest>> =
            partial_auth_paths.to_owned();
        let mut partial_tree: HashMap<u64, Digest> = HashMap::new();

        // FIXME: We find the offset from which leaf nodes occur in the tree by looking at the
        // first partial authentication path. This is a bit hacked, since what if not all
        // partial authentication paths have the same length, and what if one has a
        // different length than the tree's height?
        let auth_path_length = partial_auth_paths[0].0.len();
        let half_tree_size = 2u64.pow(auth_path_length as u32);

        // Bootstrap partial tree
        for (i, leaf_hash, partial_auth_path) in
            izip!(leaf_indices, leaf_digests, partial_auth_paths.clone())
        {
            let mut index = half_tree_size + *i as u64;

            // Insert hashes for known leaf hashes.
            partial_tree.insert(index, *leaf_hash);

            // Insert hashes for known leaves from partial authentication paths.
            for hash_option in partial_auth_path.0.iter() {
                if let Some(hash) = hash_option {
                    partial_tree.insert(index ^ 1, *hash);
                }
                index /= 2;
            }
        }

        let mut complete = false;
        while !complete {
            let mut parent_keys_mut: Vec<u64> =
                partial_tree.keys().copied().map(|x| x / 2).collect();
            parent_keys_mut.sort_by_key(|w| Reverse(*w));
            let parent_keys = parent_keys_mut.clone();
            let partial_tree_immut = partial_tree.clone();

            // Calculate indices for derivable hashes
            let mut new_derivable_digests_indices: Vec<(u64, u64, u64)> = vec![];
            for parent_key in parent_keys {
                let left_child_key = parent_key * 2;
                let right_child_key = parent_key * 2 + 1;

                if partial_tree.contains_key(&left_child_key)
                    && partial_tree.contains_key(&right_child_key)
                    && !partial_tree.contains_key(&parent_key)
                {
                    new_derivable_digests_indices.push((
                        parent_key,
                        left_child_key,
                        right_child_key,
                    ));
                }
            }

            complete = new_derivable_digests_indices.is_empty();

            // Calculate derivable digests in parallel
            let mut new_digests: Vec<(u64, Digest)> =
                Vec::with_capacity(new_derivable_digests_indices.len());
            new_derivable_digests_indices
                .par_iter()
                .map(|(parent_key, left_child_key, right_child_key)| {
                    (
                        *parent_key,
                        H::hash_pair(
                            &partial_tree_immut[left_child_key],
                            &partial_tree_immut[right_child_key],
                        ),
                    )
                })
                .collect_into_vec(&mut new_digests);
            for (parent_key, digest) in new_digests.into_iter() {
                partial_tree.insert(parent_key, digest);
            }
        }

        // Use partial tree to insert missing elements in partial authentication paths,
        // making them full authentication paths.
        for (i, partial_auth_path) in leaf_indices.iter().zip(partial_auth_paths.iter_mut()) {
            let mut index = half_tree_size + *i as u64;

            for elem in partial_auth_path.0.iter_mut() {
                let sibling = index ^ 1;

                if elem.is_none() {
                    // If the Merkle tree/proof is manipulated, the value partial_tree[&(index ^ 1)]
                    // is not guaranteed to exist. So have to  check
                    // whether it exists and return false if it does not

                    if !partial_tree.contains_key(&sibling) {
                        return false;
                    }

                    *elem = Some(partial_tree[&sibling]);
                }
                partial_tree.insert(sibling, elem.unwrap());
                index /= 2;
            }
        }

        // Remove 'Some' constructors from partial auth paths
        let reconstructed_auth_paths = partial_auth_paths
            .iter()
            .map(Self::unwrap_partial_authentication_path)
            .collect::<Vec<_>>();

        leaf_indices
            .par_iter()
            .zip(reconstructed_auth_paths.par_iter())
            .all(|(index, auth_path)| {
                Self::verify_authentication_path_from_leaf_hash_with_memoization(
                    &root_hash,
                    *index as u32,
                    auth_path,
                    &partial_tree,
                )
            })
    }

    /// Verifies a list of leaf_indices and corresponding
    /// auth_pairs (auth_path, leaf_digest) against a Merkle root.
    pub fn verify_authentication_structure(
        root_hash: Digest,
        leaf_indices: &[usize],
        auth_pairs: &[(PartialAuthenticationPath<Digest>, Digest)],
    ) -> bool {
        if leaf_indices.len() != auth_pairs.len() {
            return false;
        }

        if leaf_indices.is_empty() {
            return true;
        }
        assert_eq!(leaf_indices.len(), auth_pairs.len());

        let (auth_paths, leaves): (Vec<_>, Vec<_>) = auth_pairs.iter().cloned().unzip();

        Self::verify_authentication_structure_from_leaves(
            root_hash,
            leaf_indices,
            &leaves,
            &auth_paths,
        )
    }

    fn unwrap_partial_authentication_path(
        partial_auth_path: &PartialAuthenticationPath<Digest>,
    ) -> Vec<Digest> {
        partial_auth_path
            .clone()
            .0
            .into_iter()
            .map(|hash| hash.unwrap())
            .collect()
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
    fn from_digests(digests: &[Digest]) -> MerkleTree<H, CpuParallel> {
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
            _maker: PhantomData,
        }
    }
}

pub type SaltedAuthenticationStructure<Digest> = Vec<(PartialAuthenticationPath<Digest>, Digest)>;

#[derive(Clone, Debug)]
pub struct SaltedMerkleTree<H>
where
    H: AlgebraicHasher,
{
    internal_merkle_tree: MerkleTree<H, SaltedMaker>,
    salts: Vec<Digest>,
}

type SaltedMaker = CpuParallel;

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

        let internal_merkle_tree = MerkleTree::<H, SaltedMaker> {
            nodes,
            _hasher: PhantomData,
            _maker: PhantomData,
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
        MerkleTree::<H, SaltedMaker>::verify_authentication_path_from_leaf_hash(
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
        root_hash: Digest,
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

        MerkleTree::<H, SaltedMaker>::verify_authentication_structure_from_leaves(
            root_hash,
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
    use crate::test_shared::corrupt_digest;
    use crate::util_types::algebraic_hasher::Hashable;
    use itertools::Itertools;
    use rand::{Rng, RngCore};
    use std::iter::zip;

    /// Count the number of hashes present in all partial authentication paths
    fn count_hashes<Digest: Clone>(proof: &SaltedAuthenticationStructure<Digest>) -> usize {
        proof
            .iter()
            .map(|(partial_auth_path, _)| {
                let optional_digests = partial_auth_path.0.clone();
                optional_digests.iter().flatten().count()
            })
            .sum()
    }

    #[test]
    fn merkle_tree_test_32() {
        type H = blake3::Hasher;

        let num_leaves = 32;
        let leaves: Vec<Digest> = random_elements(num_leaves);
        let tree: MerkleTree<H> = MerkleTree::from_digests(&leaves);

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
            let random_leaves_are_members = MerkleTree::<H>::verify_authentication_structure(
                tree.get_root(),
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
            let bad_root_verifies = MerkleTree::<H>::verify_authentication_structure(
                bad_root_digest,
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
            let too_many_indices_verifies = MerkleTree::<H>::verify_authentication_structure(
                tree.get_root(),
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
            let too_few_indices_verifies = MerkleTree::<H>::verify_authentication_structure(
                tree.get_root(),
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
            let non_existent_index_verifies = MerkleTree::<H>::verify_authentication_structure(
                tree.get_root(),
                &bad_random_indices_3,
                &proof,
            );
            assert!(!non_existent_index_verifies);
        }
    }

    #[test]
    fn merkle_tree_verify_authentication_structure_degenerate_test() {
        type H = blake3::Hasher;

        let num_leaves = 32;
        let leaves: Vec<Digest> = random_elements(num_leaves);
        let tree = MerkleTree::<H>::from_digests(&leaves);

        let empty_proof = tree.get_authentication_structure(&[]);
        let auth_pairs = zip(empty_proof, leaves).collect_vec();
        let empty_proof_verifies =
            MerkleTree::<H>::verify_authentication_structure(tree.get_root(), &[], &auth_pairs);
        assert!(empty_proof_verifies);
    }

    #[test]
    fn merkle_tree_verify_authentication_structure_equivalence_test() {
        type H = blake3::Hasher;

        // This test asserts that regular merkle trees and salted merkle trees with 0 salts work equivalently.

        let num_leaves = 8;
        let leaves: Vec<Digest> = random_elements(num_leaves);
        let regular_tree = MerkleTree::<H>::from_digests(&leaves);

        let expected_path_length = 3;

        let selected_indices: Vec<usize> = vec![0, 1];
        let selected_leaves = regular_tree.get_leaves_by_indices(&selected_indices);
        let selected_auth_paths = regular_tree.get_authentication_structure(&selected_indices);
        let auth_pairs = zip(selected_auth_paths, selected_leaves.clone()).collect_vec();

        for (partial_auth_path, _digest) in auth_pairs.clone() {
            assert_eq!(
                  expected_path_length,
                  partial_auth_path.0.len(),
                  "A generated authentication path must have length the height of the tree (not counting the root)"
              );
        }

        let regular_verify = MerkleTree::<H>::verify_authentication_structure(
            regular_tree.get_root(),
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
            &selected_indices,
            &selected_leaves,
            &salted_proof,
        );

        assert_eq!(regular_verify, unsalted_salted_verify);
    }

    #[test]
    fn merkle_tree_verify_authentication_structure_test() {
        type H = blake3::Hasher;

        let num_leaves_cases = [2, 4, 8, 16, 128, 256, 512, 1024, 2048, 4096, 8192];
        let expected_path_lengths = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13]; // log2(128), root node not included
        for (num_leaves, expected_path_length) in izip!(num_leaves_cases, expected_path_lengths) {
            let leaves: Vec<Digest> = random_elements(num_leaves);
            let tree = MerkleTree::<H>::from_digests(&leaves);

            let mut rng = rand::thread_rng();
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
                    assert_eq!(expected_path_length, auth_path.0.len());
                }

                let auth_pairs: Vec<(PartialAuthenticationPath<Digest>, Digest)> =
                    zip(selected_auth_paths, selected_leaves.clone()).collect();

                let good_tree = MerkleTree::<H>::verify_authentication_structure(
                    tree.get_root(),
                    &selected_indices,
                    &auth_pairs,
                );
                assert!(
                    good_tree,
                    "An uncorrupted tree and an uncorrupted proof should verify."
                );

                // Negative: Corrupt the root and thereby the tree
                let bad_root_hash = corrupt_digest(&tree.get_root());

                let verified = MerkleTree::<H>::verify_authentication_structure(
                    bad_root_hash,
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

                let corrupted_proof_verifies = MerkleTree::<H>::verify_authentication_structure(
                    tree.get_root(),
                    &selected_indices,
                    &bad_proof,
                );
                assert!(!corrupted_proof_verifies);
            }
        }
    }

    #[test]
    fn merkle_tree_get_authentication_path_test() {
        type H = blake3::Hasher;

        // 1: Create Merkle tree
        //
        //     root
        //    /    \
        //   x      y
        //  / \    / \
        // 0   1  2   3
        let num_leaves_a = 4;
        let leaves_a: Vec<Digest> = random_elements(num_leaves_a);
        let tree_a = MerkleTree::<H>::from_digests(&leaves_a);

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
        let tree_b = MerkleTree::<H>::from_digests(&leaves_b);

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
    fn verify_authentication_path_from_leaf_hash_with_memoization_test() {
        type H = blake3::Hasher;

        // 1: Create Merkle tree
        //
        //     root
        //    /    \
        //   x      y
        //  / \    / \
        // 0   1  2   3
        let num_leaves = 4;
        let leaves_a: Vec<Digest> = random_elements(num_leaves);
        let tree_a = MerkleTree::<H>::from_digests(&leaves_a);

        let mut partial_tree: HashMap<u64, Digest> = HashMap::new();
        let leaf_index: usize = 2;
        partial_tree.insert((num_leaves + leaf_index) as u64, leaves_a[leaf_index]);
        let auth_path_leaf_index_2 = tree_a.get_authentication_path(leaf_index);

        let proof_verifies = MerkleTree::<H>::verify_authentication_path_from_leaf_hash(
            tree_a.get_root(),
            leaf_index as u32,
            leaves_a[leaf_index],
            auth_path_leaf_index_2.clone(),
        );
        assert!(proof_verifies);

        let proof_with_memoization_verifies =
            MerkleTree::<H>::verify_authentication_path_from_leaf_hash_with_memoization(
                &tree_a.get_root(),
                leaf_index as u32,
                &auth_path_leaf_index_2,
                &partial_tree,
            );
        assert!(proof_with_memoization_verifies);

        // Negative: Invalid auth path / partial tree
        let auth_path_leaf_index_3 = tree_a.get_authentication_path(3);
        let invalid_auth_path_partial_tree_verifies =
            MerkleTree::<H>::verify_authentication_path_from_leaf_hash_with_memoization(
                &tree_a.get_root(),
                leaf_index as u32,
                &auth_path_leaf_index_3,
                &partial_tree,
            );
        assert!(!invalid_auth_path_partial_tree_verifies);

        // Generate an entirely different Merkle tree
        let leaves_b: Vec<Digest> = random_elements(num_leaves);
        let different_tree = MerkleTree::<H>::from_digests(&leaves_b);
        let hmmm = MerkleTree::<H>::verify_authentication_path_from_leaf_hash_with_memoization(
            &different_tree.get_root(),
            leaf_index as u32,
            &auth_path_leaf_index_2,
            &partial_tree,
        );
        assert!(!hmmm, "Bad Merkle tree root must fail to validate");
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

            let expected_hashes = 3;
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
                &leaf_indices,
                &selected_leaves,
                &auth_path_and_salts
            ));

            auth_path_and_salts[1].1 = orig_digest;
            assert!(SaltedMerkleTree::<H>::verify_authentication_structure(
                root_hash_c,
                &leaf_indices,
                &selected_leaves,
                &auth_path_and_salts
            ));

            // Change root hash again, verify failue
            let another_bad_root_hash_c = corrupt_digest(&root_hash_c);
            assert!(!SaltedMerkleTree::<H>::verify_authentication_structure(
                another_bad_root_hash_c,
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
            let leaves: Vec<Digest> = values
                .iter()
                .map(|x| H::hash_slice(&x.to_sequence()))
                .collect();
            let salts_per_leaf = 3;
            let salts_preimage: Vec<BFieldElement> = random_elements(values.len() * salts_per_leaf);
            let salts: Vec<_> = salts_preimage
                .iter()
                .map(|x| H::hash_slice(&x.to_sequence()))
                .collect();
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
                    assert_eq!(*expected_path_length, path.0 .0.len());
                }

                assert!(SaltedMerkleTree::<H>::verify_authentication_structure(
                    tree.get_root(),
                    &indices,
                    &selected_leaves,
                    &proof,
                ));
                let bad_root_hash = corrupt_digest(&tree.get_root());

                assert!(!SaltedMerkleTree::<H>::verify_authentication_structure(
                    bad_root_hash,
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
                    &indices,
                    &corrupted_leaves,
                    &proof,
                ));

                corrupted_leaves[rnd_leaf_idx] = rnd_leaf;
                assert!(SaltedMerkleTree::<H>::verify_authentication_structure(
                    tree.get_root(),
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

        let leafs = values.iter().map(|leaf| H::hash_slice(leaf)).collect_vec();

        let tree = MT::from_digests(&leafs);

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

        let mut leafs: Vec<Digest> = values.iter().map(|leaf| H::hash_slice(leaf)).collect_vec();

        // A payload integrity test
        let test_leaf_idx = 42;
        let payload_offset = 317;
        let payload_leaf = vec![BFieldElement::new((test_leaf_idx + payload_offset) as u64)];

        // Embed
        leafs[test_leaf_idx] = H::hash_slice(&payload_leaf);

        let tree = MT::from_digests(&leafs[..]);

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
            H::hash_slice(&payload_leaf),
            ap,
        );
        assert_eq!(
            tree.get_leaf_by_index(test_leaf_idx),
            H::hash_slice(&payload_leaf)
        );
        assert!(
            verdict,
            "Rejected: `leaf: {:?}` at `leaf_idx: {:?}` failed to verify.",
            payload_leaf, test_leaf_idx,
        );
    }

    #[test]
    fn root_from_odd_number_of_digests_test() {
        type H = RescuePrimeRegular;

        let leafs: Vec<Digest> = random_elements(128);
        let mt = MerkleTree::<H>::from_digests(&leafs);

        println!("Merkle root (RP 1): {:?}", mt.get_root());

        assert_eq!(
            mt.get_root(),
            MerkleTree::<H>::root_from_arbitrary_number_of_digests(&leafs)
        );
    }

    #[test]
    fn root_from_arbitrary_number_of_digests_empty_test() {
        // Ensure that we can calculate a Merkle root from an empty list of digests.
        // This is needed since a block can contain an empty list of addition or
        // removal records.

        type H = RescuePrimeRegular;

        MerkleTree::<H>::root_from_arbitrary_number_of_digests(&[]);
    }
}
