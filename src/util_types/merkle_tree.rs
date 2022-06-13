use crate::shared_math::other::{self, get_height_of_complete_binary_tree, is_power_of_two};
use crate::util_types::simple_hasher::{Hasher, ToDigest};
use itertools::izip;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::marker::PhantomData;

// Chosen from a very small number of benchmark runs, optimized for a slow
// hash function (the original Rescue Prime implementation). It should probably
// be a higher number than 16 when using a faster hash function.
const PARALLELLIZATION_THRESHOLD: usize = 16;

#[derive(Debug)]
pub struct MerkleTree<H>
where
    H: Hasher,
{
    pub nodes: Vec<H::Digest>,
    _hasher: PhantomData<H>,
}

impl<H> Clone for MerkleTree<H>
where
    H: Hasher,
{
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            _hasher: PhantomData,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PartialAuthenticationPath<Digest>(pub Vec<Option<Digest>>);

/// # Design
/// The following are implemented as static methods:
///
/// - `verify_authentication_path`
/// - `verify_authentication_path_from_leaf_hash`
/// - `convert_pat`
/// - `verify_multi_proof`
/// - `verify_multi_proof_from_leaf_hashes`
/// - `unwrap_partial_authentication_path`
///
/// The reason being that they are called from the verifier, who does not have
/// the original `MerkleTree` object, but only partial information from it,
/// in the form of the quadrupples: `(root_hash, index, digest, auth_path)`.
/// These are exactly the arguments for the `verify_*` family of static methods.
impl<H> MerkleTree<H>
where
    H: Hasher + std::marker::Sync + std::marker::Send,
{
    /// Takes an array of digests and builds a MerkleTree over them.
    /// The digests are used copied over as the leaves of the tree.
    pub fn from_digests(digests: &[H::Digest]) -> Self
    where
        H::Digest: Clone,
    {
        let leaves_count = digests.len();

        assert!(
            other::is_power_of_two(leaves_count),
            "Size of input for Merkle tree must be a power of 2"
        );

        let filler = digests[0].clone();

        // nodes[0] is never used for anything.
        let mut nodes = vec![filler; 2 * leaves_count];
        nodes[leaves_count..(leaves_count + leaves_count)]
            .clone_from_slice(&digests[..leaves_count]);

        let hasher = H::new();

        // Parallel digest calculations
        let mut node_count_on_this_level: usize = digests.len() / 2;
        let mut count_acc: usize = 0;
        while node_count_on_this_level >= PARALLELLIZATION_THRESHOLD {
            let mut local_digests: Vec<H::Digest> = Vec::with_capacity(node_count_on_this_level);
            (0..node_count_on_this_level)
                .into_par_iter()
                .map(|i| {
                    let j = node_count_on_this_level + i;
                    let left_child = &nodes[j * 2];
                    let right_child = &nodes[j * 2 + 1];
                    hasher.hash_pair(left_child, right_child)
                })
                .collect_into_vec(&mut local_digests);
            nodes[node_count_on_this_level..(node_count_on_this_level + node_count_on_this_level)]
                .clone_from_slice(&local_digests[..node_count_on_this_level]);
            count_acc += node_count_on_this_level;
            node_count_on_this_level /= 2;
        }

        // Sequential digest calculations
        for i in (1..(digests.len() - count_acc)).rev() {
            nodes[i] = hasher.hash_pair(&nodes[i * 2], &nodes[i * 2 + 1]);
        }

        let _hasher = PhantomData;

        Self { nodes, _hasher }
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
    pub fn get_authentication_path(&self, leaf_index: usize) -> Vec<H::Digest> {
        let height = self.get_height();
        let mut auth_path: Vec<H::Digest> = Vec::with_capacity(height);

        let mut node_index = leaf_index + self.nodes.len() / 2;
        while node_index > 1 {
            // We get the sibling node by XOR'ing with 1.
            let sibling_i = node_index ^ 1;
            auth_path.push(self.nodes[sibling_i].clone());
            node_index /= 2;
        }

        // We don't include the root hash in the authentication path
        // because it's implied in the context of use.
        auth_path
    }

    // Consider renaming this `verify_leaf_with_authentication_path()`.
    pub fn verify_authentication_path_from_leaf_hash(
        root_hash: H::Digest,
        leaf_index: u32,
        leaf_hash: H::Digest,
        auth_path: Vec<H::Digest>,
    ) -> bool {
        let path_length = auth_path.len() as u32;
        let hasher = H::new();

        // Initialize `acc_hash' as leaf_hash
        let mut acc_hash = leaf_hash;
        let mut i = leaf_index + 2u32.pow(path_length);
        for path_hash in auth_path.iter() {
            // Use Merkle tree index parity (odd/even) to determine which
            // order to concatenate the hashes before hashing them.
            if i % 2 == 0 {
                acc_hash = hasher.hash_pair(&acc_hash, path_hash);
            } else {
                acc_hash = hasher.hash_pair(path_hash, &acc_hash);
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
        root_hash: &H::Digest,
        leaf_index: u32,
        auth_path: &[H::Digest],
        partial_tree: &HashMap<u64, H::Digest>,
        hasher: &H,
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
        let mut acc_hash = partial_tree[&(i as u64)].clone();
        while i / 2 >= 1 {
            if i % 2 == 0 {
                acc_hash = hasher.hash_pair(&acc_hash, &auth_path[level_in_tree]);
            } else {
                acc_hash = hasher.hash_pair(&auth_path[level_in_tree], &acc_hash);
            }
            i /= 2;
            level_in_tree += 1;
        }

        acc_hash == *root_hash
    }

    // Compact Merkle Multiproof Generation
    pub fn get_multi_proof(&self, indices: &[usize]) -> Vec<PartialAuthenticationPath<H::Digest>> {
        let mut calculable_indices: HashSet<usize> = HashSet::new();
        let mut output: Vec<PartialAuthenticationPath<H::Digest>> =
            Vec::with_capacity(indices.len());
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
    pub fn verify_multi_proof_from_leaves(
        root_hash: H::Digest,
        leaf_indices: &[usize],
        leaf_digests: &[H::Digest],
        partial_auth_paths: &[PartialAuthenticationPath<H::Digest>],
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

        let mut partial_auth_paths: Vec<PartialAuthenticationPath<H::Digest>> =
            partial_auth_paths.to_owned();
        let mut partial_tree: HashMap<u64, H::Digest> = HashMap::new();

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
            partial_tree.insert(index, leaf_hash.clone());

            // Insert hashes for known leaves from partial authentication paths.
            for hash_option in partial_auth_path.0.iter() {
                if let Some(hash) = hash_option {
                    partial_tree.insert(index ^ 1, hash.clone());
                }
                index /= 2;
            }
        }

        let mut complete = false;
        let hasher = H::new();
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
            let mut new_digests: Vec<(u64, H::Digest)> =
                Vec::with_capacity(new_derivable_digests_indices.len());
            new_derivable_digests_indices
                .par_iter()
                .map(|(parent_key, left_child_key, right_child_key)| {
                    (
                        *parent_key,
                        hasher.hash_pair(
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

                if *elem == None {
                    // If the Merkle tree/proof is manipulated, the value partial_tree[&(index ^ 1)]
                    // is not guaranteed to exist. So have to  check
                    // whether it exists and return false if it does not

                    if !partial_tree.contains_key(&sibling) {
                        return false;
                    }

                    *elem = Some(partial_tree[&sibling].clone());
                }
                let elem_for_reals = elem.as_ref().unwrap();
                partial_tree.insert(sibling, elem_for_reals.clone());
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
                    &hasher,
                )
            })
    }

    /// Verifies a list of leaf_indices and corresponding
    /// auth_pairs (auth_path, leaf_digest) against a Merkle root.
    pub fn verify_multi_proof(
        root_hash: H::Digest,
        leaf_indices: &[usize],
        auth_pairs: &[(PartialAuthenticationPath<H::Digest>, H::Digest)],
    ) -> bool {
        if leaf_indices.len() != auth_pairs.len() {
            return false;
        }

        if leaf_indices.is_empty() {
            return true;
        }
        assert_eq!(leaf_indices.len(), auth_pairs.len());

        let (auth_paths, leaves): (Vec<_>, Vec<_>) = auth_pairs.iter().cloned().unzip();

        Self::verify_multi_proof_from_leaves(root_hash, leaf_indices, &leaves, &auth_paths)
    }

    fn unwrap_partial_authentication_path(
        partial_auth_path: &PartialAuthenticationPath<H::Digest>,
    ) -> Vec<H::Digest> {
        partial_auth_path
            .clone()
            .0
            .into_iter()
            .map(|hash| hash.unwrap())
            .collect()
    }

    pub fn get_root(&self) -> H::Digest {
        self.nodes[1].clone()
    }

    pub fn get_leaf_count(&self) -> usize {
        let node_count = self.nodes.len();
        assert!(is_power_of_two(node_count));
        node_count / 2
    }

    pub fn get_height(&self) -> usize {
        get_height_of_complete_binary_tree(self.get_leaf_count())
    }

    pub fn get_all_leaves(&self) -> Vec<H::Digest> {
        let first_leaf = self.nodes.len() / 2;
        self.nodes[first_leaf..].to_vec()
    }

    pub fn get_leaf_by_index(&self, index: usize) -> H::Digest {
        let first_leaf_index = self.nodes.len() / 2;
        let beyond_last_leaf_index = self.nodes.len();
        assert!(
            index < first_leaf_index || beyond_last_leaf_index <= index,
            "Out of bounds index requested"
        );
        self.nodes[first_leaf_index + index].clone()
    }

    pub fn get_leaves_by_indices(&self, leaf_indices: &[usize]) -> Vec<H::Digest> {
        let leaf_count = leaf_indices.len();

        let mut result = Vec::with_capacity(leaf_count);

        for index in leaf_indices {
            result.push(self.get_leaf_by_index(*index));
        }
        result
    }
}

pub type SaltedMultiProof<Digest> = Vec<(PartialAuthenticationPath<Digest>, Vec<Digest>)>;

#[derive(Clone, Debug)]
pub struct SaltedMerkleTree<H>
where
    H: Hasher,
{
    internal_merkle_tree: MerkleTree<H>,
    salts: Vec<H::Digest>,
}

impl<H> SaltedMerkleTree<H>
where
    H: Hasher + Sync + Send,
{
    fn salts_per_leaf(&self) -> usize {
        let leaves_count = self.internal_merkle_tree.nodes.len() / 2;
        self.salts.len() / leaves_count
    }

    // Build a salted Merkle tree from a slice of serializable values
    pub fn from_digests(leaves: &[H::Digest], salts: &[H::Digest]) -> Self {
        assert!(
            other::is_power_of_two(leaves.len()),
            "Size of input for Merkle tree must be a power of 2"
        );

        let salts_per_leaf = salts.len() / leaves.len();

        let hasher = H::new();
        let filler = leaves[0].clone();

        // nodes[0] is never used for anything.
        let mut nodes: Vec<H::Digest> = vec![filler; 2 * leaves.len()];

        for i in 0..leaves.len() {
            let value = leaves[i].clone();
            let salty_slice = &salts[salts_per_leaf * i..salts_per_leaf * (i + 1)];
            let leaf_digest = hasher.hash_with_salts(value.to_digest(), salty_slice);

            nodes[leaves.len() + i] = leaf_digest;
        }

        // loop from `len(L) - 1` to 1
        for i in (1..(nodes.len() / 2)).rev() {
            let left = nodes[i * 2].clone();
            let right = nodes[i * 2 + 1].clone();
            nodes[i] = hasher.hash_pair(&left, &right);
        }

        let _hasher = PhantomData;

        let internal_merkle_tree: MerkleTree<H> = MerkleTree { nodes, _hasher };

        Self {
            internal_merkle_tree,
            salts: salts.to_vec(),
        }
    }

    pub fn get_authentication_path_and_salts(
        &self,
        index: usize,
    ) -> (Vec<H::Digest>, Vec<H::Digest>) {
        let authentication_path = self.internal_merkle_tree.get_authentication_path(index);
        let spl = self.salts_per_leaf();
        let mut salts = Vec::<H::Digest>::with_capacity(spl);
        for i in 0..spl {
            salts.push(self.salts[index * spl + i].clone());
        }

        (authentication_path, salts)
    }

    pub fn verify_authentication_path(
        root_hash: H::Digest,
        index: u32,
        leaf: H::Digest,
        auth_path: Vec<H::Digest>,
        salts_for_leaf: Vec<H::Digest>,
    ) -> bool {
        let hasher = H::new();
        let leaf_hash = hasher.hash_with_salts(leaf, &salts_for_leaf);

        // Set `leaf_hash` to H(value + salts[0..])
        MerkleTree::<H>::verify_authentication_path_from_leaf_hash(
            root_hash, index, leaf_hash, auth_path,
        )
    }

    pub fn get_multi_proof_and_salts(&self, indices: &[usize]) -> SaltedMultiProof<H::Digest> {
        // Get the partial authentication paths without salts
        let partial_authentication_paths: Vec<PartialAuthenticationPath<H::Digest>> =
            self.internal_merkle_tree.get_multi_proof(indices);

        // Get the salts associated with the leafs
        // salts are random data, so they cannot be compressed
        let mut ret: SaltedMultiProof<H::Digest> = Vec::with_capacity(indices.len());
        for (i, index) in indices.iter().enumerate() {
            let mut salts = vec![];
            for j in 0..self.salts_per_leaf() {
                salts.push(self.salts[index * self.salts_per_leaf() + j].clone());
            }
            ret.push((partial_authentication_paths[i].clone(), salts));
        }

        ret
    }

    /// To use this function the user must provide the corresponding *UNSALTED* `leaves`.
    pub fn verify_multi_proof(
        root_hash: H::Digest,
        indices: &[usize],
        unsalted_leaves: &[H::Digest],
        proof: &SaltedMultiProof<H::Digest>,
    ) -> bool {
        if indices.len() != proof.len() || indices.len() != unsalted_leaves.len() {
            debug_assert!(indices.len() == proof.len());
            debug_assert!(indices.len() == unsalted_leaves.len());

            return false;
        }

        if indices.is_empty() {
            return true;
        }

        let hasher = H::new();
        let mut leaf_hashes: Vec<H::Digest> = Vec::with_capacity(indices.len());
        for (value, proof_element) in izip!(unsalted_leaves, proof) {
            let salts_for_leaf = proof_element.1.clone();
            let leaf_hash = hasher.hash_with_salts(value.to_digest(), &salts_for_leaf);
            leaf_hashes.push(leaf_hash);
        }

        let saltless_proof: Vec<PartialAuthenticationPath<H::Digest>> =
            proof.iter().map(|x| x.0.clone()).collect();

        MerkleTree::<H>::verify_multi_proof_from_leaves(
            root_hash,
            indices,
            &leaf_hashes,
            &saltless_proof,
        )
    }

    pub fn get_root(&self) -> H::Digest {
        self.internal_merkle_tree.get_root()
    }

    pub fn get_leaf_count(&self) -> usize {
        self.internal_merkle_tree.get_leaf_count()
    }

    #[allow(dead_code)]
    pub fn get_height(&self) -> usize {
        self.internal_merkle_tree.get_height()
    }

    pub fn get_all_salted_leaves(&self) -> Vec<H::Digest> {
        let first_leaf = self.internal_merkle_tree.nodes.len() / 2;
        self.internal_merkle_tree.nodes[first_leaf..].to_vec()
    }

    pub fn get_salted_leaf_by_index(&self, index: usize) -> H::Digest {
        let first_leaf_index = self.internal_merkle_tree.nodes.len() / 2;
        let beyond_last_leaf_index = self.internal_merkle_tree.nodes.len();
        assert!(
            index < first_leaf_index || beyond_last_leaf_index <= index,
            "Out of bounds index requested"
        );
        self.internal_merkle_tree.nodes[first_leaf_index + index].clone()
    }

    pub fn get_salted_leaves_by_indices(&self, leaf_indices: &[usize]) -> Vec<H::Digest> {
        let leaf_count = leaf_indices.len();

        let mut result = Vec::with_capacity(leaf_count);

        for index in leaf_indices {
            result.push(self.get_salted_leaf_by_index(*index));
        }
        result
    }

    pub fn get_salts(&self) -> &[H::Digest] {
        &self.salts
    }
}

#[cfg(test)]
mod merkle_tree_test {
    use super::*;
    use crate::shared_math::b_field_element::BFieldElement;
    use crate::shared_math::rescue_prime_xlix::{RescuePrimeXlix, RP_DEFAULT_WIDTH};
    use crate::shared_math::traits::GetRandomElements;
    use crate::shared_math::x_field_element::XFieldElement;
    use crate::util_types::blake3_wrapper::Blake3Hash;
    use crate::util_types::simple_hasher::RescuePrimeProduction;
    use crate::utils::{generate_random_numbers, generate_random_numbers_u128};
    use itertools::{zip, Itertools};
    use rand::RngCore;

    fn count_hashes<Digest>(proof: &SaltedMultiProof<Digest>) -> usize {
        proof.iter().map(|y| y.0 .0.iter().flatten().count()).sum()
    }

    impl Blake3Hash {
        /// Corrupts a hash by flipping all bits.
        /// Run this twice to restore the original.
        fn toggle_corruption(&mut self) {
            let original = self.clone();
            let Blake3Hash(orig) = self;
            let mut bytes: [u8; 32] = orig.as_bytes().clone();

            for i in 0..bytes.len() {
                bytes[i] ^= 1;
            }

            *self = Blake3Hash(bytes.into());
            assert_ne!(
                original, *self,
                "The digest should always differ after being corrupted."
            )
        }

        /// Change a Blake3 hash value. Only used for negative tests.
        fn increment(&mut self) {
            let original = self.clone();
            let Blake3Hash(orig) = self;
            let mut bytes: [u8; 32] = orig.as_bytes().clone();

            let last_index = bytes.len() - 1;

            // Account for potential overflow
            if bytes[last_index] == u8::MAX {
                bytes[last_index] = 0;
            } else {
                bytes[last_index] += 1;
            }

            *self = Blake3Hash(bytes.into());
            assert_ne!(
                original, *self,
                "The digest should always differ after being incremented."
            )
        }

        /// Change a Blake3 hash value. Only used for negative tests.
        fn decrement(&mut self) {
            let original = self.clone();
            let Blake3Hash(orig) = self;
            let mut bytes: [u8; 32] = orig.as_bytes().clone();

            let last_index = bytes.len() - 1;

            // Account for potential overflow
            if bytes[last_index] == 0 {
                bytes[last_index] = u8::MAX;
            } else {
                bytes[last_index] -= 1;
            }

            *self = Blake3Hash(bytes.into());
            assert_ne!(
                original, *self,
                "The digest should always differ after being decremented."
            )
        }
    }

    impl<H> MerkleTree<H>
    where
        H: Hasher,
    {
        /// For writing negative tests only.
        fn set_root(&mut self, new_root: H::Digest) {
            self.nodes[1] = new_root
        }
    }

    impl<H> SaltedMerkleTree<H>
    where
        H: Hasher,
    {
        /// For writing negative tests only.
        #[allow(dead_code)]
        fn set_root(&mut self, new_root: H::Digest) {
            self.internal_merkle_tree.set_root(new_root)
        }
    }

    #[test]
    fn merkle_tree_test_32() {
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;

        let mut rng = rand::thread_rng();
        let values: Vec<BFieldElement> = BFieldElement::random_elements(32, &mut rng);
        let hasher = Hasher::new();

        let leaves: Vec<Digest> = values.iter().map(|x| hasher.hash(x)).collect();
        let mut mt_32: MerkleTree<Hasher> = MerkleTree::from_digests(&leaves);
        let mt_32_orig_root_hash = mt_32.get_root();

        for _ in 0..2 {
            for i in 0..20 {
                // Create a vector `indices_usize` of unique (i.e. non-repeated) indices
                // The first element is discarded to check that verify_multi_proof returns
                // false if this element is requested in the verification without being
                // included in the proof
                let indices_i128: Vec<i128> = generate_random_numbers(10 + i, 32);
                let mut indices_usize: Vec<usize> = vec![];
                for elem in indices_i128.iter().unique().skip(1) {
                    indices_usize.push(*elem as usize);
                }

                let selected_leaves: Vec<Digest> = mt_32.get_leaves_by_indices(&indices_usize);
                let partial_auth_paths: Vec<PartialAuthenticationPath<Digest>> =
                    mt_32.get_multi_proof(&indices_usize);
                let proof: Vec<(PartialAuthenticationPath<Digest>, Digest)> =
                    zip(partial_auth_paths, selected_leaves.clone()).collect();

                assert!(MerkleTree::<Hasher>::verify_multi_proof(
                    mt_32_orig_root_hash,
                    &indices_usize,
                    &proof
                ));

                // Verify that `get_value` returns the value for this proof
                assert!(proof
                    .iter()
                    .enumerate()
                    .all(|(i, (_auth_path, digest))| *digest == leaves[indices_usize[i]]));

                // manipulate Merkle root and verify failure
                let mut bad_root_hash = mt_32_orig_root_hash;
                bad_root_hash.toggle_corruption();

                assert!(!MerkleTree::<Hasher>::verify_multi_proof(
                    bad_root_hash,
                    &indices_usize,
                    &proof
                ));

                // Restore root and verify success
                mt_32.set_root(mt_32_orig_root_hash);
                assert!(MerkleTree::<Hasher>::verify_multi_proof(
                    mt_32.get_root().clone(),
                    &indices_usize,
                    &proof
                ));

                // Request an additional index and verify failure
                // (indices length does not match proof length)
                indices_usize.insert(0, indices_i128[0] as usize);
                assert!(!MerkleTree::<Hasher>::verify_multi_proof(
                    mt_32.get_root().clone(),
                    &indices_usize,
                    &proof
                ));

                // Request a non-existant index and verify failure
                // (indices length does match proof length)
                indices_usize.remove(0);
                indices_usize[0] = indices_i128[0] as usize;
                assert!(!MerkleTree::<Hasher>::verify_multi_proof(
                    mt_32.get_root().clone(),
                    &indices_usize,
                    &proof
                ));

                // Remove an element from the indices vector
                // and verify failure since the indices and proof
                // vectors do not match
                indices_usize.remove(0);
                assert!(!MerkleTree::<Hasher>::verify_multi_proof(
                    mt_32.get_root().clone(),
                    &indices_usize,
                    &proof
                ));
            }
        }
    }

    #[test]
    fn merkle_tree_verify_multi_proof_degenerate_test() {
        type Hasher = blake3::Hasher;
        type Digest = Blake3Hash;

        let mut rng = rand::thread_rng();

        // Number of Merkle tree leaves
        let values: Vec<BFieldElement> = BFieldElement::random_elements(32, &mut rng);
        let hasher = Hasher::new();
        let leaves: Vec<Digest> = values.iter().map(|x| hasher.hash(x)).collect();

        let tree = MerkleTree::<Hasher>::from_digests(&leaves);

        // Degenerate example
        let empty_proof: Vec<PartialAuthenticationPath<Digest>> = tree.get_multi_proof(&[]);
        let auth_pairs: Vec<(PartialAuthenticationPath<Digest>, Digest)> =
            zip(empty_proof, leaves).collect();
        assert!(MerkleTree::<Hasher>::verify_multi_proof(
            tree.get_root(),
            &[],
            &auth_pairs,
        ));
    }

    #[test]
    fn merkle_tree_verify_multi_proof_equivalence_test() {
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;
        let mut rng = rand::thread_rng();

        // This test asserts that regular merkle trees and salted merkle trees with 0 salts work equivalently.

        // Number of Merkle tree leaves
        let n_values = 8;
        let expected_path_length = 3; // log2(128), root node not included
        let values: Vec<BFieldElement> = BFieldElement::random_elements(n_values, &mut rng);

        let hasher = Hasher::new();
        let leaves: Vec<Digest> = values.iter().map(|x| hasher.hash(x)).collect();
        let regular_tree = MerkleTree::<Hasher>::from_digests(&leaves);

        let selected_indices: Vec<usize> = vec![0, 1];
        let selected_leaves = regular_tree.get_leaves_by_indices(&selected_indices);
        let selected_auth_paths = regular_tree.get_multi_proof(&selected_indices);
        let auth_pairs: Vec<(PartialAuthenticationPath<Digest>, Digest)> =
            zip(selected_auth_paths, selected_leaves.clone()).collect();

        for (partial_auth_path, _digest) in auth_pairs.clone() {
            assert_eq!(
                  expected_path_length,
                  partial_auth_path.0.len(),
                  "A generated authentication path must have length the height of the tree (not counting the root)"
              );
        }

        let regular_verify = MerkleTree::<Hasher>::verify_multi_proof(
            regular_tree.get_root().clone(),
            &selected_indices,
            &auth_pairs,
        );

        let how_many_salts_again = 0;

        let salts_per_element = 5;
        let salts_preimage: Vec<BFieldElement> =
            BFieldElement::random_elements(salts_per_element * leaves.len(), &mut rng);
        let salts: Vec<Digest> = salts_preimage
            .iter()
            .map(|x| hasher.hash(x))
            .take(how_many_salts_again)
            .collect();

        let unsalted_salted_tree: SaltedMerkleTree<Hasher> =
            SaltedMerkleTree::from_digests(&selected_leaves.clone(), &salts);

        let salted_proof = unsalted_salted_tree.get_multi_proof_and_salts(&selected_indices);

        let unsalted_salted_verify = SaltedMerkleTree::<Hasher>::verify_multi_proof(
            unsalted_salted_tree.get_root().clone(),
            &selected_indices,
            &selected_leaves,
            &salted_proof,
        );

        assert_eq!(regular_verify, unsalted_salted_verify);
    }

    #[test]
    fn merkle_tree_verify_multi_proof_test() {
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;

        let mut prng = rand::thread_rng();

        // Number of Merkle tree leaves
        let n_valuess = &[2, 4, 8, 16, 128, 256, 512, 1024, 2048, 4096, 8192];
        let expected_path_lengths = &[1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13]; // log2(128), root node not included
        for (n_values, expected_path_length) in izip!(n_valuess, expected_path_lengths) {
            let values: Vec<BFieldElement> =
                generate_random_numbers_u128(*n_values, Some(1u128 << 63))
                    .iter()
                    .map(|x| BFieldElement::new(*x as u64))
                    .collect();

            let hasher = Hasher::new();
            let leaves: Vec<Digest> = values.iter().map(|x| hasher.hash(x)).collect();
            let mut tree: MerkleTree<Hasher> = MerkleTree::<Hasher>::from_digests(&leaves);

            for _ in 0..3 {
                // Ask for an arbitrary amount of indices less than the total
                let n_indices = (prng.next_u64() % *n_values as u64 / 2) as usize + 1;

                // Generate that amount of indices in the valid index range [0,128)
                let selected_indices: Vec<usize> =
                    generate_random_numbers_u128(n_indices, Some(*n_values as u128))
                        .iter()
                        .map(|x| *x as usize)
                        .unique()
                        .collect();

                let selected_leaves = tree.get_leaves_by_indices(&selected_indices);
                let selected_auth_paths = tree.get_multi_proof(&selected_indices);

                for auth_path in selected_auth_paths.iter() {
                    assert_eq!(*expected_path_length, auth_path.0.len());
                }

                let auth_pairs: Vec<(PartialAuthenticationPath<Digest>, Digest)> =
                    zip(selected_auth_paths, selected_leaves.clone()).collect();

                let good_tree = MerkleTree::<Hasher>::verify_multi_proof(
                    tree.get_root(),
                    &selected_indices,
                    &auth_pairs,
                );
                assert!(
                    good_tree,
                    "An uncorrupted tree and an uncorrupted proof should verify."
                );

                // Begin negative tests

                // Corrupt the root and thereby the tree
                let orig_root_hash = tree.get_root();
                let mut bad_root_hash = tree.get_root();
                bad_root_hash.toggle_corruption();
                let verified = MerkleTree::<Hasher>::verify_multi_proof(
                    bad_root_hash,
                    &selected_indices,
                    &auth_pairs,
                );
                assert!(!verified, "Should not verify against bad root hash.");

                // Restore root
                tree.set_root(orig_root_hash);

                // Corrupt the proof and thus fail to verify against the (valid) tree.
                let mut bad_proof = auth_pairs.clone();
                let random_index =
                    ((prng.next_u64() % n_indices as u64 / 2) as usize) % bad_proof.len();
                bad_proof[random_index].1.toggle_corruption();

                assert!(
                    !MerkleTree::<Hasher>::verify_multi_proof(
                        tree.get_root(),
                        &selected_indices,
                        &bad_proof,
                    ),
                    "Should not verify with corrupted proof."
                );
            }
        }
    }

    fn b3h(hex: impl AsRef<[u8]>) -> Blake3Hash {
        Blake3Hash(blake3::Hash::from_hex(hex).unwrap())
    }

    #[test]
    fn merkle_tree_test_simple() {
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;

        let one = BFieldElement::ring_one();
        let two = BFieldElement::new(2);
        let three = BFieldElement::new(3);
        let four = BFieldElement::new(4);

        let hasher = Hasher::new();
        let values = vec![BFieldElement::new(1)];
        let leaves: Vec<Digest> = values.iter().map(|x| hasher.hash(x)).collect();

        let single_mt_one: MerkleTree<Hasher> = MerkleTree::from_digests(&leaves);
        let expected_root_hash: Digest =
            b3h("216eb5cf8e60e66db5dc53e5db5ded0a7c038a00b3bcbebb08b4556c830621a1");
        assert_eq!(expected_root_hash, single_mt_one.get_root());
        assert_eq!(0, single_mt_one.get_height());

        let single_mt_two: MerkleTree<Hasher> = MerkleTree::from_digests(&[hasher.hash(&two)]);
        let expected_root_hash_2: Digest =
            b3h("abb1f67fe5cf64ed79df481ffe011c541028decd98ad105afe1656aca29e5d14");
        assert_eq!(expected_root_hash_2, single_mt_two.get_root());
        assert_eq!(0, single_mt_two.get_height());

        let mt: MerkleTree<Hasher> =
            MerkleTree::from_digests(&[hasher.hash(&one), hasher.hash(&two)]);
        let expected_root_hash_3 =
            b3h("0d5c81b4ed156d9838b3be9e90e5c92ceabd10f40bfdef5c4adf93212d392b24");
        assert_eq!(expected_root_hash_3, mt.get_root());
        assert_eq!(1, mt.get_height());

        let leaf_index = 1;
        let mut auth_path: Vec<Digest> = mt.get_authentication_path(leaf_index);

        let success = MerkleTree::<Hasher>::verify_authentication_path_from_leaf_hash(
            mt.get_root(),
            1,
            hasher.hash(&two),
            auth_path.clone(),
        );
        assert!(success);
        assert_eq!(
            hasher.hash(&one),
            auth_path[0],
            "First element of authentication path is the leaf node"
        );

        let leaf_index = 0;
        auth_path = mt.get_authentication_path(leaf_index);
        assert!(
            MerkleTree::<Hasher>::verify_authentication_path_from_leaf_hash(
                mt.get_root(),
                leaf_index as u32,
                hasher.hash(&one),
                auth_path.clone()
            )
        );
        assert_eq!(hasher.hash(&two), auth_path[0]);
        assert_eq!(1, auth_path.len());

        let mt_reverse: MerkleTree<Hasher> =
            MerkleTree::from_digests(&[hasher.hash(&two), hasher.hash(&one)]);
        let expected_root_hash_4 =
            b3h("1fe7c02631dcf1e025ff34d82e3f164bc5839e8bca91814adf3c25440d9eac48");
        assert_eq!(expected_root_hash_4, mt_reverse.get_root());
        assert_eq!(1, mt_reverse.get_height());

        let leaves: Vec<Digest> = [one, two, three, four]
            .iter()
            .map(|x| hasher.hash(x))
            .collect();
        let mt_four: MerkleTree<Hasher> = MerkleTree::from_digests(&leaves);
        let expected_root_hash_5 =
            b3h("7b1084a91bbc707f367a24bf7fab7599fae17d34bd735dd80c671f54d08585ae");
        assert_eq!(expected_root_hash_5, mt_four.get_root());
        assert_ne!(mt.get_root(), mt_reverse.get_root());
        assert_eq!(2, mt_four.get_height());
        auth_path = mt_four.get_authentication_path(1);
        assert_eq!(2, auth_path.len());
        assert!(
            MerkleTree::<Hasher>::verify_authentication_path_from_leaf_hash(
                mt_four.get_root(),
                1,
                hasher.hash(&two),
                auth_path.clone()
            )
        );
        assert_eq!(hasher.hash(&one), auth_path[0]);

        auth_path[0] = hasher.hash(&three);
        assert!(
            !MerkleTree::<Hasher>::verify_authentication_path_from_leaf_hash(
                mt_four.get_root(),
                1,
                hasher.hash(&two),
                auth_path.clone()
            ),
            "Merkle tree auth path verification must fail when authentication path is wrong"
        );

        auth_path[0] = hasher.hash(&one);
        assert!(
            MerkleTree::<Hasher>::verify_authentication_path_from_leaf_hash(
                mt_four.get_root(),
                1,
                hasher.hash(&two),
                auth_path.clone()
            ),
            "MT AP must succeed with valid parameters"
        );
        assert!(
            !MerkleTree::<Hasher>::verify_authentication_path_from_leaf_hash(
                mt_four.get_root(),
                1,
                hasher.hash(&four),
                auth_path.clone()
            ),
            "Merkle tree authentication path must fail when leaf digest is wrong"
        );
        assert!(
            !MerkleTree::<Hasher>::verify_authentication_path_from_leaf_hash(
                mt_reverse.get_root(),
                1,
                hasher.hash(&two),
                auth_path.clone()
            ),
            "Merkle tree authentication path must fail when root is changed"
        );
        assert!(
            !MerkleTree::<Hasher>::verify_authentication_path_from_leaf_hash(
                mt_four.get_root(),
                2,
                hasher.hash(&two),
                auth_path.clone()
            ),
            "Merkle tree authentication path must fail when index is changed"
        );

        let auth_paths: Vec<PartialAuthenticationPath<Digest>> = mt_four.get_multi_proof(&[0]);
        assert!(MerkleTree::<Hasher>::verify_multi_proof_from_leaves(
            mt_four.get_root(),
            &[0],
            &mt_four.get_leaves_by_indices(&[0])[..],
            &auth_paths
        ));
        auth_path = mt_four.get_authentication_path(0);

        assert_eq!(
            auth_path.len(),
            auth_paths[0].0 .len(),
            "Methods `get_multi_proof` and `get_authentication_path()` should obtain the same authentication path for leaf at index 0."
        );

        let selected_indices_2 = [0, 1];
        let auth_paths_2 = mt_four.get_multi_proof(&selected_indices_2);
        assert!(MerkleTree::<Hasher>::verify_multi_proof_from_leaves(
            mt_four.get_root(),
            &selected_indices_2,
            &leaves[selected_indices_2[0]..=selected_indices_2[1]],
            &auth_paths_2
        ));

        let selected_indices_3 = [0, 1, 2];
        let auth_paths_3 = mt_four.get_multi_proof(&selected_indices_3);
        assert!(MerkleTree::<Hasher>::verify_multi_proof_from_leaves(
            mt_four.get_root(),
            &selected_indices_3,
            &leaves[selected_indices_3[0]..selected_indices_3.len()],
            &auth_paths_3
        ));

        // Verify that verification of multi-proof where the tree or the proof
        // does not have the indices requested leads to a false return value,
        // and not to a run-time panic.
        let out_of_bounds_index = 42;
        let selected_indices_4 = [0, 1, out_of_bounds_index];
        assert!(!MerkleTree::<Hasher>::verify_multi_proof_from_leaves(
            mt_four.get_root(),
            &selected_indices_4,
            &leaves[selected_indices_4[0]..selected_indices_4.len()],
            &auth_paths_3
        ));
    }

    #[test]
    fn merkle_tree_get_authentication_path_test() {
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;

        let hasher = Hasher::new();

        // 1: Create Merkle tree
        //
        //     root
        //    /    \
        //   x      y
        //  / \    / \
        // 3   6  9   12
        let values_a: Vec<BFieldElement> = vec![3, 6, 9, 12]
            .into_iter()
            .map(BFieldElement::new)
            .collect();
        let leaves_a: Vec<Digest> = values_a.iter().map(|x| hasher.hash(x)).collect();
        let tree_a = MerkleTree::<Hasher>::from_digests(&leaves_a);

        // 2: Get the path for value '9' (index: 2)
        let auth_path_a = tree_a.get_authentication_path(2);

        assert_eq!(
            2,
            auth_path_a.len(),
            "authentication path a has right length"
        );
        assert_eq!(tree_a.nodes[2], auth_path_a[1], "sibling x");
        assert_eq!(tree_a.nodes[7], auth_path_a[0], "sibling 12");

        //        ___root___
        //       /          \
        //      e            f
        //    /   \        /   \
        //   a     b      c     d
        //  / \   / \    / \   / \
        // 3   1 4   1  5   9 8   6
        let values_b: Vec<BFieldElement> = vec![3, 1, 4, 1, 5, 9, 8, 6]
            .into_iter()
            .map(BFieldElement::new)
            .collect();
        let leaves_b: Vec<Digest> = values_b.iter().map(|x| hasher.hash(x)).collect();
        let tree_b = MerkleTree::<Hasher>::from_digests(&leaves_b);

        // merkle leaf index: 5
        // merkle leaf value: 9
        // auth path: 5 ~> d ~> e
        let auth_path_b = tree_b.get_authentication_path(5);

        assert_eq!(3, auth_path_b.len());
        assert_eq!(tree_b.nodes[12], auth_path_b[0], "sibling 5");
        assert_eq!(tree_b.nodes[7], auth_path_b[1], "sibling d");
        assert_eq!(tree_b.nodes[2], auth_path_b[2], "sibling e");
    }

    #[test]
    fn verify_authentication_path_from_leaf_hash_with_memoization_test() {
        // This is a test of the helper function for verification of a collection
        // of partial authentication paths. This function is more thoroughly tested
        // through tests of `verify_multi_proof_from_leaves` from which it is called.
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;

        let hasher = Hasher::new();

        // 1: Create Merkle tree
        //
        //     root
        //    /    \
        //   x      y
        //  / \    / \
        // 3   6  9   12
        let values_a: Vec<BFieldElement> = vec![3, 6, 9, 12]
            .into_iter()
            .map(BFieldElement::new)
            .collect();
        let leaves_a: Vec<Digest> = values_a.iter().map(|x| hasher.hash(x)).collect();
        let tree_a = MerkleTree::<Hasher>::from_digests(&leaves_a);
        let mut partial_tree: HashMap<u64, Digest> = HashMap::new();
        let leaf_index: usize = 2;
        partial_tree.insert(4 + leaf_index as u64, leaves_a[leaf_index]);
        let auth_path_leaf_index_2 = tree_a.get_authentication_path(leaf_index);

        assert!(
            MerkleTree::<Hasher>::verify_authentication_path_from_leaf_hash(
                tree_a.get_root(),
                leaf_index as u32,
                leaves_a[leaf_index],
                auth_path_leaf_index_2.clone()
            )
        );

        assert!(
            MerkleTree::<Hasher>::verify_authentication_path_from_leaf_hash_with_memoization(
                &tree_a.get_root(),
                leaf_index as u32,
                &auth_path_leaf_index_2,
                &partial_tree,
                &hasher,
            ),
            "Valid auth path/partial tree must validate"
        );
        let auth_path_leaf_index_3 = tree_a.get_authentication_path(3);
        assert!(
            !MerkleTree::<Hasher>::verify_authentication_path_from_leaf_hash_with_memoization(
                &tree_a.get_root(),
                leaf_index as u32,
                &auth_path_leaf_index_3,
                &partial_tree,
                &hasher,
            ),
            "Invalid auth path/partial tree must not validate"
        );

        let values_b: Vec<BFieldElement> = vec![3, 6, 9, 100]
            .into_iter()
            .map(BFieldElement::new)
            .collect();
        let leaves_b: Vec<Digest> = values_b.iter().map(|x| hasher.hash(x)).collect();
        let tree_b = MerkleTree::<Hasher>::from_digests(&leaves_b);
        assert!(
            !MerkleTree::<Hasher>::verify_authentication_path_from_leaf_hash_with_memoization(
                &tree_b.get_root(),
                leaf_index as u32,
                &auth_path_leaf_index_2,
                &partial_tree,
                &hasher,
            ),
            "Bad Merkle tree root must fail to validate"
        );
    }

    // Test of salted Merkle trees
    #[test]
    fn salted_merkle_tree_get_authentication_path_test() {
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;
        type SMT = SaltedMerkleTree<Hasher>;

        // 1: Create Merkle tree
        //
        //     root
        //    /    \
        //   x      y
        //  / \    / \
        // 3   6  9   12
        let hasher = Hasher::new();
        let values_a = [
            BFieldElement::new(3),
            BFieldElement::new(1),
            BFieldElement::new(4),
            BFieldElement::new(5),
        ];
        let salts_per_leaf = 3;

        let mut rng = rand::thread_rng();
        let salts_a_preimage: Vec<BFieldElement> =
            BFieldElement::random_elements(values_a.len() * salts_per_leaf, &mut rng);
        let salts_a: Vec<_> = salts_a_preimage.iter().map(|x| hasher.hash(x)).collect();
        let leaves_a: Vec<Digest> = values_a.iter().map(|x| hasher.hash(x)).collect();
        let tree_a = SMT::from_digests(&leaves_a, &salts_a);
        assert_eq!(tree_a.get_leaf_count() * salts_per_leaf, tree_a.salts.len());

        // 2: Get the path for value '4' (index: 2)
        let leaf_a_idx = 2;
        let leaf_a_digest = leaves_a[leaf_a_idx];
        let auth_path_a_and_salt = tree_a.get_authentication_path_and_salts(leaf_a_idx);

        // 3: Verify that the proof, along with the salt, works
        assert!(SMT::verify_authentication_path(
            tree_a.get_root(),
            leaf_a_idx as u32,
            leaf_a_digest, // Must be unsalted
            auth_path_a_and_salt.0.clone(),
            auth_path_a_and_salt.1.clone(),
        ));

        assert_eq!(
            leaf_a_idx,
            auth_path_a_and_salt.0.len(),
            "authentication path a has right length"
        );
        assert_eq!(
            3,
            auth_path_a_and_salt.1.len(),
            "Proof contains expected number of salts"
        );
        assert_eq!(
            tree_a.internal_merkle_tree.nodes[2], auth_path_a_and_salt.0[1],
            "sibling x"
        );
        assert_eq!(
            tree_a.internal_merkle_tree.nodes[7], auth_path_a_and_salt.0[0],
            "sibling 12"
        );

        // 4: Change salt and verify that the proof does not work
        let mut auth_path_a_and_salt_bad_salt = tree_a.get_authentication_path_and_salts(2);
        auth_path_a_and_salt_bad_salt.1[0].increment();
        assert!(!SMT::verify_authentication_path(
            tree_a.get_root(),
            leaf_a_idx as u32,
            leaf_a_digest,
            auth_path_a_and_salt_bad_salt.0.clone(),
            auth_path_a_and_salt_bad_salt.1.clone(),
        ));
        auth_path_a_and_salt_bad_salt.1[0].decrement();
        assert!(SMT::verify_authentication_path(
            tree_a.get_root(),
            leaf_a_idx as u32,
            leaf_a_digest,
            auth_path_a_and_salt_bad_salt.0.clone(),
            auth_path_a_and_salt_bad_salt.1.clone(),
        ));
        auth_path_a_and_salt_bad_salt.1[1].increment();
        assert!(!SMT::verify_authentication_path(
            tree_a.get_root(),
            leaf_a_idx as u32,
            leaf_a_digest,
            auth_path_a_and_salt_bad_salt.0.clone(),
            auth_path_a_and_salt_bad_salt.1.clone(),
        ));
        auth_path_a_and_salt_bad_salt.1[1].decrement();
        assert!(SMT::verify_authentication_path(
            tree_a.get_root(),
            leaf_a_idx as u32,
            leaf_a_digest,
            auth_path_a_and_salt_bad_salt.0.clone(),
            auth_path_a_and_salt_bad_salt.1.clone(),
        ));
        auth_path_a_and_salt_bad_salt.1[2].decrement();
        assert!(!SMT::verify_authentication_path(
            tree_a.get_root(),
            leaf_a_idx as u32,
            leaf_a_digest,
            auth_path_a_and_salt_bad_salt.0.clone(),
            auth_path_a_and_salt_bad_salt.1.clone(),
        ));
        auth_path_a_and_salt_bad_salt.1[2].increment();
        assert!(SMT::verify_authentication_path(
            tree_a.get_root(),
            leaf_a_idx as u32,
            leaf_a_digest,
            auth_path_a_and_salt_bad_salt.0.clone(),
            auth_path_a_and_salt_bad_salt.1.clone(),
        ));

        // 5: Change the value and verify that the proof does not work
        let mut corrupt_leaf_a_digest = leaf_a_digest;
        corrupt_leaf_a_digest.increment();
        assert!(!SMT::verify_authentication_path(
            tree_a.get_root(),
            leaf_a_idx as u32,
            corrupt_leaf_a_digest,
            auth_path_a_and_salt_bad_salt.0.clone(),
            auth_path_a_and_salt_bad_salt.1.clone(),
        ));
        corrupt_leaf_a_digest.decrement();
        assert!(SMT::verify_authentication_path(
            tree_a.get_root(),
            leaf_a_idx as u32,
            corrupt_leaf_a_digest,
            auth_path_a_and_salt_bad_salt.0.clone(),
            auth_path_a_and_salt_bad_salt.1.clone(),
        ));
        corrupt_leaf_a_digest.decrement();
        assert!(!SMT::verify_authentication_path(
            tree_a.get_root(),
            leaf_a_idx as u32,
            corrupt_leaf_a_digest,
            auth_path_a_and_salt_bad_salt.0.clone(),
            auth_path_a_and_salt_bad_salt.1.clone(),
        ));

        //        ___root___
        //       /          \
        //      e            f
        //    /   \        /   \
        //   a     b      c     d
        //  / \   / \    / \   / \
        // 3   1 4   1  5   9 8   6

        let mut rng = rand::thread_rng();
        let values_b = &[3, 1, 4, 1, 5, 9, 8, 6].map(BFieldElement::new);
        let leaves_b: Vec<Digest> = values_b.iter().map(|x| hasher.hash(x)).collect();
        let salts_per_leaf = 4;
        let salts_b_preimage: Vec<BFieldElement> =
            BFieldElement::random_elements(values_b.len() * salts_per_leaf, &mut rng);
        let salts_b: Vec<_> = salts_b_preimage.iter().map(|x| hasher.hash(x)).collect();
        let tree_b = SMT::from_digests(&leaves_b, &salts_b);

        // auth path: 8 ~> c ~> e
        let mut auth_path_and_salts_b = tree_b.get_authentication_path_and_salts(6);

        assert_eq!(3, auth_path_and_salts_b.0.len());
        assert_eq!(8 * salts_per_leaf, tree_b.get_salts().len());

        // 6: Ensure that all salts are unique (statistically, they should
        // be)
        // Just how likely is this when taking the Birthday Paradox into account and noticing we
        // have 8x3 salts.
        assert_eq!(
            tree_b.get_salts().len(),
            tree_b
                .get_salts()
                .to_vec()
                .clone()
                .into_iter()
                .unique()
                .count(),
            "This can fail with nonzero probability.  In this case try run the test again."
        );
        assert_eq!(
            tree_b.internal_merkle_tree.nodes[15], auth_path_and_salts_b.0[0],
            "sibling 6"
        );
        assert_eq!(
            tree_b.internal_merkle_tree.nodes[6], auth_path_and_salts_b.0[1],
            "sibling c"
        );
        assert_eq!(
            tree_b.internal_merkle_tree.nodes[2], auth_path_and_salts_b.0[2],
            "sibling e"
        );

        let mut value_b = hasher.hash(&BFieldElement::new(8));

        let root_hash_b = &tree_b.get_root();
        assert!(SMT::verify_authentication_path(
            *root_hash_b,
            6,
            value_b,
            auth_path_and_salts_b.0.clone(),
            auth_path_and_salts_b.1.clone(),
        ));

        // 7: Change the value and verify that it fails
        value_b.decrement();
        assert!(!SMT::verify_authentication_path(
            *root_hash_b,
            6,
            value_b,
            auth_path_and_salts_b.0.clone(),
            auth_path_and_salts_b.1.clone(),
        ));
        value_b.increment();

        // 8: Change salt and verify that verification fails
        auth_path_and_salts_b.1[3].decrement();
        assert!(!SMT::verify_authentication_path(
            *root_hash_b,
            6,
            value_b,
            auth_path_and_salts_b.0.clone(),
            auth_path_and_salts_b.1.clone(),
        ));
        auth_path_and_salts_b.1[3].increment();
        assert!(SMT::verify_authentication_path(
            *root_hash_b,
            6,
            value_b,
            auth_path_and_salts_b.0.clone(),
            auth_path_and_salts_b.1.clone(),
        ));

        // 9: Verify that simple multipath authentication paths work
        let auth_path_and_salts_b_multi_0: SaltedMultiProof<Digest> =
            tree_b.get_multi_proof_and_salts(&[0, 1]);
        let multi_values_0 = vec![BFieldElement::new(3), BFieldElement::new(1)];
        let multi_digests_0: Vec<Digest> = multi_values_0.iter().map(|x| hasher.hash(x)).collect();
        assert!(SMT::verify_multi_proof(
            *root_hash_b,
            &[0, 1],
            &multi_digests_0,
            &auth_path_and_salts_b_multi_0
        ));
        assert_eq!(
            2,
            count_hashes(&auth_path_and_salts_b_multi_0),
            "paths [0,1] need two hashes"
        );

        // Verify that we get values
        let auth_paths_and_salts = tree_b.get_multi_proof_and_salts(&[0, 1]);
        assert_eq!(2, auth_paths_and_salts.len());
        assert_eq!(
            auth_path_and_salts_b_multi_0[0].0,
            auth_paths_and_salts[0].0
        );
        assert_eq!(
            auth_path_and_salts_b_multi_0[0].1,
            auth_paths_and_salts[0].1
        );
        assert_eq!(
            auth_path_and_salts_b_multi_0[1].0,
            auth_paths_and_salts[1].0
        );
        assert_eq!(
            auth_path_and_salts_b_multi_0[1].1,
            auth_paths_and_salts[1].1
        );

        // Verify that the composite verification works
        assert!(SMT::verify_multi_proof(
            *root_hash_b,
            &[0, 1],
            &multi_digests_0,
            &auth_paths_and_salts
        ));

        let mut bad_root_hash_b = tree_b.get_root();
        bad_root_hash_b.toggle_corruption();

        assert!(!SMT::verify_multi_proof(
            bad_root_hash_b,
            &[0, 1],
            &multi_digests_0,
            &auth_paths_and_salts
        ));

        let auth_path_b_multi_1 = tree_b.get_multi_proof_and_salts(&[1]);
        let multi_values_1 = vec![BFieldElement::new(1)];
        let multi_digests_1: Vec<Digest> = multi_values_1.iter().map(|x| hasher.hash(x)).collect();
        assert!(SMT::verify_multi_proof(
            *root_hash_b,
            &[1],
            &multi_digests_1,
            &auth_path_b_multi_1
        ));
        assert_eq!(
            3,
            count_hashes(&auth_path_b_multi_1),
            "paths [1] need two hashes"
        );

        let auth_path_b_multi_2 = tree_b.get_multi_proof_and_salts(&[1, 0]);
        let multi_values_2 = vec![BFieldElement::new(1), BFieldElement::new(3)];
        let multi_digests_2: Vec<Digest> = multi_values_2.iter().map(|x| hasher.hash(x)).collect();
        assert!(SMT::verify_multi_proof(
            *root_hash_b,
            &[1, 0],
            &multi_digests_2,
            &auth_path_b_multi_2
        ));

        let mut auth_path_b_multi_3 = tree_b.get_multi_proof_and_salts(&[0, 1, 2, 4, 7]);
        let mut multi_values_3 = vec![
            BFieldElement::new(3),
            BFieldElement::new(1),
            BFieldElement::new(4),
            BFieldElement::new(5),
            BFieldElement::new(6),
        ];
        let multi_digests_3: Vec<Digest> = multi_values_3.iter().map(|x| hasher.hash(x)).collect();
        assert!(SMT::verify_multi_proof(
            *root_hash_b,
            &[0, 1, 2, 4, 7],
            &multi_digests_3,
            &auth_path_b_multi_3
        ));

        let multi_digests_3b = [
            leaves_b[0],
            leaves_b[1],
            leaves_b[2],
            leaves_b[4],
            leaves_b[7],
        ];

        let temp = tree_b.get_multi_proof_and_salts(&[0, 1, 2, 4, 7]);
        assert_eq!(
            3,
            count_hashes(&temp),
            "paths [0, 1, 2, 4, 7] need three hashes"
        );

        // 10: change a hash, verify failure
        auth_path_b_multi_3[1].1[0].increment();
        assert!(!SMT::verify_multi_proof(
            *root_hash_b,
            &[0, 1, 2, 4, 7],
            &multi_digests_3b,
            &auth_path_b_multi_3
        ));

        auth_path_b_multi_3[1].1[0].decrement();
        assert!(SMT::verify_multi_proof(
            *root_hash_b,
            &[0, 1, 2, 4, 7],
            &multi_digests_3b,
            &auth_path_b_multi_3
        ));

        // 11: change a value, verify failure
        multi_values_3[0].increment();
        assert!(!SMT::verify_multi_proof(
            *root_hash_b,
            &[1, 0, 4, 7, 2],
            &multi_digests_3b,
            &auth_path_b_multi_3
        ));

        multi_values_3[0].decrement();
        assert!(SMT::verify_multi_proof(
            *root_hash_b,
            &[0, 1, 2, 4, 7],
            &multi_digests_3b,
            &auth_path_b_multi_3
        ));

        // Change root hash again, verify failue
        let mut another_bad_root_hash_b = *root_hash_b;
        another_bad_root_hash_b.toggle_corruption();
        assert!(!SMT::verify_multi_proof(
            another_bad_root_hash_b.into(),
            &[0, 1, 2, 4, 7],
            &multi_digests_3b,
            &auth_path_b_multi_3
        ));
    }

    #[test]
    fn salted_merkle_tree_get_authentication_path_xfields_test() {
        type Hasher = blake3::Hasher;
        type SMT = SaltedMerkleTree<Hasher>;
        type Digest = Blake3Hash;

        // 1: Create Merkle tree
        //
        //     root
        //    /    \
        //   x      y
        //  / \    / \
        // 3   6  9   12
        let mut rng = rand::thread_rng();
        let hasher = Hasher::new();

        let values_a = &[
            XFieldElement::new([3, 3, 3].map(BFieldElement::new)),
            XFieldElement::new([1, 1, 1].map(BFieldElement::new)),
            XFieldElement::new([4, 4, 4].map(BFieldElement::new)),
            XFieldElement::new([5, 5, 5].map(BFieldElement::new)),
        ];
        let leaves_a: Vec<Digest> = values_a.iter().map(|x| hasher.hash(x)).collect();
        let salts_per_leaf = 3;
        let salts_a_preimage: Vec<BFieldElement> =
            BFieldElement::random_elements(values_a.len() * salts_per_leaf, &mut rng);
        let salts_a: Vec<_> = salts_a_preimage.iter().map(|x| hasher.hash(x)).collect();
        let tree_a = SMT::from_digests(&leaves_a, &salts_a);

        assert_eq!(3, salts_per_leaf);
        assert_eq!(3 * 4, tree_a.salts.len());

        // 2: Get the path for value '4' (index: 2)
        let mut auth_path_a_and_salt = tree_a.get_authentication_path_and_salts(2);

        // 3: Verify that the proof, along with the salt, works
        let root_hash_a = tree_a.get_root();
        let value = XFieldElement::new([4, 4, 4].map(BFieldElement::new));
        let mut leaf = hasher.hash(&value);
        assert!(SMT::verify_authentication_path(
            root_hash_a,
            2,
            leaf,
            auth_path_a_and_salt.0.clone(),
            auth_path_a_and_salt.1.clone(),
        ));

        // 4: Change value and verify that it fails
        leaf.increment();
        assert!(!SMT::verify_authentication_path(
            root_hash_a,
            2,
            leaf,
            auth_path_a_and_salt.0.clone(),
            auth_path_a_and_salt.1.clone(),
        ));
        leaf.decrement();
        assert!(SMT::verify_authentication_path(
            root_hash_a,
            2,
            leaf,
            auth_path_a_and_salt.0.clone(),
            auth_path_a_and_salt.1.clone(),
        ));
        leaf.decrement();
        assert!(!SMT::verify_authentication_path(
            root_hash_a,
            2,
            leaf,
            auth_path_a_and_salt.0.clone(),
            auth_path_a_and_salt.1.clone(),
        ));
        leaf.increment();
        assert!(SMT::verify_authentication_path(
            root_hash_a,
            2,
            leaf,
            auth_path_a_and_salt.0.clone(),
            auth_path_a_and_salt.1.clone(),
        ));
        leaf.increment();
        assert!(!SMT::verify_authentication_path(
            root_hash_a,
            2,
            leaf,
            auth_path_a_and_salt.0.clone(),
            auth_path_a_and_salt.1.clone(),
        ));
        leaf.decrement();
        assert!(SMT::verify_authentication_path(
            root_hash_a,
            2,
            leaf,
            auth_path_a_and_salt.0.clone(),
            auth_path_a_and_salt.1.clone(),
        ));

        // 5: Change salt and verify that it fails
        auth_path_a_and_salt.1[0].decrement();
        assert!(!SMT::verify_authentication_path(
            root_hash_a,
            2,
            leaf,
            auth_path_a_and_salt.0.clone(),
            auth_path_a_and_salt.1.clone(),
        ));
        auth_path_a_and_salt.1[0].increment();
        assert!(SMT::verify_authentication_path(
            root_hash_a,
            2,
            leaf,
            auth_path_a_and_salt.0.clone(),
            auth_path_a_and_salt.1.clone(),
        ));
    }

    #[test]
    fn salted_merkle_tree_regression_test_0() {
        type Hasher = blake3::Hasher;
        type SMT = SaltedMerkleTree<Hasher>;
        type Digest = Blake3Hash;

        let hasher = Hasher::new();

        // This test was used to catch a bug in the implementation of
        // `SaltedMerkleTree::get_leafless_multi_proof_with_salts_and_values`
        // The bug that this test caught was *fixed* in 5ad285bd867bf8c6c4be380d8539ba37f4a7409a
        // and introduced in 89cfb194f02903534b1621b03a047c128af7d6c2.

        // Build a representation of the SMT made from values
        // `451282252958277131` and `3796554602848593414` as this is where the error was
        // first caught.
        let mut rng = rand::thread_rng();

        let values_reg0 = vec![
            BFieldElement::new(451282252958277131),
            BFieldElement::new(3796554602848593414),
        ];
        let leaves_reg0: Vec<Digest> = values_reg0.iter().map(|x| hasher.hash(x)).collect();
        let salts_per_leaf = 3;
        let salts_reg0_preimage: Vec<BFieldElement> =
            BFieldElement::random_elements(values_reg0.len() * salts_per_leaf, &mut rng);
        let salts_reg0: Vec<_> = salts_reg0_preimage.iter().map(|x| hasher.hash(x)).collect();
        let tree_reg0 = SMT::from_digests(&leaves_reg0, &salts_reg0);

        let selected_leaf_indices_reg0 = [0];
        let selected_leaves_reg0 = leaves_reg0[0]; // tree_reg0.get_salted_leaves_by_indices(&selected_leaf_indices_reg0);
        let proof_reg0 = tree_reg0.get_multi_proof_and_salts(&selected_leaf_indices_reg0);

        assert!(SMT::verify_multi_proof(
            tree_reg0.get_root(),
            &selected_leaf_indices_reg0,
            &[selected_leaves_reg0],
            &proof_reg0,
        ));

        let selected_leaf_indices_reg1 = vec![1];
        let selected_leaves_reg1 = leaves_reg0[1];
        let proof_reg1 = tree_reg0.get_multi_proof_and_salts(&selected_leaf_indices_reg1);

        assert!(SMT::verify_multi_proof(
            tree_reg0.get_root(),
            &selected_leaf_indices_reg1,
            &[selected_leaves_reg1],
            &proof_reg1,
        ));
    }

    #[test]
    fn salted_merkle_tree_verify_multi_proof_test() {
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;
        type SMT = SaltedMerkleTree<Hasher>;

        let hasher = Hasher::new();
        let mut prng = rand::thread_rng();

        // Number of Merkle tree leaves
        let n_valuess = &[2, 4, 8, 16, 128, 256, 512, 1024, 2048, 4096, 8192];
        let expected_path_lengths = &[1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13]; // log2(128), root node not included

        for (n_values, expected_path_length) in izip!(n_valuess, expected_path_lengths) {
            let values: Vec<BFieldElement> =
                generate_random_numbers_u128(*n_values, Some(1u128 << 63))
                    .iter()
                    .map(|x| BFieldElement::new(*x as u64))
                    .collect();

            let leaves: Vec<Digest> = values.iter().map(|x| hasher.hash(x)).collect();
            let salts_per_leaf = 3;
            let salts_preimage: Vec<BFieldElement> =
                BFieldElement::random_elements(values.len() * salts_per_leaf, &mut prng);
            let salts: Vec<_> = salts_preimage.iter().map(|x| hasher.hash(x)).collect();
            let tree = SMT::from_digests(&leaves, &salts);

            for _ in 0..3 {
                // Ask for an arbitrary amount of indices less than the total
                let max_indices = (prng.next_u64() % *n_values as u64 / 2) as usize + 1;

                // Generate that amount of indices in the valid index range [0,128)
                let indices: Vec<usize> =
                    generate_random_numbers_u128(max_indices, Some(*n_values as u128))
                        .iter()
                        .map(|x| *x as usize)
                        .unique()
                        .collect();
                let actual_number_of_indices = indices.len();

                let selected_leaves: Vec<_> = indices.iter().map(|i| leaves[*i]).collect();
                let mut proof: SaltedMultiProof<Digest> = tree.get_multi_proof_and_salts(&indices);

                for path in proof.iter() {
                    assert_eq!(*expected_path_length, path.0 .0.len());
                }

                assert!(SMT::verify_multi_proof(
                    tree.get_root(),
                    &indices,
                    &selected_leaves,
                    &proof,
                ));
                let mut bad_root_hash = tree.get_root();
                bad_root_hash.toggle_corruption();
                assert!(!SMT::verify_multi_proof(
                    bad_root_hash,
                    &indices,
                    &selected_leaves,
                    &proof,
                ));

                // Verify that an invalid leaf fails verification

                let pick = (prng.next_u64() % actual_number_of_indices as u64) as usize;

                let rnd_leaf_idx = (prng.next_u64() % selected_leaves.len() as u64) as usize;
                let mut corrupted_leaves = selected_leaves.clone();
                corrupted_leaves[rnd_leaf_idx].increment();
                assert!(!SMT::verify_multi_proof(
                    tree.get_root(),
                    &indices,
                    &corrupted_leaves,
                    &proof,
                ));

                corrupted_leaves[rnd_leaf_idx].decrement();
                assert!(SMT::verify_multi_proof(
                    tree.get_root(),
                    &indices,
                    &corrupted_leaves,
                    &proof,
                ));

                // Verify that an invalid salt fails verification
                proof[(pick + 1) % actual_number_of_indices].1[1].decrement();
                let mut corrupted_leaves = selected_leaves.clone();
                corrupted_leaves[rnd_leaf_idx].increment();
                assert!(!SMT::verify_multi_proof(
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

        type Hasher = RescuePrimeProduction;
        type MT = MerkleTree<Hasher>;
        type Digest = Vec<BFieldElement>;

        let hasher = Hasher::new();

        let exponent = 6;
        let num_leaves = usize::pow(2, exponent);
        assert!(
            other::is_power_of_two(num_leaves),
            "Size of input for Merkle tree must be a power of 2"
        );

        let offset = 17;

        let values: Vec<BFieldElement> = (offset..num_leaves + offset)
            .map(|i| BFieldElement::new(i as u64))
            .collect();

        let leaves: Vec<Digest> = values.iter().map(|x| hasher.hash(x)).collect();

        let tree = MT::from_digests(&leaves);

        assert_eq!(
            tree.get_leaf_count(),
            num_leaves,
            "All leaves should have been added to the Merkle tree."
        );

        let root_hash = tree.get_root().to_owned();

        for (leaf_idx, leaf) in leaves.iter().enumerate() {
            let ap = tree.get_authentication_path(leaf_idx);
            let verdict = MT::verify_authentication_path_from_leaf_hash(
                root_hash.clone(),
                leaf_idx as u32,
                (*leaf).clone(),
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
        /// This tests that we do not confuse indices and payloads in the
        /// test `verify_all_leaves_individually`.

        type Hasher = RescuePrimeProduction;
        type MT = MerkleTree<Hasher>;
        type Digest = Vec<BFieldElement>;

        let hasher = Hasher::new();

        let exponent = 6;
        let num_leaves = usize::pow(2, exponent);
        assert!(
            other::is_power_of_two(num_leaves),
            "Size of input for Merkle tree must be a power of 2"
        );

        let offset = 17 * 17;

        let values: Vec<BFieldElement> = (offset..num_leaves as u64 + offset)
            .map(|i| BFieldElement::new(i))
            .collect();

        let mut leaves: Vec<Digest> = values.iter().map(|x| hasher.hash(x)).collect();

        // A payload integrity test
        let test_leaf = 42;
        let payload_offset = 317;
        let payload = BFieldElement::new((test_leaf + payload_offset) as u64);

        // Embed
        leaves[test_leaf] = hasher.hash(&payload);

        let tree = MT::from_digests(&leaves[..]);

        assert_eq!(
            tree.get_leaf_count(),
            num_leaves,
            "All leaves should have been added to the Merkle tree."
        );

        let root_hash = tree.get_root().to_owned();

        (|leaf_idx: usize, leaf: &BFieldElement| {
            let ap = tree.get_authentication_path(leaf_idx);
            let verdict = MT::verify_authentication_path_from_leaf_hash(
                root_hash.clone(),
                leaf_idx as u32,
                hasher.hash(leaf),
                ap,
            );
            assert_eq!(tree.get_leaf_by_index(test_leaf), hasher.hash(&payload));
            assert!(
                verdict,
                "Rejected: `leaf: {:?}` at `leaf_idx: {:?}` failed to verify.",
                { leaf },
                { leaf_idx }
            );
        })(test_leaf, &payload)
    }

    #[test]
    fn build_xlix_merkle_tree() {
        type RP = RescuePrimeProduction;
        type RPXLIX = RescuePrimeXlix<RP_DEFAULT_WIDTH>;
        type Hasher = RPXLIX;

        let mut rng = rand::thread_rng();
        let hasher = Hasher::new();
        let elements = BFieldElement::random_elements(128, &mut rng);
        let max_length = 5;
        let leaves: Vec<_> = elements
            .iter()
            .map(|x| hasher.hash(&[*x], max_length))
            .collect();

        let mt = MerkleTree::<RP>::from_digests(&leaves);
        let mt_xlix = MerkleTree::<RPXLIX>::from_digests(&leaves);

        println!("Merkle root (RP 1): {:?}", mt.get_root());
        println!("Merkle root (RP 2): {:?}", mt_xlix.get_root());
        assert!(true, "If we make it this far, we are good.")
    }

    #[test]
    fn increment_overflow() {
        let mut max_hash = Blake3Hash::from(u128::MAX);
        max_hash.increment();
        // Verifies that increment does not overflow.
    }

    #[test]
    fn decrement_overflow() {
        let minu128 = u128::MIN;
        let mut zero_hash = Blake3Hash::from(minu128);
        zero_hash.decrement();
        // Verifies that decrement does not overflow.
    }
}
