use crate::shared_math::other::{self, log_2_floor};
use crate::shared_math::traits::GetRandomElements;
use crate::utils::blake3_digest_serialize;
use itertools::izip;
use rand::prelude::ThreadRng;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

pub type Blake3Digest = [u8; 32];

pub const BLAKE3ZERO: Blake3Digest = [0u8; 32];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node<T> {
    pub value: Option<T>,
    pub hash: Blake3Digest,
}

#[derive(Clone, Debug)]
pub struct MerkleTree<T> {
    pub root_hash: Blake3Digest,
    pub nodes: Vec<Node<T>>,
    pub height: u8,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PartialAuthenticationPath<T: Clone + Debug + PartialEq + Serialize>(
    pub Vec<Option<Node<T>>>,
);

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct LeaflessPartialAuthenticationPath(pub Vec<Option<Blake3Digest>>);

/// Method for extracting the value for which a compressed Merkle proof element is for.
impl<T: Clone + Debug + Serialize + PartialEq> PartialAuthenticationPath<T> {
    /// Given a proof_element: CompressedAuthenticationPath<T>, this returns the value
    /// `proof_element.0[0].clone().unwrap().value.unwrap();`
    pub fn get_value(&self) -> T {
        match self.0.first() {
            None => panic!("CompressedAuthenticationPath was empty"),
            Some(option) => match option {
                None => panic!("First element of CompressedAuthenticationPath was pruned"),
                Some(node) => match &node.value {
                    None => panic!("No value of first element of CompressedAuthenticationPath"),
                    Some(val) => val.clone(),
                },
            },
        }
    }
}

impl<T: Clone + Serialize + Debug + PartialEq> MerkleTree<T> {
    pub fn verify_proof(root_hash: Blake3Digest, index: u64, proof: Vec<Node<T>>) -> bool {
        let mut mut_index = index + 2u64.pow(proof.len() as u32);
        let mut v = proof[0].clone();
        let mut hasher = blake3::Hasher::new();
        for node in proof.iter().skip(1) {
            if mut_index % 2 == 0 {
                hasher.update(&v.hash[..]);
                hasher.update(&node.hash[..]);
            } else {
                hasher.update(&node.hash[..]);
                hasher.update(&v.hash[..]);
            }
            v.hash = *hasher.finalize().as_bytes();
            hasher.reset();
            mut_index /= 2;
        }
        let expected_hash = blake3_digest_serialize(&proof[0].value.clone().unwrap());
        v.hash == root_hash && expected_hash == proof[0].hash
    }

    fn get_value_by_index(&self, index: usize) -> T {
        assert!(
            index < self.nodes.len() / 2,
            "Out of bounds index requested"
        );
        self.nodes[self.nodes.len() / 2 + index]
            .value
            .clone()
            .unwrap()
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.nodes[self.nodes.len() / 2..self.nodes.len()]
            .iter()
            .map(|x| x.value.clone().unwrap())
            .collect()
    }

    pub fn from_vec(values: &[T]) -> Self {
        assert!(
            other::is_power_of_two(values.len()),
            "Size of input for Merkle tree must be a power of 2"
        );

        let mut nodes: Vec<Node<T>> = vec![
            Node {
                value: None,
                hash: BLAKE3ZERO,
            };
            2 * values.len()
        ];
        for i in 0..values.len() {
            nodes[values.len() + i].hash = blake3_digest_serialize(&values[i]);
            nodes[values.len() + i].value = Some(values[i].clone());
        }

        // loop from `len(L) - 1` to 1
        let mut hasher = blake3::Hasher::new();
        for i in (1..(nodes.len() / 2)).rev() {
            hasher.update(&nodes[i * 2].hash[..]);
            hasher.update(&nodes[i * 2 + 1].hash[..]);
            nodes[i].hash = *hasher.finalize().as_bytes();
            hasher.reset();
        }

        // nodes[0] is never used for anything.
        MerkleTree {
            root_hash: nodes[1].hash,
            nodes,
            height: log_2_floor(values.len() as u64) as u8 + 1,
        }
    }

    pub fn get_proof(&self, mut index: usize) -> Vec<Node<T>> {
        let mut proof: Vec<Node<T>> = Vec::with_capacity(self.height as usize);
        index += self.nodes.len() / 2;
        proof.push(self.nodes[index].clone());
        while index > 1 {
            proof.push(self.nodes[index ^ 1].clone());
            index /= 2;
        }
        proof
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
    pub fn get_authentication_path(&self, index: usize) -> Vec<Blake3Digest> {
        let mut auth_path: Vec<Blake3Digest> = Vec::with_capacity(self.height as usize);

        let mut i = index + self.nodes.len() / 2;
        while i > 1 {
            // We get the sibling node by XOR'ing with 1.
            let sibling_i = i ^ 1;
            auth_path.push(self.nodes[sibling_i].hash);
            i /= 2;
        }

        // We don't include the root hash in the authentication path
        // because it's implied in the context of use.
        //auth_path.push(self.root_hash);

        auth_path
    }

    fn verify_authentication_path_from_leaf_hash(
        root_hash: Blake3Digest,
        index: u32,
        leaf_hash: Blake3Digest,
        auth_path: Vec<Blake3Digest>,
    ) -> bool {
        let path_length = auth_path.len() as u32;
        let mut hasher = blake3::Hasher::new();

        // Initialize `acc_hash' as leaf_hash
        let mut acc_hash = leaf_hash;
        let mut i = index + 2u32.pow(path_length);
        for path_hash in auth_path.iter() {
            // Use Merkle tree index parity (odd/even) to determine which
            // order to concatenate the hashes before hashing them.
            if i % 2 == 0 {
                hasher.update(&acc_hash);
                hasher.update(&path_hash[..]);
            } else {
                hasher.update(&path_hash[..]);
                hasher.update(&acc_hash);
            }
            acc_hash = *hasher.finalize().as_bytes();
            hasher.reset();
            i /= 2;
        }

        acc_hash == root_hash
    }

    // Verify the `authentication path' of a `value' with an `index' from the
    // `root_hash' of a given Merkle tree. Similar to `verify_proof', but instead of
    // a `proof: Vec<Node<T>>` that contains [ValueNode, ...PathNodes..., RootNode],
    // we only pass an `auth_path: Vec<Blake3Hash>' with the hashes of the path nodes;
    // the `root_hash' is passed along separately, and the `value' hash is computed.
    //
    // The `index' is to know if a given path element is a left- or a right-sibling.
    pub fn verify_authentication_path(
        root_hash: Blake3Digest,
        index: u32,
        value: T,
        auth_path: Vec<Blake3Digest>,
    ) -> bool {
        let value_hash = blake3_digest_serialize(&value);

        // Set `leaf_hash` to H(value)
        Self::verify_authentication_path_from_leaf_hash(root_hash, index, value_hash, auth_path)
    }

    pub fn get_root(&self) -> [u8; 32] {
        self.root_hash
    }

    pub fn get_number_of_leafs(&self) -> usize {
        self.nodes.len() / 2
    }

    // Compact Merkle Multiproof Generation
    fn get_multi_proof(&self, indices: &[usize]) -> Vec<PartialAuthenticationPath<T>> {
        let mut calculable_indices: HashSet<usize> = HashSet::new();
        let mut output: Vec<PartialAuthenticationPath<T>> = Vec::with_capacity(indices.len());
        for i in indices.iter() {
            let new_branch: PartialAuthenticationPath<T> =
                PartialAuthenticationPath(self.get_proof(*i).into_iter().map(Some).collect());
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
            for elem in b.0.iter_mut().skip(1) {
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

    // TODO: Rewrite this so it doesn't throw away the values before it fetches them again.
    pub fn get_leafless_multi_proof_with_values(
        &self,
        indices: &[usize],
    ) -> Vec<(LeaflessPartialAuthenticationPath, T)> {
        // auth path and salts
        let leafless_multi_proof: Vec<LeaflessPartialAuthenticationPath> =
            self.get_leafless_multi_proof(indices);
        let mut ret: Vec<(LeaflessPartialAuthenticationPath, T)> = vec![];

        // Insert values in a for-loop
        for (i, leafless_proof) in leafless_multi_proof.into_iter().enumerate() {
            ret.push((leafless_proof, self.get_value_by_index(indices[i])));
        }

        ret
    }

    // Compact Merkle Multiproof Generation
    //
    // Leafless (produce authentication paths, not Vec<Node<T>>s with leaf values in it).
    fn get_leafless_multi_proof(
        &self,
        indices: &[usize],
    ) -> Vec<LeaflessPartialAuthenticationPath> {
        self.get_multi_proof(indices)
            .into_iter()
            .map(Self::convert_pat_leafless)
            .collect()
    }

    fn convert_pat_leafless(
        auth_path_nodes: PartialAuthenticationPath<T>,
    ) -> LeaflessPartialAuthenticationPath {
        let auth_path_hashes: Vec<Option<Blake3Digest>> = auth_path_nodes
            .0
            .iter()
            .skip(1) // drop leaf node from `get_multi_proof' output
            .map(|node_opt| node_opt.clone().map(|node| node.hash))
            .collect();

        LeaflessPartialAuthenticationPath(auth_path_hashes)
    }

    fn verify_leafless_multi_proof_from_leaf_hashes(
        root_hash: Blake3Digest,
        indices: &[usize],
        leaf_hashes: &[Blake3Digest],
        proof: &[LeaflessPartialAuthenticationPath],
    ) -> bool {
        if indices.len() != proof.len() || indices.len() != leaf_hashes.len() {
            return false;
        }

        if indices.is_empty() {
            return true;
        }

        let mut partial_auth_paths: Vec<LeaflessPartialAuthenticationPath> = proof.to_owned();
        let mut partial_tree: HashMap<u64, Blake3Digest> = HashMap::new();

        // FIXME: We find the offset from which leaf nodes occur in the tree by looking at the
        // first partial authentication path. This is a bit hacked, since what if not all
        // partial authentication paths have the same length, and what if one has a
        // different length than the tree's height?
        let auth_path_length = partial_auth_paths[0].0.len();
        let half_tree_size = 2u64.pow(auth_path_length as u32);

        // Bootstrap partial tree
        for (i, leaf_hash, partial_auth_path) in
            izip!(indices, leaf_hashes, partial_auth_paths.clone())
        {
            let mut index = half_tree_size + *i as u64;

            // Insert hashes for known leaf hashes.
            partial_tree.insert(index, *leaf_hash);

            // Insert hashes for known leaves from partial authentication paths.
            for hash_option in partial_auth_path.0.iter() {
                if let Some(hash) = *hash_option {
                    partial_tree.insert(index ^ 1, hash);
                }
                index /= 2;
            }
        }

        let mut complete = false;
        let mut hasher = blake3::Hasher::new();
        while !complete {
            complete = true;

            let mut parent_keys: Vec<u64> = partial_tree.keys().copied().map(|x| x / 2).collect();
            parent_keys.sort_by_key(|w| Reverse(*w));

            for parent_key in parent_keys {
                let left_child_key = parent_key * 2;
                let right_child_key = parent_key * 2 + 1;

                // Populate partial tree with parent key/hashes for known children
                if partial_tree.contains_key(&left_child_key)
                    && partial_tree.contains_key(&right_child_key)
                    && !partial_tree.contains_key(&parent_key)
                {
                    hasher.update(&partial_tree[&(parent_key * 2)]);
                    hasher.update(&partial_tree[&(parent_key * 2 + 1)]);

                    let hash: Blake3Digest = *hasher.finalize().as_bytes();
                    hasher.reset();

                    partial_tree.insert(parent_key, hash);
                    complete = false;
                }
            }
        }

        // Use partial tree to insert missing elements in partial authentication paths,
        // making them full authentication paths.
        for (i, partial_auth_path) in indices.iter().zip(partial_auth_paths.iter_mut()) {
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

                    *elem = Some(partial_tree[&sibling]);
                }
                partial_tree.insert(sibling, elem.unwrap());
                index /= 2;
            }
        }

        // Remove 'Some' constructors from partial auth paths
        let auth_paths = partial_auth_paths
            .iter()
            .map(Self::unwrap_leafless_partial_authentication_path);

        izip!(indices, leaf_hashes, auth_paths).all(|(index, leaf_hash, auth_path)| {
            Self::verify_authentication_path_from_leaf_hash(
                root_hash,
                *index as u32,
                *leaf_hash,
                auth_path,
            )
        })
    }

    pub fn verify_leafless_multi_proof(
        root_hash: Blake3Digest,
        indices: &[usize],
        proof: &[(LeaflessPartialAuthenticationPath, T)],
    ) -> bool {
        if indices.len() != proof.len() {
            return false;
        }

        if indices.is_empty() {
            return true;
        }

        let mut auth_paths: Vec<LeaflessPartialAuthenticationPath> =
            Vec::with_capacity(proof.len());
        let mut leaf_hashes = Vec::with_capacity(proof.len());
        for (auth_path, value) in proof.iter() {
            auth_paths.push(auth_path.clone());
            leaf_hashes.push(blake3_digest_serialize(value));
        }

        Self::verify_leafless_multi_proof_from_leaf_hashes(
            root_hash,
            indices,
            &leaf_hashes,
            &auth_paths,
        )
    }

    fn unwrap_leafless_partial_authentication_path(
        partial_auth_path: &LeaflessPartialAuthenticationPath,
    ) -> Vec<Blake3Digest> {
        partial_auth_path
            .clone()
            .0
            .into_iter()
            .map(|hash| hash.unwrap())
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct SaltedMerkleTree<T> {
    internal_merkle_tree: MerkleTree<T>,
    salts: Vec<T>,
    salts_per_value: usize,
}

impl<T: Clone + Serialize + Debug + PartialEq + GetRandomElements> SaltedMerkleTree<T> {
    // Build a salted Merkle tree from a slice of serializable values
    pub fn from_vec(values: &[T], salts_per_element: usize, rng: &mut ThreadRng) -> Self {
        assert!(
            other::is_power_of_two(values.len()),
            "Size of input for Merkle tree must be a power of 2"
        );

        let mut nodes: Vec<Node<T>> = vec![
            Node {
                value: None,
                hash: BLAKE3ZERO,
            };
            2 * values.len()
        ];

        let salts: Vec<T> = T::random_elements(salts_per_element * values.len(), rng);
        for i in 0..values.len() {
            let mut leaf_hash_preimage: Vec<u8> =
                bincode::serialize(&values[i]).expect("Encoding failed");
            for j in 0..salts_per_element {
                leaf_hash_preimage.append(
                    &mut bincode::serialize(&salts[salts_per_element * i + j])
                        .expect("Encoding failed"),
                );
            }
            nodes[values.len() + i].hash = blake3_digest_serialize(&leaf_hash_preimage);
            nodes[values.len() + i].value = Some(values[i].clone());
        }

        // loop from `len(L) - 1` to 1
        let mut hasher = blake3::Hasher::new();
        for i in (1..(nodes.len() / 2)).rev() {
            hasher.update(&nodes[i * 2].hash[..]);
            hasher.update(&nodes[i * 2 + 1].hash[..]);
            nodes[i].hash = *hasher.finalize().as_bytes();
            hasher.reset();
        }

        // nodes[0] is never used for anything.
        let internal_merkle_tree = MerkleTree {
            root_hash: nodes[1].hash,
            nodes,
            height: log_2_floor(values.len() as u64) as u8 + 1,
        };

        Self {
            internal_merkle_tree,
            salts_per_value: salts_per_element,
            salts,
        }
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.internal_merkle_tree.to_vec()
    }

    pub fn get_authentication_path(&self, index: usize) -> (Vec<Blake3Digest>, Vec<T>) {
        let authentication_path = self.internal_merkle_tree.get_authentication_path(index);
        let mut salts = Vec::<T>::with_capacity(self.salts_per_value);
        for i in 0..self.salts_per_value {
            salts.push(self.salts[index * self.salts_per_value + i].clone());
        }

        (authentication_path, salts)
    }

    pub fn verify_authentication_path(
        root_hash: Blake3Digest,
        index: u32,
        value: T,
        auth_path: Vec<Blake3Digest>,
        salts: Vec<T>,
    ) -> bool {
        let mut leaf_hash_preimage: Vec<u8> = bincode::serialize(&value).expect("Encoding failed");
        for salt in salts {
            leaf_hash_preimage.append(&mut bincode::serialize(&salt).expect("Encoding failed"));
        }

        let leaf_hash = blake3_digest_serialize(&leaf_hash_preimage);

        // Set `leaf_hash` to H(value + salts[0..])
        MerkleTree::<T>::verify_authentication_path_from_leaf_hash(
            root_hash, index, leaf_hash, auth_path,
        )
    }

    pub fn get_root(&self) -> [u8; 32] {
        self.internal_merkle_tree.root_hash
    }

    // Returns vectors of length `indices.len()` of triplets (auth path, salts, value)
    pub fn get_leafless_multi_proof_with_salts_and_values(
        &self,
        indices: &[usize],
    ) -> Vec<(LeaflessPartialAuthenticationPath, Vec<T>, T)> {
        // auth path and salts
        let leafless_multi_proof: Vec<(LeaflessPartialAuthenticationPath, Vec<T>)> =
            self.get_leafless_multi_proof_with_salts(indices);
        let mut ret: Vec<(LeaflessPartialAuthenticationPath, Vec<T>, T)> = vec![];

        // Insert values in a for-loop
        for (i, leafless_proof) in leafless_multi_proof.into_iter().enumerate() {
            ret.push((
                leafless_proof.0,
                leafless_proof.1,
                self.internal_merkle_tree.get_value_by_index(indices[i]),
            ));
        }

        ret
    }

    pub fn get_leafless_multi_proof_with_salts(
        &self,
        indices: &[usize],
    ) -> Vec<(LeaflessPartialAuthenticationPath, Vec<T>)> {
        // Get the partial authentication paths without salts
        let leafless_partial_authentication_paths: Vec<LeaflessPartialAuthenticationPath> =
            self.internal_merkle_tree.get_leafless_multi_proof(indices);

        // Get the salts associated with the leafs
        // salts are random data, so they cannot be compressed
        let mut ret: Vec<(LeaflessPartialAuthenticationPath, Vec<T>)> =
            Vec::with_capacity(indices.len());
        for (i, index) in indices.iter().enumerate() {
            let mut salts = vec![];
            for j in 0..self.salts_per_value {
                salts.push(self.salts[index * self.salts_per_value + j].clone());
            }
            ret.push((leafless_partial_authentication_paths[i].clone(), salts));
        }

        ret
    }

    pub fn verify_leafless_multi_proof_with_salts_and_values(
        root_hash: Blake3Digest,
        indices: &[usize],
        proof: &[(LeaflessPartialAuthenticationPath, Vec<T>, T)],
    ) -> bool {
        let values: Vec<T> = proof.iter().map(|x| x.2.clone()).collect();
        let auth_paths_and_salts: Vec<(LeaflessPartialAuthenticationPath, Vec<T>)> =
            proof.iter().map(|x| (x.0.clone(), x.1.clone())).collect();

        Self::verify_leafless_multi_proof(root_hash, indices, &values, &auth_paths_and_salts)
    }

    pub fn verify_leafless_multi_proof(
        root_hash: Blake3Digest,
        indices: &[usize],
        values: &[T],
        proof: &[(LeaflessPartialAuthenticationPath, Vec<T>)],
    ) -> bool {
        if indices.len() != proof.len() || indices.len() != values.len() {
            debug_assert!(indices.len() == proof.len());
            debug_assert!(indices.len() == values.len());

            return false;
        }

        if indices.is_empty() {
            return true;
        }

        let mut leaf_hashes: Vec<Blake3Digest> = Vec::with_capacity(indices.len());
        for (value, proof) in izip!(values, proof) {
            let mut leaf_hash_preimage: Vec<u8> =
                bincode::serialize(&value).expect("Encoding failed");
            for salt in proof.1.clone() {
                leaf_hash_preimage.append(&mut bincode::serialize(&salt).expect("Encoding failed"));
            }

            leaf_hashes.push(blake3_digest_serialize(&leaf_hash_preimage));
        }

        let saltless_proof: Vec<LeaflessPartialAuthenticationPath> =
            proof.iter().map(|x| x.0.clone()).collect();
        MerkleTree::<T>::verify_leafless_multi_proof_from_leaf_hashes(
            root_hash,
            indices,
            &leaf_hashes,
            &saltless_proof,
        )
    }
}

#[cfg(test)]
mod merkle_tree_test {
    use super::*;
    use crate::shared_math::b_field_element::BFieldElement;
    use crate::shared_math::prime_field_element_flexible::PrimeFieldElementFlexible;
    use crate::shared_math::x_field_element::XFieldElement;
    use crate::utils::{decode_hex, generate_random_numbers, generate_random_numbers_u128};
    use itertools::Itertools;
    use primitive_types::U256;
    use rand::RngCore;

    fn count_hashes(proof: &[(LeaflessPartialAuthenticationPath, Vec<BFieldElement>)]) -> usize {
        proof.iter().map(|y| y.0 .0.iter().flatten().count()).sum()
    }

    #[test]
    fn merkle_tree_test_32() {
        let prime: U256 = 1009.into();
        let elements: Vec<PrimeFieldElementFlexible> = generate_random_numbers(32, 1000)
            .into_iter()
            .map(|x| PrimeFieldElementFlexible::new(x.into(), prime))
            .collect();
        let mut mt_32 = MerkleTree::from_vec(&elements);

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

                let proof: Vec<(LeaflessPartialAuthenticationPath, PrimeFieldElementFlexible)> =
                    mt_32.get_leafless_multi_proof_with_values(&indices_usize);
                assert!(MerkleTree::verify_leafless_multi_proof(
                    mt_32.get_root(),
                    &indices_usize,
                    &proof
                ));

                // Verify that `get_value` returns the value for this proof
                assert!(proof
                    .iter()
                    .enumerate()
                    .all(|(i, (_auth_path, value))| *value == elements[indices_usize[i]]));

                // manipulate Merkle root and verify failure
                mt_32.root_hash[i] ^= 1;
                assert!(!MerkleTree::verify_leafless_multi_proof(
                    mt_32.get_root(),
                    &indices_usize,
                    &proof
                ));

                // Restore root and verify success
                mt_32.root_hash[i] ^= 1;
                assert!(MerkleTree::verify_leafless_multi_proof(
                    mt_32.get_root(),
                    &indices_usize,
                    &proof
                ));

                // Request an additional index and verify failure
                // (indices length does not match proof length)
                indices_usize.insert(0, indices_i128[0] as usize);
                assert!(!MerkleTree::verify_leafless_multi_proof(
                    mt_32.get_root(),
                    &indices_usize,
                    &proof
                ));

                // Request a non-existant index and verify failure
                // (indices length does match proof length)
                indices_usize.remove(0);
                indices_usize[0] = indices_i128[0] as usize;
                assert!(!MerkleTree::verify_leafless_multi_proof(
                    mt_32.get_root(),
                    &indices_usize,
                    &proof
                ));

                // Remove an element from the indices vector
                // and verify failure since the indices and proof
                // vectors do not match
                indices_usize.remove(0);
                assert!(!MerkleTree::verify_leafless_multi_proof(
                    mt_32.get_root(),
                    &indices_usize,
                    &proof
                ));
            }
        }
    }

    #[test]
    fn merkle_tree_verify_multi_proof_degenerate_test() {
        // Number of Merkle tree leaves
        let n_values = 8;
        let elements: Vec<u128> = generate_random_numbers_u128(n_values, Some(100));
        let tree = MerkleTree::from_vec(&elements);

        // Degenerate example
        let empty_leafless_proof = tree.get_leafless_multi_proof_with_values(&[]);
        assert!(MerkleTree::verify_leafless_multi_proof(
            tree.root_hash,
            &[],
            &empty_leafless_proof,
        ));
    }

    #[test]
    fn merkle_tree_verify_multi_proof_equivalence_test() {
        let mut rng = rand::thread_rng();

        // This test asserts that regular merkle trees and salted merkle trees with 0 salts work equivalently.

        // Number of Merkle tree leaves
        let n_values = 8;
        let expected_path_length = 3; // log2(128), root node not included
        let elements: Vec<BFieldElement> = BFieldElement::random_elements(n_values, &mut rng);
        let regular_tree = MerkleTree::from_vec(&elements);

        let some_indices: Vec<usize> = vec![0, 1];
        let some_leafless_proof: Vec<(LeaflessPartialAuthenticationPath, BFieldElement)> =
            regular_tree.get_leafless_multi_proof_with_values(&some_indices);

        for (partial_auth_path, _value) in some_leafless_proof.clone() {
            assert_eq!(
                expected_path_length,
                partial_auth_path.0.len(),
                "A generated authentication path must have length the height of the tree (not counting the root)"
            );
        }

        let regular_verify = MerkleTree::verify_leafless_multi_proof(
            regular_tree.get_root(),
            &some_indices,
            &some_leafless_proof,
        );

        let how_many_salts_again = 0;
        let unsalted_salted_tree: SaltedMerkleTree<BFieldElement> =
            SaltedMerkleTree::from_vec(&elements, how_many_salts_again, &mut rng);

        let salted_proof: Vec<(
            LeaflessPartialAuthenticationPath,
            Vec<BFieldElement>,
            BFieldElement,
        )> = unsalted_salted_tree.get_leafless_multi_proof_with_salts_and_values(&some_indices);

        let salted_leafless_proof: Vec<(LeaflessPartialAuthenticationPath, Vec<BFieldElement>)> =
            salted_proof
                .iter()
                .map(|(auth_path, salts, _)| (auth_path.clone(), salts.clone()))
                .collect();
        let values: Vec<BFieldElement> = salted_proof.iter().map(|(_, _, value)| *value).collect();

        let unsalted_salted_verify = SaltedMerkleTree::verify_leafless_multi_proof(
            unsalted_salted_tree.get_root(),
            &some_indices,
            &values,
            &salted_leafless_proof,
        );

        assert_eq!(regular_verify, unsalted_salted_verify);
    }

    #[test]
    fn merkle_tree_verify_leafless_multi_proof_test() {
        let mut prng = rand::thread_rng();

        // Number of Merkle tree leaves
        let n_valuess = &[2, 4, 8, 16, 128, 256, 512, 1024, 2048, 4096, 8192];
        let expected_path_lengths = &[1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13]; // log2(128), root node not included
        for (n_values, expected_path_length) in izip!(n_valuess, expected_path_lengths) {
            let elements: Vec<BFieldElement> =
                generate_random_numbers_u128(*n_values, Some(1u128 << 63))
                    .iter()
                    .map(|x| BFieldElement::new(*x))
                    .collect();
            let tree: MerkleTree<BFieldElement> = MerkleTree::from_vec(&elements);

            for _ in 0..3 {
                // Ask for an arbitrary amount of indices less than the total
                let n_indices = (prng.next_u64() % *n_values as u64 / 2) as usize + 1;

                // Generate that amount of indices in the valid index range [0,128)
                let indices: Vec<usize> =
                    generate_random_numbers_u128(n_indices, Some(*n_values as u128))
                        .iter()
                        .map(|x| *x as usize)
                        .unique()
                        .collect();

                let proof = tree.get_leafless_multi_proof_with_values(&indices);

                for (auth_path, _value) in proof.iter() {
                    assert_eq!(*expected_path_length, auth_path.0.len());
                }

                assert!(MerkleTree::verify_leafless_multi_proof(
                    tree.root_hash,
                    &indices,
                    &proof,
                ));
                let mut bad_root_hash = tree.root_hash;
                bad_root_hash[0] ^= 0x1;
                assert!(!MerkleTree::verify_leafless_multi_proof(
                    bad_root_hash,
                    &indices,
                    &proof,
                ));

                let mut bad_proof = proof.clone();
                let random_index = (prng.next_u64() % n_indices as u64 / 2) as usize;
                bad_proof[random_index].1.decrement();
                assert!(!MerkleTree::verify_leafless_multi_proof(
                    tree.root_hash,
                    &indices,
                    &bad_proof,
                ));
            }
        }
    }

    #[test]
    fn merkle_tree_test_simple() {
        let single_mt_one: MerkleTree<i128> = MerkleTree::from_vec(&[1i128]);
        assert_eq!(
            decode_hex("74500697761748e7dc0302d36778f89c6ab324ef942773976b92a7bbefa18cd2")
                .expect("Decoding failed"),
            single_mt_one.root_hash
        );
        assert_eq!(1u8, single_mt_one.height);
        let single_mt_two: MerkleTree<i128> = MerkleTree::from_vec(&[2i128]);
        assert_eq!(
            decode_hex("65706bf07e4e656de8a6b898dfbc64c076e001253f384043a40c437e1d5fb124")
                .expect("Decoding failed"),
            single_mt_two.root_hash
        );
        assert_eq!(1u8, single_mt_two.height);

        let mt: MerkleTree<i128> = MerkleTree::from_vec(&[1i128, 2]);
        assert_eq!(
            decode_hex("c19af4447b81b6ea9b76328441b963e6076d2e787b3fad956aa35c66f8ede2c4")
                .expect("Decoding failed"),
            mt.root_hash
        );
        assert_eq!(2u8, mt.height);
        let mut proof: Vec<Node<i128>> = mt.get_proof(1);
        assert!(MerkleTree::verify_proof(mt.root_hash, 1, proof.clone()));
        assert_eq!(Some(2), proof[0].value);
        proof = mt.get_proof(0);
        assert!(MerkleTree::verify_proof(mt.root_hash, 0, proof.clone()));
        assert_eq!(Some(1), proof[0].value);
        assert_eq!(2usize, proof.len());

        let mt_reverse: MerkleTree<i128> = MerkleTree::from_vec(&[2i128, 1]);
        assert_eq!(
            decode_hex("189d788c8539945c368d54e9f61847b05a847f350b925ea499eadb0007130d93")
                .expect("Decoding failed"),
            mt_reverse.root_hash
        );
        assert_eq!(2u8, mt_reverse.height);

        let mut mt_four: MerkleTree<i128> = MerkleTree::from_vec(&[1i128, 2, 3, 4]);
        assert_eq!(
            decode_hex("44bdb434be4895b977ef91f419f16df22a9c65eeefa3843aae55f81e0e102777").unwrap(),
            mt_four.root_hash
        );
        assert_ne!(mt.root_hash, mt_reverse.root_hash);
        assert_eq!(3u8, mt_four.height);
        proof = mt_four.get_proof(1);
        assert_eq!(3usize, proof.len());
        assert!(MerkleTree::verify_proof(
            mt_four.root_hash,
            1,
            proof.clone()
        ));
        assert_eq!(Some(2), proof[0].value);
        proof[0].value = Some(3);
        assert!(!MerkleTree::verify_proof(
            mt_four.root_hash,
            1,
            proof.clone()
        ));
        proof[0].value = Some(2);
        proof[0].hash = [0u8; 32];
        assert!(!MerkleTree::verify_proof(
            mt_four.root_hash,
            1,
            proof.clone()
        ));

        proof = mt_four.get_proof(1);
        assert!(MerkleTree::verify_proof(
            mt_four.root_hash,
            1,
            proof.clone()
        ));
        let original_root = mt_four.get_root();
        mt_four.root_hash = [0u8; 32];
        assert!(!MerkleTree::verify_proof(
            mt_four.root_hash,
            1,
            proof.clone()
        ));
        println!("get_proof(mt_four) = {:x?}", proof);
        mt_four.root_hash = original_root;

        println!("root_hash = {:?}", mt_four.root_hash);
        proof = mt_four.get_proof(0);
        println!("root_hash = {:?}", mt_four.root_hash);
        println!("\n\n\n\n proof(0) = {:?} \n\n\n\n", proof);
        assert!(MerkleTree::verify_proof(mt_four.root_hash, 0, proof));
        let mut compressed_proof: Vec<(LeaflessPartialAuthenticationPath, i128)> =
            mt_four.get_leafless_multi_proof_with_values(&[0]);
        assert_eq!(1i128, compressed_proof[0].1);
        assert!(MerkleTree::verify_leafless_multi_proof(
            mt_four.root_hash,
            &[0],
            &compressed_proof
        ));
        proof = mt_four.get_proof(0);

        assert_eq!(
            proof.len() - 1,
            compressed_proof[0].0 .0.len(),
            "get_proof() produces one more Node<T> (with the value in) than is contained in a leafless proof"
        );

        compressed_proof = mt_four.get_leafless_multi_proof_with_values(&[0, 1]);
        assert_eq!(1i128, compressed_proof[0].1);
        assert_eq!(2i128, compressed_proof[1].1);
        println!("{:?}", compressed_proof);
        assert!(MerkleTree::verify_leafless_multi_proof(
            mt_four.root_hash,
            &[0, 1],
            &compressed_proof
        ));

        compressed_proof = mt_four.get_leafless_multi_proof_with_values(&[0, 1, 2]);
        assert_eq!(1i128, compressed_proof[0].1);
        assert_eq!(2i128, compressed_proof[1].1);
        assert_eq!(3i128, compressed_proof[2].1);
        println!("{:?}", compressed_proof);
        assert!(MerkleTree::verify_leafless_multi_proof(
            mt_four.root_hash,
            &[0, 1, 2],
            &compressed_proof
        ));

        // Verify that verification of multi-proof where the tree or the proof
        // does not have the indices requested leads to a false return value,
        // and not to a run-time panic.
        assert!(!MerkleTree::verify_leafless_multi_proof(
            mt_four.root_hash,
            &[2, 3],
            &compressed_proof
        ));
    }

    #[test]
    fn merkle_tree_get_authentication_path_test() {
        // 1: Create Merkle tree
        //
        //     root
        //    /    \
        //   x      y
        //  / \    / \
        // 3   6  9   12
        let tree_a = MerkleTree::from_vec(&[3, 6, 9, 12]);

        // 2: Get the path for value '9' (index: 2)
        let auth_path_a = tree_a.get_authentication_path(2);

        assert_eq!(
            2,
            auth_path_a.len(),
            "authentication path a has right length"
        );
        assert_eq!(tree_a.nodes[2].hash, auth_path_a[1], "sibling x");
        assert_eq!(tree_a.nodes[7].hash, auth_path_a[0], "sibling 12");

        //        ___root___
        //       /          \
        //      e            f
        //    /   \        /   \
        //   a     b      c     d
        //  / \   / \    / \   / \
        // 3   1 4   1  5   9 8   6
        let tree_b = MerkleTree::from_vec(&[3, 1, 4, 1, 5, 9, 8, 6]);

        // merkle leaf index: 5
        // merkle leaf value: 9
        // auth path: 5 ~> d ~> e
        let auth_path_b = tree_b.get_authentication_path(5);

        assert_eq!(3, auth_path_b.len());
        assert_eq!(tree_b.nodes[12].hash, auth_path_b[0], "sibling 5");
        assert_eq!(tree_b.nodes[7].hash, auth_path_b[1], "sibling d");
        assert_eq!(tree_b.nodes[2].hash, auth_path_b[2], "sibling e");
    }

    // Test of salted Merkle trees
    #[test]
    fn salted_merkle_tree_get_authentication_path_test() {
        // 1: Create Merkle tree
        //
        //     root
        //    /    \
        //   x      y
        //  / \    / \
        // 3   6  9   12
        let mut rng = rand::thread_rng();
        let tree_a = SaltedMerkleTree::from_vec(
            &[
                BFieldElement::new(3),
                BFieldElement::new(1),
                BFieldElement::new(4),
                BFieldElement::new(5),
            ],
            3,
            &mut rng,
        );
        assert_eq!(3, tree_a.salts_per_value);
        assert_eq!(3 * 4, tree_a.salts.len());

        // 2: Get the path for value '4' (index: 2)
        let auth_path_a_and_salt = tree_a.get_authentication_path(2);

        // 3: Verify that the proof, along with the salt, works
        let root_hash_a = tree_a.get_root();
        let mut value = BFieldElement::new(4);
        assert!(SaltedMerkleTree::verify_authentication_path(
            root_hash_a,
            2,
            value,
            auth_path_a_and_salt.0.clone(),
            auth_path_a_and_salt.1.clone(),
        ));

        assert_eq!(
            2,
            auth_path_a_and_salt.0.len(),
            "authentication path a has right length"
        );
        assert_eq!(
            3,
            auth_path_a_and_salt.1.len(),
            "Proof contains expected number of salts"
        );
        assert_eq!(
            tree_a.internal_merkle_tree.nodes[2].hash, auth_path_a_and_salt.0[1],
            "sibling x"
        );
        assert_eq!(
            tree_a.internal_merkle_tree.nodes[7].hash, auth_path_a_and_salt.0[0],
            "sibling 12"
        );

        // 4: Change salt and verify that the proof does not work
        let mut auth_path_a_and_salt_bad_salt = tree_a.get_authentication_path(2);
        auth_path_a_and_salt_bad_salt.1[0].increment();
        assert!(!SaltedMerkleTree::verify_authentication_path(
            root_hash_a,
            2,
            value,
            auth_path_a_and_salt_bad_salt.0.clone(),
            auth_path_a_and_salt_bad_salt.1.clone(),
        ));
        auth_path_a_and_salt_bad_salt.1[0].decrement();
        assert!(SaltedMerkleTree::verify_authentication_path(
            root_hash_a,
            2,
            value,
            auth_path_a_and_salt_bad_salt.0.clone(),
            auth_path_a_and_salt_bad_salt.1.clone(),
        ));
        auth_path_a_and_salt_bad_salt.1[1].increment();
        assert!(!SaltedMerkleTree::verify_authentication_path(
            root_hash_a,
            2,
            value,
            auth_path_a_and_salt_bad_salt.0.clone(),
            auth_path_a_and_salt_bad_salt.1.clone(),
        ));
        auth_path_a_and_salt_bad_salt.1[1].decrement();
        assert!(SaltedMerkleTree::verify_authentication_path(
            root_hash_a,
            2,
            value,
            auth_path_a_and_salt_bad_salt.0.clone(),
            auth_path_a_and_salt_bad_salt.1.clone(),
        ));
        auth_path_a_and_salt_bad_salt.1[2].decrement();
        assert!(!SaltedMerkleTree::verify_authentication_path(
            root_hash_a,
            2,
            value,
            auth_path_a_and_salt_bad_salt.0.clone(),
            auth_path_a_and_salt_bad_salt.1.clone(),
        ));
        auth_path_a_and_salt_bad_salt.1[2].increment();
        assert!(SaltedMerkleTree::verify_authentication_path(
            root_hash_a,
            2,
            value,
            auth_path_a_and_salt_bad_salt.0.clone(),
            auth_path_a_and_salt_bad_salt.1.clone(),
        ));

        // 5: Change the value and verify that the proof does not work
        value.increment();
        assert!(!SaltedMerkleTree::verify_authentication_path(
            root_hash_a,
            2,
            value,
            auth_path_a_and_salt_bad_salt.0.clone(),
            auth_path_a_and_salt_bad_salt.1.clone(),
        ));
        value.decrement();
        assert!(SaltedMerkleTree::verify_authentication_path(
            root_hash_a,
            2,
            value,
            auth_path_a_and_salt_bad_salt.0.clone(),
            auth_path_a_and_salt_bad_salt.1.clone(),
        ));
        value.decrement();
        assert!(!SaltedMerkleTree::verify_authentication_path(
            root_hash_a,
            2,
            value,
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
        let tree_b: SaltedMerkleTree<BFieldElement> = SaltedMerkleTree::from_vec(
            &[3, 1, 4, 1, 5, 9, 8, 6].map(BFieldElement::new),
            4,
            &mut rng,
        );

        // auth path: 8 ~> c ~> e
        let mut auth_path_b = tree_b.get_authentication_path(6);
        let mut value_b = BFieldElement::new(8);

        assert_eq!(3, auth_path_b.0.len());
        assert_eq!(8 * 4, tree_b.salts.len());

        // 6: Ensure that all salts are unique (statistically, they should be)
        assert_eq!(
            tree_b.salts.len(),
            tree_b.salts.clone().into_iter().unique().count()
        );
        assert_eq!(
            tree_b.internal_merkle_tree.nodes[15].hash, auth_path_b.0[0],
            "sibling 6"
        );
        assert_eq!(
            tree_b.internal_merkle_tree.nodes[6].hash, auth_path_b.0[1],
            "sibling c"
        );
        assert_eq!(
            tree_b.internal_merkle_tree.nodes[2].hash, auth_path_b.0[2],
            "sibling e"
        );

        let mut root_hash_b = tree_b.get_root();
        assert!(SaltedMerkleTree::verify_authentication_path(
            root_hash_b,
            6,
            value_b,
            auth_path_b.0.clone(),
            auth_path_b.1.clone(),
        ));

        // 7: Change the value and verify that it fails
        value_b.decrement();
        assert!(!SaltedMerkleTree::verify_authentication_path(
            root_hash_b,
            6,
            value_b,
            auth_path_b.0.clone(),
            auth_path_b.1.clone(),
        ));
        value_b.increment();

        // 8: Change salt and verify that verification fails
        auth_path_b.1[3].decrement();
        assert!(!SaltedMerkleTree::verify_authentication_path(
            root_hash_b,
            6,
            value_b,
            auth_path_b.0.clone(),
            auth_path_b.1.clone(),
        ));
        auth_path_b.1[3].increment();
        assert!(SaltedMerkleTree::verify_authentication_path(
            root_hash_b,
            6,
            value_b,
            auth_path_b.0.clone(),
            auth_path_b.1.clone(),
        ));

        // 9: Verify that simple multipath authentication paths work
        let auth_path_b_multi_0: Vec<(LeaflessPartialAuthenticationPath, Vec<BFieldElement>)> =
            tree_b.get_leafless_multi_proof_with_salts(&[0, 1]);
        let multi_values_0 = vec![BFieldElement::new(3), BFieldElement::new(1)];
        assert!(SaltedMerkleTree::verify_leafless_multi_proof(
            root_hash_b,
            &[0, 1],
            &multi_values_0,
            &auth_path_b_multi_0
        ));
        assert_eq!(
            2,
            count_hashes(&auth_path_b_multi_0),
            "paths [0,1] need two hashes"
        );

        // Verify that we get values
        let auth_paths_salts_values =
            tree_b.get_leafless_multi_proof_with_salts_and_values(&[0, 1]);
        assert_eq!(2, auth_paths_salts_values.len());
        assert_eq!(auth_path_b_multi_0[0].0, auth_paths_salts_values[0].0);
        assert_eq!(auth_path_b_multi_0[0].1, auth_paths_salts_values[0].1);
        assert_eq!(auth_path_b_multi_0[1].0, auth_paths_salts_values[1].0);
        assert_eq!(auth_path_b_multi_0[1].1, auth_paths_salts_values[1].1);
        assert_eq!(BFieldElement::new(3), auth_paths_salts_values[0].2);
        assert_eq!(BFieldElement::new(1), auth_paths_salts_values[1].2);

        // Verify that the composite verification works
        assert!(
            SaltedMerkleTree::verify_leafless_multi_proof_with_salts_and_values(
                root_hash_b,
                &[0, 1],
                &auth_paths_salts_values
            )
        );
        let mut bad_root_hash_b = root_hash_b;
        bad_root_hash_b[5] ^= 0x1;
        assert!(
            !SaltedMerkleTree::verify_leafless_multi_proof_with_salts_and_values(
                bad_root_hash_b,
                &[0, 1],
                &auth_paths_salts_values
            )
        );

        let auth_path_b_multi_1 = tree_b.get_leafless_multi_proof_with_salts(&[1]);
        let multi_values_1 = vec![BFieldElement::new(1)];
        assert!(SaltedMerkleTree::verify_leafless_multi_proof(
            root_hash_b,
            &[1],
            &multi_values_1,
            &auth_path_b_multi_1
        ));
        assert_eq!(
            3,
            count_hashes(&auth_path_b_multi_1),
            "paths [1] need two hashes"
        );

        let auth_path_b_multi_2 = tree_b.get_leafless_multi_proof_with_salts(&[1, 0]);
        let multi_values_2 = vec![BFieldElement::new(1), BFieldElement::new(3)];
        assert!(SaltedMerkleTree::verify_leafless_multi_proof(
            root_hash_b,
            &[1, 0],
            &multi_values_2,
            &auth_path_b_multi_2
        ));

        let mut auth_path_b_multi_3 = tree_b.get_leafless_multi_proof_with_salts(&[0, 1, 2, 4, 7]);
        let mut multi_values_3 = vec![
            BFieldElement::new(3),
            BFieldElement::new(1),
            BFieldElement::new(4),
            BFieldElement::new(5),
            BFieldElement::new(6),
        ];
        assert!(SaltedMerkleTree::verify_leafless_multi_proof(
            root_hash_b,
            &[0, 1, 2, 4, 7],
            &multi_values_3,
            &auth_path_b_multi_3
        ));

        let temp = tree_b.get_leafless_multi_proof_with_salts(&[0, 1, 2, 4, 7]);
        assert_eq!(
            3,
            count_hashes(&temp),
            "paths [0, 1, 2, 4, 7] need three hashes"
        );

        // 10: change a hash, verify failure
        auth_path_b_multi_3[1].1[0].increment();
        assert!(!SaltedMerkleTree::verify_leafless_multi_proof(
            root_hash_b,
            &[0, 1, 2, 4, 7],
            &multi_values_3,
            &auth_path_b_multi_3
        ));

        auth_path_b_multi_3[1].1[0].decrement();
        assert!(SaltedMerkleTree::verify_leafless_multi_proof(
            root_hash_b,
            &[0, 1, 2, 4, 7],
            &multi_values_3,
            &auth_path_b_multi_3
        ));

        // 11: change a value, verify failure
        multi_values_3[0].increment();
        assert!(!SaltedMerkleTree::verify_leafless_multi_proof(
            root_hash_b,
            &[1, 0, 4, 7, 2],
            &multi_values_3,
            &auth_path_b_multi_3
        ));

        multi_values_3[0].decrement();
        assert!(SaltedMerkleTree::verify_leafless_multi_proof(
            root_hash_b,
            &[0, 1, 2, 4, 7],
            &multi_values_3,
            &auth_path_b_multi_3
        ));

        // Change root hash, verify failue
        root_hash_b[4] ^= 1;
        assert!(!SaltedMerkleTree::verify_leafless_multi_proof(
            root_hash_b,
            &[0, 1, 2, 4, 7],
            &multi_values_3,
            &auth_path_b_multi_3
        ));

        root_hash_b[4] ^= 1;
        assert!(SaltedMerkleTree::verify_leafless_multi_proof(
            root_hash_b,
            &[0, 1, 2, 4, 7],
            &multi_values_3,
            &auth_path_b_multi_3
        ));
    }

    #[test]
    fn salted_merkle_tree_get_authentication_path_xfields_test() {
        // 1: Create Merkle tree
        //
        //     root
        //    /    \
        //   x      y
        //  / \    / \
        // 3   6  9   12
        let mut rng = rand::thread_rng();
        let tree_a = SaltedMerkleTree::from_vec(
            &[
                XFieldElement::new([3, 3, 3].map(BFieldElement::new)),
                XFieldElement::new([1, 1, 1].map(BFieldElement::new)),
                XFieldElement::new([4, 4, 4].map(BFieldElement::new)),
                XFieldElement::new([5, 5, 5].map(BFieldElement::new)),
            ],
            3,
            &mut rng,
        );
        assert_eq!(3, tree_a.salts_per_value);
        assert_eq!(3 * 4, tree_a.salts.len());

        // 2: Get the path for value '4' (index: 2)
        let mut auth_path_a_and_salt = tree_a.get_authentication_path(2);

        // 3: Verify that the proof, along with the salt, works
        let root_hash_a = tree_a.get_root();
        let mut value = XFieldElement::new([4, 4, 4].map(BFieldElement::new));
        assert!(SaltedMerkleTree::verify_authentication_path(
            root_hash_a,
            2,
            value,
            auth_path_a_and_salt.0.clone(),
            auth_path_a_and_salt.1.clone(),
        ));

        // 4: Change value and verify that it fails
        value.incr(1);
        assert!(!SaltedMerkleTree::verify_authentication_path(
            root_hash_a,
            2,
            value,
            auth_path_a_and_salt.0.clone(),
            auth_path_a_and_salt.1.clone(),
        ));
        value.decr(1);
        assert!(SaltedMerkleTree::verify_authentication_path(
            root_hash_a,
            2,
            value,
            auth_path_a_and_salt.0.clone(),
            auth_path_a_and_salt.1.clone(),
        ));
        value.decr(2);
        assert!(!SaltedMerkleTree::verify_authentication_path(
            root_hash_a,
            2,
            value,
            auth_path_a_and_salt.0.clone(),
            auth_path_a_and_salt.1.clone(),
        ));
        value.incr(2);
        assert!(SaltedMerkleTree::verify_authentication_path(
            root_hash_a,
            2,
            value,
            auth_path_a_and_salt.0.clone(),
            auth_path_a_and_salt.1.clone(),
        ));
        value.incr(0);
        assert!(!SaltedMerkleTree::verify_authentication_path(
            root_hash_a,
            2,
            value,
            auth_path_a_and_salt.0.clone(),
            auth_path_a_and_salt.1.clone(),
        ));
        value.decr(0);
        assert!(SaltedMerkleTree::verify_authentication_path(
            root_hash_a,
            2,
            value,
            auth_path_a_and_salt.0.clone(),
            auth_path_a_and_salt.1.clone(),
        ));

        // 5: Change salt and verify that it fails
        auth_path_a_and_salt.1[0].decr(1);
        assert!(!SaltedMerkleTree::verify_authentication_path(
            root_hash_a,
            2,
            value,
            auth_path_a_and_salt.0.clone(),
            auth_path_a_and_salt.1.clone(),
        ));
        auth_path_a_and_salt.1[0].incr(1);
        assert!(SaltedMerkleTree::verify_authentication_path(
            root_hash_a,
            2,
            value,
            auth_path_a_and_salt.0.clone(),
            auth_path_a_and_salt.1.clone(),
        ));
    }

    #[test]
    fn salted_merkle_tree_bug_catching_test() {
        // This test was used to catch a bug in the implementation of
        // `SaltedMerkleTree::get_leafless_multi_proof_with_salts_and_values`
        // The bug that this test caught was *fixed* in 5ad285bd867bf8c6c4be380d8539ba37f4a7409a
        // and introduced in 89cfb194f02903534b1621b03a047c128af7d6c2.

        // Build a representation of the SMT made from values
        // `451282252958277131` and `3796554602848593414` as this is where the error was
        // first caught.
        let tree = SaltedMerkleTree {
            salts_per_value: 3,
            salts: vec![
                BFieldElement::new(16852022745602243699),
                BFieldElement::new(18192741254792895208),
                BFieldElement::new(6108973982441768052),
                BFieldElement::new(968230590020974542),
                BFieldElement::new(17104237224288853866),
                BFieldElement::new(9841916779293573099),
            ],
            internal_merkle_tree: MerkleTree {
                height: 2,
                root_hash: [
                    159, 150, 101, 39, 134, 168, 140, 214, 58, 157, 141, 212, 151, 254, 245, 58,
                    117, 98, 14, 112, 190, 175, 119, 118, 28, 217, 54, 40, 243, 81, 114, 253,
                ],
                nodes: vec![
                    Node {
                        hash: [
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                        ],
                        value: None,
                    },
                    Node {
                        hash: [
                            159, 150, 101, 39, 134, 168, 140, 214, 58, 157, 141, 212, 151, 254,
                            245, 58, 117, 98, 14, 112, 190, 175, 119, 118, 28, 217, 54, 40, 243,
                            81, 114, 253,
                        ],
                        value: None,
                    },
                    Node {
                        hash: [
                            176, 133, 158, 205, 44, 56, 95, 238, 155, 222, 76, 72, 133, 248, 13,
                            238, 243, 139, 246, 241, 122, 118, 233, 76, 245, 184, 197, 8, 240, 243,
                            192, 36,
                        ],
                        value: Some(BFieldElement::new(451282252958277131)),
                    },
                    Node {
                        hash: [
                            176, 122, 54, 202, 237, 4, 73, 88, 103, 26, 64, 17, 246, 1, 70, 45, 20,
                            216, 20, 231, 61, 51, 227, 221, 6, 234, 253, 217, 153, 89, 10, 171,
                        ],
                        value: Some(BFieldElement::new(3796554602848593414)),
                    },
                ],
            },
        };
        let proof_1 = tree.get_leafless_multi_proof_with_salts_and_values(&[1]);

        // Let's first verify the value returned in the proof as this was where the bug was
        assert_eq!(
            BFieldElement::new(3796554602848593414),
            proof_1[0].2,
            "Value returned in proof must match what we put into the SMT"
        );
        assert!(
            SaltedMerkleTree::verify_leafless_multi_proof_with_salts_and_values(
                tree.get_root(),
                &[1],
                &proof_1,
            )
        );

        let proof_0 = tree.get_leafless_multi_proof_with_salts_and_values(&[0]);
        assert_eq!(
            BFieldElement::new(451282252958277131),
            proof_0[0].2,
            "Value returned in proof must match what we put into the SMT"
        );
        assert!(
            SaltedMerkleTree::verify_leafless_multi_proof_with_salts_and_values(
                tree.get_root(),
                &[0],
                &proof_0,
            )
        );
    }

    #[test]
    fn salted_merkle_tree_verify_leafless_multi_proof_test() {
        let mut prng = rand::thread_rng();

        // Number of Merkle tree leaves
        let n_valuess = &[2, 4, 8, 16, 128, 256, 512, 1024, 2048, 4096, 8192];
        let expected_path_lengths = &[1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13]; // log2(128), root node not included
        let mut rng = rand::thread_rng();
        for (n_values, expected_path_length) in izip!(n_valuess, expected_path_lengths) {
            let elements: Vec<BFieldElement> =
                generate_random_numbers_u128(*n_values, Some(1u128 << 63))
                    .iter()
                    .map(|x| BFieldElement::new(*x))
                    .collect();
            let tree = SaltedMerkleTree::from_vec(&elements, 3, &mut rng);

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

                let values: Vec<BFieldElement> = indices.iter().map(|i| elements[*i]).collect();
                let mut proof: Vec<(
                    LeaflessPartialAuthenticationPath,
                    Vec<BFieldElement>,
                    BFieldElement,
                )> = tree.get_leafless_multi_proof_with_salts_and_values(&indices);

                for (i, path) in proof.iter().enumerate() {
                    assert_eq!(*expected_path_length, path.0 .0.len());
                    assert_eq!(values[i], path.2);
                }

                assert!(
                    SaltedMerkleTree::verify_leafless_multi_proof_with_salts_and_values(
                        tree.get_root(),
                        &indices,
                        &proof,
                    )
                );
                let mut bad_root_hash = tree.get_root();
                bad_root_hash[5] ^= 0x1;
                assert!(
                    !SaltedMerkleTree::verify_leafless_multi_proof_with_salts_and_values(
                        bad_root_hash,
                        &indices,
                        &proof,
                    )
                );

                // Verify that an invalid value fails verification
                let pick = (prng.next_u64() % actual_number_of_indices as u64) as usize;
                proof[pick].2.increment();
                assert!(
                    !SaltedMerkleTree::verify_leafless_multi_proof_with_salts_and_values(
                        tree.get_root(),
                        &indices,
                        &proof,
                    )
                );
                proof[pick].2.decrement();
                assert!(
                    SaltedMerkleTree::verify_leafless_multi_proof_with_salts_and_values(
                        tree.get_root(),
                        &indices,
                        &proof,
                    )
                );

                // Verify that an invalid salt fails verification
                proof[(pick + 1) % actual_number_of_indices].1[1].decrement();
                assert!(
                    !SaltedMerkleTree::verify_leafless_multi_proof_with_salts_and_values(
                        tree.get_root(),
                        &indices,
                        &proof,
                    )
                );
            }
        }
    }
}
