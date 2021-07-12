use crate::shared_math::other::log_2_floor;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node<T> {
    pub value: Option<T>,
    hash: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct MerkleTree<T> {
    root_hash: [u8; 32],
    nodes: Vec<Node<T>>,
    height: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CompressedProofElement<T: Clone + Debug + PartialEq + Serialize>(
    pub Vec<Option<Node<T>>>,
);

impl<T: Clone + Serialize + Debug + PartialEq> MerkleTree<T> {
    pub fn verify_proof(root_hash: [u8; 32], index: u64, proof: Vec<Node<T>>) -> bool {
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
        let expected_hash = *blake3::hash(
            bincode::serialize(&proof[0].value.clone().unwrap())
                .expect("Encoding failed")
                .as_slice(),
        )
        .as_bytes();
        // println!("root_hash = {:?}", root_hash);
        // println!("v.hash = {:?}", v.hash);
        v.hash == root_hash && expected_hash == proof[0].hash
    }

    pub fn from_vec(values: &[T]) -> Self {
        // verify that length of input is power of 2
        if values.len() & (values.len() - 1) != 0 {
            panic!("Size of input for Merkle tree must be a power of 2");
        }

        let mut nodes: Vec<Node<T>> = vec![
            Node {
                value: None,
                hash: [0u8; 32],
            };
            2 * values.len()
        ];
        for i in 0..values.len() {
            nodes[values.len() + i].hash =
                *blake3::hash(bincode::serialize(&values[i]).unwrap().as_slice()).as_bytes();
            nodes[values.len() + i].value = Some(values[i].clone());
        }

        // loop from `len(L) - 1` to 1
        let mut hasher = blake3::Hasher::new();
        for i in (1..(values.len())).rev() {
            hasher.update(&nodes[i * 2].hash[..]);
            hasher.update(&nodes[i * 2 + 1].hash[..]);
            nodes[i].hash = *hasher.finalize().as_bytes();
            hasher.reset();
        }

        // nodes[0] is never used for anything.
        MerkleTree {
            root_hash: nodes[1].hash,
            nodes,
            height: log_2_floor(values.len() as u64) + 1,
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

    pub fn get_root(&self) -> [u8; 32] {
        self.root_hash
    }

    pub fn get_number_of_leafs(&self) -> usize {
        self.nodes.len() / 2
    }

    pub fn verify_multi_proof(
        root_hash: [u8; 32],
        indices: &[usize],
        proof: &[CompressedProofElement<T>],
    ) -> bool {
        // compressed proofs can only be verified for all indices,
        // meaning that all indices for the proof values must be known.
        // This restriction is put in since the pruned parts of the
        // multi proof are currently reassembled using the indices
        // and some parts of the proof would be missing if all the proof
        // elements were not represented in the indices argument.
        if indices.len() != proof.len() {
            return false;
        }

        let mut partial_tree: HashMap<u64, Node<T>> = HashMap::new();
        let mut proof_clone: Vec<CompressedProofElement<T>> = proof.to_owned();
        let half_tree_size = 2u64.pow(proof_clone[0].0.len() as u32 - 1);
        for (i, b) in indices.iter().zip(proof_clone.iter_mut()) {
            let mut index = half_tree_size + *i as u64;
            partial_tree.insert(index, b.0[0].clone().unwrap());
            for elem in b.0.iter_mut().skip(1) {
                if let Some(i) = elem.clone() {
                    partial_tree.insert(index ^ 1, i);
                }
                index /= 2;
            }
        }

        let mut complete = false;
        let mut hasher = blake3::Hasher::new();
        while !complete {
            complete = true;
            //let mut keys: Vec<usize> = partial_tree.iter().copied().map(|x| x / 2).collect();
            let mut keys: Vec<u64> = partial_tree.keys().copied().map(|x| x / 2).collect();
            keys.sort_by_key(|w| Reverse(*w));
            for key in keys {
                if partial_tree.contains_key(&(key * 2))
                    && partial_tree.contains_key(&(key * 2 + 1))
                    && !partial_tree.contains_key(&key)
                {
                    hasher.update(&partial_tree[&(key * 2)].hash[..]);
                    hasher.update(&partial_tree[&(key * 2 + 1)].hash[..]);
                    partial_tree.insert(
                        key,
                        Node {
                            value: None,
                            hash: *hasher.finalize().as_bytes(),
                        },
                    );
                    hasher.reset();
                    complete = false;
                }
            }
        }

        for (i, b) in indices.iter().zip(proof_clone.iter_mut()) {
            let mut index = half_tree_size + *i as u64;
            for elem in b.0.iter_mut().skip(1) {
                if *elem == None {
                    // If the Merkle tree/proof is manipulated, the value partial_tree[&(index ^ 1)]
                    // is not guaranteed to exist. So have to  check
                    // whether it exists and return false if it does not
                    if !partial_tree.contains_key(&(index ^ 1)) {
                        return false;
                    }

                    *elem = Some(partial_tree[&(index ^ 1)].clone());
                }
                partial_tree.insert(index ^ 1, elem.clone().unwrap());
                index /= 2;
            }
        }

        for i in 0..indices.len() {
            let proof_clone_unwrapped: Vec<Node<T>> = proof_clone[i]
                .0
                .clone()
                .into_iter()
                .map(|x| x.unwrap())
                .collect();
            // println!("input_proof = {:?}", proof[i]);
            // println!("proof_clone_unwrapped = {:?}", proof_clone_unwrapped);
            if !Self::verify_proof(root_hash, indices[i] as u64, proof_clone_unwrapped) {
                return false;
            }
        }
        true
    }

    pub fn get_multi_proof(&self, indices: &[usize]) -> Vec<CompressedProofElement<T>> {
        let mut calculable_indices: HashSet<usize> = HashSet::new();
        let mut output: Vec<CompressedProofElement<T>> = Vec::with_capacity(indices.len());
        for i in indices.iter() {
            let new_branch: CompressedProofElement<T> =
                CompressedProofElement(self.get_proof(*i).into_iter().map(Some).collect());
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
}

#[cfg(test)]
mod merkle_tree_test {
    use super::*;
    use crate::shared_math::prime_field_element::{PrimeField, PrimeFieldElement};
    use crate::utils::{decode_hex, generate_random_numbers};
    use itertools::Itertools;

    #[test]
    fn merkle_tree_test_32() {
        let field = PrimeField::new(1009);
        let elements: Vec<PrimeFieldElement> = generate_random_numbers(32, 1000)
            .iter()
            .map(|x| PrimeFieldElement::new(*x, &field))
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

                let proof: Vec<CompressedProofElement<PrimeFieldElement>> =
                    mt_32.get_multi_proof(&indices_usize);
                assert!(MerkleTree::verify_multi_proof(
                    mt_32.get_root(),
                    &indices_usize,
                    &proof
                ));

                // manipulate Merkle root and verify failure
                mt_32.root_hash[i] ^= 1;
                assert!(!MerkleTree::verify_multi_proof(
                    mt_32.get_root(),
                    &indices_usize,
                    &proof
                ));

                // Restore root and verify success
                mt_32.root_hash[i] ^= 1;
                assert!(MerkleTree::verify_multi_proof(
                    mt_32.get_root(),
                    &indices_usize,
                    &proof
                ));

                // Request an additional index and verify failure
                // (indices length does not match proof length)
                indices_usize.insert(0, indices_i128[0] as usize);
                assert!(!MerkleTree::verify_multi_proof(
                    mt_32.get_root(),
                    &indices_usize,
                    &proof
                ));

                // Request a non-existant index and verify failure
                // (indices length does match proof length)
                indices_usize.remove(0);
                indices_usize[0] = indices_i128[0] as usize;
                assert!(!MerkleTree::verify_multi_proof(
                    mt_32.get_root(),
                    &indices_usize,
                    &proof
                ));

                // Remove an element from the indices vector
                // and verify failure since the indices and proof
                // vectors do not match
                indices_usize.remove(0);
                assert!(!MerkleTree::verify_multi_proof(
                    mt_32.get_root(),
                    &indices_usize,
                    &proof
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
        assert_eq!(1u64, single_mt_one.height);
        let single_mt_two: MerkleTree<i128> = MerkleTree::from_vec(&[2i128]);
        assert_eq!(
            decode_hex("65706bf07e4e656de8a6b898dfbc64c076e001253f384043a40c437e1d5fb124")
                .expect("Decoding failed"),
            single_mt_two.root_hash
        );
        assert_eq!(1u64, single_mt_two.height);

        let mt: MerkleTree<i128> = MerkleTree::from_vec(&[1i128, 2]);
        assert_eq!(
            decode_hex("c19af4447b81b6ea9b76328441b963e6076d2e787b3fad956aa35c66f8ede2c4")
                .expect("Decoding failed"),
            mt.root_hash
        );
        assert_eq!(2u64, mt.height);
        let mut proof = mt.get_proof(1);
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
        assert_eq!(2u64, mt_reverse.height);

        let mut mt_four: MerkleTree<i128> = MerkleTree::from_vec(&[1i128, 2, 3, 4]);
        assert_eq!(
            decode_hex("44bdb434be4895b977ef91f419f16df22a9c65eeefa3843aae55f81e0e102777").unwrap(),
            mt_four.root_hash
        );
        assert_ne!(mt.root_hash, mt_reverse.root_hash);
        assert_eq!(3u64, mt_four.height);
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
        let mut compressed_proof = mt_four.get_multi_proof(&[0]);
        assert!(MerkleTree::verify_multi_proof(
            mt_four.root_hash,
            &[0],
            &compressed_proof
        ));
        proof = mt_four.get_proof(0);
        assert_eq!(proof.len(), compressed_proof[0].0.len());
        let unwrapped_compressed_proof: Vec<Node<i128>> = compressed_proof[0]
            .0
            .clone()
            .into_iter()
            .map(|x| x.unwrap())
            .collect();
        assert_eq!(proof, unwrapped_compressed_proof);
        println!("{:?}", compressed_proof);

        compressed_proof = mt_four.get_multi_proof(&[0, 1]);
        println!("{:?}", compressed_proof);
        assert!(MerkleTree::verify_multi_proof(
            mt_four.root_hash,
            &[0, 1],
            &compressed_proof
        ));
        compressed_proof = mt_four.get_multi_proof(&[0, 1, 2]);
        println!("{:?}", compressed_proof);
        assert!(MerkleTree::verify_multi_proof(
            mt_four.root_hash,
            &[0, 1, 2],
            &compressed_proof
        ));

        // Verify that verification of multi-proof where the tree or the proof
        // does not have the indices requested leads to a false return value,
        // and not to a run-time panic.
        assert!(!MerkleTree::verify_multi_proof(
            mt_four.root_hash,
            &[2, 3],
            &compressed_proof
        ));
    }
}
