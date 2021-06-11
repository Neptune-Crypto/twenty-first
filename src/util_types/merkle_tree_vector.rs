use crate::utils::decode_hex;
use ring::digest::{digest, Algorithm, Digest, SHA256};
use serde::{Deserialize, Serialize};
use std::convert::TryInto;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::iter::Chain;
use std::slice::Iter;

#[derive(Clone, Debug)]
pub struct Node<T> {
    value: Option<T>,
    hash: [u8; 32],
    // Leaf { value: T, hash: [u8; 32] },
    // HashValue { hash: [u8; 32] },
}

#[derive(Clone, Debug)]
pub struct MerkleTreeVector<T> {
    root_hash: [u8; 32],
    nodes: Vec<Node<T>>,
    // TODO: Add field for holding all hashed values. Should
    // make it very quick to calculate all hash values. This field's
    // length will be 32 * count.
    // hashes: Vec<u8>,
}

// pub struct Node<T> {
//     // For now we only use Blake3. So the hash function is not a dynamic field
//     // pub hasher: Hasher;
//     /// The root of the inner binary tree
//     root: Tree<T>,

//     /// The height of the tree
//     height: usize,

//     /// The number of leaf nodes in the tree
//     count: usize,
// }

impl<T: Clone + Serialize + Debug> MerkleTreeVector<T> {
    pub fn from_vec(values: &[T]) -> Self {
        if values.is_empty() {
            let empty_hash = *blake3::hash(b"").as_bytes();
            return MerkleTreeVector {
                nodes: vec![],
                root_hash: empty_hash,
                // hashes: empty_hash.to_vec(),
            };
        }

        let mut nodes: Vec<Node<T>> = vec![
            Node {
                value: None,
                hash: [0u8; 32],
            };
            2 * values.len()
        ];
        for i in 0..values.len() {
            // println!(
            //     "{:?} => {:?}",
            //     &values[i],
            //     bincode::serialize(&values[i]).unwrap().as_slice()
            // );
            nodes[values.len() + i].hash =
                *blake3::hash(bincode::serialize(&values[i]).unwrap().as_slice()).as_bytes();
            nodes[values.len() + i].value = Some(values[i].clone());
        }

        // loop from `len(L) - 1` to 1
        let mut hasher = blake3::Hasher::new();
        for i in (1..(values.len())).rev() {
            hasher.update(&nodes[i * 2].hash[..]);
            hasher.update(&nodes[i * 2 + 1].hash[..]);
            nodes[i].value = None;
            nodes[i].hash = *hasher.finalize().as_bytes();
            hasher.reset();
        }

        // nodes[0] is never used for anything.
        // TODO: Remove it?
        MerkleTreeVector {
            root_hash: nodes[1].hash,
            nodes,
        }
    }
}

#[cfg(test)]
mod merkle_tree_vector_test {
    use super::*;

    #[test]
    fn merkle_tree_vector_test_simple() {
        let empty_mt: MerkleTreeVector<i128> = MerkleTreeVector::from_vec(&[]);
        assert_eq!(
            decode_hex("af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262")
                .expect("Decoding failed"),
            empty_mt.root_hash
        );
        let single_mt_one: MerkleTreeVector<i128> = MerkleTreeVector::from_vec(&[1i128]);
        assert_eq!(
            decode_hex("74500697761748e7dc0302d36778f89c6ab324ef942773976b92a7bbefa18cd2")
                .expect("Decoding failed"),
            single_mt_one.root_hash
        );
        let single_mt_two: MerkleTreeVector<i128> = MerkleTreeVector::from_vec(&[2i128]);
        assert_eq!(
            decode_hex("65706bf07e4e656de8a6b898dfbc64c076e001253f384043a40c437e1d5fb124")
                .expect("Decoding failed"),
            single_mt_two.root_hash
        );
        let mt: MerkleTreeVector<i128> = MerkleTreeVector::from_vec(&[1i128, 2]);
        assert_eq!(
            decode_hex("c19af4447b81b6ea9b76328441b963e6076d2e787b3fad956aa35c66f8ede2c4")
                .expect("Decoding failed"),
            mt.root_hash
        );
        let mt_reverse: MerkleTreeVector<i128> = MerkleTreeVector::from_vec(&[2i128, 1]);
        assert_eq!(
            decode_hex("189d788c8539945c368d54e9f61847b05a847f350b925ea499eadb0007130d93")
                .expect("Decoding failed"),
            mt_reverse.root_hash
        );
        let mt_four: MerkleTreeVector<i128> = MerkleTreeVector::from_vec(&[1i128, 2, 3, 4]);
        assert_eq!(
            decode_hex("44bdb434be4895b977ef91f419f16df22a9c65eeefa3843aae55f81e0e102777")
                .expect("Decoding failed"),
            mt_four.root_hash
        );
        assert_ne!(mt.root_hash, mt_reverse.root_hash);
        println!("{:x?}", mt.root_hash);
        println!("{:x?}", mt_reverse.root_hash);
    }
}
