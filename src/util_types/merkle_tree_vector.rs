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
        #[allow(unused_assignments)]
        //let mut nodes: Vec<Node<T>> = Vec::with_capacity(2 * values.len());
        let mut nodes: Vec<Node<T>> = vec![
            Node {
                value: None,
                hash: [0u8; 32],
            };
            2 * values.len()
        ];
        for i in 0..values.len() {
            // hash: *blake3::hash(bincode::serialize(&x).unwrap().as_slice()).as_bytes(),
            nodes[values.len() + i].hash =
                *blake3::hash(bincode::serialize(&values[i]).unwrap().as_slice()).as_bytes();
            // digest(&SHA256, bincode::serialize(&values[i]).unwrap().as_slice())
            //     .as_ref()
            //     .try_into()
            //     .expect("");
            nodes[values.len() + i].value = Some(values[i].clone());
        }
        // println!("first hashing: {:?}", nodes);
        // nodes = values
        //     .iter()
        //     .map(|x| Node {
        //         value: Some(x.clone()),
        //         hash: digest(&SHA256, bincode::serialize(&x).unwrap().as_slice())
        //             .as_ref()
        //             .try_into()
        //             .expect(""),
        //         // hash: *blake3::hash(bincode::serialize(&x).unwrap().as_slice()).as_bytes(),
        //     })
        //     .collect::<Vec<Node<T>>>();

        // loop from `len(L) - 1` to zero
        let mut hasher = blake3::Hasher::new();
        for i in (1..(values.len())).rev() {
            hasher.update(&nodes[i * 2].hash[..]);
            hasher.update(&nodes[i * 2 + 1].hash[..]);
            // println!("{}", i);
            // let concat: Vec<u8> = nodes[i * 2]
            //     .hash
            //     .iter()
            //     .chain(nodes[i * 2 + 1].hash.iter())
            //     .cloned()
            //     .collect::<Vec<u8>>();
            // println!(
            //     "{:?} + {:?} = {:?}",
            //     nodes[i * 2].hash,
            //     nodes[i * 2 + 1].hash,
            //     concat
            // );
            nodes[i].value = None;
            // nodes[i].hash = digest(&SHA256, &concat).as_ref().try_into().expect("");
            // nodes[i].hash = *blake3::hash(&concat).as_bytes();
            nodes[i].hash = *hasher.finalize().as_bytes();
            hasher.reset();
        }

        // nodes[0] is never used for anything.
        // TODO: Remove it
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
        //let empty_mt: MerkleTreeVector<i128> = MerkleTreeVector::from_vec(&vec![]);
        //let single_mt: MerkleTreeVector<i128> = MerkleTreeVector::from_vec(&vec![1i128]);
        let mt: MerkleTreeVector<i128> = MerkleTreeVector::from_vec(&vec![1i128, 2]);
        let mt_reverse: MerkleTreeVector<i128> = MerkleTreeVector::from_vec(&vec![2i128, 1]);
        // let mt_three: MerkleTreeVector<i128> = MerkleTreeVector::from_vec(&vec![1i128, 2, 3]);
        // let mt_twelve: MerkleTreeVector<i128> =
        //     MerkleTreeVector::from_vec(&vec![1i128, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        // let mt_eight: MerkleTreeVector<i128> =
        //     MerkleTreeVector::from_vec(&vec![1i128, 2, 3, 4, 5, 6, 7, 8]);
        // assert_ne!(mt.root_hash, empty_mt.root_hash);
        // assert_ne!(mt.root_hash, single_mt.root_hash);
        println!("{:x?}", mt);
        println!("{:x?}", mt_reverse);
        assert_ne!(mt.root_hash, mt_reverse.root_hash);
        // assert_ne!(mt.root_hash, mt_eight.root_hash);
        // assert_ne!(mt.root_hash, mt_three.root_hash);
        // assert_eq!(0, empty_mt.count);
        // assert_eq!(1, single_mt.count);
        // assert_eq!(2, mt.count);
        // assert_eq!(2, mt_reverse.count);
        // assert_eq!(3, mt_three.count);
        // println!("{:x?}", empty_mt.root_hash);
        // println!("{:x?}", single_mt.root_hash);
        println!("{:x?}", mt.root_hash);
        println!("{:x?}", mt_reverse.root_hash);
        // println!("{:x?}", mt_eight.root_hash);
        // println!("{:x?}", mt_three.root_hash);
        // println!("{:x?}", mt_twelve.root_hash);
    }
}
