use super::hash_utils::{HashUtils, Hashable};
pub use super::proof::{Lemma, Positioned, Proof};
use super::tree::{LeavesIntoIterator, LeavesIterator, Tree};
use ring::digest::{Algorithm, Digest};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// A Merkle tree is a binary tree, with values of type `T` at the leafs,
/// and where every internal node holds the hash of the concatenation of the hashes of its children nodes.
#[derive(Clone, Debug)]
pub struct MerkleTree<T> {
    /// The hashing algorithm used by this Merkle tree
    pub algorithm: &'static Algorithm,

    /// The root of the inner binary tree
    root: Tree<T>,

    /// The height of the tree
    height: usize,

    /// The number of leaf nodes in the tree
    count: usize,
}

impl<T: PartialEq> PartialEq for MerkleTree<T> {
    #[allow(trivial_casts)]
    fn eq(&self, other: &MerkleTree<T>) -> bool {
        self.root == other.root
    }
}

impl<T: Eq> Eq for MerkleTree<T> {}

impl<T: Ord> PartialOrd for MerkleTree<T> {
    fn partial_cmp(&self, other: &MerkleTree<T>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Ord> Ord for MerkleTree<T> {
    // Ordering is an enum with values: Less, Equal, and Greater
    // `then` chains together multiple Ordering by first doing the 1st
    // comparison, then the second etc. So it is monadic.
    // `then_with` takes a function whereas `then` takes an ordering.
    #[allow(trivial_casts)]
    fn cmp(&self, other: &MerkleTree<T>) -> Ordering {
        self.height
            .cmp(&other.height)
            .then(self.count.cmp(&other.count))
            .then_with(|| self.root.cmp(&other.root))
    }
}

impl<T: Hash> Hash for MerkleTree<T> {
    #[allow(trivial_casts)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        <Tree<T> as Hash>::hash(&self.root, state);
        self.height.hash(state);
        self.count.hash(state);
        (self.algorithm as *const Algorithm).hash(state);
    }
}

impl<T> MerkleTree<T> {
    // pub fn new_blake3_merkle_tree(values: Vec<T>) -> Self
    // where
    //     T: Hashable,
    // {
    //     let mut hasher = blake3::Hasher::new();
    //     Self::from_vec(&hasher, values)
    // }

    pub fn new_sha256_merkle_tree(values: Vec<T>) -> Self
    where
        T: Hashable,
    {
        Self::from_vec(&ring::digest::SHA256, values)
    }

    /// Constructs a Merkle Tree from a vector of data blocks.
    /// Returns `None` if `values` is empty.
    pub fn from_vec(algorithm: &'static Algorithm, values: Vec<T>) -> Self
    where
        T: Hashable,
    {
        if values.is_empty() {
            return MerkleTree {
                algorithm,
                root: Tree::empty(algorithm.hash_empty()),
                height: 0,
                count: 0,
            };
        }

        let count = values.len();
        let mut height = 0;
        let mut cur = Vec::with_capacity(count);

        for v in values {
            let leaf = Tree::new_leaf(algorithm, v);
            cur.push(leaf);
        }

        let mut mut_count = count;
        while cur.len() > 1 {
            cur.reverse();
            debug_assert!(cur.len() == mut_count);
            let mut next = Vec::with_capacity(mut_count);
            while !cur.is_empty() {
                if cur.len() == 1 {
                    next.push(cur.pop().unwrap());
                    mut_count += 2;
                } else {
                    let left = cur.pop().unwrap();
                    let right = cur.pop().unwrap();

                    let combined_hash: Digest = algorithm.hash_nodes(left.hash(), right.hash());

                    let node = Tree::Node {
                        hash: combined_hash.as_ref().into(),
                        left: Box::new(left),
                        right: Box::new(right),
                    };

                    next.push(node);
                }
            }

            height += 1;

            cur = next;
            mut_count /= 2;
        }

        debug_assert!(cur.len() == 1);

        let root = cur.pop().unwrap();

        MerkleTree {
            algorithm,
            root,
            height,
            count,
        }
    }

    /// Returns the root hash of Merkle tree
    pub fn root_hash(&self) -> &Vec<u8> {
        self.root.hash()
    }

    /// Returns the height of Merkle tree
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the number of leaves in the Merkle tree
    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns whether the Merkle tree is empty or not
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Generate an inclusion proof for the given value.
    /// Returns `None` if the given value is not found in the tree.
    pub fn gen_proof(&self, value: T) -> Option<Proof<T>>
    where
        T: Hashable,
    {
        let root_hash = self.root_hash().clone();
        let leaf_hash = self.algorithm.hash_leaf(&value);

        Lemma::new(&self.root, leaf_hash.as_ref())
            .map(|lemma| Proof::new(self.algorithm, root_hash, lemma, value))
    }

    /// Generate an inclusion proof for the `n`-th leaf value.
    pub fn gen_nth_proof(&self, n: usize) -> Option<Proof<T>>
    where
        T: Hashable + Clone,
    {
        let root_hash = self.root_hash().clone();
        Lemma::new_by_index(&self.root, n, self.count)
            .map(|(lemma, value)| Proof::new(self.algorithm, root_hash, lemma, value.clone()))
    }

    /// Creates an `Iterator` over the values contained in this Merkle tree.
    pub fn iter(&self) -> LeavesIterator<T> {
        self.root.iter()
    }
}

impl<T> IntoIterator for MerkleTree<T> {
    type Item = T;
    type IntoIter = LeavesIntoIterator<T>;

    /// Creates a consuming iterator, that is, one that moves each value out of the Merkle tree.
    /// The tree cannot be used after calling this.
    fn into_iter(self) -> Self::IntoIter {
        self.root.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a MerkleTree<T> {
    type Item = &'a T;
    type IntoIter = LeavesIterator<'a, T>;

    /// Creates a borrowing `Iterator` over the values contained in this Merkle tree.
    fn into_iter(self) -> Self::IntoIter {
        self.root.iter()
    }
}

#[cfg(test)]
mod merkle_tree_test {
    use super::*;

    #[test]
    fn merkle_tree_test_simple() {
        let empty_mt: MerkleTree<i128> = MerkleTree::new_sha256_merkle_tree(vec![]);
        let single_mt: MerkleTree<i128> = MerkleTree::new_sha256_merkle_tree(vec![1i128]);
        let single2_mt: MerkleTree<i128> = MerkleTree::new_sha256_merkle_tree(vec![2i128]);
        let mt: MerkleTree<i128> = MerkleTree::new_sha256_merkle_tree(vec![1i128, 2]);
        let mt_reverse: MerkleTree<i128> = MerkleTree::new_sha256_merkle_tree(vec![2i128, 1]);
        let mt_three: MerkleTree<i128> = MerkleTree::new_sha256_merkle_tree(vec![1i128, 2, 3]);
        let mt_twelve: MerkleTree<i128> =
            MerkleTree::new_sha256_merkle_tree(vec![1i128, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        assert_ne!(mt.root_hash(), empty_mt.root_hash());
        assert_ne!(mt.root_hash(), single_mt.root_hash());
        assert_ne!(mt.root_hash(), mt_reverse.root_hash());
        assert_ne!(mt.root_hash(), mt_three.root_hash());
        assert_eq!(0, empty_mt.count());
        assert_eq!(1, single_mt.count());
        assert_eq!(2, mt.count());
        assert_eq!(2, mt_reverse.count());
        assert_eq!(3, mt_three.count());
        println!("empty_mt = {:x?}", empty_mt.root_hash());
        println!("single_mt = {:x?}", single_mt.root_hash());
        println!("single2_mt = {:x?}", single2_mt.root_hash());
        println!("mt = {:x?}", mt.root_hash());
        println!("mt_reverse = {:x?}", mt_reverse.root_hash());
        println!("mt_three = {:x?}", mt_three.root_hash());
        println!("mt_twelve = {:x?}", mt_twelve.root_hash());
    }
}
