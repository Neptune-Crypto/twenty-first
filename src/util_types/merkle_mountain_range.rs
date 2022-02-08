use super::simple_hasher::{Hasher, ToDigest};
use std::marker::PhantomData;

/// Type parameters:
///
/// - `D`: a hash digest
/// - `H`: a `Hasher<D>`
pub struct MerkleMountainRange<D, H> {
    pub digests: Vec<D>,
    _hasher: PhantomData<H>,
}

impl<D, H> MerkleMountainRange<D, H>
where
    D: ToDigest<D> + PartialEq + Copy,
    H: Hasher<Digest = D>,
{
    /// Create MMR from `leaf_hashes` by appending each progressively.
    pub fn from_leaf_hashes(leaf_hashes: &[D]) -> Self {
        // FIXME: Assess if this initial capacity is exact.
        let digests = Vec::with_capacity(2 * leaf_hashes.len());
        let _hasher = PhantomData;
        let mut mmr = Self { digests, _hasher };

        for leaf_hash in leaf_hashes {
            mmr.append_leaf(*leaf_hash);
        }

        mmr
    }

    /// Push a leaf hash into the MMR. Compute hashes for new MMR peaks.
    /// Return the total number of hashes (leaves + non-leaves) after.
    pub fn append_leaf(&mut self, leaf_hash: D) -> usize {
        if self.digests.is_empty() {
            self.digests.push(leaf_hash);
            return 1;
        }

        let mut pos = self.digests.len();
        let (peak_map, height) = peak_map_height(pos);

        // TODO: Convert assertion to something safer.
        assert_eq!(0, height, "Can only append leaves at bottom");

        self.digests.push(leaf_hash);
        let mut prev_hash = leaf_hash;

        let mut hasher = H::new();
        let mut peak = 1;

        // While we've not exceeded the top peak
        while (peak_map & peak) != 0 {
            let left_sibling = left_sibling(pos, peak);
            let left_hash = self.digests[left_sibling];
            prev_hash = hasher.hash_two(&left_hash, &prev_hash);
            self.digests.push(prev_hash);

            peak <<= 1;
            pos += 1;
        }

        pos
    }

    pub fn verify(&self) -> bool {
        let mut hasher = H::new();
        for i in 0..self.digests.len() {
            let height = bintree_height(i);
            if height == 0 {
                continue;
            }

            let left_child_hash = self.digests[left_child(i, height)];
            let right_child_hash = self.digests[right_child(i)];
            let hash_check = hasher.hash_two(&left_child_hash, &right_child_hash);
            let hash = self.digests[i];

            if hash != hash_check {
                return false;
            }
        }

        true
    }
}

const ALL_ONES: usize = std::usize::MAX;

/// Compute (peak_map, height) where
///
/// - `peak_map`: bitmask that includes highest peak
/// - `height`: height of element at pos (0 for leaves)
///
/// Note that when `height != 0`, an element cannot be appended.
pub fn peak_map_height(mut pos: usize) -> (usize, usize) {
    if pos == 0 {
        return (0, 0);
    }
    let mut peak_size = ALL_ONES >> pos.leading_zeros();
    let mut bitmap = 0;
    while peak_size != 0 {
        bitmap <<= 1;
        if pos >= peak_size {
            pos -= peak_size;
            bitmap |= 1;
        }
        peak_size >>= 1;
    }
    (bitmap, pos)
}

/// Given a `pos`, determine the smallest full binary tree (mountain) that
/// contains it, and return its height in that tree.
#[inline(always)]
pub fn bintree_height(pos: usize) -> usize {
    if pos == 0 {
        return 0;
    }

    let (_peak_map, height) = peak_map_height(pos);

    height
}

/// Given a `pos`, compute the position of a left sibling under a `peak`
fn left_sibling(pos: usize, peak: usize) -> usize {
    pos + 1 - 2 * peak
}

fn left_child(n: usize, height: usize) -> usize {
    n - (1 << height)
}

fn right_child(n: usize) -> usize {
    n - 1
}

#[cfg(test)]
mod merkle_mountain_range_test {
    use super::*;
    use crate::shared_math::b_field_element::BFieldElement;
    use crate::shared_math::traits::GetRandomElements;
    use crate::util_types::simple_hasher::RescuePrimeProduction;
    use rand::prelude::ThreadRng;
    use rand::RngCore;

    // For testing only!
    impl GetRandomElements for blake3::Hash {
        fn random_elements(n: usize, rng: &mut ThreadRng) -> Vec<Self> {
            let mut hasher = blake3::Hasher::new();
            let seed = rng.next_u64().to_ne_bytes();
            hasher.reset();
            hasher.update(&seed);

            let mut result = Vec::with_capacity(n);
            for _ in 0..n {
                let hash = *hasher.finalize().as_bytes();
                result.push(hash.into());
                hasher.update(&hash);
            }

            result
        }
    }

    #[test]
    pub fn gotta_go_fast_test() {
        for n in [0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 127, 128, 129] {
            // 1. Create a bunch of hashes
            let mut rng = rand::thread_rng();
            let hashes = blake3::Hash::random_elements(n, &mut rng);

            // 2. Insert them into MMR
            type B3Hash = blake3::Hash;
            type B3Hasher = blake3::Hasher;
            let mmr = MerkleMountainRange::<B3Hash, B3Hasher>::from_leaf_hashes(&hashes);

            // 3. Validate MMR
            assert!(mmr.verify());
        }
    }

    #[test]
    pub fn gotta_go_bfield_test() {
        // 1. Create a bunch of hashes
        let mut rng = rand::thread_rng();
        let hashes = vec![
            BFieldElement::random_elements(5, &mut rng),
            BFieldElement::random_elements(5, &mut rng),
            BFieldElement::random_elements(5, &mut rng),
        ];

        // 2. Insert them into MMR
        type BFHash = Vec<BFieldElement>;
        type BFHasher = RescuePrimeProduction;
        let mmr = MerkleMountainRange::<BFHash, BFHasher>::from_leaf_hashes(&hashes);

        // 3. Validate MMR
        assert!(mmr.verify());
    }
}
