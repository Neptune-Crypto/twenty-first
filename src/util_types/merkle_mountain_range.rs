// TODO: Parameterise this.
type Hash = [u8; 32];

pub struct MerkleMountainRange {
    pub hashes: Vec<Hash>,
}

impl MerkleMountainRange {
    pub fn from_leaf_hashes(leaf_hashes: &[Hash]) -> Self {
        // FIXME: Assess if this initial capacity is exact.
        let hashes = Vec::with_capacity(2 * leaf_hashes.len());
        let mut mmr = Self { hashes };

        for leaf_hash in leaf_hashes {
            mmr.append_leaf(*leaf_hash);
        }

        mmr
    }

    /// Push a leaf hash into the MMR. Compute hashes for new MMR peaks.
    /// Return the total number of hashes (leaves + non-leaves) after.
    pub fn append_leaf(&mut self, leaf_hash: Hash) -> usize {
        if self.hashes.is_empty() {
            self.hashes.push(leaf_hash);
            return 1;
        }

        let mut pos = self.hashes.len();
        let (peak_map, height) = peak_map_height(pos);

        // TODO: Convert assertion to something safer.
        assert_eq!(0, height, "Can only append leaves at bottom");

        self.hashes.push(leaf_hash);
        let mut prev_hash = leaf_hash;

        // TODO: Parameterise this.
        let mut hasher = blake3::Hasher::new();

        let mut peak = 1;
        while (peak_map & peak) != 0 {
            let left_sibling = left_sibling(pos, peak);
            assert!(left_sibling < pos);
            let left_hash = self.hashes[left_sibling];

            hasher.reset();
            hasher.update(&left_hash);
            hasher.update(&prev_hash);
            prev_hash = *hasher.finalize().as_bytes();
            self.hashes.push(prev_hash);

            peak <<= 1;
            pos += 1;
        }

        pos
    }

    pub fn verify(&self) -> bool {
        // TODO: Parameterise this.
        let mut hasher = blake3::Hasher::new();

        for i in 0..self.hashes.len() {
            let height = bintree_height(i);
            if height == 0 {
                continue;
            }
            let hash = self.hashes[i];
            let left_child_hash = self.hashes[left_child(i, height)];
            let right_child_hash = self.hashes[right_child(i)];

            hasher.reset();
            hasher.update(&left_child_hash);
            hasher.update(&right_child_hash);
            let hash_check = *hasher.finalize().as_bytes();
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
/// - `peak_map`: bitmask of highest peak
/// - `height`: height of element at pos
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

/// Given a `pos`, determine the smallest full binary tree that contains it, and return its height in that tree.
#[inline(always)]
pub fn bintree_height(pos: usize) -> usize {
    if pos == 0 {
        return 0;
    }

    let (_peak_map, height) = peak_map_height(pos);

    height
}

/// Given a `pos`, compute the corresponding position of a left sibling under a `peak`
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
    use crate::shared_math::traits::GetRandomElements;
    use rand::prelude::ThreadRng;
    use rand::RngCore;

    // For testing only!
    impl GetRandomElements for Hash {
        fn random_elements(n: usize, rng: &mut ThreadRng) -> Vec<Self> {
            let mut hasher = blake3::Hasher::new();
            let seed = rng.next_u64().to_ne_bytes();
            hasher.reset();
            hasher.update(&seed);

            let mut result = Vec::with_capacity(n);
            for _ in 0..n {
                let hash = *hasher.finalize().as_bytes();
                result.push(hash);
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
            let hashes = Hash::random_elements(n, &mut rng);

            // 2. Insert them into MMR
            let mmr = MerkleMountainRange::from_leaf_hashes(&hashes);

            // 3. Validate MMR
            assert!(mmr.verify());
        }
    }
}
