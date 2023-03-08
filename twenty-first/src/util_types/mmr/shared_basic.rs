use super::mmr_membership_proof::MmrMembershipProof;
use crate::shared_math::rescue_prime_digest::Digest;
use crate::util_types::algebraic_hasher::AlgebraicHasher;

#[inline]
pub fn left_child(node_index: u64, height: u32) -> u64 {
    node_index - (1 << height)
}

#[inline]
pub fn right_child(node_index: u64) -> u64 {
    node_index - 1
}

/// Convert the leaf index into a Merkle tree index where the index refers to the tree that the leaf
/// is located in as if it were a Merkle tree. Also returns a peak index which points to which Merkle
/// tree this leaf is contained in.
pub fn leaf_index_to_mt_index_and_peak_index(leaf_index: u64, leaf_count: u64) -> (u64, u32) {
    // This assert also guarantees that leaf_count is never zero
    assert!(
        leaf_index < leaf_count,
        "Leaf index must be stricly smaller than leaf count"
    );

    let max_tree_height = u64::BITS - leaf_count.leading_zeros() - 1;
    let mut h = max_tree_height;
    let mut ret = leaf_index;
    let mut maybe_pow;
    let mut peak_index: u32 = 0;
    loop {
        let pow = 1 << h;
        maybe_pow = pow & leaf_count;
        if h == 0 || (ret < maybe_pow) {
            break;
        }
        ret -= maybe_pow;
        peak_index += (maybe_pow != 0) as u32;
        h -= 1;
    }

    ret += maybe_pow;

    (ret, peak_index)
}

/// Count the number of non-leaf nodes that were inserted *prior* to
/// the insertion of this leaf.
pub fn non_leaf_nodes_left(leaf_index: u64) -> u64 {
    // This formula is derived as follows:
    // To get the heights of peaks before this leaf index was inserted, bit-decompose
    // the number of leaves before it was inserted.
    // Number of leaves in tree of height h = 2^h
    // Number of nodes in tree of height h = 2^(h + 1) - 1
    // Number of non-leaves is `#(nodes) - #(leaves)`.
    // Thus: f(x) = sum_{h}(2^h - 1)

    // An upper limit for the loop iterator is the log_2_floor(leaf_index)
    let log_2_floor_plus_one = u64::BITS - leaf_index.leading_zeros();
    let mut h = 0;
    let mut ret = 0;
    while h != log_2_floor_plus_one {
        let pow = (1 << h) & leaf_index;
        if pow != 0 {
            ret += pow - 1;
        }
        h += 1;
    }

    ret
}

#[inline]
/// Return the number of parents that need to be added when a new leaf is inserted
pub fn right_lineage_length_from_leaf_index(leaf_index: u64) -> u32 {
    // Identify the last (least significant) nonzero bit
    let pow2 = (leaf_index + 1) & !leaf_index;

    // Get the index of that bit, counting from least significant bit
    u64::BITS - pow2.leading_zeros() - 1
}

/// Return the new peaks of the MMR after adding `new_leaf` as well as the membership
/// proof for the added leaf.
/// Returns None if configuration is impossible (too small `old_peaks` input vector)
pub fn calculate_new_peaks_from_append<H: AlgebraicHasher>(
    old_leaf_count: u64,
    old_peaks: Vec<Digest>,
    new_leaf: Digest,
) -> Option<(Vec<Digest>, MmrMembershipProof<H>)> {
    let mut peaks = old_peaks;
    peaks.push(new_leaf);
    let mut right_lineage_count = right_lineage_length_from_leaf_index(old_leaf_count);
    let mut membership_proof = MmrMembershipProof::<H>::new(old_leaf_count, vec![]);
    while right_lineage_count != 0 {
        let new_hash = peaks.pop().unwrap();
        let previous_peak_res = peaks.pop();
        let previous_peak = match previous_peak_res {
            None => return None,
            Some(peak) => peak,
        };
        membership_proof.authentication_path.push(previous_peak);
        peaks.push(H::hash_pair(&previous_peak, &new_hash));
        right_lineage_count -= 1;
    }

    Some((peaks, membership_proof))
}

/// Calculate a new peak list given the mutation of a leaf
/// The new peak list will only (max) have *one* element different
/// than `old_peaks`
pub fn calculate_new_peaks_from_leaf_mutation<H: AlgebraicHasher>(
    old_peaks: &[Digest],
    new_leaf: &Digest,
    leaf_count: u64,
    membership_proof: &MmrMembershipProof<H>,
) -> Option<Vec<Digest>> {
    let (mut acc_mt_index, peak_index) =
        leaf_index_to_mt_index_and_peak_index(membership_proof.leaf_index, leaf_count);
    let mut acc_hash: Digest = new_leaf.to_owned();
    let mut i = 0;
    while acc_mt_index != 1 {
        let ap_element = membership_proof.authentication_path[i];
        if acc_mt_index % 2 == 1 {
            // Node with `acc_hash` is a right child
            acc_hash = H::hash_pair(&ap_element, &acc_hash);
        } else {
            // Node with `acc_hash` is a left child
            acc_hash = H::hash_pair(&acc_hash, &ap_element);
        }

        acc_mt_index /= 2;
        i += 1;
    }

    let mut calculated_peaks: Vec<Digest> = old_peaks.to_vec();
    calculated_peaks[peak_index as usize] = acc_hash;

    Some(calculated_peaks)
}

#[cfg(test)]
mod mmr_test {
    use super::*;

    #[test]
    fn right_lineage_length_from_leaf_index_test() {
        assert_eq!(0, right_lineage_length_from_leaf_index(0));
        assert_eq!(1, right_lineage_length_from_leaf_index(1));
        assert_eq!(0, right_lineage_length_from_leaf_index(2));
        assert_eq!(2, right_lineage_length_from_leaf_index(3));
        assert_eq!(0, right_lineage_length_from_leaf_index(4));
        assert_eq!(1, right_lineage_length_from_leaf_index(5));
        assert_eq!(0, right_lineage_length_from_leaf_index(6));
        assert_eq!(3, right_lineage_length_from_leaf_index(7));
        assert_eq!(0, right_lineage_length_from_leaf_index(8));
        assert_eq!(1, right_lineage_length_from_leaf_index(9));
        assert_eq!(0, right_lineage_length_from_leaf_index(10));
        assert_eq!(32, right_lineage_length_from_leaf_index((1 << 32) - 1));
        assert_eq!(63, right_lineage_length_from_leaf_index((1 << 63) - 1));
    }

    #[test]
    fn leaf_index_to_mt_index_test() {
        // Leaf count = 1
        assert_eq!((1, 0), leaf_index_to_mt_index_and_peak_index(0, 1));

        // Leaf count = 2
        assert_eq!((2, 0), leaf_index_to_mt_index_and_peak_index(0, 2));
        assert_eq!((3, 0), leaf_index_to_mt_index_and_peak_index(1, 2));

        // Leaf count = 3
        assert_eq!((2, 0), leaf_index_to_mt_index_and_peak_index(0, 3));
        assert_eq!((3, 0), leaf_index_to_mt_index_and_peak_index(1, 3));
        assert_eq!((1, 1), leaf_index_to_mt_index_and_peak_index(2, 3));

        // Leaf count = 4
        assert_eq!((4, 0), leaf_index_to_mt_index_and_peak_index(0, 4));
        assert_eq!((5, 0), leaf_index_to_mt_index_and_peak_index(1, 4));
        assert_eq!((6, 0), leaf_index_to_mt_index_and_peak_index(2, 4));
        assert_eq!((7, 0), leaf_index_to_mt_index_and_peak_index(3, 4));

        // Leaf count = 14
        assert_eq!((8, 0), leaf_index_to_mt_index_and_peak_index(0, 14));
        assert_eq!((9, 0), leaf_index_to_mt_index_and_peak_index(1, 14));
        assert_eq!((10, 0), leaf_index_to_mt_index_and_peak_index(2, 14));
        assert_eq!((11, 0), leaf_index_to_mt_index_and_peak_index(3, 14));
        assert_eq!((12, 0), leaf_index_to_mt_index_and_peak_index(4, 14));
        assert_eq!((13, 0), leaf_index_to_mt_index_and_peak_index(5, 14));
        assert_eq!((14, 0), leaf_index_to_mt_index_and_peak_index(6, 14));
        assert_eq!((15, 0), leaf_index_to_mt_index_and_peak_index(7, 14));
        assert_eq!((4, 1), leaf_index_to_mt_index_and_peak_index(8, 14));
        assert_eq!((5, 1), leaf_index_to_mt_index_and_peak_index(9, 14));
        assert_eq!((6, 1), leaf_index_to_mt_index_and_peak_index(10, 14));
        assert_eq!((7, 1), leaf_index_to_mt_index_and_peak_index(11, 14));
        assert_eq!((7, 1), leaf_index_to_mt_index_and_peak_index(11, 14));

        // Leaf count = 32
        for i in 0..32 {
            assert_eq!((32 + i, 0), leaf_index_to_mt_index_and_peak_index(i, 32));
        }

        // Leaf count = 33
        for i in 0..32 {
            assert_eq!((32 + i, 0), leaf_index_to_mt_index_and_peak_index(i, 33));
        }
        assert_eq!((1, 1), leaf_index_to_mt_index_and_peak_index(32, 33));

        // Leaf count = 34
        for i in 0..32 {
            assert_eq!((32 + i, 0), leaf_index_to_mt_index_and_peak_index(i, 34));
        }
        assert_eq!((2, 1), leaf_index_to_mt_index_and_peak_index(32, 34));
        assert_eq!((3, 1), leaf_index_to_mt_index_and_peak_index(33, 34));

        // Leaf count = 35
        for i in 0..32 {
            assert_eq!((32 + i, 0), leaf_index_to_mt_index_and_peak_index(i, 35));
        }
        assert_eq!((2, 1), leaf_index_to_mt_index_and_peak_index(32, 35));
        assert_eq!((3, 1), leaf_index_to_mt_index_and_peak_index(33, 35));
        assert_eq!((1, 2), leaf_index_to_mt_index_and_peak_index(34, 35));

        // Leaf count = 36
        for i in 0..32 {
            assert_eq!((32 + i, 0), leaf_index_to_mt_index_and_peak_index(i, 36));
        }
        assert_eq!((4, 1), leaf_index_to_mt_index_and_peak_index(32, 36));
        assert_eq!((5, 1), leaf_index_to_mt_index_and_peak_index(33, 36));
        assert_eq!((6, 1), leaf_index_to_mt_index_and_peak_index(34, 36));
        assert_eq!((7, 1), leaf_index_to_mt_index_and_peak_index(35, 36));

        // Leaf count = 37
        for i in 0..32 {
            assert_eq!((32 + i, 0), leaf_index_to_mt_index_and_peak_index(i, 37));
        }
        assert_eq!((4, 1), leaf_index_to_mt_index_and_peak_index(32, 37));
        assert_eq!((5, 1), leaf_index_to_mt_index_and_peak_index(33, 37));
        assert_eq!((6, 1), leaf_index_to_mt_index_and_peak_index(34, 37));
        assert_eq!((7, 1), leaf_index_to_mt_index_and_peak_index(35, 37));
        assert_eq!((1, 2), leaf_index_to_mt_index_and_peak_index(36, 37));

        for i in 10..20 {
            assert_eq!(
                (14 + (1 << i), 0),
                leaf_index_to_mt_index_and_peak_index(14, 1 << i)
            );
            assert_eq!(
                (3, 2),
                leaf_index_to_mt_index_and_peak_index((1 << i) + 9, (1 << i) + 11)
            );
            assert_eq!(
                (1, 3),
                leaf_index_to_mt_index_and_peak_index((1 << i) + 10, (1 << i) + 11)
            );
        }
    }

    #[test]
    fn non_leaf_nodes_left_test() {
        assert_eq!(0, non_leaf_nodes_left(0));
        assert_eq!(0, non_leaf_nodes_left(1));
        assert_eq!(1, non_leaf_nodes_left(2));
        assert_eq!(1, non_leaf_nodes_left(3));
        assert_eq!(3, non_leaf_nodes_left(4));
        assert_eq!(3, non_leaf_nodes_left(5));
        assert_eq!(4, non_leaf_nodes_left(6));
        assert_eq!(4, non_leaf_nodes_left(7));
        assert_eq!(7, non_leaf_nodes_left(8));
        assert_eq!(7, non_leaf_nodes_left(9));

        assert_eq!(8, non_leaf_nodes_left(10));
        assert_eq!(8, non_leaf_nodes_left(11));
        assert_eq!(10, non_leaf_nodes_left(12));
        assert_eq!(10, non_leaf_nodes_left(13));
        assert_eq!(11, non_leaf_nodes_left(14));
        assert_eq!(11, non_leaf_nodes_left(15));
        assert_eq!(15, non_leaf_nodes_left(16));
        assert_eq!(15, non_leaf_nodes_left(17));
        assert_eq!(16, non_leaf_nodes_left(18));
    }

    #[test]
    fn peak_index_test() {
        // Verify that the function to find the Merkle tree index returns the correct peak index
        assert_eq!(0, leaf_index_to_mt_index_and_peak_index(0, 1).1);
        assert_eq!(0, leaf_index_to_mt_index_and_peak_index(0, 2).1);
        assert_eq!(0, leaf_index_to_mt_index_and_peak_index(1, 2).1);
        assert_eq!(0, leaf_index_to_mt_index_and_peak_index(0, 3).1);
        assert_eq!(0, leaf_index_to_mt_index_and_peak_index(1, 3).1);
        assert_eq!(1, leaf_index_to_mt_index_and_peak_index(2, 3).1);
        assert_eq!(0, leaf_index_to_mt_index_and_peak_index(0, 4).1);
        assert_eq!(0, leaf_index_to_mt_index_and_peak_index(1, 4).1);
        assert_eq!(0, leaf_index_to_mt_index_and_peak_index(2, 4).1);
        assert_eq!(0, leaf_index_to_mt_index_and_peak_index(3, 4).1);
        assert_eq!(0, leaf_index_to_mt_index_and_peak_index(0, 5).1);
        assert_eq!(0, leaf_index_to_mt_index_and_peak_index(1, 5).1);
        assert_eq!(0, leaf_index_to_mt_index_and_peak_index(2, 5).1);
        assert_eq!(0, leaf_index_to_mt_index_and_peak_index(3, 5).1);
        assert_eq!(1, leaf_index_to_mt_index_and_peak_index(4, 5).1);
        assert_eq!(0, leaf_index_to_mt_index_and_peak_index(0, 7).1);
        assert_eq!(0, leaf_index_to_mt_index_and_peak_index(1, 7).1);
        assert_eq!(0, leaf_index_to_mt_index_and_peak_index(2, 7).1);
        assert_eq!(0, leaf_index_to_mt_index_and_peak_index(3, 7).1);
        assert_eq!(1, leaf_index_to_mt_index_and_peak_index(4, 7).1);
        assert_eq!(1, leaf_index_to_mt_index_and_peak_index(5, 7).1);
        assert_eq!(2, leaf_index_to_mt_index_and_peak_index(6, 7).1);
        assert!(leaf_index_to_mt_index_and_peak_index(0, (1 << 32) - 1).1 == 0);
        assert!(leaf_index_to_mt_index_and_peak_index(1, (1 << 32) - 1).1 == 0);
        assert!(leaf_index_to_mt_index_and_peak_index((1 << 31) - 1, (1 << 32) - 1).1 == 0);
        assert!(
            leaf_index_to_mt_index_and_peak_index((1 << 31) + (1 << 30) - 1, (1 << 32) - 1).1 == 1
        );
        assert!(
            leaf_index_to_mt_index_and_peak_index((1 << 31) + (1 << 29) - 1, (1 << 32) - 1).1 == 1
        );
        assert!(leaf_index_to_mt_index_and_peak_index(1 << 31, (1 << 32) - 1).1 == 1);
        assert!(
            leaf_index_to_mt_index_and_peak_index(
                (1 << 31) + (1 << 30) + (1 << 29) - 1,
                (1 << 32) - 1
            )
            .1 == 2
        );
        assert!(leaf_index_to_mt_index_and_peak_index((1 << 31) + (1 << 30), (1 << 32) - 1).1 == 2);
    }
}
