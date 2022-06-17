use super::mmr_membership_proof::MmrMembershipProof;
use crate::shared_math::other::log_2_floor;
use crate::util_types::simple_hasher::{Hasher, ToDigest};
use std::marker::PhantomData;

#[inline]
pub fn left_child(node_index: u128, height: u128) -> u128 {
    node_index - (1 << height)
}

#[inline]
pub fn right_child(node_index: u128) -> u128 {
    node_index - 1
}

/// Get (index, height) of leftmost ancestor
/// This ancestor does *not* have to be in the MMR
#[inline]
pub fn leftmost_ancestor(node_index: u128) -> (u128, u128) {
    let mut h = 0;
    let mut ret = 1;
    while ret < node_index {
        h += 1;
        ret = (1 << (h + 1)) - 1;
    }

    (ret, h)
}

/// Return the tuple: (is_right_child, height)
#[inline]
pub fn right_child_and_height(node_index: u128) -> (bool, u128) {
    // 1. Find leftmost_ancestor(n), if leftmost_ancestor(n) == n => left_child (false)
    // 2. Let node = leftmost_ancestor(n)
    // 3. while(true):
    //    if n == left_child(node):
    //        return false
    //    if n < left_child(node):
    //        node = left_child(node)
    //    if n == right_child(node):
    //        return true
    //    else:
    //        node = right_child(node);

    // 1.
    let (leftmost_ancestor, ancestor_height) = leftmost_ancestor(node_index);
    if leftmost_ancestor == node_index {
        return (false, ancestor_height);
    }

    let mut node = leftmost_ancestor;
    let mut height = ancestor_height;
    loop {
        let left_child = left_child(node, height);
        height -= 1;
        if node_index == left_child {
            return (false, height);
        }
        if node_index < left_child {
            node = left_child;
        } else {
            let right_child = right_child(node);
            if node_index == right_child {
                return (true, height);
            }
            node = right_child;
        }
    }
}

/// Get the node_index of the parent
#[inline]
pub fn parent(node_index: u128) -> u128 {
    let (right, height) = right_child_and_height(node_index);

    if right {
        node_index + 1
    } else {
        node_index + (1 << (height + 1))
    }
}

#[inline]
pub fn left_sibling(node_index: u128, height: u128) -> u128 {
    node_index - (1 << (height + 1)) + 1
}

#[inline]
pub fn right_sibling(node_index: u128, height: u128) -> u128 {
    node_index + (1 << (height + 1)) - 1
}

fn get_height_from_data_index(data_index: u128) -> u128 {
    log_2_floor(data_index as u64 + 1) as u128
}

pub fn leaf_count_to_node_count(leaf_count: u128) -> u128 {
    if leaf_count == 0 {
        return 0;
    }

    let rightmost_leaf_data_index = leaf_count - 1;
    let non_leaf_nodes_left = non_leaf_nodes_left(rightmost_leaf_data_index);
    let node_index_of_rightmost_leaf = data_index_to_node_index(rightmost_leaf_data_index);

    let mut non_leaf_nodes_after = 0u128;
    let mut node_index = node_index_of_rightmost_leaf;
    let (mut is_right, mut _height) = right_child_and_height(node_index);
    while is_right {
        non_leaf_nodes_after += 1;
        node_index = parent(node_index);
        is_right = right_child_and_height(node_index).0;
    }

    // Number of nodes is: non-leafs after, non-leafs before, and leaf count
    non_leaf_nodes_after + non_leaf_nodes_left + leaf_count
}

/// leaf_index_to_peak_index
/// Compute the index of the peak that contains the given leaf index,
/// given the leaf count.
pub fn leaf_index_to_peak_index(leaf_index: u128, leaf_count: u128) -> Option<u128> {
    if leaf_index >= leaf_count {
        None
    } else {
        let mut num_set_bits = 0;
        let mut leaf_count_copy = leaf_count;
        while leaf_count_copy != 0 {
            if leaf_count_copy < leaf_index + 1 {
                num_set_bits += 1;
            }
            leaf_count_copy &= leaf_count_copy - 1;
        }
        Some(num_set_bits)
    }
}

/// Return the indices of the nodes added by an append, including the
/// peak that this append gave rise to
pub fn node_indices_added_by_append(old_leaf_count: u128) -> Vec<u128> {
    let mut node_index = data_index_to_node_index(old_leaf_count);
    let mut added_node_indices = vec![node_index];
    let mut is_right_child: bool = right_child_and_height(node_index).0;
    while is_right_child {
        // a right child's parent is found by adding 1 to the node index
        node_index += 1;
        added_node_indices.push(node_index);
        is_right_child = right_child_and_height(node_index).0;
    }

    added_node_indices
}

/// Get the node indices of the authentication path hash digest needed
/// to calculate the digest of `peak_node_index` from `start_node_index`
pub fn get_authentication_path_node_indices(
    start_node_index: u128,
    peak_node_index: u128,
    node_count: u128,
) -> Option<Vec<u128>> {
    let mut authentication_path_node_indices = vec![];
    let mut node_index = start_node_index;
    while node_index <= node_count && node_index != peak_node_index {
        let (is_right, height) = right_child_and_height(node_index);
        let sibling_node_index = if is_right {
            left_sibling(node_index, height)
        } else {
            right_sibling(node_index, height)
        };
        authentication_path_node_indices.push(sibling_node_index);
        node_index = parent(node_index);
    }

    if node_index == peak_node_index {
        Some(authentication_path_node_indices)
    } else {
        None
    }
}

/// Given leaf count, return a vector representing the height of
/// the peaks. Input is the number of leafs in the MMR
pub fn get_peak_heights_and_peak_node_indices(leaf_count: u128) -> (Vec<u128>, Vec<u128>) {
    if leaf_count == 0 {
        return (vec![], vec![]);
    }

    let node_index_of_rightmost_leaf = data_index_to_node_index(leaf_count - 1);
    let node_count = leaf_count_to_node_count(leaf_count);
    let (mut top_peak, mut top_height) = leftmost_ancestor(node_index_of_rightmost_leaf);
    if top_peak > node_count {
        top_peak = left_child(top_peak, top_height);
        top_height -= 1;
    }

    let mut heights: Vec<u128> = vec![top_height];
    let mut node_indices: Vec<u128> = vec![top_peak];
    let mut height = top_height;
    let mut candidate = right_sibling(top_peak, height);
    'outer: while height > 0 {
        '_inner: while candidate > node_count && height > 0 {
            candidate = left_child(candidate, height);
            height -= 1;
            if candidate <= node_count {
                heights.push(height);
                node_indices.push(candidate);
                candidate = right_sibling(candidate, height);
                continue 'outer;
            }
        }
    }

    (heights, node_indices)
}

/// Count the number of non-leaf nodes that were inserted *prior* to
/// the insertion of this leaf.
fn non_leaf_nodes_left(data_index: u128) -> u128 {
    if data_index == 0 {
        return 0;
    }

    let mut acc = 0;
    let mut data_index_acc = data_index;
    while data_index_acc > 0 {
        // Accumulate how many nodes in the tree of the nearest left neighbor that are not leafs.
        // We count this number for the nearest left neighbor since only the non-leafs in that
        // tree were inserted prior to the leaf this function is called for.
        // For a tree of height 2, there are 2^2 - 1 non-leaf nodes, note that height starts at
        // 0.
        // Since more than one subtree left of the requested index can contain non-leafs, we have
        // to run this accumulater untill data_index_acc is zero.
        let left_data_height = get_height_from_data_index(data_index_acc - 1);
        acc += (1 << left_data_height) - 1;
        data_index_acc -= 1 << left_data_height;
    }

    acc
}

/// Convert from data index to node index
pub fn data_index_to_node_index(data_index: u128) -> u128 {
    let diff = non_leaf_nodes_left(data_index);

    data_index + diff + 1
}

/// Convert from node index to data index in log(size) time
pub fn node_index_to_data_index(node_index: u128) -> Option<u128> {
    let (_right, own_height) = right_child_and_height(node_index);
    if own_height != 0 {
        return None;
    }

    let (mut node, mut node_height) = leftmost_ancestor(node_index);
    let mut data_index = 0;
    while node_height > 0 {
        let left_child = left_child(node, node_height);
        if node_index <= left_child {
            node = left_child;
            node_height -= 1;
        } else {
            node = right_child(node);
            node_height -= 1;
            data_index += 1 << node_height;
        }
    }

    Some(data_index)
}

/// Return the new peaks of the MMR after adding `new_leaf` as well as the membership
/// proof for the added leaf.
/// Returns None if configuration is impossible (too small `old_peaks` input vector)
pub fn calculate_new_peaks_from_append<H: Hasher>(
    old_leaf_count: u128,
    old_peaks: Vec<H::Digest>,
    new_leaf: H::Digest,
) -> Option<(Vec<H::Digest>, MmrMembershipProof<H>)> {
    let mut peaks = old_peaks;
    let mut new_node_index = data_index_to_node_index(old_leaf_count);
    let (mut new_node_is_right_child, _height) = right_child_and_height(new_node_index);
    peaks.push(new_leaf);
    let hasher = H::new();
    let mut membership_proof: MmrMembershipProof<H> = MmrMembershipProof {
        authentication_path: vec![],
        data_index: old_leaf_count,
        _hasher: PhantomData,
    };
    while new_node_is_right_child {
        let new_hash = peaks.pop().unwrap();
        let previous_peak_res = peaks.pop();
        let previous_peak = match previous_peak_res {
            None => return None,
            Some(peak) => peak,
        };
        membership_proof
            .authentication_path
            .push(previous_peak.clone());
        peaks.push(hasher.hash_pair(&previous_peak, &new_hash));
        new_node_index += 1;
        new_node_is_right_child = right_child_and_height(new_node_index).0;
    }

    Some((peaks, membership_proof))
}

/// Calculate a new peak list given the mutation of a leaf
/// The new peak list will only (max) have *one* element different
/// than `old_peaks`
pub fn calculate_new_peaks_from_leaf_mutation<H: Hasher>(
    old_peaks: &[H::Digest],
    new_leaf: &H::Digest,
    leaf_count: u128,
    membership_proof: &MmrMembershipProof<H>,
) -> Option<Vec<H::Digest>>
where
    u128: ToDigest<H::Digest>,
{
    let node_index = data_index_to_node_index(membership_proof.data_index);
    let hasher = H::new();
    let mut acc_hash: H::Digest = new_leaf.to_owned();
    let mut acc_index: u128 = node_index;
    for hash in membership_proof.authentication_path.iter() {
        let (acc_right, _acc_height) = right_child_and_height(acc_index);
        acc_hash = if acc_right {
            hasher.hash_pair(hash, &acc_hash)
        } else {
            hasher.hash_pair(&acc_hash, hash)
        };
        acc_index = parent(acc_index);
    }

    // Find the correct peak to update
    let peak_index_res = leaf_index_to_peak_index(membership_proof.data_index, leaf_count);
    let peak_index = match peak_index_res {
        None => return None,
        Some(pi) => pi,
    };

    let mut calculated_peaks: Vec<H::Digest> = old_peaks.to_vec();
    calculated_peaks[peak_index as usize] = acc_hash;

    Some(calculated_peaks)
}

/// Get a root commitment to the entire MMR
pub fn bag_peaks<H>(peaks: &[H::Digest]) -> H::Digest
where
    H: Hasher,
    u128: ToDigest<H::Digest>,
{
    // Follows the description on
    // https://github.com/mimblewimble/grin/blob/master/doc/mmr.md#hashing-and-bagging
    // to calculate a root from a list of peaks and the size of the MMR. Note, however,
    // that the node count described on that website is not used here, as we don't need
    // the extra bits of security that that would provide.
    let peaks_count: usize = peaks.len();
    let hasher: H = H::new();

    if peaks_count == 0 {
        return hasher.hash(&0u128.to_digest());
    }

    if peaks_count == 1 {
        return peaks[0].to_owned();
    }

    let mut acc: H::Digest = hasher.hash_pair(&peaks[peaks_count - 1], &peaks[peaks_count - 2]);
    for i in 2..peaks_count {
        acc = hasher.hash_pair(&acc, &peaks[peaks_count - 1 - i]);
    }

    acc
}

#[cfg(test)]
mod mmr_test {
    use rand::RngCore;

    use crate::{
        shared_math::b_field_element::BFieldElement,
        util_types::{
            mmr::{
                archival_mmr::ArchivalMmr, mmr_trait::Mmr,
                shared::calculate_new_peaks_from_leaf_mutation,
            },
            simple_hasher::{Hasher, RescuePrimeProduction},
        },
    };

    use super::*;

    #[test]
    fn data_index_to_node_index_test() {
        assert_eq!(1, data_index_to_node_index(0));
        assert_eq!(2, data_index_to_node_index(1));
        assert_eq!(4, data_index_to_node_index(2));
        assert_eq!(5, data_index_to_node_index(3));
        assert_eq!(8, data_index_to_node_index(4));
        assert_eq!(9, data_index_to_node_index(5));
        assert_eq!(11, data_index_to_node_index(6));
        assert_eq!(12, data_index_to_node_index(7));
        assert_eq!(16, data_index_to_node_index(8));
        assert_eq!(17, data_index_to_node_index(9));
        assert_eq!(19, data_index_to_node_index(10));
        assert_eq!(20, data_index_to_node_index(11));
        assert_eq!(23, data_index_to_node_index(12));
        assert_eq!(24, data_index_to_node_index(13));
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
    }

    #[test]
    fn get_height_from_data_index_test() {
        assert_eq!(0, get_height_from_data_index(0));
        assert_eq!(1, get_height_from_data_index(1));
        assert_eq!(1, get_height_from_data_index(2));
        assert_eq!(2, get_height_from_data_index(3));
        assert_eq!(2, get_height_from_data_index(4));
        assert_eq!(2, get_height_from_data_index(5));
        assert_eq!(2, get_height_from_data_index(6));
        assert_eq!(3, get_height_from_data_index(7));
        assert_eq!(3, get_height_from_data_index(8));
    }

    #[test]
    fn node_indices_added_by_append_test() {
        assert_eq!(vec![1], node_indices_added_by_append(0));
        assert_eq!(vec![2, 3], node_indices_added_by_append(1));
        assert_eq!(vec![4], node_indices_added_by_append(2));
        assert_eq!(vec![5, 6, 7], node_indices_added_by_append(3));
        assert_eq!(vec![8], node_indices_added_by_append(4));
        assert_eq!(vec![9, 10], node_indices_added_by_append(5));
        assert_eq!(vec![11], node_indices_added_by_append(6));
        assert_eq!(vec![12, 13, 14, 15], node_indices_added_by_append(7));
        assert_eq!(vec![16], node_indices_added_by_append(8));
        assert_eq!(vec![17, 18], node_indices_added_by_append(9));
        assert_eq!(vec![19], node_indices_added_by_append(10));
        assert_eq!(vec![20, 21, 22], node_indices_added_by_append(11));
        assert_eq!(vec![23], node_indices_added_by_append(12));
        assert_eq!(vec![24, 25], node_indices_added_by_append(13));
        assert_eq!(vec![26], node_indices_added_by_append(14));
        assert_eq!(vec![27, 28, 29, 30, 31], node_indices_added_by_append(15));
        assert_eq!(vec![32], node_indices_added_by_append(16));
        assert_eq!(vec![33, 34], node_indices_added_by_append(17));
        assert_eq!(vec![35], node_indices_added_by_append(18));
        assert_eq!(vec![36, 37, 38], node_indices_added_by_append(19));
        assert_eq!(
            vec![58, 59, 60, 61, 62, 63],
            node_indices_added_by_append(31)
        );
        assert_eq!(vec![64], node_indices_added_by_append(32));
    }

    #[test]
    fn leaf_index_to_peak_index_test() {
        assert_eq!(0, leaf_index_to_peak_index(0, 1).unwrap());
        assert_eq!(0, leaf_index_to_peak_index(0, 2).unwrap());
        assert_eq!(0, leaf_index_to_peak_index(1, 2).unwrap());
        assert_eq!(0, leaf_index_to_peak_index(0, 3).unwrap());
        assert_eq!(0, leaf_index_to_peak_index(1, 3).unwrap());
        assert_eq!(1, leaf_index_to_peak_index(2, 3).unwrap());
        assert_eq!(0, leaf_index_to_peak_index(0, 4).unwrap());
        assert_eq!(0, leaf_index_to_peak_index(1, 4).unwrap());
        assert_eq!(0, leaf_index_to_peak_index(2, 4).unwrap());
        assert_eq!(0, leaf_index_to_peak_index(3, 4).unwrap());
        assert_eq!(0, leaf_index_to_peak_index(0, 5).unwrap());
        assert_eq!(0, leaf_index_to_peak_index(1, 5).unwrap());
        assert_eq!(0, leaf_index_to_peak_index(2, 5).unwrap());
        assert_eq!(0, leaf_index_to_peak_index(3, 5).unwrap());
        assert_eq!(1, leaf_index_to_peak_index(4, 5).unwrap());
        assert_eq!(0, leaf_index_to_peak_index(0, 7).unwrap());
        assert_eq!(0, leaf_index_to_peak_index(1, 7).unwrap());
        assert_eq!(0, leaf_index_to_peak_index(2, 7).unwrap());
        assert_eq!(0, leaf_index_to_peak_index(3, 7).unwrap());
        assert_eq!(1, leaf_index_to_peak_index(4, 7).unwrap());
        assert_eq!(1, leaf_index_to_peak_index(5, 7).unwrap());
        assert_eq!(2, leaf_index_to_peak_index(6, 7).unwrap());
        assert!(leaf_index_to_peak_index(0, (1 << 32) - 1).unwrap() == 0);
        assert!(leaf_index_to_peak_index(1, (1 << 32) - 1).unwrap() == 0);
        assert!(leaf_index_to_peak_index((1 << 31) - 1, (1 << 32) - 1).unwrap() == 0);
        assert!(leaf_index_to_peak_index((1 << 31) + (1 << 30) - 1, (1 << 32) - 1).unwrap() == 1);
        assert!(leaf_index_to_peak_index((1 << 31) + (1 << 29) - 1, (1 << 32) - 1).unwrap() == 1);
        assert!(leaf_index_to_peak_index(1 << 31, (1 << 32) - 1).unwrap() == 1);
        assert!(
            leaf_index_to_peak_index((1 << 31) + (1 << 30) + (1 << 29) - 1, (1 << 32) - 1).unwrap()
                == 2
        );
        assert!(leaf_index_to_peak_index((1 << 31) + (1 << 30), (1 << 32) - 1).unwrap() == 2);

        // Verify that None is returned if leaf index exceeds leaf count
        assert!(leaf_index_to_peak_index(1, 0).is_none());
        assert!(leaf_index_to_peak_index(1, 1).is_none());
        assert!(leaf_index_to_peak_index(2, 1).is_none());
        assert!(leaf_index_to_peak_index(100, 1).is_none());
        assert!(leaf_index_to_peak_index(100, 99).is_none());
        assert!(leaf_index_to_peak_index(100, 100).is_none());
        assert!(leaf_index_to_peak_index(100, 101).is_some());
    }

    #[test]
    fn data_index_node_index_pbt() {
        let mut rng = rand::thread_rng();
        for _ in 0..10000 {
            let rand = rng.next_u32();
            let inversion_result = node_index_to_data_index(data_index_to_node_index(rand as u128));
            match inversion_result {
                None => panic!(),
                Some(inversion) => assert_eq!(rand, inversion as u32),
            }
        }
    }

    #[test]
    fn is_right_child_test() {
        // Consider this a 1-indexed list of the expected result where the input to the function is the
        // (1-indexed) element of the list
        let anticipations: Vec<bool> = vec![
            false, true, false, false, true, true, false, false, true, false, false, true, true,
            //1      2     3      4      5     6     7      8      9     10     11     12    13
            true, false, false, true, false, false, true, true, false, false, true, false, false,
            //14     15    16     17    18     19    20    21     22    23     24     25    26
            true, true, true, true, false, false,
            //27    28   29    30    31     32
            true,
            //33
        ];

        for (i, anticipation) in anticipations.iter().enumerate() {
            assert!(right_child_and_height(i as u128 + 1).0 == *anticipation);
        }
    }

    #[test]
    fn leftmost_ancestor_test() {
        assert_eq!((1, 0), leftmost_ancestor(1));
        assert_eq!((3, 1), leftmost_ancestor(2));
        assert_eq!((3, 1), leftmost_ancestor(3));
        assert_eq!((7, 2), leftmost_ancestor(4));
        assert_eq!((7, 2), leftmost_ancestor(5));
        assert_eq!((7, 2), leftmost_ancestor(6));
        assert_eq!((7, 2), leftmost_ancestor(7));
        assert_eq!((15, 3), leftmost_ancestor(8));
        assert_eq!((15, 3), leftmost_ancestor(9));
        assert_eq!((15, 3), leftmost_ancestor(10));
        assert_eq!((15, 3), leftmost_ancestor(11));
        assert_eq!((15, 3), leftmost_ancestor(12));
        assert_eq!((15, 3), leftmost_ancestor(13));
        assert_eq!((15, 3), leftmost_ancestor(14));
        assert_eq!((15, 3), leftmost_ancestor(15));
        assert_eq!((31, 4), leftmost_ancestor(16));
    }

    #[test]
    fn left_sibling_test() {
        assert_eq!(3, left_sibling(6, 1));
        assert_eq!(1, left_sibling(2, 0));
        assert_eq!(4, left_sibling(5, 0));
        assert_eq!(15, left_sibling(30, 3));
        assert_eq!(22, left_sibling(29, 2));
        assert_eq!(7, left_sibling(14, 2));
    }

    #[test]
    fn node_index_to_data_index_test() {
        assert_eq!(Some(0), node_index_to_data_index(1));
        assert_eq!(Some(1), node_index_to_data_index(2));
        assert_eq!(None, node_index_to_data_index(3));
        assert_eq!(Some(2), node_index_to_data_index(4));
        assert_eq!(Some(3), node_index_to_data_index(5));
        assert_eq!(None, node_index_to_data_index(6));
        assert_eq!(None, node_index_to_data_index(7));
        assert_eq!(Some(4), node_index_to_data_index(8));
        assert_eq!(Some(5), node_index_to_data_index(9));
        assert_eq!(None, node_index_to_data_index(10));
        assert_eq!(Some(6), node_index_to_data_index(11));
        assert_eq!(Some(7), node_index_to_data_index(12));
        assert_eq!(None, node_index_to_data_index(13));
        assert_eq!(None, node_index_to_data_index(14));
        assert_eq!(None, node_index_to_data_index(15));
        assert_eq!(Some(8), node_index_to_data_index(16));
        assert_eq!(Some(9), node_index_to_data_index(17));
        assert_eq!(None, node_index_to_data_index(18));
        assert_eq!(Some(10), node_index_to_data_index(19));
        assert_eq!(Some(11), node_index_to_data_index(20));
        assert_eq!(None, node_index_to_data_index(21));
        assert_eq!(None, node_index_to_data_index(22));
    }

    #[test]
    fn leaf_count_to_node_count_test() {
        let node_counts: Vec<u128> = vec![
            0, 1, 3, 4, 7, 8, 10, 11, 15, 16, 18, 19, 22, 23, 25, 26, 31, 32, 34, 35, 38, 39, 41,
            42, 46, 47, 49, 50, 53, 54, 56, 57, 63, 64,
        ];
        for (i, node_count) in node_counts.iter().enumerate() {
            assert_eq!(*node_count, leaf_count_to_node_count(i as u128));
        }
    }

    #[test]
    fn get_peak_heights_and_peak_node_indices_test() {
        let leaf_count_and_expected: Vec<(u128, (Vec<u128>, Vec<u128>))> = vec![
            (0, (vec![], vec![])),
            (1, (vec![0], vec![1])),
            (2, (vec![1], vec![3])),
            (3, (vec![1, 0], vec![3, 4])),
            (4, (vec![2], vec![7])),
            (5, (vec![2, 0], vec![7, 8])),
            (6, (vec![2, 1], vec![7, 10])),
            (7, (vec![2, 1, 0], vec![7, 10, 11])),
            (8, (vec![3], vec![15])),
            (9, (vec![3, 0], vec![15, 16])),
            (10, (vec![3, 1], vec![15, 18])),
            (11, (vec![3, 1, 0], vec![15, 18, 19])),
            (12, (vec![3, 2], vec![15, 22])),
            (13, (vec![3, 2, 0], vec![15, 22, 23])),
            (14, (vec![3, 2, 1], vec![15, 22, 25])),
            (15, (vec![3, 2, 1, 0], vec![15, 22, 25, 26])),
            (16, (vec![4], vec![31])),
            (17, (vec![4, 0], vec![31, 32])),
            (18, (vec![4, 1], vec![31, 34])),
            (19, (vec![4, 1, 0], vec![31, 34, 35])),
        ];
        for (leaf_count, (expected_heights, expected_indices)) in leaf_count_and_expected {
            assert_eq!(
                (expected_heights, expected_indices),
                get_peak_heights_and_peak_node_indices(leaf_count)
            );
        }
    }

    #[test]
    fn get_authentication_path_node_indices_test() {
        let start_end_node_count_expected: Vec<((u128, u128, u128), Option<Vec<u128>>)> = vec![
            ((1, 31, 31), Some(vec![2, 6, 14, 30])),
            ((2, 31, 31), Some(vec![1, 6, 14, 30])),
            ((3, 31, 31), Some(vec![6, 14, 30])),
            ((4, 31, 31), Some(vec![5, 3, 14, 30])),
            ((21, 31, 31), Some(vec![18, 29, 15])),
            ((21, 31, 32), Some(vec![18, 29, 15])),
            ((32, 32, 32), Some(vec![])),
            ((1, 32, 32), None),
        ];
        for ((start, end, node_count), expected) in start_end_node_count_expected {
            assert_eq!(
                expected,
                get_authentication_path_node_indices(start, end, node_count)
            );
        }
    }

    #[test]
    fn calculate_new_peaks_from_leaf_mutation_empty_mmr_test() {
        type Hasher = RescuePrimeProduction;

        // Verify that the helper function `calculate_new_peaks_from_leaf_mutation` does
        // not crash if called on an empty list of peaks
        let rp = Hasher::new();
        let new_leaf = rp.hash(&vec![BFieldElement::new(10000)]);
        let acc = ArchivalMmr::<Hasher>::new(vec![new_leaf.clone()]);
        let mp = acc.prove_membership(0).0;
        assert!(
            calculate_new_peaks_from_leaf_mutation::<RescuePrimeProduction>(
                &vec![],
                &new_leaf,
                0,
                &mp,
            )
            .is_none()
        );
    }
}
