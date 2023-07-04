use super::shared_basic::*;
use crate::shared_math::other::{indices_of_set_bits, log_2_floor};

/// Get (index, height) of leftmost ancestor
/// This ancestor does *not* have to be in the MMR
/// This algorithm finds the closest $2^n - 1$ that's bigger than
/// or equal to `node_index`.
#[inline]
pub fn leftmost_ancestor(node_index: u64) -> (u64, u32) {
    let h = u64::BITS - node_index.leading_zeros() - 1;
    let ret = (1 << (h + 1)) - 1;

    (ret, h)
}

/// Traversing from this node upwards, count how many of the ancestor (including itself)
/// is a right child. Also returns node's height.
pub fn right_lineage_length_and_own_height(node_index: u64) -> (u32, u32) {
    let (mut candidate, mut candidate_height) = leftmost_ancestor(node_index);

    // leftmost ancestor is always a left node, so count starts at 0.
    let mut right_ancestor_count = 0;

    loop {
        if candidate == node_index {
            return (right_ancestor_count, candidate_height);
        }

        let left_child = left_child(candidate, candidate_height);
        let candidate_is_right_child = left_child < node_index;
        if candidate_is_right_child {
            candidate = right_child(candidate);
            right_ancestor_count += 1;
        } else {
            candidate = left_child;
            right_ancestor_count = 0;
        };

        candidate_height -= 1;
    }
}

pub fn right_lineage_length_from_node_index(node_index: u64) -> u32 {
    let bit_width = u64::BITS - node_index.leading_zeros();
    let npo2 = 1 << bit_width;

    let dist = npo2 - node_index;

    if (bit_width as u64) < dist {
        right_lineage_length_from_node_index(node_index - (npo2 >> 1) + 1)
    } else {
        (dist - 1) as u32
    }
}

/// Convert from leaf index to node index
pub fn leaf_index_to_node_index(leaf_index: u64) -> u64 {
    let hamming_weight = leaf_index.count_ones() as u64;
    2 * leaf_index - hamming_weight + 1
}

/// Get the node_index of the parent
#[inline]
pub fn parent(node_index: u64) -> u64 {
    let (right_ancestor_count, height) = right_lineage_length_and_own_height(node_index);

    if right_ancestor_count != 0 {
        node_index + 1
    } else {
        node_index + (1 << (height + 1))
    }
}

#[inline]
pub fn left_sibling(node_index: u64, height: u32) -> u64 {
    node_index - (1 << (height + 1)) + 1
}

#[inline]
pub fn right_sibling(node_index: u64, height: u32) -> u64 {
    node_index + (1 << (height + 1)) - 1
}

pub fn get_height_from_leaf_index(leaf_index: u64) -> u32 {
    // This should be a safe cast as 2^(u32::MAX) is a *very* big number
    log_2_floor(leaf_index as u128 + 1) as u32
}

/// The number of nodes in an MMR with `leaf_count` leaves.
pub fn leaf_count_to_node_count(leaf_count: u64) -> u64 {
    let hamming_weight = leaf_count.count_ones() as u64;
    2 * leaf_count - hamming_weight
}

/// Return the indices of the nodes added by an append, including the
/// peak that this append gave rise to
pub fn node_indices_added_by_append(old_leaf_count: u64) -> Vec<u64> {
    let mut node_index = leaf_index_to_node_index(old_leaf_count);
    let mut added_node_indices = vec![node_index];
    let mut right_count = right_lineage_length_from_node_index(node_index);
    while right_count != 0 {
        // a right child's parent is found by adding 1 to the node index
        node_index += 1;
        added_node_indices.push(node_index);
        right_count -= 1;
    }

    added_node_indices
}

/// Get the node indices of the authentication path hash digest needed
/// to calculate the digest of `peak_node_index` from `start_node_index`
pub fn get_authentication_path_node_indices(
    start_node_index: u64,
    peak_node_index: u64,
    node_count: u64,
) -> Option<Vec<u64>> {
    let mut authentication_path_node_indices = vec![];
    let mut node_index = start_node_index;
    while node_index <= node_count && node_index != peak_node_index {
        // TODO: Consider if this function can be written better, or discard
        // it entirely.
        let (right_ancestor_count, height) = right_lineage_length_and_own_height(node_index);
        let sibling_node_index: u64;
        if right_ancestor_count != 0 {
            sibling_node_index = left_sibling(node_index, height);

            // parent of right child is +1
            node_index += 1;
        } else {
            sibling_node_index = right_sibling(node_index, height);

            // parent of left child:
            node_index += 1 << (height + 1);
        }

        authentication_path_node_indices.push(sibling_node_index);
    }

    if node_index == peak_node_index {
        Some(authentication_path_node_indices)
    } else {
        None
    }
}

/// Return a list of the peak heights for a given leaf count
pub fn get_peak_heights(leaf_count: u64) -> Vec<u8> {
    // The peak heights in an MMR can be read directly from the bit-decomposition
    // of the leaf count.
    indices_of_set_bits(leaf_count)
}

/// Given leaf count, return a vector representing the height of
/// the peaks. Input is the number of leafs in the MMR
pub fn get_peak_heights_and_peak_node_indices(leaf_count: u64) -> (Vec<u32>, Vec<u64>) {
    if leaf_count == 0 {
        return (vec![], vec![]);
    }

    let node_index_of_rightmost_leaf = leaf_index_to_node_index(leaf_count - 1);
    let node_count = leaf_count_to_node_count(leaf_count);
    let (mut top_peak, mut top_height) = leftmost_ancestor(node_index_of_rightmost_leaf);
    if top_peak > node_count {
        top_peak = left_child(top_peak, top_height);
        top_height -= 1;
    }

    let mut heights: Vec<u32> = vec![top_height];
    let mut node_indices: Vec<u64> = vec![top_peak];
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

/// Convert from node index to leaf index in log(size) time
pub fn node_index_to_leaf_index(node_index: u64) -> Option<u64> {
    let (_right, own_height) = right_lineage_length_and_own_height(node_index);
    if own_height != 0 {
        return None;
    }

    let (mut node, mut node_height) = leftmost_ancestor(node_index);
    let mut leaf_index = 0;
    while node_height > 0 {
        let left_child = left_child(node, node_height);
        if node_index <= left_child {
            node = left_child;
            node_height -= 1;
        } else {
            node = right_child(node);
            node_height -= 1;
            leaf_index += 1 << node_height;
        }
    }

    Some(leaf_index)
}

#[cfg(test)]
mod mmr_test {
    use rand::RngCore;

    use super::*;

    #[test]
    fn get_height_from_leaf_index_test() {
        assert_eq!(0, get_height_from_leaf_index(0));
        assert_eq!(1, get_height_from_leaf_index(1));
        assert_eq!(1, get_height_from_leaf_index(2));
        assert_eq!(2, get_height_from_leaf_index(3));
        assert_eq!(2, get_height_from_leaf_index(4));
        assert_eq!(2, get_height_from_leaf_index(5));
        assert_eq!(2, get_height_from_leaf_index(6));
        assert_eq!(3, get_height_from_leaf_index(7));
        assert_eq!(3, get_height_from_leaf_index(8));
    }

    #[test]
    fn leaf_index_to_node_index_test() {
        assert_eq!(1, leaf_index_to_node_index(0));
        assert_eq!(2, leaf_index_to_node_index(1));
        assert_eq!(4, leaf_index_to_node_index(2));
        assert_eq!(5, leaf_index_to_node_index(3));
        assert_eq!(8, leaf_index_to_node_index(4));
        assert_eq!(9, leaf_index_to_node_index(5));
        assert_eq!(11, leaf_index_to_node_index(6));
        assert_eq!(12, leaf_index_to_node_index(7));
        assert_eq!(16, leaf_index_to_node_index(8));
        assert_eq!(17, leaf_index_to_node_index(9));
        assert_eq!(19, leaf_index_to_node_index(10));
        assert_eq!(20, leaf_index_to_node_index(11));
        assert_eq!(23, leaf_index_to_node_index(12));
        assert_eq!(24, leaf_index_to_node_index(13));
        assert_eq!(26, leaf_index_to_node_index(14));
        assert_eq!(27, leaf_index_to_node_index(15));
        assert_eq!(32, leaf_index_to_node_index(16));
        assert_eq!(33, leaf_index_to_node_index(17));
        assert_eq!(35, leaf_index_to_node_index(18));
        assert_eq!(36, leaf_index_to_node_index(19));
        assert_eq!(39, leaf_index_to_node_index(20));
        assert_eq!(40, leaf_index_to_node_index(21));
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
    fn leaf_index_node_index_pbt() {
        let mut rng = rand::thread_rng();
        for _ in 0..10000 {
            let rand = rng.next_u32();
            let inversion_result = node_index_to_leaf_index(leaf_index_to_node_index(rand as u64));
            match inversion_result {
                None => panic!(),
                Some(inversion) => assert_eq!(rand, inversion as u32),
            }
        }
    }

    #[test]
    fn right_ancestor_count_test() {
        assert_eq!((0, 0), right_lineage_length_and_own_height(1)); // 0b1 => 0
        assert_eq!((1, 0), right_lineage_length_and_own_height(2)); // 0b10 => 1
        assert_eq!((0, 1), right_lineage_length_and_own_height(3)); // 0b11 => 0
        assert_eq!((0, 0), right_lineage_length_and_own_height(4)); // 0b100 => 0
        assert_eq!((2, 0), right_lineage_length_and_own_height(5)); // 0b101 => 2
        assert_eq!((1, 1), right_lineage_length_and_own_height(6)); // 0b110 => 1
        assert_eq!((0, 2), right_lineage_length_and_own_height(7)); // 0b111 => 0
        assert_eq!((0, 0), right_lineage_length_and_own_height(8)); // 0b1000 => 0
        assert_eq!((1, 0), right_lineage_length_and_own_height(9)); // 0b1001 => 1
        assert_eq!((0, 1), right_lineage_length_and_own_height(10)); // 0b1010 => 0
        assert_eq!((0, 0), right_lineage_length_and_own_height(11)); // 0b1011 => 0
        assert_eq!((3, 0), right_lineage_length_and_own_height(12)); // 0b1100 => 3
        assert_eq!((2, 1), right_lineage_length_and_own_height(13)); // 0b1101 => 2
        assert_eq!((1, 2), right_lineage_length_and_own_height(14)); // 0b1110 => 1
        assert_eq!((0, 3), right_lineage_length_and_own_height(15)); // 0b1111 => 0
        assert_eq!((0, 0), right_lineage_length_and_own_height(16)); // 0b10000 => 0
        assert_eq!((1, 0), right_lineage_length_and_own_height(17)); // 0b10001 => 1
        assert_eq!((0, 1), right_lineage_length_and_own_height(18)); // 0b10010 => 0
        assert_eq!((0, 0), right_lineage_length_and_own_height(19)); // 0b10011 => 0
        assert_eq!((2, 0), right_lineage_length_and_own_height(20)); // 0b10100 => 2
        assert_eq!((1, 1), right_lineage_length_and_own_height(21)); // 0b10101 => 1
        assert_eq!((0, 2), right_lineage_length_and_own_height(22)); // 0b10110 => 0
        assert_eq!((0, 0), right_lineage_length_and_own_height(23)); // 0b10111 => 0
        assert_eq!((1, 0), right_lineage_length_and_own_height(24)); // 0b11000 => 1
        assert_eq!((0, 1), right_lineage_length_and_own_height(25)); // 0b11001 => 0
        assert_eq!((0, 0), right_lineage_length_and_own_height(26)); // 0b11010 => 0
        assert_eq!((4, 0), right_lineage_length_and_own_height(27)); // 0b11011 => 4
        assert_eq!((3, 1), right_lineage_length_and_own_height(28)); // 0b11100 => 3
        assert_eq!((2, 2), right_lineage_length_and_own_height(29)); // 0b11101 => 2
        assert_eq!((1, 3), right_lineage_length_and_own_height(30)); // 0b11110 => 1
        assert_eq!((0, 4), right_lineage_length_and_own_height(31)); // 0b11111 => 0
        assert_eq!((0, 0), right_lineage_length_and_own_height(32)); // 0b100000 => 0
        assert_eq!((1, 0), right_lineage_length_and_own_height(33)); // 0b100001 => 1
        assert_eq!((0, 1), right_lineage_length_and_own_height(34)); // 0b100010 => 0
        assert_eq!((0, 0), right_lineage_length_and_own_height(35)); // 0b100011 => 0
        assert_eq!((2, 0), right_lineage_length_and_own_height(36)); // 0b100100 => 2
        assert_eq!((1, 1), right_lineage_length_and_own_height(37)); // 0b100101 => 1
        assert_eq!((0, 2), right_lineage_length_and_own_height(38)); // 0b100110 => 0
        assert_eq!((0, 0), right_lineage_length_and_own_height(39)); // 0b100111 => 0
        assert_eq!((1, 0), right_lineage_length_and_own_height(40)); // 0b101000 => 1
        assert_eq!((0, 1), right_lineage_length_and_own_height(41)); // 0b101001 => 0

        assert_eq!(
            (61, 1),
            right_lineage_length_and_own_height(u64::MAX / 2 - 61)
        ); // 0b111...11 => 0
        assert_eq!(
            (3, 59),
            right_lineage_length_and_own_height(u64::MAX / 2 - 3)
        ); // 0b111...11 => 0
        assert_eq!(
            (2, 60),
            right_lineage_length_and_own_height(u64::MAX / 2 - 2)
        ); // 0b111...11 => 0
        assert_eq!(
            (1, 61),
            right_lineage_length_and_own_height(u64::MAX / 2 - 1)
        ); // 0b111...11 => 0
        assert_eq!((0, 62), right_lineage_length_and_own_height(u64::MAX / 2)); // 0b111...11 => 0
    }

    #[test]
    fn right_lineage_length_pbt() {
        let mut rng = rand::thread_rng();
        for _ in 0..10000 {
            let rand_leaf_index = rng.next_u64() / 4;
            println!("{rand_leaf_index}");
            let rll = right_lineage_length_from_leaf_index(rand_leaf_index);
            let rac =
                right_lineage_length_and_own_height(leaf_index_to_node_index(rand_leaf_index)).0;
            assert_eq!(rac, rll);
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
    fn node_index_to_leaf_index_test() {
        assert_eq!(Some(0), node_index_to_leaf_index(1));
        assert_eq!(Some(1), node_index_to_leaf_index(2));
        assert_eq!(None, node_index_to_leaf_index(3));
        assert_eq!(Some(2), node_index_to_leaf_index(4));
        assert_eq!(Some(3), node_index_to_leaf_index(5));
        assert_eq!(None, node_index_to_leaf_index(6));
        assert_eq!(None, node_index_to_leaf_index(7));
        assert_eq!(Some(4), node_index_to_leaf_index(8));
        assert_eq!(Some(5), node_index_to_leaf_index(9));
        assert_eq!(None, node_index_to_leaf_index(10));
        assert_eq!(Some(6), node_index_to_leaf_index(11));
        assert_eq!(Some(7), node_index_to_leaf_index(12));
        assert_eq!(None, node_index_to_leaf_index(13));
        assert_eq!(None, node_index_to_leaf_index(14));
        assert_eq!(None, node_index_to_leaf_index(15));
        assert_eq!(Some(8), node_index_to_leaf_index(16));
        assert_eq!(Some(9), node_index_to_leaf_index(17));
        assert_eq!(None, node_index_to_leaf_index(18));
        assert_eq!(Some(10), node_index_to_leaf_index(19));
        assert_eq!(Some(11), node_index_to_leaf_index(20));
        assert_eq!(None, node_index_to_leaf_index(21));
        assert_eq!(None, node_index_to_leaf_index(22));
    }

    #[test]
    fn leaf_count_to_node_count_test() {
        let node_counts: Vec<u64> = vec![
            0, 1, 3, 4, 7, 8, 10, 11, 15, 16, 18, 19, 22, 23, 25, 26, 31, 32, 34, 35, 38, 39, 41,
            42, 46, 47, 49, 50, 53, 54, 56, 57, 63, 64,
        ];
        for (i, node_count) in node_counts.iter().enumerate() {
            assert_eq!(*node_count, leaf_count_to_node_count(i as u64));
        }
    }

    #[test]
    fn get_peak_heights_and_peak_node_indices_test() {
        type TestCase = (u64, (Vec<u32>, Vec<u64>));
        let leaf_count_and_expected: Vec<TestCase> = vec![
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
                (expected_heights.clone(), expected_indices),
                get_peak_heights_and_peak_node_indices(leaf_count)
            );

            assert_eq!(
                expected_heights
                    .iter()
                    .map(|x| *x as u8)
                    .collect::<Vec<_>>(),
                get_peak_heights(leaf_count)
            );
        }
    }

    #[test]
    fn get_authentication_path_node_indices_test() {
        type Interval = (u64, u64, u64);
        type TestCase = (Interval, Option<Vec<u64>>);
        let start_end_node_count_expected: Vec<TestCase> = vec![
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
}
