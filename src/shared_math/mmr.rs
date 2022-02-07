use serde::Serialize;
use std::hash::Hash;

use super::other::log_2_floor;

type HashDigest = [u8; 32];
const BLAKE3ZERO: [u8; 32] = [0u8; 32];

#[derive(Debug, Clone)]
pub struct Mmr {
    digests: Vec<HashDigest>,
}

impl Mmr {
    pub fn init<V>(data: &[V]) -> Self
    where
        V: Clone + Hash + Serialize,
    {
        let mut new_mmr: Self = Mmr {
            digests: vec![BLAKE3ZERO],
        };
        for datum in data {
            new_mmr.append_with_value(datum);
        }

        new_mmr
    }

    fn get_height_from_data_index(data_index: usize) -> usize {
        log_2_floor(data_index as u64 + 1) as usize
    }

    fn non_leaf_nodes_left(data_index: usize) -> usize {
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
            let left_data_height = Self::get_height_from_data_index(data_index_acc - 1);
            acc += (1 << left_data_height) - 1;
            data_index_acc -= 1 << left_data_height;
        }

        acc
    }

    pub fn data_index_to_node_index(data_index: usize) -> usize {
        let diff = Self::non_leaf_nodes_left(data_index);

        data_index + diff + 1
    }

    /// Convert from node index to data index in log(size) time
    pub fn node_index_to_data_index(node_index: usize) -> Option<usize> {
        let (_right, height) = Self::right_child_and_height(node_index);
        if height != 0 {
            return None;
        }

        let (mut node, mut height) = Self::leftmost_ancestor(node_index);
        let mut data_index = 0;
        while height > 0 {
            let left_child = Self::left_child(node, height);
            if node_index <= left_child {
                node = left_child;
                height -= 1;
            } else {
                node = Self::right_child(node);
                height -= 1;
                data_index += 1 << height;
            }
        }

        Some(data_index)
    }

    pub fn count_nodes(&self) -> usize {
        self.digests.len() - 1
    }

    /// Return the number of leaves in the tree
    pub fn count_leaves(&self) -> usize {
        let peaks_and_heights: Vec<(_, usize)> = self.get_peaks_with_heights();
        let mut acc = 0;
        for (_, height) in peaks_and_heights {
            acc += 1 << height
        }

        acc
    }

    pub fn verify<V: Clone + Hash + Serialize>(
        root: HashDigest,
        authentication_path: &[HashDigest],
        peaks: &[HashDigest],
        size: usize,
        value: V,
        node_index: usize,
    ) -> bool {
        // Verify that peaks match root
        let matching_root = root == Self::get_root_from_peaks(peaks, size);

        let mut hasher = blake3::Hasher::new();
        // hasher.update(&index.to_be_bytes());
        hasher.update(&bincode::serialize(&value).expect("Encoding failed"));
        let value_hash = *hasher.finalize().as_bytes();
        hasher.reset();
        let mut acc_hash = value_hash;
        let mut acc_index = node_index;
        let mut hasher = blake3::Hasher::new();
        for hash in authentication_path {
            let (acc_right, _acc_height) = Self::right_child_and_height(acc_index);
            // hasher.update(&acc_index.to_be_bytes());
            if acc_right {
                hasher.update(hash);
                hasher.update(&acc_hash);
            } else {
                hasher.update(&acc_hash);
                hasher.update(hash);
            }
            acc_hash = *hasher.finalize().as_bytes();
            hasher.reset();
            acc_index = Self::parent(acc_index);
        }

        peaks.iter().any(|peak| *peak == acc_hash) && matching_root
    }

    /// Return (authentication_path, peaks)
    pub fn get_proof(&self, node_index: usize) -> (Vec<HashDigest>, Vec<HashDigest>) {
        // A proof consists of an authentication path
        // and a list of peaks that must hash to the root

        // Find out how long the authentication path is
        let mut top_height: i32 = -1;
        let mut parent_index = node_index;
        while parent_index < self.digests.len() {
            parent_index = Self::parent(parent_index);
            top_height += 1;
        }

        // Build the authentication path
        let mut authentication_path: Vec<HashDigest> = vec![];
        let mut index = node_index;
        let (mut index_is_right_child, mut index_height) = Self::right_child_and_height(index);
        while index_height < top_height as usize {
            if index_is_right_child {
                let left_sibling_index = Self::left_sibling(index, index_height);
                authentication_path.push(self.digests[left_sibling_index]);
            } else {
                let right_sibling_index = Self::right_sibling(index, index_height);
                authentication_path.push(self.digests[right_sibling_index]);
            }
            index = Self::parent(index);
            let next_index_info = Self::right_child_and_height(index);
            index_is_right_child = next_index_info.0;
            index_height = next_index_info.1;
        }

        let peaks: Vec<HashDigest> = self.get_peaks_with_heights().iter().map(|x| x.0).collect();

        (authentication_path, peaks)
    }

    fn get_root_from_peaks(peaks: &[HashDigest], node_count: usize) -> HashDigest {
        let peaks_count: usize = peaks.len();
        let mut hasher = blake3::Hasher::new();
        hasher.update(&node_count.to_be_bytes());
        hasher.update(&peaks[peaks_count - 1]);
        let mut acc: HashDigest = *hasher.finalize().as_bytes();
        for i in 1..peaks_count {
            hasher.update(&peaks[peaks_count - 1 - i]);
            acc = *hasher.finalize().as_bytes();
            hasher.reset();
            hasher.update(&node_count.to_be_bytes());
            hasher.update(&acc);
        }

        acc
    }

    // Calculate the root for the entire MMR
    pub fn get_root(&self) -> HashDigest {
        // Follows the description for "bagging" on
        // https://github.com/mimblewimble/grin/blob/master/doc/mmr.md#hashing-and-bagging
        let peaks: Vec<HashDigest> = self.get_peaks_with_heights().iter().map(|x| x.0).collect();

        Self::get_root_from_peaks(&peaks, self.count_nodes())
    }

    /// Return a list of tuples (peaks, height)
    pub fn get_peaks_with_heights(&self) -> Vec<(HashDigest, usize)> {
        // 1. Find top peak
        // 2. Jump to right sibling (will not be included)
        // 3. Take left child of sibling, continue until a node in tree is found
        // 4. Once new node is found, jump to right sibling (will not be included)
        // 5. Take left child of sibling, continue until a node in tree is found
        let mut peaks_and_heights: Vec<(HashDigest, usize)> = vec![];
        let (mut top_peak, mut top_height) = Self::leftmost_ancestor(self.digests.len() - 1);
        if top_peak > self.digests.len() - 1 {
            top_peak = Self::left_child(top_peak, top_height);
            top_height -= 1;
        }
        peaks_and_heights.push((self.digests[top_peak], top_height)); // No clone needed bc array
        let mut height = top_height;
        let mut candidate = Self::right_sibling(top_peak, height);
        'outer: while height > 0 {
            '_inner: while candidate > self.digests.len() && height > 0 {
                candidate = Self::left_child(candidate, height);
                height -= 1;
                if candidate < self.digests.len() {
                    peaks_and_heights.push((self.digests[candidate], height));
                    candidate = Self::right_sibling(candidate, height);
                    continue 'outer;
                }
            }
        }

        peaks_and_heights
    }

    #[inline]
    fn left_child(node_index: usize, height: usize) -> usize {
        node_index - (1 << height)
    }

    #[inline]
    fn right_child(node_index: usize) -> usize {
        node_index - 1
    }

    /// Get (index, height) of leftmost ancestor
    // This ancestor does *not* have to be in the MMR
    fn leftmost_ancestor(node_index: usize) -> (usize, usize) {
        let mut h = 0;
        let mut ret = 1;
        while ret < node_index {
            h += 1;
            ret = (1 << (h + 1)) - 1;
        }

        (ret, h)
    }

    /// Return the tuple: (is_right_child, height)
    fn right_child_and_height(node_index: usize) -> (bool, usize) {
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
        let (leftmost_ancestor, ancestor_height) = Self::leftmost_ancestor(node_index);
        if leftmost_ancestor == node_index {
            return (false, ancestor_height);
        }

        let mut node = leftmost_ancestor;
        let mut height = ancestor_height;
        loop {
            let left_child = Self::left_child(node, height);
            height -= 1;
            if node_index == left_child {
                return (false, height);
            }
            if node_index < left_child {
                node = left_child;
            } else {
                let right_child = Self::right_child(node);
                if node_index == right_child {
                    return (true, height);
                }
                node = right_child;
            }
        }
    }

    fn parent(node_index: usize) -> usize {
        let (right, height) = Self::right_child_and_height(node_index);

        if right {
            node_index + 1
        } else {
            node_index + (1 << (height + 1))
        }
    }

    #[inline]
    fn left_sibling(node_index: usize, height: usize) -> usize {
        node_index - (1 << (height + 1)) + 1
    }

    #[inline]
    fn right_sibling(node_index: usize, height: usize) -> usize {
        node_index + (1 << (height + 1)) - 1
    }

    fn append_parents(&mut self, left_child: &HashDigest, right_child: &HashDigest) {
        let node_index = self.digests.len();
        let mut hasher = blake3::Hasher::new();
        // hasher.update(&own_index.to_be_bytes());
        hasher.update(left_child);
        hasher.update(right_child);
        let own_hash = *hasher.finalize().as_bytes();
        self.digests.push(own_hash);
        let (parent_needed, own_height) = Self::right_child_and_height(node_index);
        if parent_needed {
            let left_sibling_hash = self.digests[Self::left_sibling(node_index, own_height)];
            self.append_parents(&left_sibling_hash, &own_hash);
        }
    }

    pub fn append_with_value<V>(&mut self, value: &V)
    where
        V: Clone + Hash + Serialize,
    {
        let node_index = self.digests.len();
        let mut hasher = blake3::Hasher::new();
        // hasher.update(&own_index.to_be_bytes());
        hasher.update(&bincode::serialize(value).expect("Encoding failed"));
        let own_hash = *hasher.finalize().as_bytes();
        self.digests.push(own_hash);
        let (parent_needed, own_height) = Self::right_child_and_height(node_index);
        if parent_needed {
            let left_sibling_hash = self.digests[Self::left_sibling(node_index, own_height)];
            self.append_parents(&left_sibling_hash, &own_hash);
        }
    }
}

#[cfg(test)]
mod mmr_test {
    use rand::RngCore;

    use super::*;
    use crate::shared_math::b_field_element::BFieldElement;

    #[test]
    fn data_index_to_node_index_test() {
        assert_eq!(1, Mmr::data_index_to_node_index(0));
        assert_eq!(2, Mmr::data_index_to_node_index(1));
        assert_eq!(4, Mmr::data_index_to_node_index(2));
        assert_eq!(5, Mmr::data_index_to_node_index(3));
        assert_eq!(8, Mmr::data_index_to_node_index(4));
        assert_eq!(9, Mmr::data_index_to_node_index(5));
        assert_eq!(11, Mmr::data_index_to_node_index(6));
        assert_eq!(12, Mmr::data_index_to_node_index(7));
        assert_eq!(16, Mmr::data_index_to_node_index(8));
        assert_eq!(17, Mmr::data_index_to_node_index(9));
        assert_eq!(19, Mmr::data_index_to_node_index(10));
        assert_eq!(20, Mmr::data_index_to_node_index(11));
        assert_eq!(23, Mmr::data_index_to_node_index(12));
        assert_eq!(24, Mmr::data_index_to_node_index(13));
    }

    #[test]
    fn non_leaf_nodes_left_test() {
        assert_eq!(0, Mmr::non_leaf_nodes_left(0));
        assert_eq!(0, Mmr::non_leaf_nodes_left(1));
        assert_eq!(1, Mmr::non_leaf_nodes_left(2));
        assert_eq!(1, Mmr::non_leaf_nodes_left(3));
        assert_eq!(3, Mmr::non_leaf_nodes_left(4));
        assert_eq!(3, Mmr::non_leaf_nodes_left(5));
        assert_eq!(4, Mmr::non_leaf_nodes_left(6));
        assert_eq!(4, Mmr::non_leaf_nodes_left(7));
        assert_eq!(7, Mmr::non_leaf_nodes_left(8));
        assert_eq!(7, Mmr::non_leaf_nodes_left(9));
        assert_eq!(8, Mmr::non_leaf_nodes_left(10));
        assert_eq!(8, Mmr::non_leaf_nodes_left(11));
        assert_eq!(10, Mmr::non_leaf_nodes_left(12));
        assert_eq!(10, Mmr::non_leaf_nodes_left(13));
    }

    #[test]
    fn get_height_from_data_index_test() {
        assert_eq!(0, Mmr::get_height_from_data_index(0));
        assert_eq!(1, Mmr::get_height_from_data_index(1));
        assert_eq!(1, Mmr::get_height_from_data_index(2));
        assert_eq!(2, Mmr::get_height_from_data_index(3));
        assert_eq!(2, Mmr::get_height_from_data_index(4));
        assert_eq!(2, Mmr::get_height_from_data_index(5));
        assert_eq!(2, Mmr::get_height_from_data_index(6));
        assert_eq!(3, Mmr::get_height_from_data_index(7));
        assert_eq!(3, Mmr::get_height_from_data_index(8));
    }

    #[test]
    fn data_index_node_index_pbt() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let rand = rng.next_u32();
            let inversion_result =
                Mmr::node_index_to_data_index(Mmr::data_index_to_node_index(rand as usize));
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
            assert!(Mmr::right_child_and_height(i + 1).0 == *anticipation);
        }
    }

    #[test]
    fn leftmost_ancestor_test() {
        assert_eq!((1, 0), Mmr::leftmost_ancestor(1));
        assert_eq!((3, 1), Mmr::leftmost_ancestor(2));
        assert_eq!((3, 1), Mmr::leftmost_ancestor(3));
        assert_eq!((7, 2), Mmr::leftmost_ancestor(4));
        assert_eq!((7, 2), Mmr::leftmost_ancestor(5));
        assert_eq!((7, 2), Mmr::leftmost_ancestor(6));
        assert_eq!((7, 2), Mmr::leftmost_ancestor(7));
        assert_eq!((15, 3), Mmr::leftmost_ancestor(8));
        assert_eq!((15, 3), Mmr::leftmost_ancestor(9));
        assert_eq!((15, 3), Mmr::leftmost_ancestor(10));
        assert_eq!((15, 3), Mmr::leftmost_ancestor(11));
        assert_eq!((15, 3), Mmr::leftmost_ancestor(12));
        assert_eq!((15, 3), Mmr::leftmost_ancestor(13));
        assert_eq!((15, 3), Mmr::leftmost_ancestor(14));
        assert_eq!((15, 3), Mmr::leftmost_ancestor(15));
        assert_eq!((31, 4), Mmr::leftmost_ancestor(16));
    }

    #[test]
    fn left_sibling_test() {
        assert_eq!(3, Mmr::left_sibling(6, 1));
        assert_eq!(1, Mmr::left_sibling(2, 0));
        assert_eq!(4, Mmr::left_sibling(5, 0));
        assert_eq!(15, Mmr::left_sibling(30, 3));
        assert_eq!(22, Mmr::left_sibling(29, 2));
        assert_eq!(7, Mmr::left_sibling(14, 2));
    }

    #[test]
    fn node_index_to_data_index_test() {
        assert_eq!(Some(0), Mmr::node_index_to_data_index(1));
        assert_eq!(Some(1), Mmr::node_index_to_data_index(2));
        assert_eq!(None, Mmr::node_index_to_data_index(3));
        assert_eq!(Some(2), Mmr::node_index_to_data_index(4));
        assert_eq!(Some(3), Mmr::node_index_to_data_index(5));
        assert_eq!(None, Mmr::node_index_to_data_index(6));
        assert_eq!(None, Mmr::node_index_to_data_index(7));
        assert_eq!(Some(4), Mmr::node_index_to_data_index(8));
        assert_eq!(Some(5), Mmr::node_index_to_data_index(9));
        assert_eq!(None, Mmr::node_index_to_data_index(10));
        assert_eq!(Some(6), Mmr::node_index_to_data_index(11));
        assert_eq!(Some(7), Mmr::node_index_to_data_index(12));
        assert_eq!(None, Mmr::node_index_to_data_index(13));
        assert_eq!(None, Mmr::node_index_to_data_index(14));
        assert_eq!(None, Mmr::node_index_to_data_index(15));
        assert_eq!(Some(8), Mmr::node_index_to_data_index(16));
        assert_eq!(Some(9), Mmr::node_index_to_data_index(17));
        assert_eq!(None, Mmr::node_index_to_data_index(18));
        assert_eq!(Some(10), Mmr::node_index_to_data_index(19));
        assert_eq!(Some(11), Mmr::node_index_to_data_index(20));
        assert_eq!(None, Mmr::node_index_to_data_index(21));
        assert_eq!(None, Mmr::node_index_to_data_index(22));
    }

    #[test]
    fn one_input_mmr_test() {
        let element = BFieldElement::new(14);
        let input: Vec<BFieldElement> = vec![element];
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(1, mmr.count_leaves());
        assert_eq!(1, mmr.count_nodes());
        let peaks_and_heights = mmr.get_peaks_with_heights();
        assert_eq!(1, peaks_and_heights.len());
        assert_eq!(0, peaks_and_heights[0].1);
        let (authentication_path, peaks) = mmr.get_proof(1);
        let root = mmr.get_root();

        let valid = Mmr::verify(root, &authentication_path, &peaks, 1, element, 1);
        assert!(valid);
    }

    #[test]
    fn two_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..2).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(2, mmr.count_leaves());
        assert_eq!(3, mmr.count_nodes());
        let peaks = mmr.get_peaks_with_heights();
        assert_eq!(1, peaks.len());
        let (authentication_path, peaks) = mmr.get_proof(1);
        let root = mmr.get_root();
        let size = mmr.count_nodes();

        let valid = Mmr::verify(
            root,
            &authentication_path,
            &peaks,
            size,
            BFieldElement::ring_zero(),
            1,
        );
        assert!(valid);
    }

    #[test]
    fn three_input_mmr_test() {
        let input: Vec<BFieldElement> = (4..=6).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(3, mmr.count_leaves());
        assert_eq!(4, mmr.count_nodes());
        let peaks = mmr.get_peaks_with_heights();
        assert_eq!(2, peaks.len());

        for index in vec![1, 2, 3] {
            let (authentication_path, peaks) = mmr.get_proof(index);
            let valid = Mmr::verify(
                mmr.get_root(),
                &authentication_path,
                &peaks,
                mmr.count_nodes(),
                BFieldElement::new(index as u128 + 3),
                index,
            );
            assert!(valid);
        }
    }

    #[test]
    fn four_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..4).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(4, mmr.count_leaves());
        assert_eq!(7, mmr.count_nodes());
        let peaks = mmr.get_peaks_with_heights();
        assert_eq!(1, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.count_nodes(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn fiveinput_mmr_test() {
        let input: Vec<BFieldElement> = (0..5).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(5, mmr.count_leaves());
        assert_eq!(8, mmr.count_nodes());
        let peaks = mmr.get_peaks_with_heights();
        assert_eq!(2, peaks.len());

        for (i, index) in vec![1, 2, 4, 5, 8].iter().enumerate() {
            let (authentication_path, peaks) = mmr.get_proof(*index);
            let valid = Mmr::verify(
                mmr.get_root(),
                &authentication_path,
                &peaks,
                mmr.count_nodes(),
                input[i],
                *index,
            );
            assert!(valid);
        }
    }

    #[test]
    fn sixinput_mmr_test() {
        let input: Vec<BFieldElement> = (0..6).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(6, mmr.count_leaves());
        assert_eq!(10, mmr.count_nodes());
        let peaks = mmr.get_peaks_with_heights();
        assert_eq!(2, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.count_nodes(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn seven_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..7).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(7, mmr.count_leaves());
        assert_eq!(11, mmr.count_nodes());
        let peaks = mmr.get_peaks_with_heights();
        assert_eq!(3, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.count_nodes(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn eight_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..8).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(8, mmr.count_leaves());
        assert_eq!(15, mmr.count_nodes());
        let peaks = mmr.get_peaks_with_heights();
        assert_eq!(1, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.count_nodes(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn nine_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..9).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(16, mmr.count_nodes());
        let peaks = mmr.get_peaks_with_heights();
        assert_eq!(2, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.count_nodes(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn ten_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..10).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(18, mmr.count_nodes());
        let peaks = mmr.get_peaks_with_heights();
        assert_eq!(2, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.count_nodes(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn eleven_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..11).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(19, mmr.count_nodes());
        let peaks = mmr.get_peaks_with_heights();
        assert_eq!(3, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.count_nodes(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn twelve_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..12).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(22, mmr.count_nodes());
        let peaks = mmr.get_peaks_with_heights();
        assert_eq!(2, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.count_nodes(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn thirteen_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..13).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(23, mmr.count_nodes());
        let peaks = mmr.get_peaks_with_heights();
        assert_eq!(3, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.count_nodes(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn fourteen_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..14).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(25, mmr.count_nodes());
        let peaks = mmr.get_peaks_with_heights();
        assert_eq!(3, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.count_nodes(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn fifteen_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..15).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(26, mmr.count_nodes());
        let peaks = mmr.get_peaks_with_heights();
        assert_eq!(4, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.count_nodes(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn sixteen_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..16).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(31, mmr.count_nodes());
        let peaks = mmr.get_peaks_with_heights();
        assert_eq!(1, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.count_nodes(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn seventeen_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..17).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(32, mmr.count_nodes());
        let peaks = mmr.get_peaks_with_heights();
        assert_eq!(2, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.count_nodes(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn eighteen_input_mmr_test() {
        let input: Vec<BFieldElement> = (10..28).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(34, mmr.count_nodes());
        let peaks = mmr.get_peaks_with_heights();
        assert_eq!(2, peaks.len());

        let index = 16;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.count_nodes(),
            BFieldElement::new(18),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn nineteen_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..19).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(35, mmr.count_nodes());
        let peaks = mmr.get_peaks_with_heights();
        assert_eq!(3, peaks.len());

        let (authentication_path, peaks) = mmr.get_proof(32);
        assert_eq!(1, authentication_path.len());
        assert_eq!(3, peaks.len());
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.count_nodes(),
            BFieldElement::new(16),
            32,
        );
        assert!(valid);

        for (i, index) in vec![1, 2, 4, 5, 8, 9, 11, 12, 16, 17, 19, 20, 23, 24, 26, 27, 32]
            .iter()
            .enumerate()
        {
            let (authentication_path, peaks) = mmr.get_proof(*index as usize);
            let valid = Mmr::verify(
                mmr.get_root(),
                &authentication_path,
                &peaks,
                mmr.count_nodes(),
                input[i],
                *index as usize,
            );
            assert!(valid);
        }
    }
}
