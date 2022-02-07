use serde::Serialize;
use std::hash::Hash;

type HashDigest = [u8; 32];
const BLAKE3ZERO: [u8; 32] = [0u8; 32];

#[derive(Debug, Clone)]
pub struct Mmr {
    digests: Vec<[u8; 32]>,
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

    pub fn get_digest_count(&self) -> usize {
        self.digests.len()
    }

    pub fn verify<V: Clone + Hash + Serialize>(
        root: HashDigest,
        authentication_path: &[HashDigest],
        peaks: &[HashDigest],
        size: usize,
        value: V,
        index: usize,
    ) -> bool {
        // Verify that peaks match root
        let matching_root = root == Self::get_root_from_peaks(peaks, size);

        let mut hasher = blake3::Hasher::new();
        // hasher.update(&index.to_be_bytes());
        hasher.update(&bincode::serialize(&value).expect("Encoding failed"));
        let value_hash = *hasher.finalize().as_bytes();
        hasher.reset();
        let mut acc_hash = value_hash;
        let mut acc_index = index;
        let mut hasher = blake3::Hasher::new();
        for hash in authentication_path {
            let (acc_right, _acc_height) = Self::is_right_child(acc_index);
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
    pub fn get_proof(&self, n: usize) -> (Vec<HashDigest>, Vec<HashDigest>) {
        // A proof consists of an authentication path
        // and a list of peaks that must hash to the root

        // Find out how long the authentication path is
        let mut top_height: i32 = -1;
        let mut parent_index = n;
        while parent_index < self.digests.len() {
            parent_index = Self::parent(parent_index);
            top_height += 1;
        }

        // Build the authentication path
        let mut authentication_path: Vec<HashDigest> = vec![];
        let mut index = n;
        let (mut index_is_right_child, mut index_height) = Self::is_right_child(index);
        while index_height < top_height as usize {
            if index_is_right_child {
                let left_sibling_index = Self::left_sibling(index, index_height);
                authentication_path.push(self.digests[left_sibling_index]);
            } else {
                let right_sibling_index = Self::right_sibling(index, index_height);
                authentication_path.push(self.digests[right_sibling_index]);
            }
            index = Self::parent(index);
            let next_index_info = Self::is_right_child(index);
            index_is_right_child = next_index_info.0;
            index_height = next_index_info.1;
        }

        (authentication_path, self.get_peaks())
    }

    fn get_root_from_peaks(peaks: &[HashDigest], size: usize) -> HashDigest {
        let peaks_count: usize = peaks.len();
        let mut hasher = blake3::Hasher::new();
        hasher.update(&size.to_be_bytes());
        hasher.update(&peaks[peaks_count - 1]);
        let mut acc: HashDigest = *hasher.finalize().as_bytes();
        for i in 1..peaks_count {
            hasher.update(&peaks[peaks_count - 1 - i]);
            acc = *hasher.finalize().as_bytes();
            hasher.reset();
            hasher.update(&size.to_be_bytes());
            hasher.update(&acc);
        }

        acc
    }

    // Calculate the root for the entire MMR
    pub fn get_root(&self) -> HashDigest {
        // Follows the description for "bagging" on
        // https://github.com/mimblewimble/grin/blob/master/doc/mmr.md#hashing-and-bagging
        let peaks: Vec<[u8; 32]> = self.get_peaks();

        Self::get_root_from_peaks(&peaks, self.get_digest_count())
    }

    pub fn get_peaks(&self) -> Vec<HashDigest> {
        // 1. Find top peak
        // 2. Jump to right sibling (will not be included)
        // 3. Take left child of sibling, continue until a node in tree is found
        // 4. Once new node is found, jump to right sibling (will not be included)
        // 5. Take left child of sibling, continue until a node in tree is found
        let mut peaks: Vec<HashDigest> = vec![];
        let (mut top_peak, mut top_height) = Self::own_leftmost_peak(self.digests.len() - 1);
        if top_peak > self.digests.len() - 1 {
            top_peak = Self::left_child(top_peak, top_height);
            top_height -= 1;
        }
        peaks.push(self.digests[top_peak]); // No clone needed bc array
        let mut height = top_height;
        let mut candidate = Self::right_sibling(top_peak, height);
        'outer: while height > 0 {
            '_inner: while candidate > self.digests.len() && height > 0 {
                candidate = Self::left_child(candidate, height);
                height -= 1;
                if candidate < self.digests.len() {
                    peaks.push(self.digests[candidate]);
                    candidate = Self::right_sibling(candidate, height);
                    continue 'outer;
                }
            }
        }

        peaks
    }

    #[inline]
    fn left_child(own_index: usize, own_height: usize) -> usize {
        own_index - (1 << own_height)
    }

    #[inline]
    fn right_child(own_index: usize) -> usize {
        own_index - 1
    }

    // Get index and height of own leftmost peak
    fn own_leftmost_peak(own_index: usize) -> (usize, usize) {
        let mut h = 0;
        let mut ret = 1;
        while ret < own_index {
            h += 1;
            ret = (1 << (h + 1)) - 1;
        }

        (ret, h)
    }

    fn is_right_child(n: usize) -> (bool, usize) {
        // 1. Find own_leftmost_peak(n), if own_leftmost_peak(n) == n => left_child (false)
        // 2. Let node = own_leftmost_peak(n)
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
        let (own_leftmost_peak, peak_height) = Self::own_leftmost_peak(n);
        if own_leftmost_peak == n {
            return (false, peak_height);
        }

        let mut node = own_leftmost_peak;
        let mut height = peak_height;
        loop {
            let left_child = Self::left_child(node, height);
            height -= 1;
            if n == left_child {
                return (false, height);
            }
            if n < left_child {
                node = left_child;
            } else {
                let right_child = Self::right_child(node);
                if n == right_child {
                    return (true, height);
                }
                node = right_child;
            }
        }
    }

    fn parent(n: usize) -> usize {
        let (right, height) = Self::is_right_child(n);

        if right {
            n + 1
        } else {
            n + (1 << (height + 1))
        }
    }

    #[inline]
    fn left_sibling(own_index: usize, own_height: usize) -> usize {
        own_index - (1 << (own_height + 1)) + 1
    }

    #[inline]
    fn right_sibling(own_index: usize, own_height: usize) -> usize {
        own_index + (1 << (own_height + 1)) - 1
    }

    fn append_parents(&mut self, left_child: &HashDigest, right_child: &HashDigest) {
        let own_index = self.digests.len();
        let mut hasher = blake3::Hasher::new();
        // hasher.update(&own_index.to_be_bytes());
        hasher.update(left_child);
        hasher.update(right_child);
        let own_hash = *hasher.finalize().as_bytes();
        self.digests.push(own_hash);
        let (parent_needed, own_height) = Self::is_right_child(own_index);
        if parent_needed {
            let left_sibling_hash = self.digests[Self::left_sibling(own_index, own_height)];
            self.append_parents(&left_sibling_hash, &own_hash);
        }
    }

    pub fn append_with_value<V>(&mut self, value: &V)
    where
        V: Clone + Hash + Serialize,
    {
        let own_index = self.digests.len();
        let mut hasher = blake3::Hasher::new();
        // hasher.update(&own_index.to_be_bytes());
        hasher.update(&bincode::serialize(value).expect("Encoding failed"));
        let own_hash = *hasher.finalize().as_bytes();
        self.digests.push(own_hash);
        let (parent_needed, own_height) = Self::is_right_child(own_index);
        if parent_needed {
            let left_sibling_hash = self.digests[Self::left_sibling(own_index, own_height)];
            self.append_parents(&left_sibling_hash, &own_hash);
        }
    }
}

#[cfg(test)]
mod mmr_test {
    use super::*;
    use crate::shared_math::b_field_element::BFieldElement;

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
            assert!(Mmr::is_right_child(i + 1).0 == *anticipation);
        }
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
    fn one_input_mmr_test() {
        let element = BFieldElement::new(14);
        let input: Vec<BFieldElement> = vec![element];
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(2, mmr.get_digest_count());
        let peaks = mmr.get_peaks();
        assert_eq!(1, peaks.len());
        let (authentication_path, peaks) = mmr.get_proof(1);
        let root = mmr.get_root();
        let size = mmr.get_digest_count();

        let valid = Mmr::verify(root, &authentication_path, &peaks, size, element, 1);
        assert!(valid);
    }

    #[test]
    fn two_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..2).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(4, mmr.get_digest_count());
        let peaks = mmr.get_peaks();
        assert_eq!(1, peaks.len());
        let (authentication_path, peaks) = mmr.get_proof(1);
        let root = mmr.get_root();
        let size = mmr.get_digest_count();

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
        assert_eq!(5, mmr.get_digest_count());
        let peaks = mmr.get_peaks();
        assert_eq!(2, peaks.len());

        for index in vec![1, 2, 3] {
            let (authentication_path, peaks) = mmr.get_proof(index);
            let valid = Mmr::verify(
                mmr.get_root(),
                &authentication_path,
                &peaks,
                mmr.get_digest_count(),
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
        assert_eq!(8, mmr.get_digest_count());
        let peaks = mmr.get_peaks();
        assert_eq!(1, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.get_digest_count(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn fiveinput_mmr_test() {
        let input: Vec<BFieldElement> = (0..5).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(9, mmr.get_digest_count());
        let peaks = mmr.get_peaks();
        assert_eq!(2, peaks.len());

        for (i, index) in vec![1, 2, 4, 5, 8].iter().enumerate() {
            let (authentication_path, peaks) = mmr.get_proof(*index);
            let valid = Mmr::verify(
                mmr.get_root(),
                &authentication_path,
                &peaks,
                mmr.get_digest_count(),
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
        assert_eq!(11, mmr.get_digest_count());
        let peaks = mmr.get_peaks();
        assert_eq!(2, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.get_digest_count(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn seven_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..7).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(12, mmr.get_digest_count());
        let peaks = mmr.get_peaks();
        assert_eq!(3, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.get_digest_count(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn eight_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..8).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(16, mmr.get_digest_count());
        let peaks = mmr.get_peaks();
        assert_eq!(1, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.get_digest_count(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn nine_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..9).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(17, mmr.get_digest_count());
        let peaks = mmr.get_peaks();
        assert_eq!(2, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.get_digest_count(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn ten_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..10).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(19, mmr.get_digest_count());
        let peaks = mmr.get_peaks();
        assert_eq!(2, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.get_digest_count(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn eleven_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..11).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(20, mmr.get_digest_count());
        let peaks = mmr.get_peaks();
        assert_eq!(3, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.get_digest_count(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn twelve_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..12).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(23, mmr.get_digest_count());
        let peaks = mmr.get_peaks();
        assert_eq!(2, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.get_digest_count(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn thirteen_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..13).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(24, mmr.get_digest_count());
        let peaks = mmr.get_peaks();
        assert_eq!(3, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.get_digest_count(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn fourteen_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..14).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(26, mmr.get_digest_count());
        let peaks = mmr.get_peaks();
        assert_eq!(3, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.get_digest_count(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn fifteen_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..15).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(27, mmr.get_digest_count());
        let peaks = mmr.get_peaks();
        assert_eq!(4, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.get_digest_count(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn sixteen_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..16).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(32, mmr.get_digest_count());
        let peaks = mmr.get_peaks();
        assert_eq!(1, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.get_digest_count(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn seventeen_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..17).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(33, mmr.get_digest_count());
        let peaks = mmr.get_peaks();
        assert_eq!(2, peaks.len());

        let index = 1;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.get_digest_count(),
            BFieldElement::ring_zero(),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn eighteen_input_mmr_test() {
        let input: Vec<BFieldElement> = (10..28).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(35, mmr.get_digest_count());
        let peaks = mmr.get_peaks();
        assert_eq!(2, peaks.len());

        let index = 16;
        let (authentication_path, peaks) = mmr.get_proof(index);
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.get_digest_count(),
            BFieldElement::new(18),
            index,
        );
        assert!(valid);
    }

    #[test]
    fn nineteen_input_mmr_test() {
        let input: Vec<BFieldElement> = (0..19).map(BFieldElement::new).collect();
        let mmr: Mmr = Mmr::init(&input);
        assert_eq!(36, mmr.get_digest_count());
        let peaks = mmr.get_peaks();
        assert_eq!(3, peaks.len());

        let (authentication_path, peaks) = mmr.get_proof(32);
        assert_eq!(1, authentication_path.len());
        assert_eq!(3, peaks.len());
        let valid = Mmr::verify(
            mmr.get_root(),
            &authentication_path,
            &peaks,
            mmr.get_digest_count(),
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
                mmr.get_digest_count(),
                input[i],
                *index as usize,
            );
            assert!(valid);
        }
    }
}
