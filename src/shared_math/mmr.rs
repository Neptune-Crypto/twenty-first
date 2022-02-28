use std::fmt::Debug;
use std::marker::PhantomData;

use crate::util_types::simple_hasher::{Hasher, ToDigest};

use super::other::log_2_floor;

#[inline]
fn left_child(node_index: u128, height: u128) -> u128 {
    node_index - (1 << height)
}

#[inline]
fn right_child(node_index: u128) -> u128 {
    node_index - 1
}

/// Get (index, height) of leftmost ancestor
// This ancestor does *not* have to be in the MMR
fn leftmost_ancestor(node_index: u128) -> (u128, u128) {
    let mut h = 0;
    let mut ret = 1;
    while ret < node_index {
        h += 1;
        ret = (1 << (h + 1)) - 1;
    }

    (ret, h)
}

/// Return the tuple: (is_right_child, height)
fn right_child_and_height(node_index: u128) -> (bool, u128) {
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
fn parent(node_index: u128) -> u128 {
    let (right, height) = right_child_and_height(node_index);

    if right {
        node_index + 1
    } else {
        node_index + (1 << (height + 1))
    }
}

#[inline]
fn left_sibling(node_index: u128, height: u128) -> u128 {
    node_index - (1 << (height + 1)) + 1
}

#[inline]
fn right_sibling(node_index: u128, height: u128) -> u128 {
    node_index + (1 << (height + 1)) - 1
}

fn get_height_from_data_index(data_index: u128) -> u128 {
    log_2_floor(data_index as u64 + 1) as u128
}

fn leaf_count_to_node_count(leaf_count: u128) -> u128 {
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

pub fn get_peak_height(leaf_count: u128, data_index: u128) -> Option<u128> {
    if data_index >= leaf_count {
        return None;
    }

    let mut node_index = data_index_to_node_index(data_index);
    let node_count = leaf_count_to_node_count(leaf_count);
    let mut height = 0;
    while node_index < node_count + 1 {
        height += 1;
        node_index = parent(node_index);
    }

    Some(height - 1)
}

/// Given node count, return a vector representing the height of
/// the peaks. Input is the number of leafs in the MMR
pub fn get_peak_heights(leaf_count: u128) -> Vec<u128> {
    let node_index_of_rightmost_leaf = data_index_to_node_index(leaf_count - 1);
    let node_count = leaf_count_to_node_count(leaf_count);
    let (mut top_peak, mut top_height) = leftmost_ancestor(node_index_of_rightmost_leaf);
    if top_peak > node_count {
        top_peak = left_child(top_peak, top_height);
        top_height -= 1;
    }

    let mut heights: Vec<u128> = vec![top_height];
    let mut height = top_height;
    let mut candidate = right_sibling(top_peak, height);
    'outer: while height > 0 {
        '_inner: while candidate > node_count && height > 0 {
            candidate = left_child(candidate, height);
            height -= 1;
            if candidate <= node_count {
                heights.push(height);
                candidate = right_sibling(candidate, height);
                continue 'outer;
            }
        }
    }

    heights
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

pub fn data_index_to_node_index(data_index: u128) -> u128 {
    let diff = non_leaf_nodes_left(data_index);

    data_index + diff + 1
}

/// Convert from node index to data index in log(size) time
pub fn node_index_to_data_index(node_index: u128) -> Option<u128> {
    let (_right, height) = right_child_and_height(node_index);
    if height != 0 {
        return None;
    }

    let (mut node, mut height) = leftmost_ancestor(node_index);
    let mut data_index = 0;
    while height > 0 {
        let left_child = left_child(node, height);
        if node_index <= left_child {
            node = left_child;
            height -= 1;
        } else {
            node = right_child(node);
            height -= 1;
            data_index += 1 << height;
        }
    }

    Some(data_index)
}

/// Return the new peaks of the MMR after adding `new_leaf`
pub fn calculate_new_peaks<
    H: Hasher<Digest = HashDigest>,
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
>(
    old_leaf_count: u128,
    old_peaks: Vec<HashDigest>,
    new_leaf: HashDigest,
) -> Vec<HashDigest> {
    let mut peaks = old_peaks;
    let mut new_node_index = data_index_to_node_index(old_leaf_count);
    let (mut new_node_is_right_child, _height) = right_child_and_height(new_node_index);
    peaks.push(new_leaf);
    let mut hasher = H::new();
    while new_node_is_right_child {
        let new_hash = peaks.pop().unwrap();
        let previous_peak = peaks.pop().unwrap();
        peaks.push(hasher.hash_two(&previous_peak, &new_hash));
        new_node_index += 1;
        new_node_is_right_child = right_child_and_height(new_node_index).0;
    }

    peaks
}

#[derive(Debug, Clone)]
pub struct LightMmr<HashDigest, H> {
    leaf_count: u128,
    peaks: Vec<HashDigest>,
    _hasher: PhantomData<H>,
}

// TODO: Write tests for the light MMR functions
// 0. Create an (empty?) light MMR
// 1. append a value to this
// 2. verify that the before state, after state,
//    and leaf hash constitute an append-proof
//    that can be verified with `verify_append`.
// 3. Repeat 2 n times.
// 4. Run prove/verify_membership with some values
//    But how do we get the authentication paths?
// 5. update hashes though `modify`
// 6. verify that this results in proofs that can
//    be verified with the verify_modify function.
impl<HashDigest, H> LightMmr<HashDigest, H>
where
    H: Hasher<Digest = HashDigest> + Clone,
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
    u128: ToDigest<HashDigest>,
{
    pub fn bag_peaks(&self) -> HashDigest {
        let node_count = leaf_count_to_node_count(self.leaf_count);
        ArchivalMmr::<HashDigest, H>::get_root_from_peaks(&self.peaks, node_count)
    }

    /// Initialize a shallow MMR (only storing peaks) from a list of hash digests
    pub fn from_leafs(hashes: Vec<HashDigest>) -> Self {
        // If all the hash digests already exist in memory, we might as well
        // build the shallow MMR from an archival MMR, since it doesn't give
        // asymptotically higher RAM consumption than building it without storing
        // all digests. At least, I think that's the case.
        // Clearly, this function could use less RAM if we don't build the entire
        // archival MMR.
        let leaf_count = hashes.len() as u128;
        let archival = ArchivalMmr::init(hashes);
        let peaks_and_heights = archival.get_peaks_with_heights();
        Self {
            _hasher: archival._hasher,
            leaf_count,
            peaks: peaks_and_heights.iter().map(|x| x.0.clone()).collect(),
        }
    }

    pub fn append(&mut self, new_leaf: HashDigest) {
        self.peaks =
            calculate_new_peaks::<H, HashDigest>(self.leaf_count, self.peaks.clone(), new_leaf);
        self.leaf_count += 1;
    }

    pub fn prove_append(
        _old_peaks: Vec<HashDigest>,
        _old_size: u128,
        _new_leaf: HashDigest,
        _new_peaks: Vec<HashDigest>,
    ) {
    }

    pub fn verify_append(
        old_peaks: Vec<HashDigest>,
        old_leaf_count: u128,
        new_leaf: HashDigest,
        new_peaks_expected: Vec<HashDigest>,
    ) -> bool {
        let new_peaks_calculated =
            calculate_new_peaks::<H, HashDigest>(old_leaf_count, old_peaks, new_leaf);

        new_peaks_calculated == new_peaks_expected
    }

    // Do we need a function to update an authentication path?

    /// Update a hash in the existing light MMR
    pub fn modify() {
        todo!()
    }

    /// Construct a proof of the integral update of a hash in an existing light MMR
    pub fn prove_modify() {
        todo!()
    }

    /// Verify the integral update of a leaf hash
    // TODO: Consider make this into a class method instead
    pub fn verify_modify(
        old_peaks: &[HashDigest],
        old_authentication_path: &[HashDigest],
        new_peaks: &[HashDigest],
        new_authentication_path: &[HashDigest],
        new_leaf: &HashDigest,
        data_index: u128,
        leaf_count: u128,
    ) -> bool {
        ArchivalMmr::<HashDigest, H>::verify_modify(
            old_peaks,
            old_authentication_path,
            new_peaks,
            new_authentication_path,
            new_leaf,
            data_index,
            leaf_count,
        )
    }

    /// Prove that a specific leaf hash belongs in an MMR
    pub fn prove_membership(
        _authentication_path: &[HashDigest],
        _data_index: u128,
        _leaf_hash: HashDigest,
    ) {
    }

    /// Verify an authentication path showing that a specific leaf hash is stored in index `data_index`
    pub fn verify_membership(
        &self,
        authentication_path: &[HashDigest],
        data_index: u128,
        leaf_hash: &HashDigest,
    ) -> bool {
        let node_index = data_index_to_node_index(data_index);
        let mut hasher = H::new();
        let mut acc_hash: HashDigest = leaf_hash.to_owned();
        let mut acc_index: u128 = node_index;
        for hash in authentication_path.iter() {
            let (acc_right, _acc_height) = right_child_and_height(acc_index);
            acc_hash = if acc_right {
                hasher.hash_two(hash, &acc_hash)
            } else {
                hasher.hash_two(&acc_hash, hash)
            };
            acc_index = parent(acc_index);
        }

        // This function is *not* secure when verified against *any* peak.
        // It **must** be compared against the correct peak.
        // Otherwise you could lie leaf_hash, data_index, authentication path
        let peak_heights = get_peak_heights(self.leaf_count);
        let expected_peak_height_res = get_peak_height(self.leaf_count, data_index);
        let expected_peak_height = match expected_peak_height_res {
            None => return false,
            Some(eph) => eph,
        };

        let peak_height_index_res = peak_heights.iter().position(|x| *x == expected_peak_height);
        let peak_height_index = match peak_height_index_res {
            None => return false,
            Some(phi) => phi,
        };

        self.peaks[peak_height_index] == acc_hash
    }
}

/// A Merkle Mountain Range is a datastructure for storing a list of hashes.
///
/// Merkle Mountain Ranges only know about hashes. When values are to be associated with
/// MMRs, these values must be stored by the caller, or in a wrapper to this data structure.
#[derive(Debug, Clone)]
pub struct ArchivalMmr<HashDigest, H: Clone> {
    digests: Vec<HashDigest>,
    _hasher: PhantomData<H>,
}

impl<HashDigest, H> ArchivalMmr<HashDigest, H>
where
    H: Hasher<Digest = HashDigest> + Clone,
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
    u128: ToDigest<HashDigest>,
{
    pub fn init(hashes: Vec<HashDigest>) -> Self {
        let dummy = H::new().dummy_output();
        let mut new_mmr: Self = Self {
            digests: vec![dummy],
            _hasher: PhantomData,
        };
        for hash in hashes {
            new_mmr.archive_append(hash);
        }

        new_mmr
    }

    #[allow(clippy::type_complexity)]
    /// Create a proof for the integral modification of a leaf, without mutating the
    /// archival MMR.
    /// Output: (Old peaks, old authentication path), (new peaks, new authentication path)
    pub fn prove_modify(
        &self,
        data_index: u128,
        new_leaf: &HashDigest,
    ) -> (
        (Vec<HashDigest>, Vec<HashDigest>),
        (Vec<HashDigest>, Vec<HashDigest>),
    ) {
        // TODO: MAKE SURE THIS FUNCTION IS TESTED FOR LOW AND HIGH PEAKS!
        // For low peaks: Make sure it is tested where the peak is just a leaf,
        // i.e. when there is a peak of height 0.
        let (old_authentication_path, old_peaks): (Vec<HashDigest>, Vec<HashDigest>) =
            self.prove_membership(data_index);
        let mut new_archival_mmr: ArchivalMmr<HashDigest, H> = self.to_owned();
        let node_index_of_updated_leaf = data_index_to_node_index(data_index);

        new_archival_mmr.digests[node_index_of_updated_leaf as usize] = new_leaf.clone();

        // All parent's hashes must be recalculated when a leaf hash changes
        let mut parent_index = parent(node_index_of_updated_leaf);
        let mut node_index = node_index_of_updated_leaf;
        let mut acc_hash = new_leaf.clone();
        let mut hasher = H::new();
        while parent_index < self.digests.len() as u128 {
            let (is_right, height) = right_child_and_height(node_index);
            acc_hash = if is_right {
                hasher.hash_two(
                    &self.digests[left_sibling(node_index, height) as usize],
                    &acc_hash,
                )
            } else {
                hasher.hash_two(
                    &acc_hash,
                    &self.digests[right_sibling(node_index, height) as usize],
                )
            };
            new_archival_mmr.digests[parent_index as usize] = acc_hash.clone();
            node_index = parent_index;
            parent_index = parent(parent_index);
        }
        let (new_authentication_path, new_peaks): (Vec<HashDigest>, Vec<HashDigest>) =
            new_archival_mmr.prove_membership(data_index);

        (
            (old_peaks, old_authentication_path),
            (new_peaks, new_authentication_path),
        )
    }

    pub fn verify_modify(
        old_peaks: &[HashDigest],
        old_authentication_path: &[HashDigest],
        new_peaks: &[HashDigest],
        new_authentication_path: &[HashDigest],
        new_leaf: &HashDigest,
        data_index: u128,
        leaf_count: u128,
    ) -> bool {
        // We need to verify that
        // 1: authentication path is unchanged
        // 2: New authentication path is valid
        // 3: Only the targeted peak is changed, all other must remain unchanged

        // 1: authentication path is unchanged
        if old_authentication_path != new_authentication_path {
            return false;
        }

        // 2: New authentication path is valid
        let (new_valid, sub_tree_root_res) = Self::verify_membership(
            new_authentication_path,
            new_peaks,
            new_leaf,
            data_index,
            leaf_count,
        );
        if !new_valid {
            return false;
        }

        // 3: Only the targeted peak is changed, all other must remain unchanged
        let sub_tree_root = sub_tree_root_res.unwrap();
        let modified_peak_index_res = new_peaks.iter().position(|peak| *peak == sub_tree_root);
        let modified_peak_index = match modified_peak_index_res {
            None => return false,
            Some(index) => index,
        };
        let mut calculated_new_peaks: Vec<HashDigest> = old_peaks.to_owned();
        calculated_new_peaks[modified_peak_index] = sub_tree_root;

        if calculated_new_peaks != new_peaks {
            return false;
        }

        true
    }

    pub fn verify_membership(
        authentication_path: &[HashDigest],
        peaks: &[HashDigest],
        value_hash: &HashDigest,
        data_index: u128,
        leaf_count: u128,
    ) -> (bool, Option<HashDigest>) {
        // Verify that peaks match root
        // let matching_root = *root == Self::get_root_from_peaks(peaks, node_count);
        let node_index = data_index_to_node_index(data_index);

        let mut hasher = H::new();
        let mut acc_hash: HashDigest = value_hash.to_owned();
        let mut acc_index: u128 = node_index;
        for hash in authentication_path.iter() {
            let (acc_right, _acc_height) = right_child_and_height(acc_index);
            acc_hash = if acc_right {
                hasher.hash_two(hash, &acc_hash)
            } else {
                hasher.hash_two(&acc_hash, hash)
            };
            acc_index = parent(acc_index);
        }

        // Find the correct peak index
        let heights = get_peak_heights(leaf_count);
        if heights.len() != peaks.len() {
            return (false, None);
        }
        let expected_peak_height_res = get_peak_height(leaf_count, data_index);
        let expected_peak_height = match expected_peak_height_res {
            None => return (false, None),
            Some(eph) => eph,
        };
        let peak_index_res = heights.into_iter().position(|x| x == expected_peak_height);
        let peak_index = match peak_index_res {
            None => return (false, None),
            Some(pi) => pi,
        };

        // Compare the peak at the expected index with accumulated hash
        if peaks[peak_index] != acc_hash {
            return (false, None);
        }

        (true, Some(acc_hash))
    }

    /// Return (authentication_path, peaks)
    pub fn prove_membership(&self, data_index: u128) -> (Vec<HashDigest>, Vec<HashDigest>) {
        // A proof consists of an authentication path
        // and a list of peaks that must hash to the root

        // Find out how long the authentication path is
        let node_index = data_index_to_node_index(data_index);
        let mut top_height: i32 = -1;
        let mut parent_index = node_index;
        while (parent_index as usize) < self.digests.len() {
            parent_index = parent(parent_index);
            top_height += 1;
        }

        // Build the authentication path
        let mut authentication_path: Vec<HashDigest> = vec![];
        let mut index = node_index;
        let (mut index_is_right_child, mut index_height): (bool, u128) =
            right_child_and_height(index);
        while index_height < top_height as u128 {
            if index_is_right_child {
                let left_sibling_index = left_sibling(index, index_height);
                authentication_path.push(self.digests[left_sibling_index as usize].clone());
            } else {
                let right_sibling_index = right_sibling(index, index_height);
                authentication_path.push(self.digests[right_sibling_index as usize].clone());
            }
            index = parent(index);
            let next_index_info = right_child_and_height(index);
            index_is_right_child = next_index_info.0;
            index_height = next_index_info.1;
        }

        let peaks: Vec<HashDigest> = self
            .get_peaks_with_heights()
            .iter()
            .map(|x| x.0.clone())
            .collect();

        (authentication_path, peaks)
    }

    /// Calculate root from a list of peaks and from the node count
    fn get_root_from_peaks(peaks: &[HashDigest], node_count: u128) -> HashDigest {
        // Follows the description for "bagging" on
        // https://github.com/mimblewimble/grin/blob/master/doc/mmr.md#hashing-and-bagging
        // Note that their "size" is the node count
        let peaks_count: usize = peaks.len();
        let mut hasher: H = H::new();

        let mut acc: HashDigest = hasher.hash_two(&node_count.to_digest(), &peaks[peaks_count - 1]);
        for i in 1..peaks_count {
            acc = hasher.hash_two(&peaks[peaks_count - 1 - i], &acc);
        }

        acc
    }

    /// Calculate the root for the entire MMR
    pub fn bag_peaks(&self) -> HashDigest {
        let peaks: Vec<HashDigest> = self
            .get_peaks_with_heights()
            .iter()
            .map(|x| x.0.clone())
            .collect();

        Self::get_root_from_peaks(&peaks, self.count_nodes() as u128)
    }

    /// Return a list of tuples (peaks, height)
    pub fn get_peaks_with_heights(&self) -> Vec<(HashDigest, u128)> {
        // 1. Find top peak
        // 2. Jump to right sibling (will not be included)
        // 3. Take left child of sibling, continue until a node in tree is found
        // 4. Once new node is found, jump to right sibling (will not be included)
        // 5. Take left child of sibling, continue until a node in tree is found
        let mut peaks_and_heights: Vec<(HashDigest, u128)> = vec![];
        let (mut top_peak, mut top_height) = leftmost_ancestor(self.digests.len() as u128 - 1);
        if top_peak > self.digests.len() as u128 - 1 {
            top_peak = left_child(top_peak, top_height);
            top_height -= 1;
        }
        peaks_and_heights.push((self.digests[top_peak as usize].clone(), top_height)); // No clone needed bc array
        let mut height = top_height;
        let mut candidate = right_sibling(top_peak, height);
        'outer: while height > 0 {
            '_inner: while candidate > (self.digests.len() as u128) && height > 0 {
                candidate = left_child(candidate, height);
                height -= 1;
                if candidate < (self.digests.len() as u128) {
                    peaks_and_heights.push((self.digests[candidate as usize].clone(), height));
                    candidate = right_sibling(candidate, height);
                    continue 'outer;
                }
            }
        }

        peaks_and_heights
    }

    pub fn count_nodes(&self) -> u128 {
        self.digests.len() as u128 - 1
    }

    /// Return the number of leaves in the tree
    pub fn count_leaves(&self) -> u128 {
        let peaks_and_heights: Vec<(_, u128)> = self.get_peaks_with_heights();
        let mut acc = 0;
        for (_, height) in peaks_and_heights {
            acc += 1 << height
        }

        acc
    }

    fn archive_append(&mut self, hash: HashDigest) {
        let node_index = self.digests.len() as u128;
        self.digests.push(hash.clone());
        let (parent_needed, own_height) = right_child_and_height(node_index);
        if parent_needed {
            let left_sibling_hash =
                self.digests[left_sibling(node_index, own_height) as usize].clone();
            let mut hasher = H::new();
            let parent_hash: HashDigest = hasher.hash_two(&left_sibling_hash, &hash);
            self.archive_append(parent_hash);
        }
    }

    /// With knowledge of old peaks, old size (leaf count), new leaf hash, and new peaks, verify that
    /// append is correct.
    pub fn verify_append(
        old_root: HashDigest,
        old_peaks: &[HashDigest],
        old_leaf_count: u128,
        new_root: HashDigest,
        new_leaf_hash: HashDigest,
        new_peaks: &[HashDigest],
    ) -> bool {
        let first_new_node_index = data_index_to_node_index(old_leaf_count);
        let (mut new_node_is_right_child, _height) = right_child_and_height(first_new_node_index);

        // If new node is not a right child, the new peak list is just the old one
        // with the new leaf hash appended
        let mut calculated_peaks: Vec<HashDigest> = old_peaks.to_vec();
        calculated_peaks.push(new_leaf_hash);
        let mut new_node_index = first_new_node_index;
        let mut hasher = H::new();
        while new_node_is_right_child {
            let new_hash = calculated_peaks.pop().unwrap();
            let previous_peak = calculated_peaks.pop().unwrap();
            calculated_peaks.push(hasher.hash_two(&previous_peak, &new_hash));
            new_node_index += 1;
            new_node_is_right_child = right_child_and_height(new_node_index).0;
        }

        let calculated_new_root =
            Self::get_root_from_peaks(&calculated_peaks, new_node_index as u128);
        let calculated_old_root =
            Self::get_root_from_peaks(old_peaks, first_new_node_index as u128 - 1);

        calculated_peaks == new_peaks
            && calculated_new_root == new_root
            && calculated_old_root == old_root
    }
}

#[cfg(test)]
mod mmr_test {
    use itertools::izip;
    use rand::RngCore;

    use super::*;
    use crate::{
        shared_math::{
            b_field_element::BFieldElement, rescue_prime::RescuePrime, rescue_prime_params,
        },
        util_types::simple_hasher::RescuePrimeProduction,
    };

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
    fn data_index_node_index_pbt() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
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
            1, 3, 4, 7, 8, 10, 11, 15, 16, 18, 19, 22, 23, 25, 26, 31, 32, 34, 35, 38, 39, 41, 42,
            46, 47, 49, 50, 53, 54, 56, 57, 63, 64,
        ];
        for (i, node_count) in node_counts.iter().enumerate() {
            assert_eq!(*node_count, leaf_count_to_node_count(i as u128 + 1));
        }
    }

    #[test]
    fn get_peak_height_test() {
        assert_eq!(Some(1), get_peak_height(2, 0));
        assert_eq!(Some(1), get_peak_height(2, 1));
        assert_eq!(Some(0), get_peak_height(1, 0));
        assert_eq!(Some(1), get_peak_height(3, 0));
        assert_eq!(Some(1), get_peak_height(3, 1));
        assert_eq!(Some(0), get_peak_height(3, 2));
        assert_eq!(None, get_peak_height(3, 3));
        assert_eq!(None, get_peak_height(3, 4));
        assert_eq!(Some(2), get_peak_height(4, 0));
        assert_eq!(Some(2), get_peak_height(4, 1));
        assert_eq!(Some(2), get_peak_height(4, 2));
        assert_eq!(Some(2), get_peak_height(4, 3));
        assert_eq!(None, get_peak_height(4, 4));
        assert_eq!(Some(2), get_peak_height(5, 0));
        assert_eq!(Some(2), get_peak_height(5, 1));
        assert_eq!(Some(2), get_peak_height(5, 2));
        assert_eq!(Some(2), get_peak_height(5, 3));
        assert_eq!(Some(0), get_peak_height(5, 4));
        assert_eq!(None, get_peak_height(5, 5));
        assert_eq!(Some(5), get_peak_height(33, 0));
        assert_eq!(Some(5), get_peak_height(33, 1));
        assert_eq!(Some(5), get_peak_height(33, 2));
        assert_eq!(Some(5), get_peak_height(33, 17));
        assert_eq!(Some(5), get_peak_height(33, 31));
        assert_eq!(Some(0), get_peak_height(33, 32));
        assert_eq!(None, get_peak_height(33, 33));
    }

    #[test]
    fn verify_against_correct_peak_test() {
        let leaf_hashes: Vec<blake3::Hash> = vec![14u128, 15u128, 16u128]
            .iter()
            .map(|x| blake3::hash(bincode::serialize(x).expect("Encoding failed").as_slice()))
            .collect();
        let archival_mmr = ArchivalMmr::<blake3::Hash, blake3::Hasher>::init(leaf_hashes.clone());
        let (auth_path, peaks): (Vec<blake3::Hash>, Vec<blake3::Hash>) =
            archival_mmr.prove_membership(0);

        // Verify that the accumulated hash in the verifier is compared against the **correct** hash,
        // not just **any** hash in the peaks list.
        assert!(
            ArchivalMmr::<blake3::Hash, blake3::Hasher>::verify_membership(
                &auth_path,
                &peaks,
                &leaf_hashes[0],
                0,
                3,
            )
            .0
        );
        assert!(
            !ArchivalMmr::<blake3::Hash, blake3::Hasher>::verify_membership(
                &auth_path,
                &peaks,
                &leaf_hashes[0],
                2,
                3,
            )
            .0
        );

        // verify the same behavior in the light MMR
        let light_mmr = LightMmr::<blake3::Hash, blake3::Hasher>::from_leafs(leaf_hashes.clone());
        assert!(light_mmr.verify_membership(&auth_path, 0, &leaf_hashes[0]));
        assert!(!light_mmr.verify_membership(&auth_path, 2, &leaf_hashes[0]));
    }

    #[test]
    fn bag_peaks_test() {
        // Verify that archival and light MMR produce the same root
        // First with blake3
        let leaf_hashes_blake3: Vec<blake3::Hash> = vec![14u128, 15u128, 16u128]
            .iter()
            .map(|x| blake3::hash(bincode::serialize(x).expect("Encoding failed").as_slice()))
            .collect();
        let archival_mmr_small =
            ArchivalMmr::<blake3::Hash, blake3::Hasher>::init(leaf_hashes_blake3.clone());
        let light_mmr_small =
            LightMmr::<blake3::Hash, blake3::Hasher>::from_leafs(leaf_hashes_blake3);
        assert_eq!(archival_mmr_small.bag_peaks(), light_mmr_small.bag_peaks());
        assert!(!light_mmr_small
            .peaks
            .iter()
            .any(|peak| *peak == light_mmr_small.bag_peaks()));

        // Then with Rescue Prime
        let leaf_hashes_rescue_prime: Vec<Vec<BFieldElement>> =
            (14..17).map(|x| vec![BFieldElement::new(x)]).collect();
        let archival_mmr_small_rp = ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::init(
            leaf_hashes_rescue_prime.clone(),
        );
        let light_mmr_small_rp = LightMmr::<Vec<BFieldElement>, RescuePrimeProduction>::from_leafs(
            leaf_hashes_rescue_prime,
        );
        assert_eq!(
            archival_mmr_small_rp.bag_peaks(),
            light_mmr_small_rp.bag_peaks()
        );
        assert!(!light_mmr_small_rp
            .peaks
            .iter()
            .any(|peak| *peak == light_mmr_small_rp.bag_peaks()));

        // Then with a bigger dataset
        let leaf_hashes_bigger_blake3: Vec<blake3::Hash> =
            vec![14u128, 15u128, 16u128, 206, 1232, 123, 9989]
                .iter()
                .map(|x| blake3::hash(bincode::serialize(x).expect("Encoding failed").as_slice()))
                .collect();
        let archival_mmr_bigger =
            ArchivalMmr::<blake3::Hash, blake3::Hasher>::init(leaf_hashes_bigger_blake3.clone());
        let light_mmr_bigger =
            LightMmr::<blake3::Hash, blake3::Hasher>::from_leafs(leaf_hashes_bigger_blake3);
        assert_eq!(
            archival_mmr_bigger.bag_peaks(),
            light_mmr_bigger.bag_peaks()
        );
        assert!(!light_mmr_bigger
            .peaks
            .iter()
            .any(|peak| *peak == light_mmr_bigger.bag_peaks()));
    }

    #[test]
    fn one_input_mmr_test() {
        let element = vec![BFieldElement::new(14)];
        let mut rp = RescuePrimeProduction::new();
        let input_hash = rp.hash_one(&element);
        let mut mmr =
            ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::init(
                vec![input_hash.clone()],
            );
        let mut leaf_count = 1;
        assert_eq!(leaf_count, mmr.count_leaves());
        assert_eq!(1, mmr.count_nodes());
        let original_peaks_and_heights: Vec<(Vec<BFieldElement>, u128)> =
            mmr.get_peaks_with_heights();
        assert_eq!(1, original_peaks_and_heights.len());
        assert_eq!(0, original_peaks_and_heights[0].1);
        let original_root: Vec<BFieldElement> = mmr.bag_peaks();

        let data_index = 0;
        let (authentication_path, peaks) = mmr.prove_membership(data_index);
        let valid_res = ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::verify_membership(
            &authentication_path,
            &peaks,
            &input_hash,
            data_index,
            leaf_count,
        );
        assert!(valid_res.0);
        assert!(valid_res.1.is_some());

        let new_input_hash = rp.hash_one(&vec![BFieldElement::new(201)]);
        mmr.archive_append(new_input_hash.clone());
        leaf_count += 1;
        let new_peaks_and_heights = mmr.get_peaks_with_heights();
        assert_eq!(1, new_peaks_and_heights.len());
        assert_eq!(1, new_peaks_and_heights[0].1);

        let original_peaks: Vec<Vec<BFieldElement>> = original_peaks_and_heights
            .iter()
            .map(|x| x.0.to_vec())
            .collect();
        let new_peaks: Vec<Vec<BFieldElement>> =
            new_peaks_and_heights.iter().map(|x| x.0.to_vec()).collect();
        let new_root = mmr.bag_peaks();
        assert!(
            ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::verify_append(
                original_root,
                &original_peaks,
                mmr.count_leaves() - 1,
                new_root,
                new_input_hash,
                &new_peaks
            )
        );

        for &data_index in &[0u128, 1] {
            let new_leaf: Vec<BFieldElement> = rp.hash_one(&vec![BFieldElement::new(987223)]);
            let ((old_peaks, old_authentication_path), (new_peaks, new_authentication_path)) =
                mmr.prove_modify(data_index, &new_leaf);
            assert!(
                ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::verify_modify(
                    &old_peaks,
                    &old_authentication_path,
                    &new_peaks,
                    &new_authentication_path,
                    &new_leaf,
                    data_index,
                    leaf_count
                )
            );
            let wrong_data_index = (data_index + 1) % mmr.count_leaves();
            assert!(
                !ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::verify_modify(
                    &old_peaks,
                    &old_authentication_path,
                    &new_peaks,
                    &new_authentication_path,
                    &new_leaf,
                    wrong_data_index,
                    leaf_count
                )
            );
        }
    }

    #[test]
    fn two_input_mmr_test() {
        let values: Vec<Vec<BFieldElement>> = (0..2).map(|x| vec![BFieldElement::new(x)]).collect();
        let mut rp = RescuePrimeProduction::new();
        let input_hashes: Vec<Vec<BFieldElement>> = values.iter().map(|x| rp.hash_one(x)).collect();
        let mut mmr =
            ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::init(input_hashes.clone());
        let mut leaf_count = 2;
        assert_eq!(leaf_count, mmr.count_leaves());
        assert_eq!(3, mmr.count_nodes());
        let original_peaks_and_heights: Vec<(Vec<BFieldElement>, u128)> =
            mmr.get_peaks_with_heights();
        assert_eq!(1, original_peaks_and_heights.len());
        let original_root = mmr.bag_peaks();

        let data_index: usize = 0;
        let (authentication_path, peaks) = mmr.prove_membership(data_index as u128);
        let valid_res = ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::verify_membership(
            &authentication_path,
            &peaks,
            &input_hashes[data_index],
            data_index as u128,
            leaf_count,
        );
        assert!(valid_res.0);
        assert!(valid_res.1.is_some());

        // Negative test for verify membership
        assert!(
            !ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::verify_membership(
                &authentication_path,
                &peaks,
                &input_hashes[data_index],
                data_index as u128 + 1,
                leaf_count
            )
            .0
        );

        let new_leaf_hash: Vec<BFieldElement> = rp.hash_one(&vec![BFieldElement::new(201)]);
        mmr.archive_append(new_leaf_hash.clone());
        let new_peaks_and_heights = mmr.get_peaks_with_heights();
        let original_peaks: Vec<Vec<BFieldElement>> = original_peaks_and_heights
            .iter()
            .map(|x| x.0.to_vec())
            .collect();
        leaf_count += 1;
        let new_peaks: Vec<Vec<BFieldElement>> =
            new_peaks_and_heights.iter().map(|x| x.0.to_vec()).collect();
        let new_root = mmr.bag_peaks();
        assert!(
            ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::verify_append(
                original_root,
                &original_peaks,
                mmr.count_leaves() - 1,
                new_root,
                new_leaf_hash,
                &new_peaks
            )
        );

        for &data_index in &[0u128, 1, 2] {
            let new_leaf: Vec<BFieldElement> = rp.hash_one(&vec![BFieldElement::new(987223)]);
            let ((old_peaks, old_authentication_path), (new_peaks, new_authentication_path)) =
                mmr.prove_modify(data_index, &new_leaf);
            assert!(
                ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::verify_modify(
                    &old_peaks,
                    &old_authentication_path,
                    &new_peaks,
                    &new_authentication_path,
                    &new_leaf,
                    data_index,
                    leaf_count
                )
            );
            let wrong_data_index = (data_index + 1) % mmr.count_leaves();
            assert!(
                !ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::verify_modify(
                    &old_peaks,
                    &old_authentication_path,
                    &new_peaks,
                    &new_authentication_path,
                    &new_leaf,
                    wrong_data_index,
                    leaf_count
                )
            );
        }
    }

    #[test]
    fn variable_size_rescue_prime_mmr_test() {
        let node_counts: Vec<u128> = vec![
            1, 3, 4, 7, 8, 10, 11, 15, 16, 18, 19, 22, 23, 25, 26, 31, 32, 34, 35, 38, 39, 41, 42,
            46, 47, 49, 50, 53, 54, 56, 57, 63, 64,
        ];
        let peak_counts: Vec<u128> = vec![
            1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4,
            4, 5, 1, 2,
        ];
        for (data_size, node_count, peak_count) in
            izip!((1u128..34).collect::<Vec<u128>>(), node_counts, peak_counts)
        {
            let input_prehashes: Vec<Vec<BFieldElement>> = (0..data_size)
                .map(|x| vec![BFieldElement::new(x as u128 + 14)])
                .collect();
            let rp: RescuePrime = rescue_prime_params::rescue_prime_params_bfield_0();
            let input_hashes: Vec<Vec<BFieldElement>> =
                input_prehashes.iter().map(|x| rp.hash(x)).collect();
            let mut mmr = ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::init(
                input_hashes.clone(),
            );
            assert_eq!(data_size, mmr.count_leaves());
            assert_eq!(node_count, mmr.count_nodes());
            let original_peaks_and_heights = mmr.get_peaks_with_heights();
            let peak_heights_1: Vec<u128> =
                original_peaks_and_heights.iter().map(|x| x.1).collect();
            let peak_heights_2: Vec<u128> = get_peak_heights(data_size);
            assert_eq!(peak_heights_1, peak_heights_2);
            assert_eq!(peak_count, original_peaks_and_heights.len() as u128);
            let original_root = mmr.bag_peaks();

            // Get an authentication path for **all** values in MMR,
            // verify that it is valid
            for index in 0..data_size {
                let (authentication_path, peaks) = mmr.prove_membership(index as u128);
                let valid_res =
                    ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::verify_membership(
                        &authentication_path,
                        &peaks,
                        &input_hashes[index as usize],
                        index,
                        data_size,
                    );
                assert!(valid_res.0);
                assert!(valid_res.1.is_some());
            }

            // Make a new MMR where we append with a value and run the verify_append
            let new_leaf_hash = rp.hash(&vec![BFieldElement::new(201)]);
            mmr.archive_append(new_leaf_hash.clone());
            let new_peaks_and_heights = mmr.get_peaks_with_heights();
            let original_peaks: Vec<Vec<BFieldElement>> = original_peaks_and_heights
                .iter()
                .map(|x| x.0.to_vec())
                .collect();
            let new_peaks: Vec<Vec<BFieldElement>> =
                new_peaks_and_heights.iter().map(|x| x.0.to_vec()).collect();
            let new_root = mmr.bag_peaks();
            assert!(
                ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::verify_append(
                    original_root,
                    &original_peaks,
                    mmr.count_leaves() - 1,
                    new_root,
                    new_leaf_hash,
                    &new_peaks
                )
            );
        }
    }

    #[test]
    fn variable_size_blake3_mmr_test() {
        let node_counts: Vec<u128> = vec![
            1, 3, 4, 7, 8, 10, 11, 15, 16, 18, 19, 22, 23, 25, 26, 31, 32, 34, 35, 38, 39, 41, 42,
            46, 47, 49, 50, 53, 54, 56, 57, 63, 64,
        ];
        let peak_counts: Vec<u128> = vec![
            1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4,
            4, 5, 1, 2,
        ];
        for (data_size, node_count, peak_count) in
            izip!((1u128..34).collect::<Vec<u128>>(), node_counts, peak_counts)
        {
            let input_prehashes: Vec<Vec<BFieldElement>> = (0..data_size)
                .map(|x| vec![BFieldElement::new(x as u128 + 14)])
                .collect();
            // let rp: RescuePrime = rescue_prime_params::rescue_prime_params_bfield_0();
            // blake3_digest(input)
            // blake3_digest_serialize()
            let input_hashes: Vec<blake3::Hash> = input_prehashes
                .iter()
                .map(|x| blake3::hash(bincode::serialize(x).expect("Encoding failed").as_slice()))
                .collect();
            let mut mmr = ArchivalMmr::<blake3::Hash, blake3::Hasher>::init(input_hashes.clone());
            assert_eq!(data_size, mmr.count_leaves());
            assert_eq!(node_count, mmr.count_nodes());
            let original_peaks_and_heights: Vec<(blake3::Hash, u128)> =
                mmr.get_peaks_with_heights();
            let peak_heights_1: Vec<u128> =
                original_peaks_and_heights.iter().map(|x| x.1).collect();
            let peak_heights_2: Vec<u128> = get_peak_heights(data_size);
            assert_eq!(peak_heights_1, peak_heights_2);
            assert_eq!(peak_count, original_peaks_and_heights.len() as u128);
            let original_root = mmr.bag_peaks();

            // Get an authentication path for **all** values in MMR,
            // verify that it is valid
            for data_index in 0..data_size {
                let (authentication_path, peaks) = mmr.prove_membership(data_index);
                let valid_res = ArchivalMmr::<blake3::Hash, blake3::Hasher>::verify_membership(
                    &authentication_path,
                    &peaks,
                    &input_hashes[data_index as usize],
                    data_index,
                    data_size,
                );
                assert!(valid_res.0);
                assert!(valid_res.1.is_some());

                let new_leaf: blake3::Hash = blake3::hash(
                    bincode::serialize(&98723u128)
                        .expect("Encoding failed")
                        .as_slice(),
                );
                let ((old_peaks, old_authentication_path), (new_peaks, new_authentication_path)) =
                    mmr.prove_modify(data_index, &new_leaf);
                assert!(ArchivalMmr::<blake3::Hash, blake3::Hasher>::verify_modify(
                    &old_peaks,
                    &old_authentication_path,
                    &new_peaks,
                    &new_authentication_path,
                    &new_leaf,
                    data_index,
                    data_size
                ));
                let wrong_data_index = (data_index + 1) % mmr.count_leaves();

                // The below verify_modify test should only fail if `wrong_data_index` is
                // different than `data_index`.
                assert!(
                    wrong_data_index == data_index
                        || !ArchivalMmr::<blake3::Hash, blake3::Hasher>::verify_modify(
                            &old_peaks,
                            &old_authentication_path,
                            &new_peaks,
                            &new_authentication_path,
                            &new_leaf,
                            wrong_data_index,
                            data_size
                        )
                );
            }

            // Make a new MMR where we append with a value and run the verify_append
            let new_leaf_hash = blake3::hash(
                blake3::Hash::from_hex(format!("{:064x}", 519u128))
                    .unwrap()
                    .as_bytes(),
            );
            mmr.archive_append(new_leaf_hash);
            let new_peaks_and_heights = mmr.get_peaks_with_heights();
            let original_peaks: Vec<blake3::Hash> =
                original_peaks_and_heights.iter().map(|x| x.0).collect();
            let new_peaks: Vec<blake3::Hash> = new_peaks_and_heights.iter().map(|x| x.0).collect();
            let new_root = mmr.bag_peaks();
            assert!(ArchivalMmr::<blake3::Hash, blake3::Hasher>::verify_append(
                original_root,
                &original_peaks,
                mmr.count_leaves() - 1,
                new_root,
                new_leaf_hash,
                &new_peaks
            ));
        }
    }
}
