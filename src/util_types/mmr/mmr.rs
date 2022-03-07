use std::fmt::Debug;
use std::marker::PhantomData;

use crate::util_types::simple_hasher::{Hasher, ToDigest};

use crate::shared_math::other::log_2_floor;

use super::membership_proof::{verify_membership_proof, MembershipProof};

#[inline]
fn left_child(node_index: u128, height: u128) -> u128 {
    node_index - (1 << height)
}

#[inline]
fn right_child(node_index: u128) -> u128 {
    node_index - 1
}

/// Get (index, height) of leftmost ancestor
/// This ancestor does *not* have to be in the MMR
#[inline]
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

/// Given a leaf count and a data index, return the height of the peak
/// that this leaf points to.
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

/// Given node count, return a vector representing the height of
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
fn node_index_to_data_index(node_index: u128) -> Option<u128> {
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

/// Return the new peaks of the MMR after adding `new_leaf` as well as the membership
/// proof for the added leaf.
/// Returns None if configuration is impossible (too small `old_peaks` input vector)
fn calculate_new_peaks_and_membership_proof<
    H: Hasher<Digest = HashDigest> + Clone,
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
>(
    old_leaf_count: u128,
    old_peaks: Vec<HashDigest>,
    new_leaf: HashDigest,
) -> Option<(Vec<HashDigest>, MembershipProof<HashDigest, H>)> {
    let mut peaks = old_peaks;
    let mut new_node_index = data_index_to_node_index(old_leaf_count);
    let (mut new_node_is_right_child, _height) = right_child_and_height(new_node_index);
    peaks.push(new_leaf);
    let mut hasher = H::new();
    let mut membership_proof: MembershipProof<HashDigest, H> = MembershipProof {
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
        peaks.push(hasher.hash_two(&previous_peak, &new_hash));
        new_node_index += 1;
        new_node_is_right_child = right_child_and_height(new_node_index).0;
    }

    Some((peaks, membership_proof))
}

/// Get a root commitment to the entire MMR
fn bag_peaks<HashDigest, H>(peaks: &[HashDigest], node_count: u128) -> HashDigest
where
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
    H: Hasher<Digest = HashDigest> + Clone,
    u128: ToDigest<HashDigest>,
{
    // Follows the description on
    // https://github.com/mimblewimble/grin/blob/master/doc/mmr.md#hashing-and-bagging
    // to calculate a root from a list of peaks and the size of the MMR.
    let peaks_count: usize = peaks.len();
    let mut hasher: H = H::new();

    if peaks_count == 0 {
        return hasher.hash_one(&0u128.to_digest());
    }

    let mut acc: HashDigest = hasher.hash_two(&node_count.to_digest(), &peaks[peaks_count - 1]);
    for i in 1..peaks_count {
        acc = hasher.hash_two(&peaks[peaks_count - 1 - i], &acc);
    }

    acc
}

/// Verify a proof for the integral update of a leaf in the MMR
pub fn verify_leaf_update_proof<HashDigest, H>(
    update_leaf_proof: &LeafUpdateProof<HashDigest, H>,
    new_leaf: &HashDigest,
    leaf_count: u128,
) -> bool
where
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
    H: Hasher<Digest = HashDigest> + Clone,
    u128: ToDigest<HashDigest>,
{
    // We need to verify that
    // 1: New authentication path is valid
    // 2: Only the targeted peak is changed, all other must remain unchanged

    // 1: New authentication path is valid
    let (new_valid, sub_tree_root_res) = verify_membership_proof(
        &update_leaf_proof.membership_proof,
        &update_leaf_proof.new_peaks,
        new_leaf,
        leaf_count,
    );
    if !new_valid {
        return false;
    }

    // 2: Only the targeted peak is changed, all other must remain unchanged
    let sub_tree_root = sub_tree_root_res.unwrap();
    let modified_peak_index_res = update_leaf_proof
        .new_peaks
        .iter()
        .position(|peak| *peak == sub_tree_root);
    let modified_peak_index = match modified_peak_index_res {
        None => return false,
        Some(index) => index,
    };
    let mut calculated_new_peaks: Vec<HashDigest> = update_leaf_proof.old_peaks.to_owned();
    calculated_new_peaks[modified_peak_index] = sub_tree_root;

    calculated_new_peaks == update_leaf_proof.new_peaks
}

#[derive(Debug, Clone, PartialEq)]
pub struct AppendProof<HashDigest>
where
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
{
    pub old_leaf_count: u128,
    pub old_peaks: Vec<HashDigest>,
    pub new_peaks: Vec<HashDigest>,
    // TODO: Add a verify method
}

/// A proof of integral updating of a leaf. The membership_proof can be either before
/// or after the update, since it does not change up the update, only all other
/// membership proofs do.
#[derive(Debug, Clone)]
pub struct LeafUpdateProof<HashDigest, H>
where
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
    H: Hasher<Digest = HashDigest> + Clone,
{
    // The membership proof for a leaf does *not* change when that leaf is updated,
    // only all other membership proofs do. So we only include *one* membership proof
    // in this data structure. In other words: This membership proof is simulataneously
    // the old *and* the new membership proof of the updated leaf.
    pub membership_proof: MembershipProof<HashDigest, H>,
    pub old_peaks: Vec<HashDigest>,
    pub new_peaks: Vec<HashDigest>,
    // TODO: Add a verify method
}

#[derive(Debug, Clone)]
pub struct MmrAccumulator<HashDigest, H> {
    leaf_count: u128,
    peaks: Vec<HashDigest>,
    _hasher: PhantomData<H>,
}

// TODO: Write tests for the accumulator MMR functions
// 0. Create an (empty?) accumulator MMR
// 1. append a value to this
// 2. verify that the before state, after state,
//    and leaf hash constitute an append-proof
//    that can be verified with `verify_append`.
// 3. Repeat (2) n times.
// 4. Run prove/verify_membership with some values
//    But how do we get the authentication paths?
// 5. update hashes though `modify`
// 6. verify that this results in proofs that can
//    be verified with the verify_modify function.
impl<HashDigest, H> MmrAccumulator<HashDigest, H>
where
    H: Hasher<Digest = HashDigest> + Clone,
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
    u128: ToDigest<HashDigest>,
{
    /// Initialize a shallow MMR (only storing peaks) from a list of hash digests
    pub fn new(hashes: Vec<HashDigest>) -> Self {
        // If all the hash digests already exist in memory, we might as well
        // build the shallow MMR from an archival MMR, since it doesn't give
        // asymptotically higher RAM consumption than building it without storing
        // all digests. At least, I think that's the case.
        // Clearly, this function could use less RAM if we don't build the entire
        // archival MMR.
        let leaf_count = hashes.len() as u128;
        let archival = MmrArchive::new(hashes);
        let peaks_and_heights = archival.get_peaks_with_heights();
        Self {
            _hasher: archival._hasher,
            leaf_count,
            peaks: peaks_and_heights.iter().map(|x| x.0.clone()).collect(),
        }
    }

    pub fn bag_peaks(&self) -> HashDigest {
        bag_peaks::<HashDigest, H>(&self.peaks, leaf_count_to_node_count(self.leaf_count))
    }

    pub fn get_peaks(&self) -> Vec<HashDigest> {
        self.peaks.clone()
    }

    pub fn count_leaves(&self) -> u128 {
        self.leaf_count
    }

    pub fn is_empty(&self) -> bool {
        self.leaf_count == 0
    }

    /// Calculate the new accumulator MMR after inserting a new leaf and return the membership
    /// proof of this new leaf.
    /// The membership proof is returned here since the accumulater MMR has no other way of
    /// retrieving a membership proof for a leaf.
    pub fn append(&mut self, new_leaf: HashDigest) -> MembershipProof<HashDigest, H> {
        let (new_peaks, membership_proof) =
            calculate_new_peaks_and_membership_proof::<H, HashDigest>(
                self.leaf_count,
                self.peaks.clone(),
                new_leaf,
            )
            .unwrap();
        self.peaks = new_peaks;
        self.leaf_count += 1;

        membership_proof
    }

    /// Create a proof for honest appending. Verifiable by `AccumulatorMmr` implementation.
    /// Returns (old_peaks, old_leaf_count, new_peaks)
    pub fn prove_append(&self, new_leaf: HashDigest) -> AppendProof<HashDigest> {
        let old_peaks = self.peaks.clone();
        let old_leaf_count = self.leaf_count;
        let new_peaks = calculate_new_peaks_and_membership_proof::<H, HashDigest>(
            old_leaf_count,
            old_peaks.clone(),
            new_leaf,
        )
        .unwrap()
        .0;

        AppendProof {
            old_peaks,
            old_leaf_count,
            new_peaks,
        }
    }

    /// Verify a proof for integral append
    pub fn verify_append_proof(
        append_proof: AppendProof<HashDigest>,
        new_leaf: HashDigest,
    ) -> bool {
        let expected_new_peaks = append_proof.new_peaks.clone();
        let new_peaks_calculated: Option<Vec<HashDigest>> =
            calculate_new_peaks_and_membership_proof::<H, HashDigest>(
                append_proof.old_leaf_count,
                append_proof.old_peaks,
                new_leaf,
            )
            .map(|x| x.0);

        match new_peaks_calculated {
            None => false,
            Some(peaks) => expected_new_peaks == peaks,
        }
    }

    /// Update a leaf hash and modify the peaks with this new hash
    pub fn update_leaf(
        &mut self,
        old_membership_proof: &MembershipProof<HashDigest, H>,
        new_leaf: &HashDigest,
    ) {
        let node_index = data_index_to_node_index(old_membership_proof.data_index);
        let mut hasher = H::new();
        let mut acc_hash: HashDigest = new_leaf.to_owned();
        let mut acc_index: u128 = node_index;
        for hash in old_membership_proof.authentication_path.iter() {
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
        let (peak_heights, _) = get_peak_heights_and_peak_node_indices(self.leaf_count);
        let expected_peak_height_res =
            get_peak_height(self.leaf_count, old_membership_proof.data_index);
        let expected_peak_height = match expected_peak_height_res {
            None => panic!("Did not find any peak height for (leaf_count, data_index) combination. Got: leaf_count = {}, data_index = {}", self.leaf_count, old_membership_proof.data_index),
            Some(eph) => eph,
        };

        let peak_height_index_res = peak_heights.iter().position(|x| *x == expected_peak_height);
        let peak_height_index = match peak_height_index_res {
            None => panic!("Did not find a matching peak"),
            Some(index) => index,
        };

        self.peaks[peak_height_index] = acc_hash;
    }

    /// Construct a proof of the integral update of a hash in an existing accumulator MMR
    /// New authentication path (membership proof) is unchanged by this operation, so
    /// it is not output. Outputs new_peaks.
    pub fn prove_update_leaf(
        &self,
        old_membership_proof: &MembershipProof<HashDigest, H>,
        new_leaf: &HashDigest,
    ) -> Vec<HashDigest> {
        let mut updated_self = self.clone();
        updated_self.update_leaf(old_membership_proof, new_leaf);

        updated_self.peaks
    }

    /// Prove that a specific leaf hash belongs in an MMR
    pub fn prove_membership(
        _membership_proof: &MembershipProof<HashDigest, H>,
        _leaf_hash: HashDigest,
    ) {
    }

    /// Verify a membership proof/leaf hash pair
    pub fn verify_membership_proof(
        &self,
        membership_proof: &MembershipProof<HashDigest, H>,
        leaf_hash: &HashDigest,
    ) -> (bool, Option<HashDigest>) {
        verify_membership_proof(membership_proof, &self.peaks, leaf_hash, self.leaf_count)
    }
}

/// A Merkle Mountain Range is a datastructure for storing a list of hashes.
///
/// Merkle Mountain Ranges only know about hashes. When values are to be associated with
/// MMRs, these values must be stored by the caller, or in a wrapper to this data structure.
#[derive(Debug, Clone)]
pub struct MmrArchive<HashDigest, H: Clone> {
    digests: Vec<HashDigest>,
    _hasher: PhantomData<H>,
}

impl<HashDigest, H> MmrArchive<HashDigest, H>
where
    H: Hasher<Digest = HashDigest> + Clone,
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
    u128: ToDigest<HashDigest>,
{
    pub fn new(hashes: Vec<HashDigest>) -> Self {
        let dummy_digest = 0u128.to_digest();
        let mut new_mmr: Self = Self {
            digests: vec![dummy_digest],
            _hasher: PhantomData,
        };
        for hash in hashes {
            new_mmr.append(hash);
        }

        new_mmr
    }

    pub fn is_empty(&self) -> bool {
        self.digests.len() == 1
    }

    /// Get a leaf from the MMR, will panic if index is out of range
    pub fn get_leaf(&self, data_index: u128) -> HashDigest {
        let node_index = data_index_to_node_index(data_index);
        self.digests[node_index as usize].clone()
    }

    /// Update a hash in the existing archival MMR
    pub fn update_leaf(&mut self, data_index: u128, new_leaf: HashDigest) {
        // 1. change the leaf value
        let mut node_index = data_index_to_node_index(data_index);
        self.digests[node_index as usize] = new_leaf.clone();

        // 2. Calculate hash changes for all parents
        let mut parent_index = parent(node_index);
        let mut acc_hash = new_leaf;
        let mut hasher = H::new();

        // While parent exists in MMR, update parent
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
            self.digests[parent_index as usize] = acc_hash.clone();
            node_index = parent_index;
            parent_index = parent(parent_index);
        }
    }

    #[allow(clippy::type_complexity)]
    /// Create a proof for the integral modification of a leaf, without mutating the
    /// archival MMR.
    /// Output: (Old peaks, old membership proof), (new peaks, new membership proof)
    pub fn prove_update_leaf(
        &self,
        data_index: u128,
        new_leaf: &HashDigest,
    ) -> LeafUpdateProof<HashDigest, H> {
        // TODO: MAKE SURE THIS FUNCTION IS TESTED FOR LOW AND HIGH PEAKS!
        // For low peaks: Make sure it is tested where the peak is just a leaf,
        // i.e. when there is a peak of height 0.
        let (old_membership_proof, old_peaks): (MembershipProof<HashDigest, H>, Vec<HashDigest>) =
            self.prove_membership(data_index);
        let mut new_archival_mmr: MmrArchive<HashDigest, H> = self.to_owned();
        let node_index_of_updated_leaf = data_index_to_node_index(data_index);

        new_archival_mmr.digests[node_index_of_updated_leaf as usize] = new_leaf.clone();

        // All parent's hashes must be recalculated when a leaf hash changes
        // TODO: Rewrite this to use the `update_leaf` method
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
        let (new_membership_proof, new_peaks): (MembershipProof<HashDigest, H>, Vec<HashDigest>) =
            new_archival_mmr.prove_membership(data_index);

        // Sanity check. Should always succeed.
        assert!(
            old_membership_proof.authentication_path == new_membership_proof.authentication_path
                && old_membership_proof.data_index == new_membership_proof.data_index
        );

        LeafUpdateProof {
            membership_proof: old_membership_proof,
            new_peaks,
            old_peaks,
        }
    }

    /// Return (membership_proof, peaks)
    pub fn prove_membership(
        &self,
        data_index: u128,
    ) -> (MembershipProof<HashDigest, H>, Vec<HashDigest>) {
        // A proof consists of an authentication path
        // and a list of peaks

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

        let membership_proof = MembershipProof {
            authentication_path,
            data_index,
            _hasher: PhantomData,
        };

        (membership_proof, peaks)
    }

    /// Verify a membership proof. Return the peak digest that the leaf points to.
    pub fn verify_membership_proof(
        &self,
        membership_proof: &MembershipProof<HashDigest, H>,
        leaf_hash: &HashDigest,
    ) -> (bool, Option<HashDigest>) {
        let res = verify_membership_proof(
            membership_proof,
            &self.get_peaks(),
            leaf_hash,
            self.count_leaves(),
        );

        if res.0
            && self.digests[data_index_to_node_index(membership_proof.data_index) as usize]
                != *leaf_hash
        {
            // This should *never* happen. It would indicate that the hash function is broken
            // since the leaf hash hashed to the correct peak, but that the digest could not
            // be found in the digests field
            panic!("Verified membership proof but did not find leaf hash at data index.");
        }

        res
    }

    /// Calculate the root for the entire MMR
    pub fn bag_peaks(&self) -> HashDigest {
        let peaks: Vec<HashDigest> = self.get_peaks();
        bag_peaks::<HashDigest, H>(&peaks, self.count_nodes() as u128)
    }

    /// Return the digests of the peaks of the MMR
    pub fn get_peaks(&self) -> Vec<HashDigest> {
        let peaks_and_heights = self.get_peaks_with_heights();
        peaks_and_heights.into_iter().map(|x| x.0).collect()
    }

    /// Return a list of tuples (peaks, height)
    pub fn get_peaks_with_heights(&self) -> Vec<(HashDigest, u128)> {
        if self.is_empty() {
            return vec![];
        }

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

    /// Return the number of nodes in all the trees in the MMR
    fn count_nodes(&self) -> u128 {
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

    /// Append an element to the archival MMR, return the membership proof of the newly added leaf.
    /// The membership proof is returned here since the accumulater MMR has no other way of
    /// retrieving a membership proof for a leaf. And the archival and accumulator MMR share
    /// this interface.
    pub fn append(&mut self, new_leaf: HashDigest) -> MembershipProof<HashDigest, H> {
        let node_index = self.digests.len() as u128;
        let data_index = node_index_to_data_index(node_index).unwrap();
        self.append_raw(new_leaf);
        self.prove_membership(data_index).0
    }

    /// Append an element to the archival MMR
    pub fn append_raw(&mut self, new_leaf: HashDigest) {
        let node_index = self.digests.len() as u128;
        self.digests.push(new_leaf.clone());
        let (parent_needed, own_height) = right_child_and_height(node_index);
        if parent_needed {
            let left_sibling_hash =
                self.digests[left_sibling(node_index, own_height) as usize].clone();
            let mut hasher = H::new();
            let parent_hash: HashDigest = hasher.hash_two(&left_sibling_hash, &new_leaf);
            self.append_raw(parent_hash);
        }
    }

    /// Create a proof for honest appending. Verifiable by `AccumulatorMmr` implementation.
    /// Returns (old_peaks, old_leaf_count, new_peaks)
    pub fn prove_append(&self, new_leaf: HashDigest) -> AppendProof<HashDigest> {
        let old_leaf_count: u128 = self.count_leaves();
        let old_peaks_and_heights: Vec<(HashDigest, u128)> = self.get_peaks_with_heights();
        let old_peaks: Vec<HashDigest> = old_peaks_and_heights.into_iter().map(|x| x.0).collect();
        let new_peaks: Vec<HashDigest> = calculate_new_peaks_and_membership_proof::<H, HashDigest>(
            old_leaf_count,
            old_peaks.clone(),
            new_leaf,
        )
        .unwrap()
        .0;

        AppendProof {
            old_peaks,
            old_leaf_count,
            new_peaks,
        }
    }

    /// With knowledge of old peaks, old size (leaf count), new leaf hash, and new peaks, verify that
    /// append is correct.
    pub fn verify_append_proof(
        append_proof: AppendProof<HashDigest>,
        new_leaf: HashDigest,
    ) -> bool {
        let first_new_node_index = data_index_to_node_index(append_proof.old_leaf_count);
        let (mut new_node_is_right_child, _height) = right_child_and_height(first_new_node_index);

        // If new node is not a right child, the new peak list is just the old one
        // with the new leaf hash appended
        let mut calculated_peaks: Vec<HashDigest> = append_proof.old_peaks.to_vec();
        calculated_peaks.push(new_leaf);
        let mut new_node_index = first_new_node_index;
        let mut hasher = H::new();
        while new_node_is_right_child {
            let new_hash = calculated_peaks.pop().unwrap();
            let previous_peak = calculated_peaks.pop().unwrap();
            calculated_peaks.push(hasher.hash_two(&previous_peak, &new_hash));
            new_node_index += 1;
            new_node_is_right_child = right_child_and_height(new_node_index).0;
        }

        calculated_peaks == append_proof.new_peaks
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
    fn empty_mmr_behavior_test() {
        let mut archival_mmr: MmrArchive<blake3::Hash, blake3::Hasher> =
            MmrArchive::<blake3::Hash, blake3::Hasher>::new(vec![]);
        let mut accumulator_mmr: MmrAccumulator<blake3::Hash, blake3::Hasher> =
            MmrAccumulator::<blake3::Hash, blake3::Hasher>::new(vec![]);
        assert_eq!(0, archival_mmr.count_leaves());
        assert_eq!(0, accumulator_mmr.leaf_count);
        assert_eq!(archival_mmr.get_peaks(), accumulator_mmr.get_peaks());
        assert_eq!(Vec::<blake3::Hash>::new(), accumulator_mmr.get_peaks());
        assert_eq!(archival_mmr.bag_peaks(), accumulator_mmr.bag_peaks());
        assert_eq!(0, archival_mmr.count_nodes());
        assert!(accumulator_mmr.is_empty());
        assert!(archival_mmr.is_empty());

        // Test behavior of appending to an empty MMR
        let new_leaf = blake3::hash(
            bincode::serialize(&0xbeefu128)
                .expect("Encoding failed")
                .as_slice(),
        );
        let archival_append_proof = archival_mmr.prove_append(new_leaf);
        let accumulator_append_proof = accumulator_mmr.prove_append(new_leaf);

        // Verify that the append proofs look as expected
        assert_eq!(archival_append_proof, accumulator_append_proof);
        assert_eq!(0, archival_append_proof.old_leaf_count);
        assert_eq!(archival_mmr.get_peaks(), archival_append_proof.old_peaks);
        assert_eq!(0, archival_append_proof.old_peaks.len());
        assert_eq!(1, archival_append_proof.new_peaks.len());

        // Verify that the proofs validate
        assert!(
            MmrArchive::<blake3::Hash, blake3::Hasher>::verify_append_proof(
                archival_append_proof.clone(),
                new_leaf,
            )
        );
        assert!(
            MmrAccumulator::<blake3::Hash, blake3::Hasher>::verify_append_proof(
                accumulator_append_proof,
                new_leaf,
            )
        );

        // Make the append and verify that the new peaks match the one from the proofs
        let archival_membership_proof = archival_mmr.append(new_leaf);
        let accumulator_membership_proof = accumulator_mmr.append(new_leaf);
        assert_eq!(archival_mmr.get_peaks(), archival_append_proof.new_peaks);
        assert_eq!(accumulator_mmr.get_peaks(), archival_append_proof.new_peaks);

        // Verify that the appended value matches the one stored in the archival MMR
        assert_eq!(new_leaf, archival_mmr.get_leaf(0));

        // Verify that the membership proofs for the inserted leafs are valid and that they agree
        assert_eq!(
            archival_membership_proof, accumulator_membership_proof,
            "accumulator and archival membership proofs must agree"
        );
        assert!(
            archival_mmr
                .verify_membership_proof(&archival_membership_proof, &new_leaf)
                .0
        );
    }

    #[test]
    fn verify_against_correct_peak_test() {
        // This test addresses a bug that was discovered late in the development process
        // where it was possible to fake a verification proof by providing a valid leaf
        // and authentication path but lying about the data index. This error occurred
        // because the derived hash was compared against all of the peaks to find a match
        // and it wasn't verified that the accumulated hash matched the *correct* peak.
        // This error was fixed and this test fails without that fix.
        let leaf_hashes: Vec<blake3::Hash> = vec![14u128, 15u128, 16u128]
            .iter()
            .map(|x| blake3::hash(bincode::serialize(x).expect("Encoding failed").as_slice()))
            .collect();
        let archival_mmr = MmrArchive::<blake3::Hash, blake3::Hasher>::new(leaf_hashes.clone());
        let (mut membership_proof, peaks): (
            MembershipProof<blake3::Hash, blake3::Hasher>,
            Vec<blake3::Hash>,
        ) = archival_mmr.prove_membership(0);

        // Verify that the accumulated hash in the verifier is compared against the **correct** hash,
        // not just **any** hash in the peaks list.
        assert!(verify_membership_proof(&membership_proof, &peaks, &leaf_hashes[0], 3,).0);
        membership_proof.data_index = 2;
        assert!(!verify_membership_proof(&membership_proof, &peaks, &leaf_hashes[0], 3,).0);
        membership_proof.data_index = 0;

        // verify the same behavior in the accumulator MMR
        let accumulator_mmr =
            MmrAccumulator::<blake3::Hash, blake3::Hasher>::new(leaf_hashes.clone());
        assert!(
            accumulator_mmr
                .verify_membership_proof(&membership_proof, &leaf_hashes[0])
                .0
        );
        membership_proof.data_index = 2;
        assert!(
            !accumulator_mmr
                .verify_membership_proof(&membership_proof, &leaf_hashes[0])
                .0
        );
    }

    #[test]
    fn update_leaf_archival_test() {
        let mut rp = RescuePrimeProduction::new();
        let leaf_hashes: Vec<Vec<BFieldElement>> = (14..17)
            .map(|x| rp.hash_one(&vec![BFieldElement::new(x)]))
            .collect();
        let mut archival_mmr =
            MmrArchive::<Vec<BFieldElement>, RescuePrimeProduction>::new(leaf_hashes.clone());
        let (mp, old_peaks): (
            MembershipProof<Vec<BFieldElement>, RescuePrimeProduction>,
            Vec<Vec<BFieldElement>>,
        ) = archival_mmr.prove_membership(2);
        assert!(verify_membership_proof(&mp, &old_peaks, &leaf_hashes[2], 3).0);
        let new_leaf = rp.hash_one(&vec![BFieldElement::new(10000)]);

        archival_mmr.update_leaf(2, new_leaf.clone());
        let new_peaks = archival_mmr.get_peaks();

        // Verify that peaks have changed as expected
        assert_ne!(old_peaks[1], new_peaks[1]);
        assert_eq!(old_peaks[0], new_peaks[0]);
        assert_eq!(2, new_peaks.len());
        assert_eq!(2, old_peaks.len());
        assert!(!verify_membership_proof(&mp, &new_peaks, &leaf_hashes[2], 3).0);
        assert!(verify_membership_proof(&mp, &new_peaks, &new_leaf, 3).0);

        // Create a new archival MMR with the same leaf hashes as in the
        // modified MMR, and verify that the two MMRs are equivalent
        let leaf_hashes_new = vec![
            rp.hash_one(&vec![BFieldElement::new(14)]),
            rp.hash_one(&vec![BFieldElement::new(15)]),
            rp.hash_one(&vec![BFieldElement::new(10000)]),
        ];
        let archival_mmr_new =
            MmrArchive::<Vec<BFieldElement>, RescuePrimeProduction>::new(leaf_hashes_new);
        assert_eq!(archival_mmr.digests, archival_mmr_new.digests);
    }

    #[test]
    fn prove_append_test() {
        let leaf_hashes_blake3: Vec<blake3::Hash> = vec![14u128, 15u128, 16u128]
            .iter()
            .map(|x| blake3::hash(bincode::serialize(x).expect("Encoding failed").as_slice()))
            .collect();
        let leaf_hashes_blake3_alt: Vec<blake3::Hash> = vec![14u128, 15u128, 17u128]
            .iter()
            .map(|x| blake3::hash(bincode::serialize(x).expect("Encoding failed").as_slice()))
            .collect();
        let new_leaf: blake3::Hash = blake3::hash(
            bincode::serialize(&1337u128)
                .expect("Encoding failed")
                .as_slice(),
        );
        let bad_leaf: blake3::Hash = blake3::hash(
            bincode::serialize(&13333337u128)
                .expect("Encoding failed")
                .as_slice(),
        );
        let mut archival_mmr_small =
            MmrArchive::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_blake3.clone());
        let archival_mmr_small_alt =
            MmrArchive::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_blake3_alt.clone());
        let mut accumulator_mmr_small =
            MmrAccumulator::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_blake3.clone());

        // Create an append proof with the archival MMR and verify it with a accumulator MMR
        let append_proof_archival = archival_mmr_small.prove_append(new_leaf);
        assert!(
            MmrAccumulator::<blake3::Hash, blake3::Hasher>::verify_append_proof(
                append_proof_archival.clone(),
                new_leaf,
            )
        );

        // Verify that accumulator and Archival creates the same proofs
        let append_proof_accumulator = accumulator_mmr_small.prove_append(new_leaf);
        assert_eq!(append_proof_archival, append_proof_accumulator);

        // Negative tests: verify failure if parameters are wrong
        assert!(
            !MmrAccumulator::<blake3::Hash, blake3::Hasher>::verify_append_proof(
                AppendProof {
                    new_peaks: append_proof_archival.new_peaks.clone(),
                    old_leaf_count: append_proof_archival.old_leaf_count + 1,
                    old_peaks: append_proof_archival.old_peaks.clone(),
                },
                new_leaf,
            )
        );
        assert!(
            !MmrAccumulator::<blake3::Hash, blake3::Hasher>::verify_append_proof(
                append_proof_archival.clone(),
                bad_leaf,
            )
        );

        // switch old and new peaks in proof, verify failure
        assert!(
            !MmrAccumulator::<blake3::Hash, blake3::Hasher>::verify_append_proof(
                AppendProof {
                    new_peaks: append_proof_archival.old_peaks.clone(),
                    old_leaf_count: append_proof_archival.old_leaf_count,
                    old_peaks: append_proof_archival.new_peaks.clone(),
                },
                new_leaf,
            )
        );

        // switch peaks back
        assert!(
            !MmrAccumulator::<blake3::Hash, blake3::Hasher>::verify_append_proof(
                AppendProof {
                    new_peaks: archival_mmr_small_alt.get_peaks(),
                    old_leaf_count: append_proof_archival.old_leaf_count,
                    old_peaks: append_proof_archival.old_peaks.clone(),
                },
                new_leaf,
            )
        );
        assert!(
            MmrAccumulator::<blake3::Hash, blake3::Hasher>::verify_append_proof(
                append_proof_archival.clone(),
                new_leaf,
            )
        );

        // Actually append the new leaf and verify that it matches the values from the proof
        archival_mmr_small.append(new_leaf);
        let new_peaks_from_archival: Vec<blake3::Hash> = archival_mmr_small.get_peaks();
        assert_eq!(append_proof_archival.new_peaks, new_peaks_from_archival);
        accumulator_mmr_small.append(new_leaf);
        let new_peaks_from_accumulator: Vec<blake3::Hash> = accumulator_mmr_small.peaks;
        assert_eq!(append_proof_archival.new_peaks, new_peaks_from_accumulator);
    }

    #[test]
    fn bag_peaks_test() {
        // Verify that archival and accumulator MMR produce the same root
        // First with blake3
        let leaf_hashes_blake3: Vec<blake3::Hash> = vec![14u128, 15u128, 16u128]
            .iter()
            .map(|x| blake3::hash(bincode::serialize(x).expect("Encoding failed").as_slice()))
            .collect();
        let archival_mmr_small =
            MmrArchive::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_blake3.clone());
        let accumulator_mmr_small =
            MmrAccumulator::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_blake3);
        assert_eq!(
            archival_mmr_small.bag_peaks(),
            accumulator_mmr_small.bag_peaks()
        );
        assert_eq!(
            archival_mmr_small.bag_peaks(),
            bag_peaks::<blake3::Hash, blake3::Hasher>(&accumulator_mmr_small.peaks, 4)
        );
        assert!(!accumulator_mmr_small
            .peaks
            .iter()
            .any(|peak| *peak == accumulator_mmr_small.bag_peaks()));

        // Then with Rescue Prime
        let leaf_hashes_rescue_prime: Vec<Vec<BFieldElement>> =
            (14..17).map(|x| vec![BFieldElement::new(x)]).collect();
        let archival_mmr_small_rp = MmrArchive::<Vec<BFieldElement>, RescuePrimeProduction>::new(
            leaf_hashes_rescue_prime.clone(),
        );
        let accumulator_mmr_small_rp =
            MmrAccumulator::<Vec<BFieldElement>, RescuePrimeProduction>::new(
                leaf_hashes_rescue_prime,
            );
        assert_eq!(
            archival_mmr_small_rp.bag_peaks(),
            accumulator_mmr_small_rp.bag_peaks()
        );
        assert!(!accumulator_mmr_small_rp
            .peaks
            .iter()
            .any(|peak| *peak == accumulator_mmr_small_rp.bag_peaks()));

        // Then with a bigger dataset
        let leaf_hashes_bigger_blake3: Vec<blake3::Hash> =
            vec![14u128, 15u128, 16u128, 206, 1232, 123, 9989]
                .iter()
                .map(|x| blake3::hash(bincode::serialize(x).expect("Encoding failed").as_slice()))
                .collect();
        let archival_mmr_bigger =
            MmrArchive::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_bigger_blake3.clone());
        let accumulator_mmr_bigger =
            MmrAccumulator::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_bigger_blake3);
        assert_eq!(
            archival_mmr_bigger.bag_peaks(),
            accumulator_mmr_bigger.bag_peaks()
        );
        assert!(!accumulator_mmr_bigger
            .peaks
            .iter()
            .any(|peak| *peak == accumulator_mmr_bigger.bag_peaks()));
    }

    #[test]
    fn accumulator_mmr_update_leaf_test() {
        // Verify that upating leafs in archival and in accumulator MMR results in the same peaks
        // and verify that updating all leafs in an MMR results in the expected MMR
        for size in 1..150 {
            let new_leaf = blake3::hash(
                bincode::serialize(&314159265358979u128)
                    .expect("Encoding failed")
                    .as_slice(),
            );
            let leaf_hashes_blake3: Vec<blake3::Hash> = (500u128..500 + size)
                .map(|x| blake3::hash(bincode::serialize(&x).expect("Encoding failed").as_slice()))
                .collect();
            let mut acc =
                MmrAccumulator::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_blake3.clone());
            let mut archival =
                MmrArchive::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_blake3.clone());
            let archival_end_state =
                MmrArchive::<blake3::Hash, blake3::Hasher>::new(vec![new_leaf; size as usize]);
            for i in 0..size {
                let (mp, _archival_peaks) = archival.prove_membership(i);
                assert_eq!(i, mp.data_index);
                acc.update_leaf(&mp, &new_leaf);
                archival.update_leaf(i, new_leaf);
                let new_archival_peaks = archival.get_peaks();
                assert_eq!(new_archival_peaks, acc.peaks);
            }

            assert_eq!(archival_end_state.get_peaks(), acc.peaks);
        }
    }

    #[test]
    fn accumulator_mmr_prove_verify_leaf_update_test() {
        for size in 1..150 {
            let new_leaf = blake3::hash(
                bincode::serialize(&314159265358979u128)
                    .expect("Encoding failed")
                    .as_slice(),
            );
            let bad_leaf = blake3::hash(
                bincode::serialize(&27182818284590452353u128)
                    .expect("Encoding failed")
                    .as_slice(),
            );
            let leaf_hashes_blake3: Vec<blake3::Hash> = (500u128..500 + size)
                .map(|x| blake3::hash(bincode::serialize(&x).expect("Encoding failed").as_slice()))
                .collect();
            let mut acc =
                MmrAccumulator::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_blake3.clone());
            let mut archival =
                MmrArchive::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_blake3.clone());
            let archival_end_state =
                MmrArchive::<blake3::Hash, blake3::Hasher>::new(vec![new_leaf; size as usize]);
            for i in 0..size {
                let (mp, _archival_peaks) = archival.prove_membership(i);
                let new_peaks_from_proof = acc.prove_update_leaf(&mp, &new_leaf);
                let update_leaf_proof = LeafUpdateProof {
                    membership_proof: mp.clone(),
                    new_peaks: new_peaks_from_proof,
                    old_peaks: acc.peaks.clone(),
                };
                assert!(verify_leaf_update_proof(
                    &update_leaf_proof,
                    &new_leaf,
                    size
                ));
                assert!(!verify_leaf_update_proof(
                    &update_leaf_proof,
                    &bad_leaf,
                    size
                ));

                archival.update_leaf(i, new_leaf);
                acc.update_leaf(&mp, &new_leaf);
                let new_archival_peaks = archival.get_peaks();
                assert_eq!(new_archival_peaks, acc.peaks);
            }
            assert_eq!(archival_end_state.get_peaks(), acc.peaks);
        }
    }

    #[test]
    fn mmr_append_test() {
        // Verify that building an MMR iteratively or in *one* function call results in the same MMR
        for size in 1..260 {
            let leaf_hashes_blake3: Vec<blake3::Hash> = (500u128..500 + size)
                .map(|x| blake3::hash(bincode::serialize(&x).expect("Encoding failed").as_slice()))
                .collect();
            let mut archival_iterative = MmrArchive::<blake3::Hash, blake3::Hasher>::new(vec![]);
            let archival_batch =
                MmrArchive::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_blake3.clone());
            let mut accumulator_iterative =
                MmrAccumulator::<blake3::Hash, blake3::Hasher>::new(vec![]);
            let accumulator_batch =
                MmrAccumulator::<blake3::Hash, blake3::Hasher>::new(leaf_hashes_blake3.clone());
            for (data_index, leaf_hash) in leaf_hashes_blake3.into_iter().enumerate() {
                let archival_membership_proof: MembershipProof<blake3::Hash, blake3::Hasher> =
                    archival_iterative.append(leaf_hash);
                let accumulator_membership_proof = accumulator_iterative.append(leaf_hash);

                // Verify membership proofs returned from the append operation
                assert_eq!(
                    accumulator_membership_proof, archival_membership_proof,
                    "membership proofs from append operation must agree"
                );
                assert!(
                    archival_iterative
                        .verify_membership_proof(&archival_membership_proof, &leaf_hash)
                        .0,
                    "membership proof from append must verify"
                );

                // Verify that membership proofs are the same as generating them from an
                // archival MMR
                let archival_membership_proof_direct =
                    archival_iterative.prove_membership(data_index as u128).0;
                assert_eq!(archival_membership_proof_direct, archival_membership_proof);
            }

            // Verify that the MMRs built iteratively from `append` and
            // in *one* batch are the same
            assert_eq!(archival_iterative.digests, archival_batch.digests);
            assert_eq!(accumulator_batch.peaks, accumulator_iterative.peaks);
            assert_eq!(
                accumulator_batch.leaf_count,
                accumulator_iterative.leaf_count
            );
            assert_eq!(size, accumulator_iterative.leaf_count);
            assert_eq!(archival_iterative.get_peaks(), accumulator_iterative.peaks);
        }
    }

    #[test]
    fn one_input_mmr_test() {
        let element = vec![BFieldElement::new(14)];
        let mut rp = RescuePrimeProduction::new();
        let input_hash = rp.hash_one(&element);
        let mut mmr =
            MmrArchive::<Vec<BFieldElement>, RescuePrimeProduction>::new(vec![input_hash.clone()]);
        let mut leaf_count = 1;
        assert_eq!(leaf_count, mmr.count_leaves());
        assert_eq!(1, mmr.count_nodes());
        let original_peaks_and_heights: Vec<(Vec<BFieldElement>, u128)> =
            mmr.get_peaks_with_heights();
        assert_eq!(1, original_peaks_and_heights.len());
        assert_eq!(0, original_peaks_and_heights[0].1);

        let data_index = 0;
        let (membership_proof, peaks) = mmr.prove_membership(data_index);
        let valid_res = verify_membership_proof(&membership_proof, &peaks, &input_hash, leaf_count);
        assert!(valid_res.0);
        assert!(valid_res.1.is_some());

        let new_input_hash = rp.hash_one(&vec![BFieldElement::new(201)]);
        mmr.append(new_input_hash.clone());
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
        let append_proof = AppendProof {
            old_peaks: original_peaks,
            old_leaf_count: mmr.count_leaves() - 1,
            new_peaks,
        };
        assert!(
            MmrArchive::<Vec<BFieldElement>, RescuePrimeProduction>::verify_append_proof(
                append_proof,
                new_input_hash,
            )
        );

        for &data_index in &[0u128, 1] {
            let new_leaf: Vec<BFieldElement> = rp.hash_one(&vec![BFieldElement::new(987223)]);
            let mut update_leaf_proof = mmr.prove_update_leaf(data_index, &new_leaf);
            assert!(verify_leaf_update_proof(
                &update_leaf_proof,
                &new_leaf,
                leaf_count
            ));
            let wrong_data_index = (data_index + 1) % mmr.count_leaves();
            update_leaf_proof.membership_proof.data_index = wrong_data_index;
            assert!(!verify_leaf_update_proof(
                &update_leaf_proof,
                &new_leaf,
                leaf_count
            ));
        }
    }

    #[test]
    fn two_input_mmr_test() {
        let values: Vec<Vec<BFieldElement>> = (0..2).map(|x| vec![BFieldElement::new(x)]).collect();
        let mut rp = RescuePrimeProduction::new();
        let input_hashes: Vec<Vec<BFieldElement>> = values.iter().map(|x| rp.hash_one(x)).collect();
        let mut mmr =
            MmrArchive::<Vec<BFieldElement>, RescuePrimeProduction>::new(input_hashes.clone());
        let mut leaf_count = 2;
        assert_eq!(leaf_count, mmr.count_leaves());
        assert_eq!(3, mmr.count_nodes());
        let original_peaks_and_heights: Vec<(Vec<BFieldElement>, u128)> =
            mmr.get_peaks_with_heights();
        assert_eq!(1, original_peaks_and_heights.len());

        let data_index: usize = 0;
        let (mut membership_proof, peaks) = mmr.prove_membership(data_index as u128);
        let valid_res = verify_membership_proof(
            &membership_proof,
            &peaks,
            &input_hashes[data_index],
            leaf_count,
        );
        assert!(valid_res.0);
        assert!(valid_res.1.is_some());

        // Negative test for verify membership
        membership_proof.data_index += 1;
        assert!(
            !verify_membership_proof(
                &membership_proof,
                &peaks,
                &input_hashes[data_index],
                leaf_count
            )
            .0
        );

        let new_leaf_hash: Vec<BFieldElement> = rp.hash_one(&vec![BFieldElement::new(201)]);
        mmr.append(new_leaf_hash.clone());
        let new_peaks_and_heights = mmr.get_peaks_with_heights();
        let original_peaks: Vec<Vec<BFieldElement>> = original_peaks_and_heights
            .iter()
            .map(|x| x.0.to_vec())
            .collect();
        leaf_count += 1;
        let new_peaks: Vec<Vec<BFieldElement>> =
            new_peaks_and_heights.iter().map(|x| x.0.to_vec()).collect();
        let append_proof = AppendProof {
            new_peaks,
            old_leaf_count: mmr.count_leaves() - 1,
            old_peaks: original_peaks,
        };
        assert!(
            MmrArchive::<Vec<BFieldElement>, RescuePrimeProduction>::verify_append_proof(
                append_proof,
                new_leaf_hash,
            )
        );

        for &data_index in &[0u128, 1, 2] {
            let new_leaf: Vec<BFieldElement> = rp.hash_one(&vec![BFieldElement::new(987223)]);
            let mut proof = mmr.prove_update_leaf(data_index, &new_leaf);

            assert!(verify_leaf_update_proof(&proof, &new_leaf, leaf_count));
            let wrong_data_index = (data_index + 1) % mmr.count_leaves();
            proof.membership_proof.data_index = wrong_data_index;

            assert!(!verify_leaf_update_proof(&proof, &new_leaf, leaf_count));
            proof.membership_proof.data_index = data_index;

            assert!(verify_leaf_update_proof(&proof, &new_leaf, leaf_count));
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
            let mut mmr =
                MmrArchive::<Vec<BFieldElement>, RescuePrimeProduction>::new(input_hashes.clone());
            assert_eq!(data_size, mmr.count_leaves());
            assert_eq!(node_count, mmr.count_nodes());
            let original_peaks_and_heights = mmr.get_peaks_with_heights();
            let peak_heights_1: Vec<u128> =
                original_peaks_and_heights.iter().map(|x| x.1).collect();
            let (peak_heights_2, _) = get_peak_heights_and_peak_node_indices(data_size);
            assert_eq!(peak_heights_1, peak_heights_2);
            assert_eq!(peak_count, original_peaks_and_heights.len() as u128);

            // Get an authentication path for **all** values in MMR,
            // verify that it is valid
            for index in 0..data_size {
                let (membership_proof, peaks) = mmr.prove_membership(index as u128);
                let valid_res = verify_membership_proof(
                    &membership_proof,
                    &peaks,
                    &input_hashes[index as usize],
                    data_size,
                );
                assert!(valid_res.0);
                assert!(valid_res.1.is_some());
            }

            // Make a new MMR where we append with a value and run the verify_append
            let new_leaf_hash = rp.hash(&vec![BFieldElement::new(201)]);
            mmr.append(new_leaf_hash.clone());
            let new_peaks_and_heights = mmr.get_peaks_with_heights();
            let original_peaks: Vec<Vec<BFieldElement>> = original_peaks_and_heights
                .iter()
                .map(|x| x.0.to_vec())
                .collect();
            let new_peaks: Vec<Vec<BFieldElement>> =
                new_peaks_and_heights.iter().map(|x| x.0.to_vec()).collect();
            let append_proof = AppendProof {
                new_peaks,
                old_leaf_count: mmr.count_leaves() - 1,
                old_peaks: original_peaks,
            };
            assert!(
                MmrArchive::<Vec<BFieldElement>, RescuePrimeProduction>::verify_append_proof(
                    append_proof,
                    new_leaf_hash,
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

            let input_hashes: Vec<blake3::Hash> = input_prehashes
                .iter()
                .map(|x| blake3::hash(bincode::serialize(x).expect("Encoding failed").as_slice()))
                .collect();
            let mut mmr = MmrArchive::<blake3::Hash, blake3::Hasher>::new(input_hashes.clone());
            assert_eq!(data_size, mmr.count_leaves());
            assert_eq!(node_count, mmr.count_nodes());
            let original_peaks_and_heights: Vec<(blake3::Hash, u128)> =
                mmr.get_peaks_with_heights();
            let peak_heights_1: Vec<u128> =
                original_peaks_and_heights.iter().map(|x| x.1).collect();
            let (peak_heights_2, _) = get_peak_heights_and_peak_node_indices(data_size);
            assert_eq!(peak_heights_1, peak_heights_2);
            assert_eq!(peak_count, original_peaks_and_heights.len() as u128);

            // Get an authentication path for **all** values in MMR,
            // verify that it is valid
            for data_index in 0..data_size {
                let (membership_proof, peaks) = mmr.prove_membership(data_index);
                let valid_res = verify_membership_proof(
                    &membership_proof,
                    &peaks,
                    &input_hashes[data_index as usize],
                    data_size,
                );
                assert!(valid_res.0);
                assert!(valid_res.1.is_some());

                let new_leaf: blake3::Hash = blake3::hash(
                    bincode::serialize(&98723u128)
                        .expect("Encoding failed")
                        .as_slice(),
                );
                let mut update_proof = mmr.prove_update_leaf(data_index, &new_leaf);
                assert!(verify_leaf_update_proof(
                    &update_proof,
                    &new_leaf,
                    data_size
                ));

                let wrong_data_index = (data_index + 1) % mmr.count_leaves();

                // The below verify_modify tests should only fail if `wrong_data_index` is
                // different than `data_index`.
                update_proof.membership_proof.data_index = wrong_data_index;
                assert!(
                    wrong_data_index == data_index
                        || !verify_leaf_update_proof(&update_proof, &new_leaf, data_size)
                );

                update_proof.membership_proof.data_index = data_index;

                // Modify an element in the MMR and run prove/verify for membership
                let old_leaf = input_hashes[data_index as usize];
                mmr.update_leaf(data_index, new_leaf.clone());
                let (new_mp, new_peaks) = mmr.prove_membership(data_index);
                assert!(verify_membership_proof(&new_mp, &new_peaks, &new_leaf, data_size).0);
                assert!(!verify_membership_proof(&new_mp, &new_peaks, &old_leaf, data_size).0);

                // Return the element to its former value and run prove/verify for membership
                mmr.update_leaf(data_index, old_leaf.clone());
                let (old_mp, old_peaks) = mmr.prove_membership(data_index);
                assert!(!verify_membership_proof(&old_mp, &old_peaks, &new_leaf, data_size).0);
                assert!(verify_membership_proof(&old_mp, &old_peaks, &old_leaf, data_size).0);
            }

            // Make a new MMR where we append with a value and run the verify_append
            let new_leaf_hash = blake3::hash(
                blake3::Hash::from_hex(format!("{:064x}", 519u128))
                    .unwrap()
                    .as_bytes(),
            );
            mmr.append(new_leaf_hash);
            let new_peaks_and_heights = mmr.get_peaks_with_heights();
            let original_peaks: Vec<blake3::Hash> =
                original_peaks_and_heights.iter().map(|x| x.0).collect();
            let new_peaks: Vec<blake3::Hash> = new_peaks_and_heights.iter().map(|x| x.0).collect();
            let append_proof = AppendProof {
                new_peaks,
                old_leaf_count: mmr.count_leaves() - 1,
                old_peaks: original_peaks,
            };
            assert!(
                MmrArchive::<blake3::Hash, blake3::Hasher>::verify_append_proof(
                    append_proof,
                    new_leaf_hash,
                )
            );
        }
    }
}
