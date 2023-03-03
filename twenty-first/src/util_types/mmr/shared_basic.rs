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

/// Convert from leaf index to node index
pub fn leaf_index_to_node_index(leaf_index: u64) -> u64 {
    let diff = non_leaf_nodes_left(leaf_index);

    leaf_index + diff + 1
}

pub fn right_lineage_length(node_index: u64) -> u32 {
    let bit_width = u64::BITS - node_index.leading_zeros();
    let npo2 = 1 << bit_width;

    let dist = npo2 - node_index;

    if (bit_width as u64) < dist {
        right_lineage_length(node_index - (npo2 >> 1) + 1)
    } else {
        (dist - 1) as u32
    }
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
    let mut new_node_index = leaf_index_to_node_index(old_leaf_count);
    let mut right_lineage_count = right_lineage_length(new_node_index);
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
        new_node_index += 1;
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
