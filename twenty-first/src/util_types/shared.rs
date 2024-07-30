use crate::math::digest::Digest;
use crate::prelude::Tip5;
use crate::util_types::algebraic_hasher::AlgebraicHasher;

/// Get a root commitment to the entire MMR/list of Merkle trees
pub fn bag_peaks(peaks: &[Digest]) -> Digest {
    // Follows the description on
    // https://github.com/mimblewimble/grin/blob/master/doc/mmr.md#hashing-and-bagging
    // to calculate a root from a list of peaks and the size of the MMR. Note, however,
    // that the node count described on that website is not used here, as we don't need
    // the extra bits of security that that would provide.
    let peaks_count: usize = peaks.len();

    if peaks_count == 0 {
        return Tip5::hash(&0u128);
    }

    if peaks_count == 1 {
        return peaks[0].to_owned();
    }

    let mut acc: Digest = Tip5::hash_pair(peaks[peaks_count - 2], peaks[peaks_count - 1]);
    for i in 2..peaks_count {
        acc = Tip5::hash_pair(peaks[peaks_count - 1 - i], acc);
    }

    acc
}
