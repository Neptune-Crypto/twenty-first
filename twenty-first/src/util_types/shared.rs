use crate::math::digest::Digest;
use crate::prelude::Tip5;

/// Get a root commitment to the entire MMR/list of Merkle trees
pub fn bag_peaks(peaks: &[Digest]) -> Digest {
    // Follows the description on
    // https://github.com/mimblewimble/grin/blob/master/doc/mmr.md#hashing-and-bagging
    // to calculate a root from a list of peaks and the size of the MMR. Note, however,
    // that the node count described on that website is not used here, as we don't need
    // the extra bits of security that that would provide.

    let mut peaks = peaks.iter();
    let Some(&last_peak) = peaks.next_back() else {
        return Tip5::hash(&0u128);
    };
    let Some(&second_to_last_peak) = peaks.next_back() else {
        return last_peak;
    };

    let accumulator = Tip5::hash_pair(second_to_last_peak, last_peak);
    peaks
        .rev()
        .fold(accumulator, |acc, &peak| Tip5::hash_pair(peak, acc))
}
