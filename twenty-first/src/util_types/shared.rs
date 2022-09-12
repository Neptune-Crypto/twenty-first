use crate::util_types::simple_hasher::Hasher;

use super::simple_hasher::Hashable;

/// Get a root commitment to the entire MMR/list of Merkle trees
pub fn bag_peaks<H>(peaks: &[H::Digest]) -> H::Digest
where
    H: Hasher + std::marker::Sync + std::marker::Send,
    u128: Hashable<<H as Hasher>::T>,
{
    // Follows the description on
    // https://github.com/mimblewimble/grin/blob/master/doc/mmr.md#hashing-and-bagging
    // to calculate a root from a list of peaks and the size of the MMR. Note, however,
    // that the node count described on that website is not used here, as we don't need
    // the extra bits of security that that would provide.
    let peaks_count: usize = peaks.len();
    let hasher: H = H::new();

    if peaks_count == 0 {
        return hasher.hash_sequence(&0u128.to_sequence());
    }

    if peaks_count == 1 {
        return peaks[0].to_owned();
    }

    // let mut acc: H::Digest = hasher.hash_pair(&peaks[peaks_count - 1], &peaks[peaks_count - 2]);
    let mut acc: H::Digest = hasher.hash_pair(&peaks[peaks_count - 2], &peaks[peaks_count - 1]);
    for i in 2..peaks_count {
        acc = hasher.hash_pair(&peaks[peaks_count - 1 - i], &acc);
    }

    acc
}
