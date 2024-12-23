use arbitrary::Arbitrary;
use itertools::Itertools;

use super::mmr_accumulator::MmrAccumulator;
use super::shared_advanced::get_peak_heights_and_peak_node_indices;
use super::shared_advanced::parent;
use super::shared_advanced::right_sibling;
use super::shared_basic::calculate_new_peaks_from_append;
use super::shared_basic::leaf_index_to_mt_index_and_peak_index;
use crate::prelude::*;
use crate::util_types::mmr::shared_advanced::left_sibling;
use crate::util_types::mmr::shared_advanced::node_indices_added_by_append;

/// An MmrSuccessorProof asserts that one MMR Accumulator is the descendant of
/// another, *i.e.*, that the second can be obtained by appending a set of leafs
/// to the first. It consists of a set of authentication paths connecting the
/// old peaks to the new peaks.
#[derive(Debug, Clone, BFieldCodec, Arbitrary)]
pub struct MmrSuccessorProof {
    pub paths: Vec<Digest>,
}

impl MmrSuccessorProof {
    /// Compute a new `MmrSuccessorProof` given the starting MMR accumulator and
    /// a list of digests to be appended.
    ///
    /// # Panics
    ///
    ///  - if the number of leafs in the MMRA is greater than or equal to 2^63
    pub fn new_from_batch_append(mmra: &MmrAccumulator, new_leafs: &[Digest]) -> Self {
        let (heights_of_old_peaks, indices_of_old_peaks) =
            get_peak_heights_and_peak_node_indices(mmra.num_leafs());
        let (_heights_of_new_peaks, indices_of_new_peaks) =
            get_peak_heights_and_peak_node_indices(mmra.num_leafs() + new_leafs.len() as u64);
        let num_old_peaks = heights_of_old_peaks.len();

        let mut needed_indices = vec![vec![]; num_old_peaks];
        for (i, (index, height)) in indices_of_old_peaks
            .iter()
            .copied()
            .zip(heights_of_old_peaks)
            .enumerate()
        {
            let mut current_index = index;
            let mut current_height = height;
            while !indices_of_new_peaks.contains(&current_index) {
                let mut sibling = right_sibling(current_index, current_height);
                let parent_index = parent(current_index);
                if parent(sibling) != parent_index {
                    sibling = left_sibling(current_index, current_height);
                };
                let list_index = needed_indices[i].len();
                needed_indices[i].push(Some((list_index, sibling)));
                current_height += 1;
                current_index = parent_index;
            }
        }

        let mut current_peaks = mmra.peaks();
        let mut current_peak_indices = indices_of_old_peaks.clone();
        let mut current_leaf_count = mmra.num_leafs();
        let mut paths = needed_indices
            .iter()
            .map(|ni| vec![Digest::default(); ni.len()])
            .collect_vec();

        for &new_leaf in new_leafs {
            let new_node_indices = node_indices_added_by_append(current_leaf_count);

            let (new_peaks, membership_proof) = calculate_new_peaks_from_append(
                current_leaf_count,
                current_peaks.clone(),
                new_leaf,
            );

            let (_new_heights, new_peak_indices) =
                get_peak_heights_and_peak_node_indices(current_leaf_count + 1);
            let new_nodes = membership_proof
                .authentication_path
                .into_iter()
                .scan(new_leaf, |runner, path_node| {
                    let yld = *runner;
                    *runner = Tip5::hash_pair(path_node, *runner);
                    Some(yld)
                })
                .collect_vec();

            for (index, node) in new_node_indices.into_iter().zip(new_nodes).chain(
                current_peak_indices
                    .into_iter()
                    .zip(current_peaks.iter().copied()),
            ) {
                for (path, path_indices) in paths.iter_mut().zip(needed_indices.iter_mut()) {
                    if let Some(wrapped_pair) = path_indices
                        .iter_mut()
                        .filter(|maybe| maybe.is_some())
                        .find(|definitely| definitely.unwrap().1 == index)
                    {
                        path[wrapped_pair.unwrap().0] = node;
                        *wrapped_pair = None;
                    }
                }
            }

            current_peaks = new_peaks;
            current_peak_indices = new_peak_indices;
            current_leaf_count += 1;
        }

        Self {
            paths: paths.concat(),
        }
    }

    /// Verify that the `old` [`MmrAccumulator`] is a predecessor of the `new` one.
    pub fn verify(&self, old: &MmrAccumulator, new: &MmrAccumulator) -> bool {
        let merkle_tree_root_index = u64::try_from(MerkleTree::ROOT_INDEX)
            .expect("internal error: type `usize` should have at most 64 bits");

        if old.num_leafs() > new.num_leafs() {
            return false;
        }

        let num_new_peaks = u32::try_from(new.peaks().len())
            .expect("internal error: Merkle Mountain Ranges should have at most 2^63 leafs");
        if num_new_peaks != new.num_leafs().count_ones() {
            return false;
        }

        let mut verified_old_leafs_count = 0;
        let mut authentication_paths = self.paths.iter();
        for old_peak in old.peaks() {
            let (merkle_tree_index_of_first_leaf_under_new_peak, new_peak_index) =
                leaf_index_to_mt_index_and_peak_index(verified_old_leafs_count, new.num_leafs());

            let num_remaining_leafs = old.num_leafs() - verified_old_leafs_count;
            let old_peak_height = num_remaining_leafs.ilog2();
            let mut current_merkle_tree_index =
                merkle_tree_index_of_first_leaf_under_new_peak >> old_peak_height;
            let mut current_node = old_peak;
            while current_merkle_tree_index > merkle_tree_root_index {
                let Some(&sibling) = authentication_paths.next() else {
                    return false;
                };
                let is_left_sibling = current_merkle_tree_index % 2 == 0;
                current_node = if is_left_sibling {
                    Tip5::hash_pair(current_node, sibling)
                } else {
                    Tip5::hash_pair(sibling, current_node)
                };
                current_merkle_tree_index >>= 1;
            }
            debug_assert_eq!(merkle_tree_root_index, current_merkle_tree_index);

            let new_peak_index = usize::try_from(new_peak_index)
                .expect("internal error: type `usize` should have at least 32 bits");
            let Some(&new_peak) = new.peaks().get(new_peak_index) else {
                return false;
            };
            if current_node != new_peak {
                return false;
            }

            verified_old_leafs_count += 1 << old_peak_height;
        }

        // ensure all digests were read
        authentication_paths.count() == 0
    }
}

#[cfg(test)]
mod test {
    use proptest::collection::vec;
    use proptest::prop_assert;
    use proptest_arbitrary_interop::arb;
    use rand::prelude::*;
    use test_strategy::proptest;

    use super::*;

    fn verification_succeeds_with_n_leafs_append_m(n: usize, m: usize, rng: &mut dyn RngCore) {
        let original_leafs = (0..n).map(|_| rng.gen::<Digest>()).collect_vec();
        let old_mmra = MmrAccumulator::new_from_leafs(original_leafs);

        let new_leafs = (0..m).map(|_| rng.gen::<Digest>()).collect_vec();
        let successor_proof = MmrSuccessorProof::new_from_batch_append(&old_mmra, &new_leafs);

        let mut new_mmra = old_mmra.clone();
        for new_leaf in new_leafs {
            new_mmra.append(new_leaf);
        }

        assert!(successor_proof.verify(&old_mmra, &new_mmra));
    }

    #[test]
    fn small_leaf_counts_unit() {
        let mut rng = thread_rng();
        let threshold = 18;
        for n in 0..threshold {
            for m in 0..threshold {
                verification_succeeds_with_n_leafs_append_m(n, m, &mut rng);
            }
        }
    }

    #[proptest(cases = 50)]
    fn verification_succeeds_positive_property(
        #[strategy(arb::<MmrAccumulator>())] old_mmr: MmrAccumulator,
        #[strategy(vec(arb::<Digest>(), 0usize..(1<<6)))] new_leafs: Vec<Digest>,
    ) {
        let mut new_mmr = old_mmr.clone();
        let mmr_successor_proof = MmrSuccessorProof::new_from_batch_append(&old_mmr, &new_leafs);
        for leaf in new_leafs {
            new_mmr.append(leaf);
        }

        prop_assert!(mmr_successor_proof.verify(&old_mmr, &new_mmr));
    }

    fn rotr(i: u64) -> u64 {
        (i >> 1) | ((i & 1) << 63)
    }

    #[proptest(cases = 50)]
    fn verification_fails_negative_properties(
        #[filter(#old_mmr.num_leafs() != rotr(#old_mmr.num_leafs()))]
        #[strategy(arb::<MmrAccumulator>())]
        old_mmr: MmrAccumulator,
        #[strategy(vec(arb::<Digest>(), 0usize..(1<<10)))] new_leafs: Vec<Digest>,
        #[strategy(arb::<usize>())] mut modify_path_element: usize,
    ) {
        let mut new_mmr = old_mmr.clone();
        let mmr_successor_proof = MmrSuccessorProof::new_from_batch_append(&old_mmr, &new_leafs);
        for leaf in new_leafs.iter() {
            new_mmr.append(*leaf);
        }

        // old MMR has wrong num leafs
        if rotr(old_mmr.num_leafs()) != old_mmr.num_leafs()
            && rotr(old_mmr.num_leafs()) < (u64::MAX >> 1)
        {
            let fake_old_mmr = MmrAccumulator::init(old_mmr.peaks(), rotr(old_mmr.num_leafs()));
            prop_assert!(!mmr_successor_proof.verify(&fake_old_mmr, &new_mmr));
        }

        // new MMR has wrong num leafs
        if rotr(new_mmr.num_leafs()) != new_mmr.num_leafs()
            && rotr(new_mmr.num_leafs()) < (u64::MAX >> 1)
        {
            let fake_new_mmr = MmrAccumulator::init(new_mmr.peaks(), rotr(new_mmr.num_leafs()));
            prop_assert!(!mmr_successor_proof.verify(&old_mmr, &fake_new_mmr));
        }

        // swap two peaks in old mmr
        if old_mmr.peaks().len() >= 2 {
            let mut old_peaks_swapped = old_mmr.peaks();
            old_peaks_swapped.swap(0, 1);
            let old_mmr_swapped = MmrAccumulator::init(old_peaks_swapped, old_mmr.num_leafs());
            prop_assert!(!mmr_successor_proof.verify(&old_mmr_swapped, &new_mmr));
        }

        // swap two peaks in new mmr
        if new_mmr.peaks().len() >= 2 {
            let mut new_peaks_swapped = new_mmr.peaks();
            new_peaks_swapped.swap(0, 1);
            let new_mmr_swapped = MmrAccumulator::init(new_peaks_swapped, new_mmr.num_leafs());
            prop_assert!(!mmr_successor_proof.verify(&old_mmr, &new_mmr_swapped));
        }

        // change one path element
        if !mmr_successor_proof.paths.is_empty() {
            let mut fake_mmr_successor_proof_3 = mmr_successor_proof.clone();
            let path_index = modify_path_element % fake_mmr_successor_proof_3.paths.len();
            modify_path_element =
                (modify_path_element - path_index) / fake_mmr_successor_proof_3.paths.len();
            let value_index = modify_path_element % Digest::LEN;
            fake_mmr_successor_proof_3.paths[path_index].0[value_index].increment();
            prop_assert!(!fake_mmr_successor_proof_3.verify(&old_mmr, &new_mmr));
        }

        // Missing path element
        if !mmr_successor_proof.paths.is_empty() {
            let mut fake_mmr_successor_proof_4 = mmr_successor_proof.clone();
            fake_mmr_successor_proof_4.paths.pop();
            prop_assert!(!fake_mmr_successor_proof_4.verify(&old_mmr, &new_mmr));
        }

        // One element too many
        let mut fake_mmr_successor_proof_5 = mmr_successor_proof.clone();
        fake_mmr_successor_proof_5.paths.push(Digest::default());
        prop_assert!(!fake_mmr_successor_proof_5.verify(&old_mmr, &new_mmr));

        // Missing peak
        let fake_new_mmr = MmrAccumulator::init(
            new_mmr
                .peaks()
                .into_iter()
                .rev()
                .skip(1)
                .rev()
                .collect_vec(),
            new_mmr.num_leafs(),
        );
        prop_assert!(!mmr_successor_proof.verify(&old_mmr, &fake_new_mmr));
    }

    #[test]
    fn verification_succeeds_unit() {
        let mut rng: StdRng = SeedableRng::from_seed(
            hex::decode("deadbeef00000000deadbeef00000000deadbeef00000000deadbeef00000000")
                .unwrap()
                .try_into()
                .unwrap(),
        );
        let num_new_leafs = rng.gen_range(0..(1 << 15));
        let old_num_leafs = rng.gen_range(0u64..(u64::MAX >> 1));
        let old_peaks = (0..old_num_leafs.count_ones())
            .map(|_| rng.gen::<Digest>())
            .collect_vec();

        let old_mmr = MmrAccumulator::init(old_peaks, old_num_leafs);
        let mut new_mmr = old_mmr.clone();

        let new_leafs = (0..num_new_leafs)
            .map(|_| rng.gen::<Digest>())
            .collect_vec();

        for &leaf in new_leafs.iter() {
            new_mmr.append(leaf);
        }

        let mmr_successor_proof = MmrSuccessorProof::new_from_batch_append(&old_mmr, &new_leafs);

        assert!(mmr_successor_proof.verify(&old_mmr, &new_mmr));
    }
}
