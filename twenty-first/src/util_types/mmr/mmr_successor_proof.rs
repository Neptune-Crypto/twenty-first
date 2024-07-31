use std::collections::VecDeque;

use bfieldcodec_derive::BFieldCodec;
use itertools::Itertools;

use crate::{
    prelude::{AlgebraicHasher, Digest, Mmr, Tip5},
    util_types::mmr::shared_advanced::{left_sibling, node_indices_added_by_append},
};

use super::{
    mmr_accumulator::MmrAccumulator,
    shared_advanced::{get_peak_heights_and_peak_node_indices, parent, right_sibling},
    shared_basic::{calculate_new_peaks_from_append, leaf_index_to_mt_index_and_peak_index},
};

/// An MmrSuccessorProof asserts that one MMR Accumulator is the descendant of
/// another, *i.e.*, that the second can be obtained by appending a set of leafs
/// to the first. It consists of a set of authentication paths connecting the
/// old peaks to the new peaks.
#[derive(Debug, Clone, BFieldCodec)]
pub struct MmrSuccessorProof {
    pub paths: Vec<Vec<Digest>>,
}

impl MmrSuccessorProof {
    /// Compute a new `MmrSuccessorProof` given the starting MMR accumulator and
    /// a list of digests to be appended.
    pub fn new_from_batch_append(mmra: &MmrAccumulator, new_leafs: &[Digest]) -> Self {
        let (heights_of_old_peaks, indices_of_old_peaks) =
            get_peak_heights_and_peak_node_indices(mmra.num_leafs());
        let (_heights_of_new_peaks, indices_of_new_peaks) =
            get_peak_heights_and_peak_node_indices(mmra.num_leafs() + new_leafs.len() as u64);
        let num_old_peaks = heights_of_old_peaks.len();

        let mut needed_indices = vec![VecDeque::new(); num_old_peaks];
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
                needed_indices[i].push_back(sibling);
                current_height += 1;
                current_index = parent_index;
            }
        }

        let mut current_peaks = mmra.peaks();
        let mut current_peak_indices = indices_of_old_peaks.clone();
        let mut current_leaf_count = mmra.num_leafs();
        let mut paths = vec![vec![]; num_old_peaks];

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

            for (index, node) in new_node_indices.into_iter().zip(new_nodes) {
                for (i, (path, path_indices)) in
                    paths.iter_mut().zip(needed_indices.iter_mut()).enumerate()
                {
                    if Some(&index) == path_indices.front() {
                        print!(
                            "found path {i} element in new nodes! path length was {}",
                            path_indices.len()
                        );
                        path.push(node);
                        path_indices.pop_front();
                        println!(" and is now {}", path_indices.len());
                    }
                }
            }

            for (index, node) in current_peak_indices
                .into_iter()
                .zip(current_peaks.iter().copied())
            {
                for (i, (path, path_indices)) in
                    paths.iter_mut().zip(needed_indices.iter_mut()).enumerate()
                {
                    if Some(&index) == path_indices.front() {
                        print!(
                            "found path {i} element in current peaks! path length was {}",
                            path_indices.len()
                        );
                        path.push(node);
                        path_indices.pop_front();
                        println!(" and is now {}", path_indices.len());
                    }
                }
            }

            current_peaks = new_peaks;
            current_peak_indices = new_peak_indices;
            current_leaf_count += 1;
        }

        if needed_indices.iter().any(|ni| !ni.is_empty()) {
            for (i, ni) in needed_indices.into_iter().enumerate() {
                if !ni.is_empty() {
                    println!(
                        "error! path {i} has {} needed indices left: [{}]",
                        ni.len(),
                        ni.iter().join(", ")
                    );
                }
            }
            panic!("could not find all necessary nodes");
        }

        Self { paths }
    }

    /// Verify that `old_mmra` is a predecessor of `new_mmra`.
    pub fn verify(&self, old_mmra: &MmrAccumulator, new_mmra: &MmrAccumulator) -> bool {
        if old_mmra.num_leafs() == 0 {
            return true;
        }

        let (old_peak_heights, old_peak_indices) =
            get_peak_heights_and_peak_node_indices(old_mmra.num_leafs());
        if old_peak_heights.len() != self.paths.len() {
            println!("number of old peaks does not match with number of paths");
            return false;
        }

        let (new_peak_heights, new_peak_indices) =
            get_peak_heights_and_peak_node_indices(new_mmra.num_leafs());

        let mut running_leaf_count = 0;
        for (i, (old_peak, (old_height, old_index))) in old_mmra
            .peaks()
            .into_iter()
            .zip(old_peak_heights.into_iter().zip(old_peak_indices))
            .enumerate()
        {
            running_leaf_count += 1 << old_height;
            let mut current_index = old_index;
            let mut current_height = old_height;
            let mut current_node = old_peak;
            let (merkle_tree_index_of_last_leaf_under_this_peak, _) =
                leaf_index_to_mt_index_and_peak_index(running_leaf_count - 1, new_mmra.num_leafs());
            let mut current_merkle_tree_index =
                merkle_tree_index_of_last_leaf_under_this_peak >> current_height;
            let mut j = 0;
            while !new_peak_indices.contains(&current_index) {
                let Some(&sibling) = self.paths[i].get(j) else {
                    println!("cannot get path[{i}] element {j}");
                    return false;
                };
                let is_left_sibling = current_merkle_tree_index & 1 == 0;
                current_node = if is_left_sibling {
                    Tip5::hash_pair(current_node, sibling)
                } else {
                    Tip5::hash_pair(sibling, current_node)
                };
                print!(
                    "current node ({}) is left sibling? {} node is ({})",
                    current_index, is_left_sibling, current_node
                );
                current_index = parent(current_index);
                println!(" with index {current_index}");
                current_merkle_tree_index >>= 1;
                current_height += 1;
                j += 1;
            }
            println!(
                "ended with node {current_index} which is in [{}]",
                new_peak_indices.iter().join(", ")
            );
            if !new_mmra
                .peaks()
                .into_iter()
                .zip(new_peak_heights.iter().zip(new_peak_indices.iter()))
                .any(|(p, (h, idx))| {
                    p == current_node && *h == current_height && *idx == current_index
                })
            {
                println!("cannot find calculated root in new peaks list.");
                println!("calculated node: ({})", current_node);
                println!("peaks: [{}]", new_mmra.peaks().into_iter().join("/"));
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use proptest::collection::vec;
    use proptest::prop_assert;
    use proptest_arbitrary_interop::arb;
    use rand::rngs::StdRng;
    use rand::thread_rng;
    use rand::Rng;
    use rand::RngCore;
    use rand::SeedableRng;
    use test_strategy::proptest;

    use super::MmrSuccessorProof;
    use crate::prelude::Digest;
    use crate::prelude::Mmr;
    use crate::util_types::mmr::mmr_accumulator::MmrAccumulator;

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
        let threshold = 7;
        for n in 0..threshold {
            for m in 0..threshold {
                verification_succeeds_with_n_leafs_append_m(n, m, &mut rng);
            }
        }
    }

    #[proptest]
    fn verification_succeeds_property(
        #[strategy(arb::<MmrAccumulator>())] old_mmr: MmrAccumulator,
        #[strategy(vec(arb::<Digest>(), 0usize..(1<<10)))] new_leafs: Vec<Digest>,
    ) {
        let mut new_mmr = old_mmr.clone();
        let mmr_successor_proof = MmrSuccessorProof::new_from_batch_append(&old_mmr, &new_leafs);
        for leaf in new_leafs {
            new_mmr.append(leaf);
        }

        prop_assert!(mmr_successor_proof.verify(&old_mmr, &new_mmr));
    }

    #[test]
    fn verification_succeeds_unit() {
        let mut rng: StdRng = SeedableRng::from_seed(
            hex::decode("deadbeef00000000deadbeef00000000deadbeef00000000deadbeef00000000")
                .unwrap()
                .try_into()
                .unwrap(),
        );
        let num_new_leafs = rng.gen_range(0..(1 << 10));
        let old_num_leafs = rng.gen_range(0u64..(u64::MAX >> 1));
        let old_peaks = (0..old_num_leafs.count_ones())
            .map(|_| rng.gen::<Digest>())
            .collect_vec();
        println!("num old peaks: {}", old_peaks.len());
        let old_mmr = MmrAccumulator::init(old_peaks, old_num_leafs);
        let mut new_mmr = old_mmr.clone();

        let new_leafs = (0..num_new_leafs)
            .map(|_| rng.gen::<Digest>())
            .collect_vec();
        let mmr_successor_proof = MmrSuccessorProof::new_from_batch_append(&old_mmr, &new_leafs);
        println!(
            "paths lengths: [{}]",
            mmr_successor_proof.paths.iter().map(|p| p.len()).join(", ")
        );
        for leaf in new_leafs {
            new_mmr.append(leaf);
        }

        assert!(mmr_successor_proof.verify(&old_mmr, &new_mmr));
    }
}
