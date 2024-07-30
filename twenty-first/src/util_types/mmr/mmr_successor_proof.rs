use bfieldcodec_derive::BFieldCodec;
use itertools::Itertools;

use crate::{
    prelude::{AlgebraicHasher, Digest, Mmr, Tip5},
    util_types::mmr::shared_advanced::node_indices_added_by_append,
};

use super::{
    mmr_accumulator::MmrAccumulator,
    shared_advanced::{get_peak_heights_and_peak_node_indices, parent, right_sibling},
    shared_basic::calculate_new_peaks_from_append,
};

#[derive(Debug, Clone, BFieldCodec)]
pub struct MmrSuccessorProof {
    pub paths: Vec<Vec<Digest>>,
}

impl MmrSuccessorProof {
    pub fn new_from_batch_append(mmra: &MmrAccumulator, new_leafs: &[Digest]) -> Self {
        let (heights_of_old_peaks, indices_of_old_peaks) =
            get_peak_heights_and_peak_node_indices(mmra.num_leafs());
        let (_heights_of_new_peaks, indices_of_new_peaks) =
            get_peak_heights_and_peak_node_indices(mmra.num_leafs() + new_leafs.len() as u64);
        let num_old_peaks = heights_of_old_peaks.len();

        let mut needed_indices = vec![vec![]; num_old_peaks];
        for (i, index) in indices_of_old_peaks.into_iter().enumerate() {
            let mut current_index = index;
            while !indices_of_new_peaks.contains(&current_index) {
                needed_indices[i].push(right_sibling(current_index, heights_of_old_peaks[i]));
                current_index = parent(current_index);
            }
        }

        let mut current_peaks = mmra.peaks();
        let mut current_leaf_count = mmra.num_leafs();
        let mut paths = vec![vec![]; num_old_peaks];

        for &new_leaf in new_leafs {
            let new_node_indices = node_indices_added_by_append(current_leaf_count);

            let (new_peaks, membership_proof) =
                calculate_new_peaks_from_append(current_leaf_count, current_peaks, new_leaf);
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
                for (i, path) in paths.iter_mut().enumerate() {
                    if needed_indices[i].contains(&index) {
                        path.push(node);
                    }
                }
            }

            current_peaks = new_peaks;
            current_leaf_count += 1;
        }

        Self { paths }
    }
}
