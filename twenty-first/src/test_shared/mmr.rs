use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use itertools::Itertools;
use rusty_leveldb::DB;

use crate::shared_math::digest::Digest;
use crate::shared_math::other::{log_2_ceil, random_elements};
use crate::util_types::mmr::mmr_accumulator::MmrAccumulator;
use crate::util_types::mmr::mmr_membership_proof::MmrMembershipProof;
use crate::util_types::mmr::shared_advanced::right_lineage_length_from_node_index;
use crate::util_types::mmr::shared_basic::{self, leaf_index_to_mt_index_and_peak_index};
use crate::util_types::storage_vec::RustyLevelDbVec;
use crate::util_types::{algebraic_hasher::AlgebraicHasher, mmr::archival_mmr::ArchivalMmr};
use crate::utils::has_unique_elements;

/// Return an empty in-memory archival MMR for testing purposes.
/// Does *not* have a unique ID, so you can't expect multiple of these
/// instances to behave independently unless you understand the
/// underlying data structure.
pub fn get_empty_rustyleveldb_ammr<H: AlgebraicHasher>() -> ArchivalMmr<H, RustyLevelDbVec<Digest>>
{
    let opt = rusty_leveldb::in_memory();
    let db = DB::open("mydatabase", opt).unwrap();
    let db = Arc::new(Mutex::new(db));
    let pv = RustyLevelDbVec::new(db, 0, "in-memory AMMR for unit tests");
    ArchivalMmr::new(pv)
}

pub fn get_rustyleveldb_ammr_from_digests<H>(
    digests: Vec<Digest>,
) -> ArchivalMmr<H, RustyLevelDbVec<Digest>>
where
    H: AlgebraicHasher,
{
    let mut ammr = get_empty_rustyleveldb_ammr();
    for digest in digests {
        ammr.append_raw(digest);
    }

    ammr
}

/// Get an MMR accumulator with a requested number of leafs, and requested leaf digests at specified indices
/// Also returns the MMR membership proofs for the specified leafs.
pub fn mmra_with_mps<H: AlgebraicHasher>(
    leaf_count: u64,
    specified_leafs: Vec<(u64, Digest)>,
) -> (MmrAccumulator<H>, Vec<MmrMembershipProof<H>>) {
    assert!(
        has_unique_elements(specified_leafs.iter().map(|x| x.0)),
        "Specified leaf indices must be unique"
    );

    // initial_setup
    let mut peaks: Vec<Digest> = random_elements(leaf_count.count_ones() as usize);
    if specified_leafs.is_empty() {
        return (MmrAccumulator::init(peaks, leaf_count), vec![]);
    }

    let (first_leaf_index, first_specified_digest) = specified_leafs[0];
    let (first_mt_index, first_peak_index) =
        leaf_index_to_mt_index_and_peak_index(first_leaf_index, leaf_count);

    // Change peaks such that the 1st specification belongs in the MMR
    let first_mt_height = log_2_ceil(first_mt_index as u128 + 1) - 1;
    let first_ap: Vec<Digest> = random_elements(first_mt_height as usize);

    let mut all_leaf_indices = vec![first_mt_index];
    let first_mp = MmrMembershipProof::<H>::new(first_leaf_index, first_ap);
    let original_node_indices = first_mp.get_node_indices();
    let mut derivable_node_values: HashMap<u64, Digest> = HashMap::default();
    let mut first_acc_hash = first_specified_digest;
    for (height, node_index_in_path) in first_mp.get_direct_path_indices().into_iter().enumerate() {
        derivable_node_values.insert(node_index_in_path, first_acc_hash);
        if first_mp.authentication_path.len() > height {
            if right_lineage_length_from_node_index(node_index_in_path) != 0 {
                first_acc_hash = H::hash_pair(first_mp.authentication_path[height], first_acc_hash);
            } else {
                first_acc_hash = H::hash_pair(first_acc_hash, first_mp.authentication_path[height]);
            }
        }
    }

    // Update root
    peaks[first_peak_index as usize] = first_acc_hash;

    let mut all_ap_elements: HashMap<u64, Digest> = original_node_indices
        .into_iter()
        .zip_eq(first_mp.authentication_path.clone())
        .collect();
    let mut all_mps = vec![first_mp];
    let mut all_leaves = vec![first_specified_digest];

    for (new_leaf_index, new_leaf) in specified_leafs.into_iter().skip(1) {
        let (new_leaf_mt_index, _new_leaf_peaks_index) =
            leaf_index_to_mt_index_and_peak_index(new_leaf_index, leaf_count);
        let height_of_new_mt = log_2_ceil(new_leaf_mt_index as u128 + 1) - 1;
        let mut new_mp = MmrMembershipProof::<H>::new(
            new_leaf_index,
            random_elements(height_of_new_mt as usize),
        );
        let new_node_indices = new_mp.get_node_indices();

        for (height, new_node_index) in new_node_indices.iter().enumerate() {
            if all_ap_elements.contains_key(new_node_index) {
                // AP element may not be mutated
                new_mp.authentication_path[height] = all_ap_elements[new_node_index];
            } else if derivable_node_values.contains_key(new_node_index) {
                // AP element must refer to both old and new leaf
                new_mp.authentication_path[height] = derivable_node_values[new_node_index];
            }
        }

        let new_peaks = shared_basic::calculate_new_peaks_from_leaf_mutation::<H>(
            &peaks, new_leaf, leaf_count, &new_mp,
        );
        assert!(new_mp.verify(&new_peaks, new_leaf, leaf_count).0);
        for (j, mp) in all_mps.iter().enumerate() {
            assert!(mp.verify(&peaks, all_leaves[j], leaf_count).0);
        }
        let mutated = MmrMembershipProof::batch_update_from_batch_leaf_mutation(
            &mut all_mps.iter_mut().collect_vec(),
            vec![(new_mp.clone(), new_leaf)],
        );

        // Sue me
        for muta in mutated.iter() {
            let mp = &all_mps[*muta];
            for (hght, idx) in mp.get_node_indices().iter().enumerate() {
                all_ap_elements.insert(*idx, mp.authentication_path[hght]);
            }
        }

        for (j, mp) in all_mps.iter().enumerate() {
            assert!(mp.verify(&new_peaks, all_leaves[j], leaf_count).0);
        }

        // Update derivable node values
        let mut acc_hash = new_leaf;
        for (height, node_index_in_path) in new_mp.get_direct_path_indices().into_iter().enumerate()
        {
            if height == new_mp.get_direct_path_indices().len() - 1 {
                break;
            }
            derivable_node_values.insert(node_index_in_path, acc_hash);
            if right_lineage_length_from_node_index(node_index_in_path) != 0 {
                acc_hash = H::hash_pair(new_mp.authentication_path[height], acc_hash);
            } else {
                acc_hash = H::hash_pair(acc_hash, new_mp.authentication_path[height]);
            }
        }

        // Update all_ap_elements
        for (node_index, ap_element) in new_node_indices
            .into_iter()
            .zip_eq(new_mp.authentication_path.clone().into_iter())
        {
            all_ap_elements.insert(node_index, ap_element);
        }

        all_mps.push(new_mp);
        peaks = new_peaks;
        all_leaves.push(new_leaf);
        all_leaf_indices.push(new_leaf_index);
    }

    (MmrAccumulator::init(peaks, leaf_count), all_mps)
}

#[cfg(test)]
mod shared_tests_tests {
    use hashbrown::HashSet;
    use rand::{random, Rng};

    use crate::{shared_math::tip5::Tip5, util_types::mmr::mmr_trait::Mmr};

    use super::*;

    #[should_panic]
    #[test]
    fn disallow_repeated_leaf_indices_in_construction() {
        type H = blake3::Hasher;
        mmra_with_mps::<H>(14, vec![(0, random()), (0, random())]);
    }

    #[test]
    fn mmra_and_mps_construct_test_cornercases() {
        type H = blake3::Hasher;
        let mut rng = rand::thread_rng();
        for leaf_count in 0..5 {
            let (_mmra, _mps) = mmra_with_mps::<H>(leaf_count, vec![]);
        }
        let some: Digest = rng.gen();
        for leaf_count in 1..10 {
            for index in 0..leaf_count {
                let (mmra, mps) = mmra_with_mps::<H>(leaf_count, vec![(index, some)]);
                assert!(mps[0].verify(&mmra.get_peaks(), some, leaf_count).0);
            }
        }

        let other: Digest = rng.gen();
        for leaf_count in 2..10 {
            for first_index in 0..leaf_count {
                for second_index in 0..leaf_count {
                    if first_index == second_index {
                        continue;
                    }
                    let (mmra, mps) = mmra_with_mps::<H>(
                        leaf_count,
                        vec![(first_index, some), (second_index, other)],
                    );
                    assert!(mps[0].verify(&mmra.get_peaks(), some, leaf_count).0);
                    assert!(mps[1].verify(&mmra.get_peaks(), other, leaf_count).0);
                }
            }
        }

        // Full specification
        for leaf_count in 3..10 {
            let specifications = (0..leaf_count).map(|i| (i, random())).collect_vec();
            let (mmra, mps) = mmra_with_mps::<H>(leaf_count, specifications.clone());
            for (mp, leaf) in mps.iter().zip(specifications.iter().map(|x| x.1)) {
                assert!(mp.verify(&mmra.get_peaks(), leaf, leaf_count).0);
            }
        }
    }

    #[test]
    fn mmra_and_mps_construct_test_small() {
        type H = blake3::Hasher;
        let mut rng = rand::thread_rng();
        let some_digest: Digest = rng.gen();
        let other_digest: Digest = rng.gen();

        let (mmra, mps) = mmra_with_mps::<H>(32, vec![(12, some_digest), (14, other_digest)]);
        assert!(
            mps[0]
                .verify(&mmra.get_peaks(), some_digest, mmra.count_leaves())
                .0
        );
        assert!(
            mps[1]
                .verify(&mmra.get_peaks(), other_digest, mmra.count_leaves())
                .0
        );
    }

    #[test]
    fn mmra_and_mps_construct_test_pbt() {
        type H = Tip5;
        let mut rng = rand::thread_rng();

        for leaf_count in 2..25 {
            for specified_count in 0..leaf_count {
                let mut specified_indices: HashSet<u64> = HashSet::default();
                for _ in 0..specified_count {
                    specified_indices.insert(rng.gen_range(0..leaf_count));
                }

                let collected_values = specified_indices.len();
                let specified_leafs: Vec<(u64, Digest)> = specified_indices
                    .into_iter()
                    .zip_eq(random_elements(collected_values))
                    .collect_vec();
                let (mmra, mps) = mmra_with_mps::<H>(leaf_count, specified_leafs.clone());

                for (mp, leaf) in mps.iter().zip_eq(specified_leafs.iter().map(|x| x.1)) {
                    assert!(mp.verify(&mmra.get_peaks(), leaf, leaf_count).0);
                }
            }
        }
    }

    #[test]
    fn mmra_and_mps_construct_test_big() {
        type H = Tip5;
        let mut rng = rand::thread_rng();
        let leaf_count = (1 << 59) + (1 << 44) + 1234567890;
        let specified_count = 40;
        let mut specified_indices: HashSet<u64> = HashSet::default();
        for _ in 0..specified_count {
            specified_indices.insert(rng.gen_range(0..leaf_count));
        }

        let collected_values = specified_indices.len();
        let specified_leafs: Vec<(u64, Digest)> = specified_indices
            .into_iter()
            .zip_eq(random_elements(collected_values))
            .collect_vec();
        let (mmra, mps) = mmra_with_mps::<H>(leaf_count, specified_leafs.clone());

        for (mp, leaf) in mps.iter().zip_eq(specified_leafs.iter().map(|x| x.1)) {
            assert!(mp.verify(&mmra.get_peaks(), leaf, leaf_count).0);
        }
    }
}
