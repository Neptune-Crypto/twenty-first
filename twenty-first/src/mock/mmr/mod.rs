mod mock_mmr;

pub use mock_mmr::MockMmr;

use crate::math::digest::Digest;
use crate::util_types::algebraic_hasher::AlgebraicHasher;

/// Return an empty in-memory archival MMR for testing purposes.
/// Does *not* have a unique ID, so you can't expect multiple of these
/// instances to behave independently unless you understand the
/// underlying data structure.
pub fn get_empty_mock_ammr<H: AlgebraicHasher>() -> MockMmr<H> {
    let pv: Vec<Digest> = Default::default();
    MockMmr::new(pv)
}

pub fn get_mock_ammr_from_digests<H>(digests: Vec<Digest>) -> MockMmr<H>
where
    H: AlgebraicHasher,
{
    let mut ammr = get_empty_mock_ammr();
    for digest in digests {
        ammr.append_raw(digest);
    }
    ammr
}

#[cfg(test)]
mod shared_tests_tests {
    use hashbrown::HashSet;
    use rand::{random, Rng};

    use crate::math::other::random_elements;
    use crate::util_types::mmr::mmr_accumulator::util::mmra_with_mps;
    use crate::{math::tip5::Tip5, util_types::mmr::mmr_trait::Mmr};
    use itertools::Itertools;

    use super::*;

    #[should_panic]
    #[test]
    fn disallow_repeated_leaf_indices_in_construction() {
        type H = Tip5;
        mmra_with_mps::<H>(14, vec![(0, random()), (0, random())]);
    }

    #[test]
    fn mmra_and_mps_construct_test_cornercases() {
        type H = Tip5;
        let mut rng = rand::thread_rng();
        for leaf_count in 0..5 {
            let (_mmra, _mps) = mmra_with_mps::<H>(leaf_count, vec![]);
        }
        let some: Digest = rng.gen();
        for leaf_count in 1..10 {
            for leaf_index in 0..leaf_count {
                let (mmra, mps) = mmra_with_mps::<H>(leaf_count, vec![(leaf_index, some)]);
                assert!(mps[0].verify(leaf_index, some, &mmra.get_peaks(), leaf_count));
            }
        }

        let other: Digest = rng.gen();
        for leaf_count in 2..10 {
            for some_index in 0..leaf_count {
                for other_index in 0..leaf_count {
                    if some_index == other_index {
                        continue;
                    }
                    let (mmra, mps) = mmra_with_mps::<H>(
                        leaf_count,
                        vec![(some_index, some), (other_index, other)],
                    );
                    assert!(mps[0].verify(some_index, some, &mmra.get_peaks(), leaf_count));
                    assert!(mps[1].verify(other_index, other, &mmra.get_peaks(), leaf_count));
                }
            }
        }

        // Full specification, set *all* leafs in MMR explicitly.
        for leaf_count in 3..10 {
            let specifications = (0..leaf_count).map(|i| (i, random())).collect_vec();
            let (mmra, mps) = mmra_with_mps::<H>(leaf_count, specifications.clone());
            for (mp, (leaf_index, leaf)) in mps.iter().zip(specifications) {
                assert!(mp.verify(leaf_index, leaf, &mmra.get_peaks(), leaf_count));
            }
        }
    }

    #[test]
    fn mmra_and_mps_construct_test_small() {
        type H = Tip5;
        let mut rng = rand::thread_rng();
        let digest_leaf_idx12: Digest = rng.gen();
        let digest_leaf_idx14: Digest = rng.gen();

        let (mmra, mps) =
            mmra_with_mps::<H>(32, vec![(12, digest_leaf_idx12), (14, digest_leaf_idx14)]);
        assert!(mps[0].verify(
            12,
            digest_leaf_idx12,
            &mmra.get_peaks(),
            mmra.count_leaves()
        ));
        assert!(mps[1].verify(
            14,
            digest_leaf_idx14,
            &mmra.get_peaks(),
            mmra.count_leaves()
        ));
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

                for (mp, (leaf_idx, leaf)) in mps.iter().zip_eq(specified_leafs) {
                    assert!(mp.verify(leaf_idx, leaf, &mmra.get_peaks(), leaf_count));
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

        for (mp, (leaf_idx, leaf)) in mps.iter().zip_eq(specified_leafs) {
            assert!(mp.verify(leaf_idx, leaf, &mmra.get_peaks(), leaf_count));
        }
    }
}
