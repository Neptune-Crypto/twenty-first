mod mock_mmr;

pub use mock_mmr::MockMmr;

use crate::shared_math::digest::Digest;
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

    use crate::shared_math::other::random_elements;
    use crate::util_types::mmr::mmr_accumulator::util::mmra_with_mps;
    use crate::{shared_math::tip5::Tip5, util_types::mmr::mmr_trait::Mmr};
    use itertools::Itertools;

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
