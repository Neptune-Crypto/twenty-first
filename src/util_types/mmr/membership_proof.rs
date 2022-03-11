use crate::util_types::simple_hasher::{Hasher, ToDigest};
use std::{
    collections::{hash_map::RandomState, hash_set::Intersection, HashMap, HashSet},
    fmt::Debug,
    iter::FromIterator,
    marker::PhantomData,
};

use super::shared::{
    data_index_to_node_index, get_authentication_path_node_indices, get_peak_height,
    get_peak_heights_and_peak_node_indices, leaf_count_to_node_count, left_sibling,
    node_indices_added_by_append, parent, right_child_and_height, right_sibling,
};

#[derive(Debug, Clone)]
pub struct MembershipProof<HashDigest, H>
where
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
    H: Hasher<Digest = HashDigest> + Clone,
{
    pub data_index: u128,
    pub authentication_path: Vec<HashDigest>,
    pub _hasher: PhantomData<H>,
}

impl<HashDigest: PartialEq, H> PartialEq for MembershipProof<HashDigest, H>
where
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
    H: Hasher<Digest = HashDigest> + Clone,
{
    // Two membership proofs are considered equal if they contain the same authentication path
    // *and* point to the same data index
    fn eq(&self, other: &Self) -> bool {
        self.data_index == other.data_index && self.authentication_path == other.authentication_path
    }
}

impl<HashDigest, H> MembershipProof<HashDigest, H>
where
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
    H: Hasher<Digest = HashDigest> + Clone,
{
    /// Verify a membership proof for an MMR
    pub fn verify(
        &self,
        peaks: &[HashDigest],
        leaf_hash: &HashDigest,
        leaf_count: u128,
    ) -> (bool, Option<HashDigest>)
    where
        HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
        H: Hasher<Digest = HashDigest> + Clone,
        u128: ToDigest<HashDigest>,
    {
        let node_index = data_index_to_node_index(self.data_index);

        let mut hasher = H::new();
        let mut acc_hash: HashDigest = leaf_hash.to_owned();
        let mut acc_index: u128 = node_index;
        for hash in self.authentication_path.iter() {
            let (acc_right, _acc_height) = right_child_and_height(acc_index);
            acc_hash = if acc_right {
                hasher.hash_two(hash, &acc_hash)
            } else {
                hasher.hash_two(&acc_hash, hash)
            };
            acc_index = parent(acc_index);
        }

        // Find the correct peak index
        let (heights, _) = get_peak_heights_and_peak_node_indices(leaf_count);
        if heights.len() != peaks.len() {
            return (false, None);
        }
        let expected_peak_height_res = get_peak_height(leaf_count, self.data_index);
        let expected_peak_height = match expected_peak_height_res {
            None => return (false, None),
            Some(eph) => eph,
        };
        let peak_index_res = heights.into_iter().position(|x| x == expected_peak_height);
        let peak_index = match peak_index_res {
            None => return (false, None),
            Some(pi) => pi,
        };

        // Compare the peak at the expected index with accumulated hash
        if peaks[peak_index] != acc_hash {
            return (false, None);
        }

        (true, Some(acc_hash))
    }

    /// Return the node indices for the authentication path in this membership proof
    fn get_node_indices(&self) -> Vec<u128> {
        let mut node_index = data_index_to_node_index(self.data_index);
        let mut node_indices = vec![];
        for _ in 0..self.authentication_path.len() {
            let (right, height) = right_child_and_height(node_index);
            if right {
                node_indices.push(left_sibling(node_index, height));
            } else {
                node_indices.push(right_sibling(node_index, height));
            }
            node_index = parent(node_index);
        }

        node_indices
    }

    /// Return the node indices for the hash values that can be derived from this proof
    fn get_direct_path_indices(&self) -> Vec<u128> {
        let mut node_index = data_index_to_node_index(self.data_index);
        let mut node_indices = vec![node_index];
        for _ in 0..self.authentication_path.len() {
            node_index = parent(node_index);
            node_indices.push(node_index);
        }

        node_indices
    }

    /// Return the node index of the peak that the membership proof is pointing
    /// to, as well as this peak's height.
    fn get_peak_index_and_height(&self) -> (u128, u32) {
        (
            *self.get_direct_path_indices().last().unwrap(),
            self.authentication_path.len() as u32,
        )
    }

    /// Update a membership proof with a `verify_append` proof. Returns `true` if an
    /// authentication path has been mutated, false otherwise.
    pub fn update_from_append(
        &mut self,
        old_leaf_count: u128,
        new_leaf: &HashDigest,
        old_peaks: &[HashDigest],
    ) -> bool {
        // 1. Get index of authentication paths's peak
        // 2. Get node indices for nodes added by the append
        // 3. Check if authentication path's peak's parent is present in the added nodes (peak can only be left child)
        //   a. If not, then we are done, return from method
        // 4. Get the indices that auth path must be extended with
        // 5. Get all derivable node digests, store in hash map
        //   a. Get the node digests from the previous peaks
        //   b. Get the node digests that can be calculated by hashing from the new leaf
        // 6. Push required digests to the authentication path

        // 1
        let (old_peak_index, old_peak_height) = self.get_peak_index_and_height();

        // 2
        let added_node_indices = node_indices_added_by_append(old_leaf_count);

        // 3
        // Any peak is a left child, so we don't have to check if it's a right or left child.
        // This means we can use a faster method to find the parent index than the generic method.
        let peak_parent_index = old_peak_index + (1 << (old_peak_height + 1));

        // 3a
        if !added_node_indices.contains(&peak_parent_index) {
            return false;
        }

        // 4 Get node indices of missing digests
        let new_peak_index: u128 = *added_node_indices.last().unwrap();
        let new_node_count: u128 = leaf_count_to_node_count(old_leaf_count + 1);
        let node_indices_for_missing_digests: Vec<u128> =
            get_authentication_path_node_indices(old_peak_index, new_peak_index, new_node_count)
                .unwrap();

        // 5 collect all derivable peaks in a hashmap indexed by node index
        // 5.a, collect all node hash digests that are present in the old peaks
        // The keys in the hash map are node indices
        let mut known_digests: HashMap<u128, HashDigest> = HashMap::new();
        let (_old_peak_heights, old_peak_indices) =
            get_peak_heights_and_peak_node_indices(old_leaf_count);
        for (old_peak_index, old_peak_digest) in old_peak_indices.iter().zip(old_peaks.iter()) {
            known_digests.insert(*old_peak_index, old_peak_digest.to_owned());
        }

        // 5.b collect all node hash digests that are derivable from `new_leaf` and
        // `old_peaks`. These are the digests of `new_leaf`'s path to the root.
        // break out of loop once *one* digest is found this way since that will
        // always suffice.
        let mut acc_hash = new_leaf.to_owned();
        let mut hasher = H::new();
        for (node_index, old_peak_digest) in added_node_indices.iter().zip(old_peaks.iter().rev()) {
            known_digests.insert(*node_index, acc_hash.to_owned());

            // peaks are always left children, so we don't have to check for that
            acc_hash = hasher.hash_two(old_peak_digest, &acc_hash);

            // once we encouter the first of the needed accumulator indices,
            // we can break. Just like we could in the update for the leaf update
            // membership proof update.
            // The reason for this break is that the authentication path consists of
            // commits to disjoint sets, so anything that can be derived from a
            // hash that is part of the missing digests cannot possible be an
            // element in an authentication path
            if node_indices_for_missing_digests.contains(node_index) {
                break;
            }
        }

        // 6
        for missing_digest_node_index in node_indices_for_missing_digests {
            self.authentication_path
                .push(known_digests[&missing_digest_node_index].clone());
        }

        true
    }

    /// Batch update multiple membership proofs.
    /// Returns the indices of the membership proofs that were modified where index refers
    /// to the order in which the membership proofs were given to this function.
    pub fn batch_update_from_append(
        membership_proofs: &mut [Self],
        old_leaf_count: u128,
        new_leaf: &HashDigest,
        old_peaks: &[HashDigest],
    ) -> Vec<u128> {
        // 1. Get node indices for nodes added by the append
        //   a. If length of this list is one, newly added leaf was a left child. Return.
        // 2. Get all derivable node digests, store in hash map
        let added_node_indices = node_indices_added_by_append(old_leaf_count);
        if added_node_indices.len() == 1 {
            return vec![];
        }

        // 2 collect all derivable peaks in a hashmap indexed by node index
        // 2.a, collect all node hash digests that are present in the old peaks
        // The keys in the hash map are node indices
        let mut known_digests: HashMap<u128, HashDigest> = HashMap::new();
        let (_old_peak_heights, old_peak_indices) =
            get_peak_heights_and_peak_node_indices(old_leaf_count);
        for (old_peak_index, old_peak_digest) in old_peak_indices.iter().zip(old_peaks.iter()) {
            known_digests.insert(*old_peak_index, old_peak_digest.to_owned());
        }

        // 2.b collect all node hash digests that are derivable from `new_leaf` and
        // `old_peaks`. These are the digests of `new_leaf`'s path to the root.
        let mut acc_hash = new_leaf.to_owned();
        let mut hasher = H::new();
        for ((count, node_index), old_peak_digest) in added_node_indices
            .iter()
            .enumerate()
            .zip(old_peaks.iter().rev())
        {
            known_digests.insert(*node_index, acc_hash.to_owned());

            // The last index in `added_node_indices` is the new peak
            // and the 2nd last will hash to the digest of the new peak,
            // so we can skip the last two values from this list
            if count == added_node_indices.len() - 2 {
                break;
            }

            // peaks are always left children, so we don't have to check for that
            acc_hash = hasher.hash_two(old_peak_digest, &acc_hash);
        }

        // Loop over all membership proofs and insert missing hashes for each
        let mut modified: Vec<u128> = vec![];
        let new_peak_index: u128 = *added_node_indices.last().unwrap();
        let new_node_count: u128 = leaf_count_to_node_count(old_leaf_count + 1);
        for (i, membership_proof) in membership_proofs.iter_mut().enumerate() {
            let (old_peak_index, old_peak_height) = membership_proof.get_peak_index_and_height();

            // Any peak is a left child, so we don't have to check if it's a right or left child.
            // This means we can use a faster method to find the parent index than the generic method.
            let peak_parent_index = old_peak_index + (1 << (old_peak_height + 1));
            if !added_node_indices.contains(&peak_parent_index) {
                continue;
            }

            modified.push(i as u128);

            let node_indices_for_missing_digests: Vec<u128> = get_authentication_path_node_indices(
                old_peak_index,
                new_peak_index,
                new_node_count,
            )
            .unwrap();

            // Sanity check
            debug_assert!(
                !node_indices_for_missing_digests.is_empty(),
                "authentication path must be missing digests at this point"
            );

            for missing_digest_node_index in node_indices_for_missing_digests {
                membership_proof
                    .authentication_path
                    .push(known_digests[&missing_digest_node_index].clone());
            }
        }

        modified
    }

    /// Update a membership proof with a `leaf_update` proof. For the `membership_proof`
    /// parameter, it doesn't matter if you use the old or new membership proof associated
    /// with the leaf update, as they are the same before and after the leaf update.
    pub fn update_from_leaf_update(
        &mut self,
        leaf_update_membership_proof: &MembershipProof<HashDigest, H>,
        new_leaf: &HashDigest,
    ) -> bool {
        // TODO: This function could also return the new peak and perhaps the peaks index.
        // this way, this function could also be used to update a `MmrAccumulator` struct
        // and not just a membership proof.
        let own_node_ap_indices = self.get_node_indices();
        let affected_node_indices = leaf_update_membership_proof.get_direct_path_indices();
        let own_node_indices_hash_set: HashSet<u128> =
            HashSet::from_iter(own_node_ap_indices.clone());
        let affected_node_indices_hash_set: HashSet<u128> =
            HashSet::from_iter(affected_node_indices);
        let mut intersection: Intersection<u128, RandomState> =
            own_node_indices_hash_set.intersection(&affected_node_indices_hash_set);

        // If intersection is empty no change is needed
        let intersection_index_res: Option<&u128> = intersection.next();
        let intersection_index: u128 = match intersection_index_res {
            None => return false,
            Some(&index) => index,
        };

        // Sanity check, should always be true, since `intersection` can at most
        // contain *one* element.
        assert!(intersection.next().is_none());

        // If intersection is **not** empty, we need to calculate all deducible node hashes from the
        // `membership_proof` until we meet the intersecting node.
        let mut deducible_hashes: HashMap<u128, HashDigest> = HashMap::new();
        let mut node_index = data_index_to_node_index(leaf_update_membership_proof.data_index);
        deducible_hashes.insert(node_index, new_leaf.clone());
        let mut hasher = H::new();
        let mut acc_hash: HashDigest = new_leaf.to_owned();

        // Calculate hashes from the bottom towards the peak. Break when
        // the intersecting node is reached.
        for hash in leaf_update_membership_proof.authentication_path.iter() {
            // It's not necessary to calculate all the way to the root since,
            // the intersection set has a size of at most one (I think).
            // So we can break the loop when we find a `node_index` that
            // is equal to the intersection index. This way we same some
            // hash calculations here.
            if intersection_index == node_index {
                break;
            }

            let (acc_right, _acc_height) = right_child_and_height(node_index);
            acc_hash = if acc_right {
                hasher.hash_two(hash, &acc_hash)
            } else {
                hasher.hash_two(&acc_hash, hash)
            };
            node_index = parent(node_index);
            deducible_hashes.insert(node_index, acc_hash.clone());
        }

        // Some of the hashes in `self` need to be updated. We can loop over
        // `own_node_indices` and check if the element is contained `deducible_hashes`.
        // If it is, then the appropriate element in `self.authentication_path` needs to
        // be replaced with an element from `deducible_hashes`.
        for (digest, own_node_index) in self
            .authentication_path
            .iter_mut()
            .zip(own_node_ap_indices.into_iter())
        {
            if !deducible_hashes.contains_key(&own_node_index) {
                continue;
            }
            *digest = deducible_hashes[&own_node_index].clone();
        }

        true
    }

    /// Update multiple membership proofs with a `leaf_update` proof. For the `membership_proof`
    /// parameter, it doesn't matter if you use the old or new membership proof associated
    /// with the leaf update, as they are the same before and after the leaf update.
    /// Returns the indices of the membership proofs that were modified where index refers
    /// to the order in which the membership proofs were given to this function.
    pub fn batch_update_from_leaf_mutation(
        membership_proofs: &mut [Self],
        leaf_update_membership_proof: &MembershipProof<HashDigest, H>,
        new_leaf: &HashDigest,
    ) -> Vec<u128> {
        // 1. Calculate all hashes that are deducible from the leaf update
        // 2. Iterate through all membership proofs and update digests that
        //    are deducible from the leaf update proof.

        let mut deducible_hashes: HashMap<u128, HashDigest> = HashMap::new();
        let mut node_index = data_index_to_node_index(leaf_update_membership_proof.data_index);
        deducible_hashes.insert(node_index, new_leaf.clone());
        let mut hasher = H::new();
        let mut acc_hash: HashDigest = new_leaf.to_owned();

        // Calculate hashes from the bottom towards the peak. Break before we
        // calculate the hash of the peak, since peaks are never included in
        // authentication paths
        for (count, hash) in leaf_update_membership_proof
            .authentication_path
            .iter()
            .enumerate()
        {
            // Do not calculate the last hash as it will always be a peak which
            // are never included in the authentication path
            if count == leaf_update_membership_proof.authentication_path.len() - 1 {
                break;
            }
            let (acc_right, _acc_height) = right_child_and_height(node_index);
            acc_hash = if acc_right {
                hasher.hash_two(hash, &acc_hash)
            } else {
                hasher.hash_two(&acc_hash, hash)
            };
            node_index = parent(node_index);
            deducible_hashes.insert(node_index, acc_hash.clone());
        }

        let mut modified_membership_proofs: Vec<u128> = vec![];
        for (i, membership_proof) in membership_proofs.iter_mut().enumerate() {
            let ap_indices = membership_proof.get_node_indices();

            // Some of the hashes in may `membership_proof` need to be updated. We can loop over
            // `authentication_path_indices` and check if the element is contained `deducible_hashes`.
            // If it is, then the appropriate element in `membership_proof.authentication_path` needs to
            // be replaced with an element from `deducible_hashes`.
            for (digest, authentication_path_indices) in membership_proof
                .authentication_path
                .iter_mut()
                .zip(ap_indices.into_iter())
            {
                // Maximum 1 digest can be updated in each authentication path
                // so if that is encountered, we might as well break and go to
                // the next membership proof
                if deducible_hashes.contains_key(&authentication_path_indices) {
                    *digest = deducible_hashes[&authentication_path_indices].clone();
                    modified_membership_proofs.push(i as u128);
                    break;
                }
            }
        }

        modified_membership_proofs
    }
}

#[cfg(test)]
mod mmr_membership_proof_test {
    use crate::{
        shared_math::b_field_element::BFieldElement,
        util_types::mmr::mmr_accumulator::MmrAccumulator,
        util_types::mmr::{archival_mmr::ArchivalMmr, mmr_trait::Mmr},
        util_types::simple_hasher::RescuePrimeProduction,
    };

    use super::*;

    #[test]
    fn equality_test() {
        let mp0: MembershipProof<blake3::Hash, blake3::Hasher> = MembershipProof {
            authentication_path: vec![],
            data_index: 4,
            _hasher: PhantomData,
        };
        let mp1: MembershipProof<blake3::Hash, blake3::Hasher> = MembershipProof {
            authentication_path: vec![],
            data_index: 4,
            _hasher: PhantomData,
        };
        let mp2: MembershipProof<blake3::Hash, blake3::Hasher> = MembershipProof {
            authentication_path: vec![],
            data_index: 3,
            _hasher: PhantomData,
        };
        let mp3: MembershipProof<blake3::Hash, blake3::Hasher> = MembershipProof {
            authentication_path: vec![blake3::hash(b"foobarbaz")],
            data_index: 4,
            _hasher: PhantomData,
        };
        let mp4: MembershipProof<blake3::Hash, blake3::Hasher> = MembershipProof {
            authentication_path: vec![blake3::hash(b"foobarbaz")],
            data_index: 4,
            _hasher: PhantomData,
        };
        assert_eq!(mp0, mp1);
        assert_ne!(mp1, mp2);
        assert_ne!(mp2, mp3);
        assert_eq!(mp3, mp4);
        assert_ne!(mp3, mp0);
    }

    #[test]
    fn get_node_indices_simple_test() {
        let leaf_hashes: Vec<blake3::Hash> = (14u128..14 + 8)
            .map(|x| blake3::hash(bincode::serialize(&x).expect("Encoding failed").as_slice()))
            .collect();
        let archival_mmr = ArchivalMmr::<blake3::Hash, blake3::Hasher>::new(leaf_hashes);
        let (membership_proof, _peaks): (
            MembershipProof<blake3::Hash, blake3::Hasher>,
            Vec<blake3::Hash>,
        ) = archival_mmr.prove_membership(4);
        assert_eq!(vec![9, 13, 7], membership_proof.get_node_indices());
        assert_eq!(
            vec![8, 10, 14, 15],
            membership_proof.get_direct_path_indices()
        );
    }

    #[test]
    fn get_peak_index_simple_test() {
        let mut mmr_size = 7;
        let leaf_hashes: Vec<blake3::Hash> = (14u128..14 + mmr_size)
            .map(|x| blake3::hash(bincode::serialize(&x).expect("Encoding failed").as_slice()))
            .collect();
        let mut archival_mmr = ArchivalMmr::<blake3::Hash, blake3::Hasher>::new(leaf_hashes);
        let mut expected_peak_indices_and_heights: Vec<(u128, u32)> =
            vec![(7, 2), (7, 2), (7, 2), (7, 2), (10, 1), (10, 1), (11, 0)];
        for (i, expected_peak_index) in
            (0..mmr_size).zip(expected_peak_indices_and_heights.into_iter())
        {
            let (membership_proof, _peaks): (
                MembershipProof<blake3::Hash, blake3::Hasher>,
                Vec<blake3::Hash>,
            ) = archival_mmr.prove_membership(i);
            assert_eq!(
                expected_peak_index,
                membership_proof.get_peak_index_and_height()
            );
        }

        // Increase size to 8 and verify that the peaks are now different
        mmr_size = 8;
        archival_mmr.append(blake3::hash(
            bincode::serialize(&1337u128)
                .expect("Encoding failed")
                .as_slice(),
        ));
        expected_peak_indices_and_heights = vec![(15, 3); mmr_size as usize];
        for (i, expected_peak_index) in
            (0..mmr_size).zip(expected_peak_indices_and_heights.into_iter())
        {
            let (membership_proof, _peaks): (
                MembershipProof<blake3::Hash, blake3::Hasher>,
                Vec<blake3::Hash>,
            ) = archival_mmr.prove_membership(i);
            assert_eq!(
                expected_peak_index,
                membership_proof.get_peak_index_and_height()
            );
        }

        // Increase size to 9 and verify that the peaks are now different
        mmr_size = 9;
        archival_mmr.append(blake3::hash(
            bincode::serialize(&13337u128)
                .expect("Encoding failed")
                .as_slice(),
        ));
        expected_peak_indices_and_heights = vec![
            (15, 3),
            (15, 3),
            (15, 3),
            (15, 3),
            (15, 3),
            (15, 3),
            (15, 3),
            (15, 3),
            (16, 0),
        ];
        for (i, expected_peak_index) in
            (0..mmr_size).zip(expected_peak_indices_and_heights.into_iter())
        {
            let (membership_proof, _peaks): (
                MembershipProof<blake3::Hash, blake3::Hasher>,
                Vec<blake3::Hash>,
            ) = archival_mmr.prove_membership(i);
            assert_eq!(
                expected_peak_index,
                membership_proof.get_peak_index_and_height()
            );
        }
    }

    #[test]
    fn update_membership_proof_from_leaf_update_test() {
        let leaf_hashes: Vec<blake3::Hash> = (14u128..14 + 8)
            .map(|x| blake3::hash(bincode::serialize(&x).expect("Encoding failed").as_slice()))
            .collect();
        let new_leaf = blake3::hash(
            bincode::serialize(&133337u128)
                .expect("Encoding failed")
                .as_slice(),
        );
        let mut accumulator_mmr =
            MmrAccumulator::<blake3::Hash, blake3::Hasher>::new(leaf_hashes.clone());
        assert_eq!(8, accumulator_mmr.count_leaves());
        let mut archival_mmr =
            ArchivalMmr::<blake3::Hash, blake3::Hasher>::new(leaf_hashes.clone());
        let original_archival_mmr = archival_mmr.clone();
        let (mut membership_proof, _peaks): (
            MembershipProof<blake3::Hash, blake3::Hasher>,
            Vec<blake3::Hash>,
        ) = archival_mmr.prove_membership(4);

        // 1. Update a leaf in both the accumulator MMR and in the archival MMR
        let old_peaks = archival_mmr.get_peaks();
        let membership_proof_for_manipulated_leaf = archival_mmr.prove_membership(2).0;
        archival_mmr.update_leaf_raw(2, new_leaf);
        accumulator_mmr.mutate_leaf(&membership_proof_for_manipulated_leaf, &new_leaf);
        assert_eq!(archival_mmr.get_peaks(), accumulator_mmr.get_peaks());
        let new_peaks = archival_mmr.get_peaks();
        assert_ne!(
            new_peaks, old_peaks,
            "Peaks must change when leaf is mutated"
        );
        let (real_membership_proof_from_archival, archival_peaks) =
            archival_mmr.prove_membership(4);
        assert_eq!(
            new_peaks, archival_peaks,
            "peaks returned from `get_peaks` must match that returned with membership proof"
        );

        // 2. Verify that the proof fails but that the one from archival works
        assert!(
            !membership_proof
                .verify(&new_peaks, &new_leaf, accumulator_mmr.count_leaves())
                .0
        );
        assert!(
            membership_proof
                .verify(&old_peaks, &leaf_hashes[4], accumulator_mmr.count_leaves())
                .0
        );

        assert!(
            real_membership_proof_from_archival
                .verify(&new_peaks, &leaf_hashes[4], accumulator_mmr.count_leaves())
                .0
        );

        // 3. Update the membership proof with the membership method
        membership_proof.update_from_leaf_update(&membership_proof_for_manipulated_leaf, &new_leaf);

        // 4. Verify that the proof now succeeds
        assert!(
            membership_proof
                .verify(&new_peaks, &leaf_hashes[4], accumulator_mmr.count_leaves())
                .0
        );

        // 5. test batch update from leaf update
        for i in 0..8 {
            let mut archival_mmr = original_archival_mmr.clone();
            let mut mps: Vec<MembershipProof<blake3::Hash, blake3::Hasher>> = vec![];
            for j in 0..8 {
                mps.push(original_archival_mmr.prove_membership(j).0);
            }
            let original_mps = mps.clone();
            let leaf_update_membership_proof = archival_mmr.prove_membership(i).0;
            archival_mmr.update_leaf_raw(i, new_leaf);
            let new_peaks = archival_mmr.get_peaks();
            let modified =
                MembershipProof::<blake3::Hash, blake3::Hasher>::batch_update_from_leaf_mutation(
                    &mut mps,
                    &leaf_update_membership_proof,
                    &new_leaf,
                );

            // when updating data index i, all authentication paths are updated
            // *except* for element i.
            let mut expected_modified = Vec::from_iter(0..8);
            expected_modified.remove(i as usize);
            assert_eq!(expected_modified, modified);

            for j in 0..8 {
                let our_leaf = if i == j {
                    &new_leaf
                } else {
                    &leaf_hashes[j as usize]
                };
                assert!(mps[j as usize].verify(&new_peaks, &our_leaf, 8,).0);

                // Verify that original membership proofs are no longer valid
                // For size = 8, all membership proofs except the one for element 0
                // will be updated since this MMR only contains a single peak.
                // An updated leaf (0 in this case) retains its authentication path after
                // the update. But all other leafs pointing to the same MMR will have updated
                // authentication paths.
                if j == i {
                    assert!(
                        original_mps[j as usize].verify(&new_peaks, &our_leaf, 8,).0,
                        "original membership proof must be valid when j = i"
                    );
                } else {
                    assert!(
                        !original_mps[j as usize].verify(&new_peaks, &our_leaf, 8,).0,
                        "original membership proof must be invalid when j != i"
                    );
                }
            }
        }
    }

    #[test]
    fn update_membership_proof_from_leaf_update_blake3_big_test() {
        // Build MMR from leaf count 0 to 17, and loop through *each*
        // leaf index for MMR, modifying its membership proof with a
        // leaf update.
        for leaf_count in 0..65 {
            let leaf_hashes: Vec<blake3::Hash> = (543217893265643843678u128
                ..543217893265643843678 + leaf_count)
                .map(|x| blake3::hash(bincode::serialize(&x).expect("Encoding failed").as_slice()))
                .collect();
            let new_leaf = blake3::hash(
                bincode::serialize(&133333333333333333333337u128)
                    .expect("Encoding failed")
                    .as_slice(),
            );
            let archival_mmr =
                ArchivalMmr::<blake3::Hash, blake3::Hasher>::new(leaf_hashes.clone());

            // Loop over all leaf indices that we want to modify in the MMR
            for i in 0..leaf_count {
                let (leaf_update_membership_proof, _old_peaks): (
                    MembershipProof<blake3::Hash, blake3::Hasher>,
                    Vec<blake3::Hash>,
                ) = archival_mmr.prove_membership(i);
                let mut modified_archival_mmr = archival_mmr.clone();
                modified_archival_mmr.update_leaf_raw(i, new_leaf);
                let new_peaks = modified_archival_mmr.get_peaks();

                // Loop over all leaf indices want a membership proof of, for modification
                for j in 0..leaf_count {
                    let mut membership_proof: MembershipProof<blake3::Hash, blake3::Hasher> =
                        archival_mmr.prove_membership(j).0;
                    let original_membership_roof = membership_proof.clone();
                    let membership_proof_was_mutated = membership_proof
                        .update_from_leaf_update(&leaf_update_membership_proof, &new_leaf);
                    let our_leaf = if i == j {
                        &new_leaf
                    } else {
                        &leaf_hashes[j as usize]
                    };
                    assert!(membership_proof.verify(&new_peaks, our_leaf, leaf_count).0);

                    // If membership proof was mutated, the original proof must fail
                    if membership_proof_was_mutated {
                        assert!(
                            !original_membership_roof
                                .verify(&new_peaks, our_leaf, leaf_count,)
                                .0
                        );
                    }

                    // Verify that modified membership proof matches that which can be
                    // fetched from the modified archival MMR
                    assert_eq!(
                        modified_archival_mmr.prove_membership(j).0,
                        membership_proof
                    );
                }
            }
        }
    }

    #[test]
    fn update_membership_proof_from_append_test_simple() {
        let leaf_count = 7;
        let leaf_hashes: Vec<blake3::Hash> = (14u128..14 + leaf_count)
            .map(|x| blake3::hash(bincode::serialize(&x).expect("Encoding failed").as_slice()))
            .collect();
        let new_leaf = blake3::hash(
            bincode::serialize(&133337u128)
                .expect("Encoding failed")
                .as_slice(),
        );
        let archival_mmr = ArchivalMmr::<blake3::Hash, blake3::Hasher>::new(leaf_hashes.clone());
        for i in 0..leaf_count {
            let (mut membership_proof, old_peaks): (
                MembershipProof<blake3::Hash, blake3::Hasher>,
                Vec<blake3::Hash>,
            ) = archival_mmr.prove_membership(i);
            let mut appended_archival_mmr = archival_mmr.clone();
            appended_archival_mmr.append(new_leaf.clone());
            let new_peaks = appended_archival_mmr.get_peaks();

            // Verify that membership proof fails before update and succeeds after
            // for the case of leaf_count 7, **all** membership proofs have to be
            // updated to be valid, so they should all fail prior to the update.
            assert!(
                !membership_proof
                    .verify(&new_peaks, &leaf_hashes[i as usize], leaf_count + 1,)
                    .0
            );
            membership_proof.update_from_append(leaf_count, &new_leaf, &old_peaks);
            assert!(
                membership_proof
                    .verify(&new_peaks, &leaf_hashes[i as usize], leaf_count + 1)
                    .0
            );

            // Verify that the appended Arhival MMR produces the same membership proof
            // as the one we got by updating the old membership proof
            assert_eq!(
                appended_archival_mmr.prove_membership(i),
                (membership_proof, new_peaks)
            );
        }
    }

    #[test]
    fn update_membership_proof_from_append_big_blake3() {
        // Build MMR from leaf count 0 to 514, and loop through *each*
        // leaf index for MMR, modifying its membership proof with an
        // append update.
        let new_leaf = blake3::hash(
            bincode::serialize(&133333333333333333333337u128)
                .expect("Encoding failed")
                .as_slice(),
        );
        for leaf_count in 0..514 {
            let leaf_hashes: Vec<blake3::Hash> = (543217893265643843678u128
                ..543217893265643843678 + leaf_count)
                .map(|x| blake3::hash(bincode::serialize(&x).expect("Encoding failed").as_slice()))
                .collect();
            let archival_mmr =
                ArchivalMmr::<blake3::Hash, blake3::Hasher>::new(leaf_hashes.clone());
            for i in 0..leaf_count {
                let (mut membership_proof, old_peaks): (
                    MembershipProof<blake3::Hash, blake3::Hasher>,
                    Vec<blake3::Hash>,
                ) = archival_mmr.prove_membership(i);
                let mut appended_archival_mmr = archival_mmr.clone();
                appended_archival_mmr.append(new_leaf.clone());
                let new_peaks = appended_archival_mmr.get_peaks();

                // Update membership proof and verify that it succeeds
                membership_proof.update_from_append(leaf_count, &new_leaf, &old_peaks);
                assert!(
                    membership_proof
                        .verify(&new_peaks, &leaf_hashes[i as usize], leaf_count + 1)
                        .0
                );

                // Verify that the appended Arhival MMR produces the same membership proof
                // as the one we got by updating the old membership proof
                assert_eq!(
                    appended_archival_mmr.prove_membership(i),
                    (membership_proof, new_peaks)
                );
            }

            // Test batch update of membership proofs
            let mut membership_proofs: Vec<MembershipProof<blake3::Hash, blake3::Hasher>> = (0
                ..leaf_count)
                .map(|i| archival_mmr.prove_membership(i).0)
                .collect();
            let original_mps = membership_proofs.clone();
            let old_peaks = archival_mmr.get_peaks();
            let mut i = 0;
            for mp in membership_proofs.iter() {
                assert!(
                    mp.verify(&old_peaks, &leaf_hashes[i as usize], leaf_count)
                        .0
                );
                i += 1;
            }
            let mut appended_archival_mmr = archival_mmr.clone();
            appended_archival_mmr.append(new_leaf.clone());
            let new_peaks = appended_archival_mmr.get_peaks();
            let indices_of_mutated_mps: Vec<u128> =
                MembershipProof::<blake3::Hash, blake3::Hasher>::batch_update_from_append(
                    &mut membership_proofs,
                    leaf_count,
                    &new_leaf,
                    &old_peaks,
                );
            let mut i = 0;
            for mp in membership_proofs {
                assert!(
                    mp.verify(&new_peaks, &leaf_hashes[i as usize], leaf_count + 1)
                        .0
                );
                i += 1;
            }

            // Verify that mutated membership proofs no longer work
            let mut i = 0;
            for index in indices_of_mutated_mps {
                assert!(
                    !original_mps[index as usize]
                        .verify(&new_peaks, &leaf_hashes[i as usize], leaf_count + 1)
                        .0
                );
                i += 1;
            }
        }
    }

    #[test]
    fn update_membership_proof_from_append_big_rescue_prime() {
        // Build MMR from leaf count 0 to 9, and loop through *each*
        // leaf index for MMR, modifying its membership proof with an
        // append update.
        let mut rp = RescuePrimeProduction::new();
        for leaf_count in 0..9 {
            let leaf_hashes: Vec<Vec<BFieldElement>> = (1001..1001 + leaf_count)
                .map(|x| rp.hash_one(&vec![BFieldElement::new(x)]))
                .collect();
            let archival_mmr =
                ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::new(leaf_hashes.clone());
            let new_leaf = rp.hash_one(&vec![BFieldElement::new(13333337)]);
            for i in 0..leaf_count {
                let (original_membership_proof, old_peaks): (
                    MembershipProof<Vec<BFieldElement>, RescuePrimeProduction>,
                    Vec<Vec<BFieldElement>>,
                ) = archival_mmr.prove_membership(i);
                let mut appended_archival_mmr = archival_mmr.clone();
                appended_archival_mmr.append(new_leaf.clone());
                let new_peaks = appended_archival_mmr.get_peaks();

                // Update membership proof and verify that it succeeds
                let mut membership_proof_mutated = original_membership_proof.clone();
                let mutated =
                    membership_proof_mutated.update_from_append(leaf_count, &new_leaf, &old_peaks);
                assert!(
                    membership_proof_mutated
                        .verify(&new_peaks, &leaf_hashes[i as usize], leaf_count + 1,)
                        .0
                );

                // If membership proof mutated, then the old proof must be invalid
                if mutated {
                    assert!(
                        !original_membership_proof
                            .verify(&new_peaks, &leaf_hashes[i as usize], leaf_count + 1,)
                            .0
                    );
                }

                // Verify that the appended Arhival MMR produces the same membership proof
                // as the one we got by updating the old membership proof
                assert_eq!(
                    appended_archival_mmr.prove_membership(i),
                    (membership_proof_mutated, new_peaks)
                );
            }
        }
    }
}
