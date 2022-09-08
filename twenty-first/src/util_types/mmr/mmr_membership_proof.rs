use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::util_types::simple_hasher::{Hashable, Hasher, ToVec};
use std::{
    collections::{hash_map::RandomState, hash_set::Intersection, HashMap, HashSet},
    fmt::Debug,
    iter::FromIterator,
};

use super::shared::{
    data_index_to_node_index, get_authentication_path_node_indices,
    get_peak_heights_and_peak_node_indices, leaf_count_to_node_count, leaf_index_to_peak_index,
    left_sibling, node_indices_added_by_append, parent, right_child_and_height, right_sibling,
};

#[derive(Debug, Serialize, Deserialize)]
pub struct MmrMembershipProof<H>
where
    H: Hasher + Sized,
{
    pub data_index: u128,
    pub authentication_path: Vec<H::Digest>,
}

impl<H> Clone for MmrMembershipProof<H>
where
    H: Hasher,
{
    fn clone(&self) -> Self {
        Self {
            data_index: self.data_index,
            authentication_path: self.authentication_path.clone(),
        }
    }
}

impl<H> PartialEq for MmrMembershipProof<H>
where
    H: Hasher,
{
    // Two membership proofs are considered equal if they contain the same authentication path
    // *and* point to the same data index
    fn eq(&self, other: &Self) -> bool {
        self.data_index == other.data_index && self.authentication_path == other.authentication_path
    }
}

impl<H> MmrMembershipProof<H>
where
    H: Hasher,
    u128: Hashable<H::T>,
{
    pub fn hash(&self) -> H::Digest {
        let data_index_hashable: Vec<H::T> = self.data_index.to_sequence();
        let hasher = H::new();
        let digest_preimage: Vec<H::T> = [
            data_index_hashable,
            self.authentication_path
                .iter()
                .flat_map(|ap| ap.to_vec())
                .collect_vec(),
        ]
        .concat();

        hasher.hash_sequence(&digest_preimage)
    }

    /**
     * verify
     * Verify a membership proof for an MMR. If verification succeeds, return the final state of the accumulator hash.
     */
    pub fn verify(
        &self,
        peaks: &[H::Digest],
        leaf_hash: &H::Digest,
        leaf_count: u128,
    ) -> (bool, Option<H::Digest>)
    where
        H: Hasher,
        u128: Hashable<H::T>,
    {
        let node_index = data_index_to_node_index(self.data_index);

        let hasher = H::new();
        let mut acc_hash: H::Digest = leaf_hash.to_owned();
        let mut acc_index: u128 = node_index;
        for hash in self.authentication_path.iter() {
            let (acc_right, _acc_height) = right_child_and_height(acc_index);
            acc_hash = if acc_right {
                hasher.hash_pair(hash, &acc_hash)
            } else {
                hasher.hash_pair(&acc_hash, hash)
            };
            acc_index = parent(acc_index);
        }

        // Find the correct peak index
        let peak_index_res = leaf_index_to_peak_index(self.data_index, leaf_count);
        let peak_index = match peak_index_res {
            None => return (false, None),
            Some(pi) => pi,
        };

        // Compare the peak at the expected index with accumulated hash
        if peaks[peak_index as usize] != acc_hash {
            return (false, None);
        }

        (true, Some(acc_hash))
    }

    /// Return the node indices for the authentication path in this membership proof
    pub fn get_node_indices(&self) -> Vec<u128> {
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
        old_mmr_leaf_count: u128,
        new_mmr_leaf: &H::Digest,
        old_mmr_peaks: &[H::Digest],
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
        let (own_old_peak_index, own_old_peak_height) = self.get_peak_index_and_height();

        // 2
        let added_node_indices = node_indices_added_by_append(old_mmr_leaf_count);

        // 3
        // Any peak is a left child, so we don't have to check if it's a right or left child.
        // This means we can use a faster method to find the parent index than the generic method.
        let peak_parent_index = own_old_peak_index + (1 << (own_old_peak_height + 1));

        // 3a
        if !added_node_indices.contains(&peak_parent_index) {
            return false;
        }

        // 4 Get node indices of missing digests
        let new_peak_index: u128 = *added_node_indices.last().unwrap();
        let new_node_count: u128 = leaf_count_to_node_count(old_mmr_leaf_count + 1);
        let node_indices_for_missing_digests: Vec<u128> = get_authentication_path_node_indices(
            own_old_peak_index,
            new_peak_index,
            new_node_count,
        )
        .unwrap();

        // 5 collect all derivable peaks in a hashmap indexed by node index
        // 5.a, collect all node hash digests that are present in the old peaks
        // The keys in the hash map are node indices
        let mut known_digests: HashMap<u128, H::Digest> = HashMap::new();
        let (_old_mmr_peak_heights, old_mmr_peak_indices) =
            get_peak_heights_and_peak_node_indices(old_mmr_leaf_count);
        for (old_peak_index, old_peak_digest) in
            old_mmr_peak_indices.iter().zip(old_mmr_peaks.iter())
        {
            known_digests.insert(*old_peak_index, old_peak_digest.to_owned());
        }

        // 5.b collect all node hash digests that are derivable from `new_leaf` and
        // `old_peaks`. These are the digests of `new_leaf`'s path to the root.
        // break out of loop once *one* digest is found this way since that will
        // always suffice.
        let mut acc_hash = new_mmr_leaf.to_owned();
        let hasher = H::new();
        for (node_index, old_peak_digest) in
            added_node_indices.iter().zip(old_mmr_peaks.iter().rev())
        {
            known_digests.insert(*node_index, acc_hash.to_owned());

            // peaks are always left children, so we don't have to check for that
            acc_hash = hasher.hash_pair(old_peak_digest, &acc_hash);

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
        membership_proofs: &mut [&mut Self],
        old_leaf_count: u128,
        new_leaf: &H::Digest,
        old_peaks: &[H::Digest],
    ) -> Vec<usize> {
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
        let mut known_digests: HashMap<u128, H::Digest> = HashMap::new();
        let (_old_peak_heights, old_peak_indices) =
            get_peak_heights_and_peak_node_indices(old_leaf_count);
        for (old_peak_index, old_peak_digest) in old_peak_indices.iter().zip(old_peaks.iter()) {
            known_digests.insert(*old_peak_index, old_peak_digest.to_owned());
        }

        // 2.b collect all node hash digests that are derivable from `new_leaf` and
        // `old_peaks`. These are the digests of `new_leaf`'s path to the root.
        let mut acc_hash = new_leaf.to_owned();
        let hasher = H::new();
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
            acc_hash = hasher.hash_pair(old_peak_digest, &acc_hash);
        }

        // Loop over all membership proofs and insert missing hashes for each
        let mut modified: Vec<usize> = vec![];
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

            modified.push(i);

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

    /// Update a membership proof with a `leaf_mutation` proof. For the `membership_proof`
    /// parameter, it doesn't matter if you use the old or new membership proof associated
    /// with the leaf update, as they are the same before and after the leaf mutation.
    pub fn update_from_leaf_mutation(
        &mut self,
        leaf_mutation_membership_proof: &MmrMembershipProof<H>,
        new_leaf: &H::Digest,
    ) -> bool {
        let own_node_ap_indices = self.get_node_indices();
        let affected_node_indices = leaf_mutation_membership_proof.get_direct_path_indices();
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
        let mut deducible_hashes: HashMap<u128, H::Digest> = HashMap::new();
        let mut node_index = data_index_to_node_index(leaf_mutation_membership_proof.data_index);
        deducible_hashes.insert(node_index, new_leaf.clone());
        let hasher = H::new();
        let mut acc_hash: H::Digest = new_leaf.to_owned();

        // Calculate hashes from the bottom towards the peak. Break when
        // the intersecting node is reached.
        for hash in leaf_mutation_membership_proof.authentication_path.iter() {
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
                hasher.hash_pair(hash, &acc_hash)
            } else {
                hasher.hash_pair(&acc_hash, hash)
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

    /// Update multiple membership proofs with a `leaf_mutation` proof. For the `leaf_mutation_membership_proof`
    /// parameter, it doesn't matter if you use the old or new membership proof associated
    /// with the leaf mutation, as they are the same before and after the leaf mutation.
    /// Returns the indices of the membership proofs that were modified where index refers
    /// to the order in which the membership proofs were given to this function.
    pub fn batch_update_from_leaf_mutation(
        membership_proofs: &mut [Self],
        leaf_mutation_membership_proof: &MmrMembershipProof<H>,
        new_leaf: &H::Digest,
    ) -> Vec<u128> {
        // 1. Calculate all hashes that are deducible from the leaf update
        // 2. Iterate through all membership proofs and update digests that
        //    are deducible from the leaf update proof.

        let mut deducible_hashes: HashMap<u128, H::Digest> = HashMap::new();
        let mut node_index = data_index_to_node_index(leaf_mutation_membership_proof.data_index);
        deducible_hashes.insert(node_index, new_leaf.clone());
        let hasher = H::new();
        let mut acc_hash: H::Digest = new_leaf.to_owned();

        // Calculate hashes from the bottom towards the peak. Break before we
        // calculate the hash of the peak, since peaks are never included in
        // authentication paths
        for (count, hash) in leaf_mutation_membership_proof
            .authentication_path
            .iter()
            .enumerate()
        {
            // Do not calculate the last hash as it will always be a peak which
            // are never included in the authentication path
            if count == leaf_mutation_membership_proof.authentication_path.len() - 1 {
                break;
            }
            let (acc_right, _acc_height) = right_child_and_height(node_index);
            acc_hash = if acc_right {
                hasher.hash_pair(hash, &acc_hash)
            } else {
                hasher.hash_pair(&acc_hash, hash)
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
                // Since this function returns the indices of the modified membership proofs,
                // a check if the new digest is actually different from the previous value is
                // needed.
                if deducible_hashes.contains_key(&authentication_path_indices)
                    && *digest != deducible_hashes[&authentication_path_indices]
                {
                    *digest = deducible_hashes[&authentication_path_indices].clone();
                    modified_membership_proofs.push(i as u128);
                    break;
                }
            }
        }

        modified_membership_proofs
    }

    /// batch_update_from_batch_leaf_mutation
    /// Update a batch of own membership proofs given a batch of
    /// authenticated leaf modifications. It is the caller's res-
    /// ponsibility to ensure that the authentication paths are
    /// valid; if not, the updated membership proofs will become
    /// invalid as well.
    /// @params
    ///  - membership_proofs -- own membership proofs, to be updated
    ///  - authentication_paths_and_leafs -- membership proofs of the mutated
    ///    leafs, and the new leaf values
    /// Returns those indices into the slice of membership proofs that were updated.
    pub fn batch_update_from_batch_leaf_mutation(
        membership_proofs: &mut [&mut Self],
        mut authentication_paths_and_leafs: Vec<(MmrMembershipProof<H>, H::Digest)>,
    ) -> Vec<usize> {
        // Calculate all derivable paths
        let mut new_ap_digests: HashMap<u128, H::Digest> = HashMap::new();
        let hasher = H::new();

        // Calculate the derivable digests from a number of leaf mutations and their
        // associated authentication paths. Notice that all authentication paths
        // are only valid *prior* to any updates. They get invalidated (unless updated)
        // throughout the updating as their neighbor leaf digests change values.
        // The hash map `new_ap_digests` takes care of that.
        while let Some((ap, new_leaf)) = authentication_paths_and_leafs.pop() {
            let mut node_index = data_index_to_node_index(ap.data_index);
            let former_value = new_ap_digests.insert(node_index, new_leaf.clone());
            assert!(
                former_value.is_none(),
                "Duplicated leaves are not allowed in membership proof updater"
            );
            let mut acc_hash: H::Digest = new_leaf.to_owned();

            for (count, hash) in ap.authentication_path.iter().enumerate() {
                // Do not calculate the last hash as it will always be a peak which
                // are never included in the authentication path
                if count == ap.authentication_path.len() - 1 {
                    break;
                }

                // If sibling node is something that has already been calculated, we use that
                // hash digest. Otherwise we use the one in our authentication path.

                let (right, height) = right_child_and_height(node_index);
                if right {
                    let left_sibling_index = left_sibling(node_index, height);
                    let sibling_hash: &H::Digest = match new_ap_digests.get(&left_sibling_index) {
                        Some(h) => h,
                        None => hash,
                    };
                    acc_hash = hasher.hash_pair(sibling_hash, &acc_hash);

                    // Find parent node index
                    node_index += 1;
                } else {
                    let right_sibling_index = right_sibling(node_index, height);
                    let sibling_hash: &H::Digest = match new_ap_digests.get(&right_sibling_index) {
                        Some(h) => h,
                        None => hash,
                    };
                    acc_hash = hasher.hash_pair(&acc_hash, sibling_hash);

                    // Find parent node index
                    node_index += 1 << (height + 1);
                }

                new_ap_digests.insert(node_index, acc_hash.clone());
            }
        }

        let mut modified_membership_proof_indices: Vec<usize> = vec![];
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
                // Any number of hashes can be updated in the authentication path, since
                // we're modifying multiple leaves in the MMR
                // Since this function returns the indices of the modified membership proofs,
                // a check if the new digest is actually different from the previous value is
                // needed.
                if new_ap_digests.contains_key(&authentication_path_indices)
                    && *digest != new_ap_digests[&authentication_path_indices]
                {
                    modified_membership_proof_indices.push(i);
                    *digest = new_ap_digests[&authentication_path_indices].clone();
                }
            }
        }

        modified_membership_proof_indices.dedup();
        modified_membership_proof_indices
    }
}

#[cfg(test)]
mod mmr_membership_proof_test {
    use itertools::Itertools;
    use rand::{thread_rng, RngCore};

    use super::*;
    use crate::shared_math::rescue_prime_regular::RescuePrimeRegular;
    use crate::shared_math::rescue_prime_xlix::{
        RescuePrimeXlix, RP_DEFAULT_OUTPUT_SIZE, RP_DEFAULT_WIDTH,
    };
    use crate::test_shared::mmr::get_archival_mmr_from_digests;
    use crate::util_types::blake3_wrapper::Blake3Hash;
    use crate::{
        shared_math::b_field_element::BFieldElement,
        util_types::blake3_wrapper,
        util_types::mmr::mmr_accumulator::MmrAccumulator,
        util_types::mmr::{archival_mmr::ArchivalMmr, mmr_trait::Mmr},
    };

    #[test]
    fn equality_and_hash_test() {
        type Hasher = blake3::Hasher;

        let mp0: MmrMembershipProof<Hasher> = MmrMembershipProof {
            authentication_path: vec![],
            data_index: 4,
        };
        let mp1: MmrMembershipProof<Hasher> = MmrMembershipProof {
            authentication_path: vec![],
            data_index: 4,
        };
        let mp2: MmrMembershipProof<Hasher> = MmrMembershipProof {
            authentication_path: vec![],
            data_index: 3,
        };
        let mp3: MmrMembershipProof<Hasher> = MmrMembershipProof {
            authentication_path: vec![blake3_wrapper::hash(b"foobarbaz")],
            data_index: 4,
        };
        let mp4: MmrMembershipProof<Hasher> = MmrMembershipProof {
            authentication_path: vec![blake3_wrapper::hash(b"foobarbaz")],
            data_index: 4,
        };
        assert_eq!(mp0, mp1);
        assert_ne!(mp1, mp2);
        assert_ne!(mp2, mp3);
        assert_eq!(mp3, mp4);
        assert_ne!(mp3, mp0);

        // test the digests. This is to verify that both fields are inputs to the hash function
        assert_eq!(mp0.hash(), mp1.hash());
        assert_ne!(mp1.hash(), mp2.hash());
        assert_ne!(mp2.hash(), mp3.hash());
        assert_eq!(mp3.hash(), mp4.hash());
        assert_ne!(mp3.hash(), mp0.hash());
    }

    #[test]
    fn get_node_indices_simple_test() {
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;

        let leaf_hashes: Vec<Digest> = (14u128..14 + 8).map(|x| x.into()).collect();
        let mut archival_mmr: ArchivalMmr<Hasher> = get_archival_mmr_from_digests(leaf_hashes);
        let (membership_proof, _peaks): (MmrMembershipProof<Hasher>, Vec<Digest>) =
            archival_mmr.prove_membership(4);
        assert_eq!(vec![9, 13, 7], membership_proof.get_node_indices());
        assert_eq!(
            vec![8, 10, 14, 15],
            membership_proof.get_direct_path_indices()
        );
    }

    #[test]
    fn get_peak_index_simple_test() {
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;

        let mut mmr_size = 7;
        let leaf_hashes: Vec<Blake3Hash> = (14u128..14 + mmr_size).map(|x| x.into()).collect();
        let mut archival_mmr: ArchivalMmr<Hasher> = get_archival_mmr_from_digests(leaf_hashes);
        let mut expected_peak_indices_and_heights: Vec<(u128, u32)> =
            vec![(7, 2), (7, 2), (7, 2), (7, 2), (10, 1), (10, 1), (11, 0)];
        for (i, expected_peak_index) in
            (0..mmr_size).zip(expected_peak_indices_and_heights.into_iter())
        {
            let (membership_proof, _peaks): (MmrMembershipProof<Hasher>, Vec<Digest>) =
                archival_mmr.prove_membership(i);
            assert_eq!(
                expected_peak_index,
                membership_proof.get_peak_index_and_height()
            );
        }

        // Increase size to 8 and verify that the peaks are now different
        mmr_size = 8;
        let leaf_hash: Digest = 1337u128.into();
        archival_mmr.append(leaf_hash);
        expected_peak_indices_and_heights = vec![(15, 3); mmr_size as usize];
        for (i, expected_peak_index) in
            (0..mmr_size).zip(expected_peak_indices_and_heights.into_iter())
        {
            let (membership_proof, _peaks): (MmrMembershipProof<Hasher>, Vec<Digest>) =
                archival_mmr.prove_membership(i);
            assert_eq!(
                expected_peak_index,
                membership_proof.get_peak_index_and_height()
            );
        }

        // Increase size to 9 and verify that the peaks are now different
        mmr_size = 9;
        let another_leaf_hash: Digest = 13337u128.into();
        archival_mmr.append(another_leaf_hash);
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
            let (membership_proof, _peaks): (MmrMembershipProof<Hasher>, Vec<Digest>) =
                archival_mmr.prove_membership(i);
            assert_eq!(
                expected_peak_index,
                membership_proof.get_peak_index_and_height()
            );
        }
    }

    #[test]
    fn update_batch_membership_proofs_from_leaf_mutations_new_test() {
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;

        let total_leaf_count = 8;
        let leaf_hashes: Vec<Digest> = (14u128..14 + total_leaf_count).map(|x| x.into()).collect();
        let mut archival_mmr: ArchivalMmr<Hasher> =
            get_archival_mmr_from_digests(leaf_hashes.clone());
        let mut membership_proofs: Vec<MmrMembershipProof<Hasher>> = vec![];
        for data_index in 0..total_leaf_count {
            membership_proofs.push(archival_mmr.prove_membership(data_index).0);
        }

        let new_leaf2: Digest = 133337u128.into();
        let new_leaf3: Digest = 12345678u128.into();
        let mutation_membership_proof_old2 = archival_mmr.prove_membership(2).0;
        let mutation_membership_proof_old3 = archival_mmr.prove_membership(3).0;
        archival_mmr.mutate_leaf_raw(2, new_leaf2);
        archival_mmr.mutate_leaf_raw(3, new_leaf3);
        for mp in membership_proofs.iter_mut() {
            mp.update_from_leaf_mutation(&mutation_membership_proof_old2, &new_leaf2);
        }
        for mp in membership_proofs.iter_mut() {
            mp.update_from_leaf_mutation(&mutation_membership_proof_old3, &new_leaf3);
        }

        let mut updated_leaf_hashes = leaf_hashes.clone();
        updated_leaf_hashes[2] = new_leaf2;
        updated_leaf_hashes[3] = new_leaf3;
        for (_i, (mp, leaf_hash)) in membership_proofs
            .iter()
            .zip(updated_leaf_hashes.iter())
            .enumerate()
        {
            mp.verify(
                &archival_mmr.get_peaks(),
                leaf_hash,
                archival_mmr.count_leaves(),
            );
        }
    }

    #[test]
    fn batch_update_from_batch_leaf_mutation_total_replacement_test() {
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;

        let total_leaf_count = 268;
        let leaf_hashes_init: Vec<Digest> =
            (14u128..14 + total_leaf_count).map(|x| x.into()).collect();
        let mut archival_mmr_init: ArchivalMmr<Hasher> =
            get_archival_mmr_from_digests(leaf_hashes_init.clone());
        let leaf_hashes_final: Vec<Digest> = (541u128..541 + total_leaf_count)
            .map(|x| x.into())
            .collect();
        let mut archival_mmr_final: ArchivalMmr<Hasher> =
            get_archival_mmr_from_digests(leaf_hashes_final.clone());
        let mut membership_proofs: Vec<MmrMembershipProof<Hasher>> = (0..total_leaf_count)
            .map(|i| archival_mmr_init.prove_membership(i).0)
            .collect();
        let membership_proofs_init_and_new_leafs: Vec<(MmrMembershipProof<Hasher>, Digest)> =
            membership_proofs
                .clone()
                .into_iter()
                .zip(leaf_hashes_final.clone().into_iter())
                .collect();
        let changed_values = MmrMembershipProof::batch_update_from_batch_leaf_mutation(
            &mut membership_proofs.iter_mut().collect::<Vec<_>>(),
            membership_proofs_init_and_new_leafs,
        );

        // This assert only works if `total_leaf_count` is an even number since there
        // otherwise is a membership proof that's an empty authentication path, and that
        // does not change
        assert_eq!(
            (0..total_leaf_count as usize).collect::<Vec<_>>(),
            changed_values,
            "All membership proofs must be indicated as changed"
        );

        for (mp, final_leaf_hash) in membership_proofs.iter().zip(leaf_hashes_final.iter()) {
            assert!(
                mp.verify(
                    &archival_mmr_final.get_peaks(),
                    final_leaf_hash,
                    total_leaf_count
                )
                .0
            );
        }
    }

    #[test]
    fn batch_update_from_batch_leaf_mutation_test() {
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;

        let total_leaf_count = 34;
        let mut leaf_hashes: Vec<Digest> =
            (14u128..14 + total_leaf_count).map(|x| x.into()).collect();
        let mut archival_mmr: ArchivalMmr<Hasher> =
            get_archival_mmr_from_digests(leaf_hashes.clone());
        let mut prng = thread_rng();
        for modified_leaf_count in 0..=total_leaf_count {
            // Pick a set of membership proofs that we want to batch-update
            let own_membership_proof_count: u128 = prng.next_u32() as u128 % total_leaf_count;
            let mut all_data_indices: Vec<u128> = (0..total_leaf_count).collect();
            let mut own_membership_proofs: Vec<MmrMembershipProof<Hasher>> = vec![];
            for _ in 0..own_membership_proof_count {
                let data_index =
                    all_data_indices.remove(prng.next_u32() as usize % all_data_indices.len());
                own_membership_proofs.push(archival_mmr.prove_membership(data_index).0);
            }

            // Set the new leafs and their associated authentication paths
            let new_leafs: Vec<Digest> = (133337..133337 + modified_leaf_count as u128)
                .map(|x| x.into())
                .collect();
            let mut all_data_indices_new: Vec<u128> = (0..total_leaf_count).collect();
            let mut authentication_paths: Vec<MmrMembershipProof<Hasher>> = vec![];
            for _ in 0..modified_leaf_count {
                let data_index = all_data_indices_new
                    .remove(prng.next_u32() as usize % all_data_indices_new.len());
                authentication_paths.push(archival_mmr.prove_membership(data_index).0);
            }

            // let the magic start
            let original_mps = own_membership_proofs.clone();
            let mutation_argument: Vec<(MmrMembershipProof<Hasher>, Digest)> = authentication_paths
                .clone()
                .into_iter()
                .zip(new_leafs.clone().into_iter())
                .collect();
            let updated_mp_indices_0 = MmrMembershipProof::batch_update_from_batch_leaf_mutation(
                &mut own_membership_proofs.iter_mut().collect::<Vec<_>>(),
                mutation_argument.clone(),
            );

            // update MMR
            for i in 0..(modified_leaf_count as usize) {
                let auth_path = authentication_paths[i].clone();
                leaf_hashes[auth_path.data_index as usize] = new_leafs[i];
                archival_mmr.mutate_leaf_raw(auth_path.data_index, new_leafs[i]);
            }

            // Let's verify that `batch_mutate_leaf_and_update_mps` from the
            // MmrAccumulator agrees
            let mut mmra: MmrAccumulator<Hasher> = (&mut archival_mmr).into();
            let mut mps_copy = original_mps;
            let updated_mp_indices_1 =
                mmra.batch_mutate_leaf_and_update_mps(&mut mps_copy, mutation_argument);
            assert_eq!(own_membership_proofs, mps_copy);
            assert_eq!(mmra.get_peaks(), archival_mmr.get_peaks());
            assert_eq!(updated_mp_indices_0, updated_mp_indices_1);

            // test that all updated membership proofs are valid under the updated MMR
            for i in 0..own_membership_proof_count {
                let membership_proof = own_membership_proofs[i as usize].clone();
                assert!(
                    membership_proof
                        .verify(
                            &archival_mmr.get_peaks(),
                            &leaf_hashes[membership_proof.data_index as usize],
                            archival_mmr.count_leaves(),
                        )
                        .0
                );
            }
        }
    }

    #[test]
    fn batch_update_from_leaf_mutation_no_change_return_value_test() {
        // This test verifies that the return value indicating changed membership proofs is empty
        // even though the mutations affect the membership proofs. The reason it is empty is that
        // the resulting membership proof digests are unchanged, since the leaf hashes mutations
        // are the identity operators. In other words: the leafs don't change.
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;

        let total_leaf_count = 8;
        let leaf_hashes: Vec<Digest> = (14u128..14 + total_leaf_count).map(|x| x.into()).collect();
        let mut archival_mmr: ArchivalMmr<Hasher> =
            get_archival_mmr_from_digests(leaf_hashes.clone());
        let mut membership_proofs: Vec<MmrMembershipProof<Hasher>> = vec![];
        for data_index in 0..total_leaf_count {
            membership_proofs.push(archival_mmr.prove_membership(data_index).0);
        }

        for i in 0..total_leaf_count as usize {
            let leaf_mutation_membership_proof = membership_proofs[i].clone();
            let new_leaf = leaf_hashes[i];
            let ret = MmrMembershipProof::batch_update_from_leaf_mutation(
                &mut membership_proofs,
                &leaf_mutation_membership_proof,
                &new_leaf,
            );

            // the return value must be empty since no membership proof has changed
            assert!(ret.is_empty());
        }

        let membership_proofs_init_and_new_leafs: Vec<(MmrMembershipProof<Hasher>, Digest)> =
            membership_proofs
                .clone()
                .into_iter()
                .zip(leaf_hashes.clone().into_iter())
                .collect();
        let ret = MmrMembershipProof::batch_update_from_batch_leaf_mutation(
            &mut membership_proofs.iter_mut().collect::<Vec<_>>(),
            membership_proofs_init_and_new_leafs.clone(),
        );

        // the return value must be empty since no membership proof has changed
        assert!(ret.is_empty());

        // Let's test the exact same for the MMR accumulator scheme
        let mut mmra: MmrAccumulator<Hasher> = MmrAccumulator::new(leaf_hashes);
        let ret_from_acc = mmra.batch_mutate_leaf_and_update_mps(
            &mut membership_proofs,
            membership_proofs_init_and_new_leafs,
        );
        assert!(ret_from_acc.is_empty());
    }

    #[test]
    fn update_batch_membership_proofs_from_batch_leaf_mutations_test() {
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;

        let total_leaf_count = 8;
        let leaf_hashes: Vec<Digest> = (14u128..14 + total_leaf_count).map(|x| x.into()).collect();
        let mut archival_mmr: ArchivalMmr<Hasher> =
            get_archival_mmr_from_digests(leaf_hashes.clone());
        let modified_leaf_count = 8;
        let mut membership_proofs: Vec<MmrMembershipProof<Hasher>> = vec![];
        for data_index in 0..modified_leaf_count {
            membership_proofs.push(archival_mmr.prove_membership(data_index).0);
        }

        let new_leafs: Vec<Digest> = (1337..1337 + modified_leaf_count as u128)
            .map(|x| x.into())
            .collect();

        for i in 0..modified_leaf_count as usize {
            let leaf_mutation_membership_proof = membership_proofs[i].clone();
            let new_leaf = new_leafs[i];
            MmrMembershipProof::batch_update_from_leaf_mutation(
                &mut membership_proofs,
                &leaf_mutation_membership_proof,
                &new_leaf,
            );
        }

        for i in 0..modified_leaf_count {
            archival_mmr.mutate_leaf_raw(i, new_leafs[i as usize]);
        }

        for (i, mp) in membership_proofs.iter().enumerate() {
            assert!(
                mp.verify(
                    &archival_mmr.get_peaks(),
                    &new_leafs[i],
                    archival_mmr.count_leaves()
                )
                .0
            );
        }
    }

    #[test]
    fn update_membership_proof_from_leaf_mutation_test() {
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;

        let leaf_hashes: Vec<Digest> = (14u128..14 + 8).map(|x| x.into()).collect();
        let new_leaf: Digest = 133337u128.into();
        let mut accumulator_mmr = MmrAccumulator::<Hasher>::new(leaf_hashes.clone());

        assert_eq!(8, accumulator_mmr.count_leaves());
        let mut archival_mmr: ArchivalMmr<Hasher> =
            get_archival_mmr_from_digests(leaf_hashes.clone());
        let mut original_archival_mmr: ArchivalMmr<Hasher> =
            get_archival_mmr_from_digests(leaf_hashes.clone());
        let (mut membership_proof, _peaks): (MmrMembershipProof<Hasher>, Vec<Digest>) =
            archival_mmr.prove_membership(4);

        // 1. Update a leaf in both the accumulator MMR and in the archival MMR
        let old_peaks = archival_mmr.get_peaks();
        let membership_proof_for_manipulated_leaf = archival_mmr.prove_membership(2).0;
        archival_mmr.mutate_leaf_raw(2, new_leaf);
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
        membership_proof
            .update_from_leaf_mutation(&membership_proof_for_manipulated_leaf, &new_leaf);

        // 4. Verify that the proof now succeeds
        assert!(
            membership_proof
                .verify(&new_peaks, &leaf_hashes[4], accumulator_mmr.count_leaves())
                .0
        );

        // 5. test batch update from leaf update
        for i in 0..8 {
            let mut archival_mmr: ArchivalMmr<Hasher> =
                get_archival_mmr_from_digests(leaf_hashes.clone());
            let mut mps: Vec<MmrMembershipProof<Hasher>> = vec![];
            for j in 0..8 {
                mps.push(original_archival_mmr.prove_membership(j).0);
            }
            let original_mps = mps.clone();
            let leaf_mutation_membership_proof = archival_mmr.prove_membership(i).0;
            archival_mmr.mutate_leaf_raw(i, new_leaf);
            let new_peaks = archival_mmr.get_peaks();
            let modified = MmrMembershipProof::<Hasher>::batch_update_from_leaf_mutation(
                &mut mps,
                &leaf_mutation_membership_proof,
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
    fn update_membership_proof_from_leaf_mutation_blake3_big_test() {
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;

        // Build MMR from leaf count 0 to 22, and loop through *each*
        // leaf index for MMR, modifying its membership proof with a
        // leaf update.
        for leaf_count in 0..=22 {
            let start: u128 = 543217893265643843678;
            let leaf_hashes: Vec<Digest> = (start..start + leaf_count).map(|x| x.into()).collect();
            let new_leaf: Digest = 133333333333333333333337u128.into();
            let mut archival_mmr: ArchivalMmr<Hasher> =
                get_archival_mmr_from_digests(leaf_hashes.clone());

            // Loop over all leaf indices that we want to modify in the MMR
            for i in 0..leaf_count {
                let (leaf_mutation_membership_proof, _old_peaks): (
                    MmrMembershipProof<Hasher>,
                    Vec<Digest>,
                ) = archival_mmr.prove_membership(i);
                let mut modified_archival_mmr: ArchivalMmr<Hasher> =
                    get_archival_mmr_from_digests(leaf_hashes.clone());
                modified_archival_mmr.mutate_leaf_raw(i, new_leaf);
                let new_peaks = modified_archival_mmr.get_peaks();

                // Loop over all leaf indices want a membership proof of, for modification
                for j in 0..leaf_count {
                    let mut membership_proof: MmrMembershipProof<Hasher> =
                        archival_mmr.prove_membership(j).0;
                    let original_membership_roof = membership_proof.clone();
                    let membership_proof_was_mutated = membership_proof
                        .update_from_leaf_mutation(&leaf_mutation_membership_proof, &new_leaf);
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
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;

        let leaf_count = 7;
        let leaf_hashes: Vec<Digest> = (14u128..14 + leaf_count).map(|x| x.into()).collect();
        let new_leaf: Digest = 133337u128.into();
        let mut archival_mmr: ArchivalMmr<Hasher> =
            get_archival_mmr_from_digests(leaf_hashes.clone());

        for i in 0..leaf_count {
            let (mut membership_proof, old_peaks): (MmrMembershipProof<Hasher>, Vec<Digest>) =
                archival_mmr.prove_membership(i);
            let mut appended_archival_mmr: ArchivalMmr<Hasher> =
                get_archival_mmr_from_digests(leaf_hashes.clone());
            // let mut appended_archival_mmr = archival_mmr.clone();
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
        type Digest = Blake3Hash;
        type Hasher = blake3::Hasher;

        // Build MMR from leaf count 0 to 68, and loop through *each*
        // leaf index for MMR, modifying its membership proof with an
        // append update.
        let new_leaf: Digest = 133333333333333333333337u128.into();

        for leaf_count in 0..68 {
            let start: u128 = 543217893265643843678;
            let leaf_hashes: Vec<Digest> = (start..start + leaf_count).map(|x| x.into()).collect();
            let mut archival_mmr: ArchivalMmr<Hasher> =
                get_archival_mmr_from_digests(leaf_hashes.clone());
            for i in 0..leaf_count {
                let (mut membership_proof, old_peaks): (MmrMembershipProof<Hasher>, Vec<Digest>) =
                    archival_mmr.prove_membership(i);
                // let mut appended_archival_mmr = archival_mmr.clone();
                let mut appended_archival_mmr: ArchivalMmr<Hasher> =
                    get_archival_mmr_from_digests(leaf_hashes.clone());
                appended_archival_mmr.append(new_leaf.clone());
                let new_peaks = appended_archival_mmr.get_peaks();

                // Update membership proof and verify that it succeeds
                let original_mp = membership_proof.clone();
                let changed =
                    membership_proof.update_from_append(leaf_count, &new_leaf, &old_peaks);
                assert!(
                    membership_proof
                        .verify(&new_peaks, &leaf_hashes[i as usize], leaf_count + 1)
                        .0
                );

                // Verify that the old membership proof fails iff a change was indicated
                if changed {
                    assert!(
                        !original_mp
                            .verify(&new_peaks, &leaf_hashes[i as usize], leaf_count + 1)
                            .0
                    );
                } else {
                    assert!(
                        original_mp
                            .verify(&new_peaks, &leaf_hashes[i as usize], leaf_count + 1)
                            .0
                    );
                }

                // Verify that the appended Arhival MMR produces the same membership proof
                // as the one we got by updating the old membership proof
                assert_eq!(
                    appended_archival_mmr.prove_membership(i),
                    (membership_proof, new_peaks)
                );
            }

            // Test batch update of membership proofs
            let mut membership_proofs: Vec<MmrMembershipProof<Hasher>> = (0..leaf_count)
                .map(|i| archival_mmr.prove_membership(i).0)
                .collect();
            let old_peaks = archival_mmr.get_peaks();
            for (i, mp) in membership_proofs.iter().enumerate() {
                assert!(
                    mp.verify(&old_peaks, &leaf_hashes[i as usize], leaf_count)
                        .0
                );
            }
            // let mut appended_archival_mmr = archival_mmr.clone();
            let mut appended_archival_mmr: ArchivalMmr<Hasher> =
                get_archival_mmr_from_digests(leaf_hashes.clone());
            appended_archival_mmr.append(new_leaf.clone());
            let new_peaks = appended_archival_mmr.get_peaks();
            let original_mps = membership_proofs.clone();
            let indices_of_mutated_mps: Vec<usize> =
                MmrMembershipProof::<Hasher>::batch_update_from_append(
                    &mut membership_proofs.iter_mut().collect::<Vec<_>>(),
                    leaf_count,
                    &new_leaf,
                    &old_peaks,
                );
            for (i, mp) in membership_proofs.iter().enumerate() {
                assert!(mp.verify(&new_peaks, &leaf_hashes[i], leaf_count + 1).0);
            }

            // Verify that mutated membership proofs no longer work and that the non-mutated
            // still work
            for (i, original_mp) in original_mps.iter().enumerate() {
                if indices_of_mutated_mps.contains(&i) {
                    assert!(
                        !original_mp
                            .verify(&new_peaks, &leaf_hashes[i], leaf_count + 1)
                            .0
                    );
                } else {
                    assert!(
                        original_mp
                            .verify(&new_peaks, &leaf_hashes[i], leaf_count + 1)
                            .0
                    );
                }
            }
        }
    }

    #[test]
    fn update_membership_proof_from_append_big_rescue_prime() {
        type Digest = [BFieldElement; 5];
        type Hasher = RescuePrimeRegular;

        // Build MMR from leaf count 0 to 9, and loop through *each*
        // leaf index for MMR, modifying its membership proof with an
        // append update.
        let rp = RescuePrimeRegular::new();
        for leaf_count in 0..9u128 {
            let leaf_hashes: Vec<Digest> = (1001..1001 + leaf_count)
                .map(|x| rp.hash_sequence(&vec![BFieldElement::new(x as u64)]))
                .collect_vec();
            // let archival_mmr = ArchivalMmr::<RescuePrimeRegular>::new(leaf_hashes.clone());
            let mut archival_mmr: ArchivalMmr<Hasher> =
                get_archival_mmr_from_digests(leaf_hashes.clone());
            let new_leaf = rp.hash_sequence(&vec![BFieldElement::new(13333337)]);
            for i in 0..leaf_count {
                let (original_membership_proof, old_peaks): (
                    MmrMembershipProof<RescuePrimeRegular>,
                    Vec<Digest>,
                ) = archival_mmr.prove_membership(i);
                // let mut appended_archival_mmr = archival_mmr.clone();
                let mut appended_archival_mmr: ArchivalMmr<Hasher> =
                    get_archival_mmr_from_digests(leaf_hashes.clone());
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

    #[test]
    fn serialization_test() {
        // You could argue that this test doesn't belong here, as it tests the behavior of
        // an imported library. I included it here, though, because the setup seems a bit clumsy
        // to me so far.
        let rp = RescuePrimeXlix::new();
        type Hasher = RescuePrimeXlix<RP_DEFAULT_WIDTH>;
        let leaf_hashes: Vec<Vec<BFieldElement>> = (1001..1001 + 3)
            .map(|x| rp.hash(&vec![BFieldElement::new(x as u64)], RP_DEFAULT_OUTPUT_SIZE))
            .collect();
        let mut archival_mmr: ArchivalMmr<Hasher> =
            get_archival_mmr_from_digests(leaf_hashes.clone());
        let mp: MmrMembershipProof<RescuePrimeXlix<RP_DEFAULT_WIDTH>> =
            archival_mmr.prove_membership(1).0;
        let json = serde_json::to_string(&mp).unwrap();
        let s_back =
            serde_json::from_str::<MmrMembershipProof<RescuePrimeXlix<RP_DEFAULT_WIDTH>>>(&json)
                .unwrap();
        assert!(
            s_back
                .verify(&archival_mmr.get_peaks(), &leaf_hashes[1], 3)
                .0
        );
    }
}
