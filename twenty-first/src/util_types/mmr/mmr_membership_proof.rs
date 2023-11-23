use get_size::GetSize;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::RandomState;
use std::collections::hash_set::Intersection;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::{fmt::Debug, iter::FromIterator};

use super::{shared_advanced, shared_basic};
use crate::shared_math::bfield_codec::BFieldCodec;
use crate::shared_math::digest::Digest;
use crate::shared_math::other::log_2_floor;
use crate::util_types::algebraic_hasher::AlgebraicHasher;

#[derive(Debug, Clone, Serialize, Deserialize, GetSize, BFieldCodec)]
pub struct MmrMembershipProof<H>
where
    H: AlgebraicHasher + Sized,
{
    pub leaf_index: u64,
    pub authentication_path: Vec<Digest>,
    #[bfield_codec(ignore)]
    pub _hasher: PhantomData<H>,
}

impl<H: AlgebraicHasher> PartialEq for MmrMembershipProof<H> {
    // Two membership proofs are considered equal if they contain the same authentication path
    // *and* point to the same data index
    fn eq(&self, other: &Self) -> bool {
        self.leaf_index == other.leaf_index && self.authentication_path == other.authentication_path
    }
}

impl<H: AlgebraicHasher> Eq for MmrMembershipProof<H> {}

impl<H: AlgebraicHasher> MmrMembershipProof<H> {
    pub fn new(leaf_index: u64, authentication_path: Vec<Digest>) -> Self {
        Self {
            leaf_index,
            authentication_path,
            _hasher: PhantomData,
        }
    }

    /// Verify a membership proof for an MMR. If verification succeeds, return the final state of the accumulator hash.
    pub fn verify(
        &self,
        peaks: &[Digest],
        leaf_hash: Digest,
        leaf_count: u64,
    ) -> (bool, Option<Digest>) {
        let (mut mt_index, peak_index) =
            shared_basic::leaf_index_to_mt_index_and_peak_index(self.leaf_index, leaf_count);

        // Verify that authentication path has correct length to fail gracefully when fed
        // a too short authentication path.
        if log_2_floor(mt_index as u128) != self.authentication_path.len() as u64 {
            return (false, None);
        }

        let mut i = 0;
        let mut acc_hash: Digest = leaf_hash.to_owned();
        while mt_index != 1 {
            let ap_element = self.authentication_path[i];
            if mt_index % 2 == 0 {
                // node of `acc_hash` is left child
                acc_hash = H::hash_pair(acc_hash, ap_element);
            } else {
                // node of `acc_hash` is right child
                acc_hash = H::hash_pair(ap_element, acc_hash);
            }

            i += 1;
            mt_index /= 2;
        }

        let expected_peak = peaks[peak_index as usize];
        if expected_peak != acc_hash {
            return (false, None);
        }

        (true, Some(acc_hash))
    }

    /// Return the node indices for the authentication path in this membership proof
    pub fn get_node_indices(&self) -> Vec<u64> {
        let mut node_index = shared_advanced::leaf_index_to_node_index(self.leaf_index);
        let mut node_indices = vec![];
        for _ in 0..self.authentication_path.len() {
            let (right_ancestor_count, height) =
                shared_advanced::right_lineage_length_and_own_height(node_index);
            let is_right_child = right_ancestor_count != 0;
            if is_right_child {
                node_indices.push(shared_advanced::left_sibling(node_index, height));

                // parent of right child is +1
                node_index += 1;
            } else {
                node_indices.push(shared_advanced::right_sibling(node_index, height));

                // parent of left child:
                node_index += 1 << (height + 1);
            }
        }

        node_indices
    }

    /// Return the node indices for the hash values that can be derived from this proof
    pub(crate) fn get_direct_path_indices(&self) -> Vec<u64> {
        let mut node_index = shared_advanced::leaf_index_to_node_index(self.leaf_index);
        let mut node_indices = vec![node_index];
        for _ in 0..self.authentication_path.len() {
            node_index = shared_advanced::parent(node_index);
            node_indices.push(node_index);
        }

        node_indices
    }

    /// Return the node index of the peak that the membership proof is pointing
    /// to, as well as this peak's height.
    fn get_peak_index_and_height(&self) -> (u64, u32) {
        (
            *self.get_direct_path_indices().last().unwrap(),
            self.authentication_path.len() as u32,
        )
    }

    /// Update a membership proof with a `verify_append` proof. Returns `true` if an
    /// authentication path has been mutated, false otherwise.
    pub fn update_from_append(
        &mut self,
        old_mmr_leaf_count: u64,
        new_mmr_leaf: Digest,
        old_mmr_peaks: &[Digest],
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
        let added_node_indices = shared_advanced::node_indices_added_by_append(old_mmr_leaf_count);

        // 3
        // Any peak is a left child, so we don't have to check if it's a right or left child.
        // This means we can use a faster method to find the parent index than the generic method.
        let peak_parent_index = own_old_peak_index + (1 << (own_old_peak_height + 1));

        // 3a
        if !added_node_indices.contains(&peak_parent_index) {
            return false;
        }

        // 4 Get node indices of missing digests
        let new_peak_index: u64 = *added_node_indices.last().unwrap();
        let new_node_count: u64 = shared_advanced::leaf_count_to_node_count(old_mmr_leaf_count + 1);
        let node_indices_for_missing_digests: Vec<u64> =
            shared_advanced::get_authentication_path_node_indices(
                own_old_peak_index,
                new_peak_index,
                new_node_count,
            )
            .unwrap();

        // 5 collect all derivable peaks in a hashmap indexed by node index
        // 5.a, collect all node hash digests that are present in the old peaks
        // The keys in the hash map are node indices
        let mut known_digests: HashMap<u64, Digest> = HashMap::new();
        let (_old_mmr_peak_heights, old_mmr_peak_indices) =
            shared_advanced::get_peak_heights_and_peak_node_indices(old_mmr_leaf_count);
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
        for (node_index, &old_peak_digest) in
            added_node_indices.iter().zip(old_mmr_peaks.iter().rev())
        {
            known_digests.insert(*node_index, acc_hash.to_owned());

            // peaks are always left children, so we don't have to check for that
            acc_hash = H::hash_pair(old_peak_digest, acc_hash);

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
                .push(known_digests[&missing_digest_node_index]);
        }

        true
    }

    /// Batch update multiple membership proofs.
    /// Returns the indices of the membership proofs that were modified where index refers
    /// to the order in which the membership proofs were given to this function.
    pub fn batch_update_from_append(
        membership_proofs: &mut [&mut Self],
        old_leaf_count: u64,
        new_leaf: Digest,
        old_peaks: &[Digest],
    ) -> Vec<usize> {
        // 1. Get node indices for nodes added by the append
        //   a. If length of this list is one, newly added leaf was a left child. Return.
        // 2. Get all derivable node digests, store in hash map
        let added_node_indices = shared_advanced::node_indices_added_by_append(old_leaf_count);
        if added_node_indices.len() == 1 {
            return vec![];
        }

        // 2 collect all derivable peaks in a hashmap indexed by node index
        // 2.a, collect all node hash digests that are present in the old peaks
        // The keys in the hash map are node indices
        let mut known_digests: HashMap<u64, Digest> = HashMap::new();
        let (_old_peak_heights, old_peak_indices) =
            shared_advanced::get_peak_heights_and_peak_node_indices(old_leaf_count);
        for (old_peak_index, old_peak_digest) in old_peak_indices.iter().zip(old_peaks.iter()) {
            known_digests.insert(*old_peak_index, old_peak_digest.to_owned());
        }

        // 2.b collect all node hash digests that are derivable from `new_leaf` and
        // `old_peaks`. These are the digests of `new_leaf`'s path to the root.
        let mut acc_hash = new_leaf.to_owned();
        for ((count, node_index), &old_peak_digest) in added_node_indices
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
            acc_hash = H::hash_pair(old_peak_digest, acc_hash);
        }

        // Loop over all membership proofs and insert missing hashes for each
        let mut modified: Vec<usize> = vec![];
        let new_peak_index: u64 = *added_node_indices.last().unwrap();
        let new_node_count: u64 = shared_advanced::leaf_count_to_node_count(old_leaf_count + 1);
        for (i, membership_proof) in membership_proofs.iter_mut().enumerate() {
            let (old_peak_index, old_peak_height) = membership_proof.get_peak_index_and_height();

            // Any peak is a left child, so we don't have to check if it's a right or left child.
            // This means we can use a faster method to find the parent index than the generic method.
            let peak_parent_index = old_peak_index + (1 << (old_peak_height + 1));
            if !added_node_indices.contains(&peak_parent_index) {
                continue;
            }

            modified.push(i);

            let node_indices_for_missing_digests: Vec<u64> =
                shared_advanced::get_authentication_path_node_indices(
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
                    .push(known_digests[&missing_digest_node_index]);
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
        new_leaf: Digest,
    ) -> bool {
        let own_node_ap_indices = self.get_node_indices();
        let affected_node_indices = leaf_mutation_membership_proof.get_direct_path_indices();
        let own_node_indices_hash_set: HashSet<u64> =
            HashSet::from_iter(own_node_ap_indices.clone());
        let affected_node_indices_hash_set: HashSet<u64> =
            HashSet::from_iter(affected_node_indices);
        let mut intersection: Intersection<u64, RandomState> =
            own_node_indices_hash_set.intersection(&affected_node_indices_hash_set);

        // If intersection is empty no change is needed
        let intersection_index_res: Option<&u64> = intersection.next();
        let intersection_index: u64 = match intersection_index_res {
            None => return false,
            Some(&index) => index,
        };

        // Sanity check, should always be true, since `intersection` can at most
        // contain *one* element.
        assert!(intersection.next().is_none());

        // If intersection is **not** empty, we need to calculate all deducible node hashes from the
        // `membership_proof` until we meet the intersecting node.
        let mut deducible_hashes: HashMap<u64, Digest> = HashMap::new();
        let mut node_index =
            shared_advanced::leaf_index_to_node_index(leaf_mutation_membership_proof.leaf_index);
        deducible_hashes.insert(node_index, new_leaf);
        let mut acc_hash: Digest = new_leaf.to_owned();

        // Calculate hashes from the bottom towards the peak. Break when
        // the intersecting node is reached.
        for &hash in leaf_mutation_membership_proof.authentication_path.iter() {
            // It's not necessary to calculate all the way to the root since,
            // the intersection set has a size of at most one (I think).
            // So we can break the loop when we find a `node_index` that
            // is equal to the intersection index. This way we same some
            // hash calculations here.
            if intersection_index == node_index {
                break;
            }

            let (acc_right_ancestor_count, acc_height) =
                shared_advanced::right_lineage_length_and_own_height(node_index);
            if acc_right_ancestor_count != 0 {
                acc_hash = H::hash_pair(hash, acc_hash);

                // parent of right child is +1
                node_index += 1;
            } else {
                acc_hash = H::hash_pair(acc_hash, hash);

                // parent of left child:
                node_index += 1 << (acc_height + 1);
            };
            deducible_hashes.insert(node_index, acc_hash);
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
            *digest = deducible_hashes[&own_node_index];
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
        new_leaf: Digest,
    ) -> Vec<u64> {
        // 1. Calculate all hashes that are deducible from the leaf update
        // 2. Iterate through all membership proofs and update digests that
        //    are deducible from the leaf update proof.

        let mut deducible_hashes: HashMap<u64, Digest> = HashMap::new();
        let mut node_index =
            shared_advanced::leaf_index_to_node_index(leaf_mutation_membership_proof.leaf_index);
        deducible_hashes.insert(node_index, new_leaf);
        let mut acc_hash: Digest = new_leaf.to_owned();

        // Calculate hashes from the bottom towards the peak. Break before we
        // calculate the hash of the peak, since peaks are never included in
        // authentication paths
        for (count, &hash) in leaf_mutation_membership_proof
            .authentication_path
            .iter()
            .enumerate()
        {
            // Do not calculate the last hash as it will always be a peak which
            // are never included in the authentication path
            if count == leaf_mutation_membership_proof.authentication_path.len() - 1 {
                break;
            }

            let (right_ancestor_count, acc_height) =
                shared_advanced::right_lineage_length_and_own_height(node_index);
            if right_ancestor_count != 0 {
                // node is right child
                acc_hash = H::hash_pair(hash, acc_hash);

                // parent of right child is +1
                node_index += 1;
            } else {
                // node is left child
                acc_hash = H::hash_pair(acc_hash, hash);

                // parent of left child:
                node_index += 1 << (acc_height + 1);
            };
            deducible_hashes.insert(node_index, acc_hash);
        }

        let mut modified_membership_proofs: Vec<u64> = vec![];
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
                    *digest = deducible_hashes[&authentication_path_indices];
                    modified_membership_proofs.push(i as u64);
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
        mut authentication_paths_and_leafs: Vec<(MmrMembershipProof<H>, Digest)>,
    ) -> Vec<usize> {
        // Calculate all derivable paths
        let mut new_ap_digests: HashMap<u64, Digest> = HashMap::new();

        // Calculate the derivable digests from a number of leaf mutations and their
        // associated authentication paths. Notice that all authentication paths
        // are only valid *prior* to any updates. They get invalidated (unless updated)
        // throughout the updating as their neighbor leaf digests change values.
        // The hash map `new_ap_digests` takes care of that.
        while let Some((ap, new_leaf)) = authentication_paths_and_leafs.pop() {
            let mut node_index = shared_advanced::leaf_index_to_node_index(ap.leaf_index);
            let former_value = new_ap_digests.insert(node_index, new_leaf);
            assert!(
                former_value.is_none(),
                "Duplicated leaves are not allowed in membership proof updater"
            );
            let mut acc_hash: Digest = new_leaf.to_owned();

            for (count, &hash) in ap.authentication_path.iter().enumerate() {
                // Do not calculate the last hash as it will always be a peak which
                // are never included in the authentication path
                if count == ap.authentication_path.len() - 1 {
                    break;
                }

                // If sibling node is something that has already been calculated, we use that
                // hash digest. Otherwise we use the one in our authentication path.

                let (right_ancestor_count, height) =
                    shared_advanced::right_lineage_length_and_own_height(node_index);
                if right_ancestor_count != 0 {
                    let left_sibling_index = shared_advanced::left_sibling(node_index, height);
                    let sibling_hash: Digest = match new_ap_digests.get(&left_sibling_index) {
                        Some(&h) => h,
                        None => hash,
                    };
                    acc_hash = H::hash_pair(sibling_hash, acc_hash);

                    // Find parent node index
                    node_index += 1;
                } else {
                    let right_sibling_index = shared_advanced::right_sibling(node_index, height);
                    let sibling_hash: Digest = match new_ap_digests.get(&right_sibling_index) {
                        Some(&h) => h,
                        None => hash,
                    };
                    acc_hash = H::hash_pair(acc_hash, sibling_hash);

                    // Find parent node index
                    node_index += 1 << (height + 1);
                }

                new_ap_digests.insert(node_index, acc_hash);
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
                    *digest = new_ap_digests[&authentication_path_indices];
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
    use rand::{random, thread_rng, Rng, RngCore};

    use crate::shared_math::b_field_element::BFieldElement;
    use crate::shared_math::digest::Digest;
    use crate::shared_math::other::random_elements;
    use crate::shared_math::tip5::Tip5;
    use crate::test_shared;
    use crate::test_shared::mmr::get_rustyleveldb_ammr_from_digests;
    use crate::util_types::mmr::archival_mmr::ArchivalMmr;
    use crate::util_types::mmr::mmr_accumulator::MmrAccumulator;
    use crate::util_types::mmr::mmr_trait::Mmr;
    use crate::util_types::storage_vec::RustyLevelDbVec;

    use super::*;

    #[test]
    fn equality_and_hash_test() {
        type H = blake3::Hasher;
        let mut rng = rand::thread_rng();
        let some_digest: Digest = rng.gen();
        let other_digest: Digest = rng.gen();

        // Assert that both membership proofs and their digests are equal
        let mp0 = MmrMembershipProof::<H>::new(4, vec![]);
        let mp1 = MmrMembershipProof::<H>::new(4, vec![]);
        assert_eq!(mp0, mp1);
        assert_eq!(H::hash(&mp0), H::hash(&mp1));

        let mp2 = MmrMembershipProof::<H>::new(4, vec![some_digest]);
        let mp3 = MmrMembershipProof::<H>::new(4, vec![some_digest]);
        assert_eq!(mp2, mp3);
        assert_eq!(H::hash(&mp2), H::hash(&mp3));

        let mp4 = MmrMembershipProof::<H>::new(3, vec![]);
        assert_ne!(mp0, mp4);
        assert_ne!(H::hash(&mp0), H::hash(&mp4));
        assert_ne!(mp2, mp4);
        assert_ne!(H::hash(&mp2), H::hash(&mp4));

        let mp5 = MmrMembershipProof::<H>::new(3, vec![other_digest]);
        assert_ne!(mp2, mp5);
        assert_ne!(H::hash(&mp2), H::hash(&mp5));

        let mp6 = MmrMembershipProof::<H>::new(3, vec![some_digest, other_digest]);
        let mp7 = MmrMembershipProof::<H>::new(3, vec![other_digest, some_digest]);
        assert_eq!(mp6, mp6);
        assert_eq!(H::hash(&mp6), H::hash(&mp6));
        assert_ne!(mp6, mp7);
        assert_ne!(H::hash(&mp6), H::hash(&mp7));
        assert_ne!(mp0, mp6);
        assert_ne!(H::hash(&mp0), H::hash(&mp6));
        assert_ne!(mp2, mp6);
        assert_ne!(H::hash(&mp2), H::hash(&mp6));
        assert_ne!(mp5, mp6);
        assert_ne!(H::hash(&mp5), H::hash(&mp6));
    }

    #[test]
    fn get_node_indices_simple_test() {
        type H = blake3::Hasher;

        let leaf_hashes: Vec<Digest> = random_elements(8);
        let archival_mmr: ArchivalMmr<H, RustyLevelDbVec<Digest>> =
            get_rustyleveldb_ammr_from_digests(leaf_hashes);
        let (membership_proof, _peaks): (MmrMembershipProof<H>, Vec<Digest>) =
            archival_mmr.prove_membership(4);
        assert_eq!(vec![9, 13, 7], membership_proof.get_node_indices());
        assert_eq!(
            vec![8, 10, 14, 15],
            membership_proof.get_direct_path_indices()
        );
    }

    #[test]
    fn get_peak_index_simple_test() {
        type H = blake3::Hasher;

        let mut mmr_size = 7;
        let leaf_digests: Vec<Digest> = random_elements(mmr_size);
        let mut archival_mmr: ArchivalMmr<H, RustyLevelDbVec<Digest>> =
            get_rustyleveldb_ammr_from_digests(leaf_digests);
        let mut expected_peak_indices_and_heights: Vec<(u64, u32)> =
            vec![(7, 2), (7, 2), (7, 2), (7, 2), (10, 1), (10, 1), (11, 0)];
        for (leaf_index, expected_peak_index) in
            (0..mmr_size as u64).zip(expected_peak_indices_and_heights.into_iter())
        {
            let (membership_proof, _peaks): (MmrMembershipProof<H>, Vec<Digest>) =
                archival_mmr.prove_membership(leaf_index);
            assert_eq!(
                expected_peak_index,
                membership_proof.get_peak_index_and_height()
            );
        }

        // Increase size to 8 and verify that the peaks are now different
        mmr_size = 8;
        let leaf_hash: Digest = H::hash(&1337u64);
        archival_mmr.append(leaf_hash);
        expected_peak_indices_and_heights = vec![(15, 3); mmr_size];
        for (leaf_index, expected_peak_index) in
            (0..mmr_size as u64).zip(expected_peak_indices_and_heights.into_iter())
        {
            let (membership_proof, _peaks): (MmrMembershipProof<H>, Vec<Digest>) =
                archival_mmr.prove_membership(leaf_index);
            assert_eq!(
                expected_peak_index,
                membership_proof.get_peak_index_and_height()
            );
        }

        // Increase size to 9 and verify that the peaks are now different
        mmr_size = 9;
        let another_leaf_hash: Digest = H::hash(&13337u64);
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
        for (leaf_index, expected_peak_index) in
            (0..mmr_size as u64).zip(expected_peak_indices_and_heights.into_iter())
        {
            let (membership_proof, _peaks): (MmrMembershipProof<H>, Vec<Digest>) =
                archival_mmr.prove_membership(leaf_index);
            assert_eq!(
                expected_peak_index,
                membership_proof.get_peak_index_and_height()
            );
        }
    }

    #[test]
    fn update_batch_membership_proofs_from_leaf_mutations_new_test() {
        type H = blake3::Hasher;

        let total_leaf_count = 8;
        let leaf_hashes: Vec<Digest> = random_elements(total_leaf_count);
        let mut archival_mmr: ArchivalMmr<H, RustyLevelDbVec<Digest>> =
            get_rustyleveldb_ammr_from_digests(leaf_hashes.clone());
        let mut membership_proofs: Vec<MmrMembershipProof<H>> = vec![];
        for i in 0..total_leaf_count {
            let leaf_index = i as u64;
            membership_proofs.push(archival_mmr.prove_membership(leaf_index).0);
        }

        let new_leaf2: Digest = H::hash(&133337u64);
        let new_leaf3: Digest = H::hash(&12345678u64);
        let mutation_membership_proof_old2 = archival_mmr.prove_membership(2).0;
        let mutation_membership_proof_old3 = archival_mmr.prove_membership(3).0;
        archival_mmr.mutate_leaf_raw(2, new_leaf2);
        archival_mmr.mutate_leaf_raw(3, new_leaf3);
        for mp in membership_proofs.iter_mut() {
            mp.update_from_leaf_mutation(&mutation_membership_proof_old2, new_leaf2);
        }
        for mp in membership_proofs.iter_mut() {
            mp.update_from_leaf_mutation(&mutation_membership_proof_old3, new_leaf3);
        }

        let mut updated_leaf_hashes = leaf_hashes;
        updated_leaf_hashes[2] = new_leaf2;
        updated_leaf_hashes[3] = new_leaf3;
        for (_i, (mp, &leaf_hash)) in membership_proofs
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
        type H = blake3::Hasher;

        let total_leaf_count = 268;
        let leaf_hashes_init: Vec<Digest> = random_elements(total_leaf_count);
        let archival_mmr_init: ArchivalMmr<H, RustyLevelDbVec<Digest>> =
            get_rustyleveldb_ammr_from_digests(leaf_hashes_init);
        let leaf_hashes_final: Vec<Digest> = random_elements(total_leaf_count);
        let archival_mmr_final: ArchivalMmr<H, RustyLevelDbVec<Digest>> =
            get_rustyleveldb_ammr_from_digests(leaf_hashes_final.clone());
        let mut membership_proofs: Vec<MmrMembershipProof<H>> = (0..total_leaf_count as u64)
            .map(|leaf_index| archival_mmr_init.prove_membership(leaf_index).0)
            .collect();
        let membership_proofs_init_and_new_leafs: Vec<(MmrMembershipProof<H>, Digest)> =
            membership_proofs
                .clone()
                .into_iter()
                .zip(leaf_hashes_final.clone())
                .collect();
        let changed_values = MmrMembershipProof::batch_update_from_batch_leaf_mutation(
            &mut membership_proofs.iter_mut().collect::<Vec<_>>(),
            membership_proofs_init_and_new_leafs,
        );

        // This assert only works if `total_leaf_count` is an even number since there
        // otherwise is a membership proof that's an empty authentication path, and that
        // does not change
        assert_eq!(
            (0..total_leaf_count).collect::<Vec<_>>(),
            changed_values,
            "All membership proofs must be indicated as changed"
        );

        for (mp, &final_leaf_hash) in membership_proofs.iter().zip(leaf_hashes_final.iter()) {
            assert!(
                mp.verify(
                    &archival_mmr_final.get_peaks(),
                    final_leaf_hash,
                    total_leaf_count as u64
                )
                .0
            );
        }
    }

    #[test]
    fn batch_update_from_batch_leaf_mutation_test() {
        type H = blake3::Hasher;

        let total_leaf_count = 34;
        let mut leaf_hashes: Vec<Digest> = random_elements(total_leaf_count);
        let mut archival_mmr: ArchivalMmr<H, RustyLevelDbVec<Digest>> =
            get_rustyleveldb_ammr_from_digests(leaf_hashes.clone());
        let mut rng = rand::thread_rng();
        for modified_leaf_count in 0..=total_leaf_count {
            // Pick a set of membership proofs that we want to batch-update
            let own_membership_proof_count = rng.gen_range(0..total_leaf_count);
            let mut all_leaf_indices: Vec<u64> = (0..total_leaf_count as u64).collect();
            let mut own_membership_proofs: Vec<MmrMembershipProof<H>> = vec![];
            for _ in 0..own_membership_proof_count {
                let leaf_index = all_leaf_indices.remove(rng.gen_range(0..all_leaf_indices.len()));
                own_membership_proofs.push(archival_mmr.prove_membership(leaf_index).0);
            }

            // Set the new leafs and their associated authentication paths
            let new_leafs: Vec<Digest> = random_elements(modified_leaf_count);
            let mut all_leaf_indices_new: Vec<u64> = (0..total_leaf_count as u64).collect();
            let mut authentication_paths: Vec<MmrMembershipProof<H>> = vec![];
            for _ in 0..modified_leaf_count {
                let leaf_index =
                    all_leaf_indices_new.remove(rng.gen_range(0..all_leaf_indices_new.len()));
                authentication_paths.push(archival_mmr.prove_membership(leaf_index).0);
            }

            // let the magic start
            let original_mps = own_membership_proofs.clone();
            let mutation_argument: Vec<(MmrMembershipProof<H>, Digest)> = authentication_paths
                .clone()
                .into_iter()
                .zip(new_leafs.clone().into_iter())
                .collect();
            let updated_mp_indices_0 = MmrMembershipProof::batch_update_from_batch_leaf_mutation(
                &mut own_membership_proofs.iter_mut().collect::<Vec<_>>(),
                mutation_argument.clone(),
            );

            // update MMR
            for i in 0..modified_leaf_count {
                let auth_path = authentication_paths[i].clone();
                leaf_hashes[auth_path.leaf_index as usize] = new_leafs[i];
                archival_mmr.mutate_leaf_raw(auth_path.leaf_index, new_leafs[i]);
            }

            // Let's verify that `batch_mutate_leaf_and_update_mps` from the
            // MmrAccumulator agrees
            let mut mmra: MmrAccumulator<H> = (&archival_mmr).into();
            let mut mps_copy = original_mps;
            let updated_mp_indices_1 = mmra.batch_mutate_leaf_and_update_mps(
                &mut mps_copy.iter_mut().collect::<Vec<_>>(),
                mutation_argument,
            );
            assert_eq!(own_membership_proofs, mps_copy);
            assert_eq!(mmra.get_peaks(), archival_mmr.get_peaks());
            assert_eq!(updated_mp_indices_0, updated_mp_indices_1);

            // test that all updated membership proofs are valid under the updated MMR
            (0..own_membership_proof_count).for_each(|i| {
                let membership_proof = own_membership_proofs[i].clone();
                assert!(
                    membership_proof
                        .verify(
                            &archival_mmr.get_peaks(),
                            leaf_hashes[membership_proof.leaf_index as usize],
                            archival_mmr.count_leaves(),
                        )
                        .0
                );
            });
        }
    }

    #[test]
    fn batch_update_from_leaf_mutation_no_change_return_value_test() {
        // This test verifies that the return value indicating changed membership proofs is empty
        // even though the mutations affect the membership proofs. The reason it is empty is that
        // the resulting membership proof digests are unchanged, since the leaf hashes mutations
        // are the identity operators. In other words: the leafs don't change.
        type H = blake3::Hasher;

        let total_leaf_count = 8;
        let (leaf_hashes, mut membership_proofs) =
            make_populated_archival_mmr::<H>(total_leaf_count);

        for i in 0..total_leaf_count {
            let leaf_mutation_membership_proof = membership_proofs[i].clone();
            let new_leaf = leaf_hashes[i];
            let ret = MmrMembershipProof::batch_update_from_leaf_mutation(
                &mut membership_proofs,
                &leaf_mutation_membership_proof,
                new_leaf,
            );

            // the return value must be empty since no membership proof has changed
            assert!(ret.is_empty());
        }

        let membership_proofs_init_and_new_leafs: Vec<(MmrMembershipProof<H>, Digest)> =
            membership_proofs
                .clone()
                .into_iter()
                .zip(leaf_hashes.clone())
                .collect();
        let ret = MmrMembershipProof::batch_update_from_batch_leaf_mutation(
            &mut membership_proofs.iter_mut().collect::<Vec<_>>(),
            membership_proofs_init_and_new_leafs.clone(),
        );

        // the return value must be empty since no membership proof has changed
        assert!(ret.is_empty());

        // Let's test the exact same for the MMR accumulator scheme
        let mut mmra: MmrAccumulator<H> = MmrAccumulator::new(leaf_hashes);
        let ret_from_acc = mmra.batch_mutate_leaf_and_update_mps(
            &mut membership_proofs.iter_mut().collect::<Vec<_>>(),
            membership_proofs_init_and_new_leafs,
        );
        assert!(ret_from_acc.is_empty());
    }

    fn make_populated_archival_mmr<H: AlgebraicHasher>(
        total_leaf_count: usize,
    ) -> (Vec<Digest>, Vec<MmrMembershipProof<H>>) {
        let leaf_hashes: Vec<Digest> = random_elements(total_leaf_count);
        let archival_mmr: ArchivalMmr<H, RustyLevelDbVec<Digest>> =
            get_rustyleveldb_ammr_from_digests(leaf_hashes.clone());
        let mut membership_proofs: Vec<MmrMembershipProof<H>> = vec![];
        for i in 0..total_leaf_count {
            let leaf_index = i as u64;
            membership_proofs.push(archival_mmr.prove_membership(leaf_index).0);
        }
        (leaf_hashes, membership_proofs)
    }

    #[test]
    fn update_batch_membership_proofs_from_batch_leaf_mutations_test() {
        type H = blake3::Hasher;

        let total_leaf_count = 8;
        let leaf_hashes: Vec<Digest> = random_elements(total_leaf_count);
        let mut archival_mmr: ArchivalMmr<H, RustyLevelDbVec<Digest>> =
            get_rustyleveldb_ammr_from_digests(leaf_hashes);
        let modified_leaf_count: usize = 8;
        let mut membership_proofs: Vec<MmrMembershipProof<H>> = vec![];
        for i in 0..modified_leaf_count {
            let leaf_index = i as u64;
            membership_proofs.push(archival_mmr.prove_membership(leaf_index).0);
        }

        let new_leafs: Vec<Digest> = random_elements(modified_leaf_count);
        for i in 0..modified_leaf_count {
            let leaf_mutation_membership_proof = membership_proofs[i].clone();
            let new_leaf = new_leafs[i];
            MmrMembershipProof::batch_update_from_leaf_mutation(
                &mut membership_proofs,
                &leaf_mutation_membership_proof,
                new_leaf,
            );
        }

        (0..modified_leaf_count).for_each(|i| {
            let leaf_index = i as u64;
            archival_mmr.mutate_leaf_raw(leaf_index, new_leafs[i]);
        });

        for (i, mp) in membership_proofs.iter().enumerate() {
            assert!(
                mp.verify(
                    &archival_mmr.get_peaks(),
                    new_leafs[i],
                    archival_mmr.count_leaves()
                )
                .0
            );
        }
    }

    #[test]
    fn update_membership_proof_from_leaf_mutation_test() {
        type H = blake3::Hasher;

        let leaf_hashes: Vec<Digest> = random_elements(8);
        let new_leaf: Digest = H::hash(&133337u64);
        let mut accumulator_mmr = MmrAccumulator::<H>::new(leaf_hashes.clone());

        assert_eq!(8, accumulator_mmr.count_leaves());
        let mut an_archival_mmr: ArchivalMmr<H, RustyLevelDbVec<Digest>> =
            get_rustyleveldb_ammr_from_digests(leaf_hashes.clone());
        let original_archival_mmr: ArchivalMmr<H, RustyLevelDbVec<Digest>> =
            get_rustyleveldb_ammr_from_digests(leaf_hashes.clone());
        let (mut membership_proof, _peaks): (MmrMembershipProof<H>, Vec<Digest>) =
            an_archival_mmr.prove_membership(4);

        // 1. Update a leaf in both the accumulator MMR and in the archival MMR
        let old_peaks = an_archival_mmr.get_peaks();
        let membership_proof_for_manipulated_leaf = an_archival_mmr.prove_membership(2).0;
        an_archival_mmr.mutate_leaf_raw(2, new_leaf);
        accumulator_mmr.mutate_leaf(&membership_proof_for_manipulated_leaf, new_leaf);
        assert_eq!(an_archival_mmr.get_peaks(), accumulator_mmr.get_peaks());

        let new_peaks_1 = an_archival_mmr.get_peaks();
        assert_ne!(
            new_peaks_1, old_peaks,
            "Peaks must change when leaf is mutated"
        );
        let (real_membership_proof_from_archival, archival_peaks) =
            an_archival_mmr.prove_membership(4);
        assert_eq!(
            new_peaks_1, archival_peaks,
            "peaks returned from `get_peaks` must match that returned with membership proof"
        );

        // 2. Verify that the proof fails but that the one from archival works
        assert!(
            !membership_proof
                .verify(&new_peaks_1, new_leaf, accumulator_mmr.count_leaves())
                .0
        );
        assert!(
            membership_proof
                .verify(&old_peaks, leaf_hashes[4], accumulator_mmr.count_leaves())
                .0
        );

        assert!(
            real_membership_proof_from_archival
                .verify(&new_peaks_1, leaf_hashes[4], accumulator_mmr.count_leaves())
                .0
        );

        // 3. Update the membership proof with the membership method
        membership_proof
            .update_from_leaf_mutation(&membership_proof_for_manipulated_leaf, new_leaf);

        // 4. Verify that the proof now succeeds
        assert!(
            membership_proof
                .verify(&new_peaks_1, leaf_hashes[4], accumulator_mmr.count_leaves())
                .0
        );

        // 5. test batch update from leaf update
        for i in 0..8 {
            let mut archival_mmr: ArchivalMmr<H, RustyLevelDbVec<Digest>> =
                get_rustyleveldb_ammr_from_digests(leaf_hashes.clone());
            let mut mps: Vec<MmrMembershipProof<H>> = vec![];

            for j in 0..8 {
                mps.push(original_archival_mmr.prove_membership(j).0);
            }
            let original_mps = mps.clone();
            let leaf_mutation_membership_proof = archival_mmr.prove_membership(i).0;
            archival_mmr.mutate_leaf_raw(i, new_leaf);
            let new_peaks_2 = archival_mmr.get_peaks();
            let modified = MmrMembershipProof::<H>::batch_update_from_leaf_mutation(
                &mut mps,
                &leaf_mutation_membership_proof,
                new_leaf,
            );

            // when updating data index i, all authentication paths are updated
            // *except* for element i.
            let mut expected_modified = Vec::from_iter(0..8);
            expected_modified.remove(i as usize);
            assert_eq!(expected_modified, modified);

            for j in 0..8 {
                let our_leaf = if i == j {
                    new_leaf
                } else {
                    leaf_hashes[j as usize]
                };
                assert!(mps[j as usize].verify(&new_peaks_2, our_leaf, 8).0);

                // Verify that original membership proofs are no longer valid
                // For size = 8, all membership proofs except the one for element 0
                // will be updated since this MMR only contains a single peak.
                // An updated leaf (0 in this case) retains its authentication path after
                // the update. But all other leafs pointing to the same MMR will have updated
                // authentication paths.
                if j == i {
                    assert!(
                        original_mps[j as usize].verify(&new_peaks_2, our_leaf, 8).0,
                        "original membership proof must be valid when j = i"
                    );
                } else {
                    assert!(
                        !original_mps[j as usize].verify(&new_peaks_2, our_leaf, 8).0,
                        "original membership proof must be invalid when j != i"
                    );
                }
            }
        }
    }

    #[test]
    fn update_membership_proof_from_leaf_mutation_blake3_big_test() {
        type H = blake3::Hasher;

        // Build MMR from leaf count 0 to 22, and loop through *each*
        // leaf index for MMR, modifying its membership proof with a
        // leaf update.
        for leaf_count in 0..=22 {
            let leaf_hashes: Vec<Digest> = random_elements(leaf_count);
            let new_leaf: Digest = random();
            let archival_mmr: ArchivalMmr<H, RustyLevelDbVec<Digest>> =
                get_rustyleveldb_ammr_from_digests(leaf_hashes.clone());

            // Loop over all leaf indices that we want to modify in the MMR
            for i in 0..leaf_count {
                let leaf_index_i = i as u64;
                let (leaf_mutation_membership_proof, _old_peaks): (
                    MmrMembershipProof<H>,
                    Vec<Digest>,
                ) = archival_mmr.prove_membership(leaf_index_i);
                let mut modified_archival_mmr: ArchivalMmr<H, RustyLevelDbVec<Digest>> =
                    get_rustyleveldb_ammr_from_digests(leaf_hashes.clone());
                modified_archival_mmr.mutate_leaf_raw(leaf_index_i, new_leaf);
                let new_peaks = modified_archival_mmr.get_peaks();

                // Loop over all leaf indices want a membership proof of, for modification
                for j in 0..leaf_count {
                    let leaf_index_j = j as u64;
                    let mut membership_proof: MmrMembershipProof<H> =
                        archival_mmr.prove_membership(leaf_index_j).0;
                    let original_membership_roof = membership_proof.clone();
                    let membership_proof_was_mutated = membership_proof
                        .update_from_leaf_mutation(&leaf_mutation_membership_proof, new_leaf);
                    let our_leaf = if leaf_index_i == leaf_index_j {
                        new_leaf
                    } else {
                        leaf_hashes[leaf_index_j as usize]
                    };
                    assert!(
                        membership_proof
                            .verify(&new_peaks, our_leaf, leaf_count as u64)
                            .0
                    );

                    // If membership proof was mutated, the original proof must fail
                    if membership_proof_was_mutated {
                        assert!(
                            !original_membership_roof
                                .verify(&new_peaks, our_leaf, leaf_count as u64)
                                .0
                        );
                    }

                    // Verify that modified membership proof matches that which can be
                    // fetched from the modified archival MMR
                    assert_eq!(
                        modified_archival_mmr.prove_membership(leaf_index_j).0,
                        membership_proof
                    );
                }
            }
        }
    }

    #[test]
    fn update_membership_proof_from_append_simple_with_bit_mmra() {
        type H = blake3::Hasher;
        let original_leaf_count = (1 << 35) + (1 << 7) - 1;

        let specified_count = 40;
        let mut rng = rand::thread_rng();
        let mut specified_indices: HashSet<u64> = HashSet::default();
        for _ in 0..specified_count {
            specified_indices.insert(rng.gen_range(0..original_leaf_count));
        }

        // Ensure that at least *one* MP will be mutated upon insertion of a new leaf
        specified_indices.insert(original_leaf_count - 1);

        let collected_values = specified_indices.len();
        let specified_leafs: Vec<(u64, Digest)> = specified_indices
            .into_iter()
            .zip_eq(random_elements(collected_values))
            .collect_vec();
        let (mut mmra, mut mps) =
            test_shared::mmr::mmra_with_mps::<H>(original_leaf_count, specified_leafs.clone());

        let new_leaf: Digest = random();
        let old_peaks = mmra.get_peaks();
        mmra.append(new_leaf);
        assert!(!mps
            .iter()
            .zip_eq(specified_leafs.iter())
            .all(|(mp, (_, digest))| mp.verify(&mmra.get_peaks(), *digest, mmra.count_leaves()).0));
        MmrMembershipProof::batch_update_from_append(
            &mut mps.iter_mut().collect_vec(),
            original_leaf_count,
            new_leaf,
            &old_peaks,
        );

        assert!(mps
            .iter()
            .zip_eq(specified_leafs.iter())
            .all(|(mp, (_, digest))| mp.verify(&mmra.get_peaks(), *digest, mmra.count_leaves()).0));
    }

    #[test]
    fn update_membership_proof_from_append_simple() {
        type H = blake3::Hasher;

        let leaf_count = 7;
        let leaf_hashes: Vec<Digest> = random_elements(leaf_count);
        let new_leaf: Digest = H::hash(&133337u64);
        let archival_mmr: ArchivalMmr<H, RustyLevelDbVec<Digest>> =
            get_rustyleveldb_ammr_from_digests(leaf_hashes.clone());

        for i in 0..leaf_count {
            let leaf_index = i as u64;
            let (mut membership_proof, old_peaks): (MmrMembershipProof<H>, Vec<Digest>) =
                archival_mmr.prove_membership(leaf_index);
            let mut appended_archival_mmr: ArchivalMmr<H, RustyLevelDbVec<Digest>> =
                get_rustyleveldb_ammr_from_digests(leaf_hashes.clone());
            // let mut appended_archival_mmr = archival_mmr.clone();
            appended_archival_mmr.append(new_leaf);
            let new_peaks = appended_archival_mmr.get_peaks();

            // Verify that membership proof fails before update and succeeds after
            // for the case of leaf_count 7, **all** membership proofs have to be
            // updated to be valid, so they should all fail prior to the update.
            let last_leaf_index = leaf_count as u64;
            assert!(
                !membership_proof
                    .verify(
                        &new_peaks,
                        leaf_hashes[leaf_index as usize],
                        last_leaf_index + 1
                    )
                    .0
            );
            membership_proof.update_from_append(last_leaf_index, new_leaf, &old_peaks);
            assert!(
                membership_proof
                    .verify(
                        &new_peaks,
                        leaf_hashes[leaf_index as usize],
                        last_leaf_index + 1
                    )
                    .0
            );

            // Verify that the appended Arhival MMR produces the same membership proof
            // as the one we got by updating the old membership proof
            assert_eq!(
                appended_archival_mmr.prove_membership(leaf_index),
                (membership_proof, new_peaks)
            );
        }
    }

    #[test]
    fn update_membership_proof_from_append_big_tests() {
        type H = blake3::Hasher;

        for leaf_count in 0..68u64 {
            // 1. Build an ArchivalMmr with a variable amount of leaves
            let leaf_digests: Vec<Digest> = random_elements(leaf_count as usize);
            let new_leaf_digest: Digest = rand::thread_rng().gen();
            let archival_mmr: ArchivalMmr<H, RustyLevelDbVec<Digest>> =
                get_rustyleveldb_ammr_from_digests(leaf_digests.clone());

            // For every valid data index
            for leaf_index in 0..leaf_count {
                // 2. Create an equivalent ArchivalMmr, but with `new_leaf_digest` appended
                let mut appended_archival_mmr: ArchivalMmr<H, RustyLevelDbVec<Digest>> =
                    get_rustyleveldb_ammr_from_digests(leaf_digests.clone());
                appended_archival_mmr.append(new_leaf_digest);
                let new_peaks = appended_archival_mmr.get_peaks();

                // 3. Create membership proof for ArchivalMmr from before `new_leaf_digest`
                let (original_membership_proof, original_peaks) =
                    archival_mmr.prove_membership(leaf_index);

                // 4. Update (a copy of) membership proof and verify it
                let mut updated_membership_proof = original_membership_proof.clone();
                let is_updated_membership_proof_changed = updated_membership_proof
                    .update_from_append(leaf_count, new_leaf_digest, &original_peaks);

                let (updated_membership_proof_verifies, _acc_hash_1) = updated_membership_proof
                    .verify(
                        &new_peaks,
                        leaf_digests[leaf_index as usize],
                        leaf_count + 1,
                    );
                assert!(updated_membership_proof_verifies);

                // 5. Assert that original membership proof fails iff a change was indicated
                let (original_membership_proof_verifies, _acc_hash_2) = original_membership_proof
                    .verify(
                        &new_peaks,
                        leaf_digests[leaf_index as usize],
                        leaf_count + 1,
                    );
                if is_updated_membership_proof_changed {
                    assert!(!original_membership_proof_verifies,);
                } else {
                    assert!(original_membership_proof_verifies,);
                }

                // 6. Assert that updating an old membership proof is equivalent to getting a proof for an appended ArchivalMmr
                let expected = (updated_membership_proof, new_peaks);
                assert_eq!(expected, appended_archival_mmr.prove_membership(leaf_index));
            }

            // 7. Test batch update of membership proofs...
            let old_peaks = archival_mmr.get_peaks();
            let original_membership_proofs = (0..leaf_count)
                .map(|leaf_index| archival_mmr.prove_membership(leaf_index).0)
                .collect_vec();

            // ...first by asserting that all leaves are members...
            for (leaf_index, membership_proof) in original_membership_proofs.iter().enumerate() {
                let (membership_proof_verifies_before_batch_update, _acc_hash) =
                    membership_proof.verify(&old_peaks, leaf_digests[leaf_index], leaf_count);
                assert!(membership_proof_verifies_before_batch_update);
            }

            // ...and then create an equivalent ArchivalMmr, but with `new_leaf_digest` appended...
            let mut appended_archival_mmr: ArchivalMmr<H, RustyLevelDbVec<Digest>> =
                get_rustyleveldb_ammr_from_digests(leaf_digests.clone());
            appended_archival_mmr.append(new_leaf_digest);
            let new_peaks = appended_archival_mmr.get_peaks();
            let mut mutated_membership_proofs = original_membership_proofs.clone();

            // ...run batch_update_from_append()...
            let indices_of_mutated_membership_proofs: Vec<usize> =
                MmrMembershipProof::<H>::batch_update_from_append(
                    &mut mutated_membership_proofs.iter_mut().collect_vec(),
                    leaf_count,
                    new_leaf_digest,
                    &old_peaks,
                );

            // ...and assert that mutated membership proofs with new peaks verify!
            for (leaf_index, mutated_membership_proof) in
                mutated_membership_proofs.iter().enumerate()
            {
                let (mutated_membership_proof_verifies_with_new_peaks, _acc_hash) =
                    mutated_membership_proof.verify(
                        &new_peaks,
                        leaf_digests[leaf_index],
                        leaf_count + 1,
                    );
                assert!(mutated_membership_proof_verifies_with_new_peaks);
            }

            // Finally, assert that original membership proofs only work if they're not mutated.
            for (leaf_index, original_membership_proof) in
                original_membership_proofs.iter().enumerate()
            {
                let (original_membership_proof_verifies, _acc_hash) = original_membership_proof
                    .verify(&new_peaks, leaf_digests[leaf_index], leaf_count + 1);

                if indices_of_mutated_membership_proofs.contains(&leaf_index) {
                    assert!(!original_membership_proof_verifies)
                } else {
                    assert!(original_membership_proof_verifies);
                }
            }
        }
    }

    #[test]
    fn update_membership_proof_from_append_big_tip5() {
        type H = Tip5;

        // Build MMR from leaf count 0 to 9, and loop through *each*
        // leaf index for MMR, modifying its membership proof with an
        // append update.
        for leaf_count in 0..9 {
            let leaf_hashes: Vec<Digest> = random_elements(leaf_count);
            let archival_mmr: ArchivalMmr<H, RustyLevelDbVec<Digest>> =
                get_rustyleveldb_ammr_from_digests(leaf_hashes.clone());
            let new_leaf = H::hash(&BFieldElement::new(13333337));
            for i in 0..leaf_count {
                let leaf_index = i as u64;
                let leaf_count_index = leaf_count as u64;
                let (original_membership_proof, old_peaks): (MmrMembershipProof<H>, Vec<Digest>) =
                    archival_mmr.prove_membership(leaf_index);
                let mut appended_archival_mmr: ArchivalMmr<H, RustyLevelDbVec<Digest>> =
                    get_rustyleveldb_ammr_from_digests(leaf_hashes.clone());
                appended_archival_mmr.append(new_leaf);
                let new_peaks = appended_archival_mmr.get_peaks();

                // Update membership proof and verify that it succeeds
                let mut membership_proof_mutated = original_membership_proof.clone();
                let mutated = membership_proof_mutated.update_from_append(
                    leaf_count_index,
                    new_leaf,
                    &old_peaks,
                );
                assert!(
                    membership_proof_mutated
                        .verify(&new_peaks, leaf_hashes[i], leaf_count_index + 1)
                        .0
                );

                // If membership proof mutated, then the old proof must be invalid
                if mutated {
                    assert!(
                        !original_membership_proof
                            .verify(&new_peaks, leaf_hashes[i], leaf_count_index + 1)
                            .0
                    );
                }

                // Verify that the appended Arhival MMR produces the same membership proof
                // as the one we got by updating the old membership proof
                assert_eq!(
                    appended_archival_mmr.prove_membership(leaf_index),
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
        type H = Tip5;

        let leaf_hashes: Vec<Digest> = random_elements(3);
        let archival_mmr: ArchivalMmr<H, RustyLevelDbVec<Digest>> =
            get_rustyleveldb_ammr_from_digests(leaf_hashes.clone());
        let mp: MmrMembershipProof<H> = archival_mmr.prove_membership(1).0;
        let json = serde_json::to_string(&mp).unwrap();
        let s_back = serde_json::from_str::<MmrMembershipProof<H>>(&json).unwrap();
        assert!(
            s_back
                .verify(&archival_mmr.get_peaks(), leaf_hashes[1], 3)
                .0
        );
    }

    #[test]
    fn test_decode_mmr_membership_proof() {
        let mut rng = thread_rng();
        type H = Tip5;
        for _ in 0..100 {
            let num_leafs = 2 + (rng.next_u32() as usize % 1000);
            let leaf_hashes = random_elements(num_leafs);
            let archival_mmr = get_rustyleveldb_ammr_from_digests::<H>(leaf_hashes);
            let leaf_index = (rng.next_u32() as usize % num_leafs) as u64;
            let mp = archival_mmr.prove_membership(leaf_index).0;
            let mp_encoded = mp.encode();
            let mp_decoded = *MmrMembershipProof::decode(&mp_encoded).unwrap();
            assert_eq!(mp, mp_decoded);
        }
    }
}
