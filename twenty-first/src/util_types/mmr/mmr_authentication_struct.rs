use std::collections::HashMap;
use std::collections::HashSet;

use itertools::Itertools;
use num_traits::One;

use crate::bfe;
use crate::prelude::AlgebraicHasher;
use crate::prelude::BFieldCodec;
use crate::prelude::BFieldElement;
use crate::prelude::Digest;
use crate::prelude::Inverse;
use crate::prelude::MerkleTree;
use crate::prelude::Mmr;
use crate::prelude::MmrMembershipProof;
use crate::prelude::Sponge;
use crate::prelude::Tip5;
use crate::prelude::XFieldElement;
use crate::util_types::mmr::shared_advanced::get_peak_heights;
use crate::util_types::mmr::shared_basic::leaf_index_to_mt_index_and_peak_index;

use super::mmr_accumulator::MmrAccumulator;

const ROOT_MT_INDEX: u64 = 1;

pub struct MerkleAuthenticationStructAuthenticityWitness {
    // All indices are Merkle tree node indices
    nd_auth_struct_indices: Vec<u64>,
    nd_sibling_indices: Vec<(u64, u64)>,
    nd_siblings: Vec<(Digest, Digest)>,
}

impl MerkleAuthenticationStructAuthenticityWitness {
    /// Return the Merkle tree node indices of the digests required to prove
    /// membership for the specified leaf indices
    fn authentication_structure_mt_indices(
        num_leafs: u64,
        leaf_indices: &[u64],
    ) -> impl ExactSizeIterator<Item = u64> {
        // The set of indices of nodes that need to be included in the authentications
        // structure. In principle, every node of every authentication path is needed.
        // The root is never needed. Hence, it is not considered below.
        let mut node_is_needed = HashSet::new();

        // The set of indices of nodes that can be computed from other nodes in the
        // authentication structure or the leafs that are explicitly supplied during
        // verification. Every node on the direct path from the leaf to the root can
        // be computed by the very nature of “authentication path”.
        let mut node_can_be_computed = HashSet::new();

        for &leaf_index in leaf_indices {
            assert!(num_leafs > leaf_index, "Leaf index must be less than number of leafs. Got leaf_index = {leaf_index}; num_leafs = {num_leafs}");

            let mut node_index = leaf_index + num_leafs;
            while node_index > ROOT_MT_INDEX {
                let sibling_index = node_index ^ 1;
                node_can_be_computed.insert(node_index);
                node_is_needed.insert(sibling_index);
                node_index /= 2;
            }
        }

        let set_difference = node_is_needed.difference(&node_can_be_computed).copied();
        set_difference.sorted_unstable().rev()
    }

    pub fn root_from_authentication_struct(
        &self,
        tree_height: u32,
        auth_struct: Vec<Digest>,
        indexed_leafs: Vec<(u64, Digest)>,
    ) -> Digest {
        fn digest_to_xfe(digest: Digest, challenge: XFieldElement) -> XFieldElement {
            let leaf_xfe_lo = XFieldElement::new([digest.0[0], digest.0[1], digest.0[2]]);
            let leaf_xfe_hi =
                challenge * XFieldElement::new([digest.0[3], digest.0[4], BFieldElement::one()]);

            leaf_xfe_lo + leaf_xfe_hi
        }

        fn node_index_to_bfe(node_index: u64) -> BFieldElement {
            BFieldElement::new(node_index)
        }

        // Sanity check
        assert_eq!(
            self.nd_auth_struct_indices.len(),
            auth_struct.len(),
            "Provided auth struct length must match that specified in receiver"
        );

        // Get challenges
        let (alpha, gamma, delta) = {
            let mut sponge = Tip5::init();
            sponge.pad_and_absorb_all(&indexed_leafs.encode());
            sponge.pad_and_absorb_all(&auth_struct.encode());
            let challenges = sponge.sample_scalars(3);
            (challenges[0], challenges[1], challenges[2])
        };

        // Accumulate `p` from public data
        let mut p = XFieldElement::one();
        for i in (0..indexed_leafs.len()).rev() {
            let leaf_index_as_bfe = node_index_to_bfe((1 << tree_height) ^ indexed_leafs[i].0);
            let leaf_as_xfe = digest_to_xfe(indexed_leafs[i].1, alpha);
            let fact = leaf_as_xfe - gamma + delta * leaf_index_as_bfe;
            p *= fact;
        }

        let mut prev = 0;
        for i in (0..auth_struct.len()).rev() {
            let auth_struct_index = self.nd_auth_struct_indices[i];
            assert!(auth_struct_index > prev);
            prev = auth_struct_index;

            let auth_struct_index_as_bfe = node_index_to_bfe(auth_struct_index);

            let auth_str_elem_as_xfe = digest_to_xfe(auth_struct[i], alpha);
            let fact = auth_str_elem_as_xfe - gamma + delta * auth_struct_index_as_bfe;
            p *= fact;
        }

        // Use secret data to invert `p` back and to calculate the root
        let mut t = auth_struct
            .first()
            .copied()
            .unwrap_or_else(|| indexed_leafs.first().unwrap().1);
        let mut t_xfe = digest_to_xfe(t, alpha);
        let mut parent_index_bfe = BFieldElement::one();
        for ((l, r), (left_index, right_index)) in self
            .nd_siblings
            .iter()
            .zip_eq(self.nd_sibling_indices.clone())
        {
            assert_eq!(left_index + 1, right_index);

            t = Tip5::hash_pair(*l, *r);

            let l_xfe = digest_to_xfe(*l, alpha);
            let r_xfe = digest_to_xfe(*r, alpha);
            t_xfe = digest_to_xfe(t, alpha);

            let left_index_bfe = node_index_to_bfe(left_index);
            let right_index_bfe = node_index_to_bfe(right_index);
            parent_index_bfe = left_index_bfe / bfe!(2);

            let fact1 = l_xfe - gamma + delta * left_index_bfe;
            let fact2 = r_xfe - gamma + delta * right_index_bfe;
            let fact_parent = t_xfe - gamma + delta * parent_index_bfe;

            p *= fact1.inverse() * fact2.inverse() * fact_parent;
        }

        assert_eq!(t_xfe - gamma + delta, p);
        assert!(parent_index_bfe.is_one());

        t
    }

    /// Return the authentication structure authenticity witness,
    /// authentication structure, and the (leaf-index, leaf-digest) pairs
    /// from a list of MMR membership proofs. All MMR membership proofs must
    /// belong under the same peak, i.e., be part of the same Merkle tree in
    /// the list of Merkle trees that the MMR contains.
    ///
    /// Panics if the input list of MMR-membership proofs is empty, or if they
    /// do not all belong under the same peak.
    pub fn new_from_mmr_membership_proofs(
        mmra: &MmrAccumulator,
        indexed_mmr_mps: Vec<(u64, Digest, MmrMembershipProof)>,
    ) -> (Self, Vec<Digest>, Vec<(u64, Digest)>) {
        // TODO: Consider rewriting this to return a list of authenticated
        // authentication structs, one for each peak in question.

        // Verify that MMR leaf indices belong to the same peak
        let num_mmr_leafs = mmra.num_leafs();
        let mt_and_peak_indices = indexed_mmr_mps
            .iter()
            .map(|(mmr_leaf_index, _leaf, _mmr_mp)| {
                leaf_index_to_mt_index_and_peak_index(*mmr_leaf_index, num_mmr_leafs)
            })
            .collect_vec();

        assert!(
            mt_and_peak_indices
                .iter()
                .map(|(_mt_index, peak_index)| peak_index)
                .unique()
                .count()
                < 2
        );
        assert!(!mt_and_peak_indices.is_empty(), "");

        let peak_index = mt_and_peak_indices[0].1;
        let mt_indices = mt_and_peak_indices
            .into_iter()
            .map(|(mt_index, _peak_index)| mt_index)
            .collect_vec();

        let peak_index: usize = peak_index.try_into().unwrap();
        let height_of_local_mt = get_peak_heights(num_mmr_leafs)[peak_index];
        let num_leafs_in_local_mt = 1 << height_of_local_mt;
        let local_mt_leaf_indices = mt_indices
            .iter()
            .map(|mt_index| mt_index - num_leafs_in_local_mt)
            .collect_vec();

        let nd_auth_struct_indices = Self::authentication_structure_mt_indices(
            num_leafs_in_local_mt,
            &local_mt_leaf_indices,
        )
        .collect_vec();

        let mut nd_left_indices = mt_indices
            .iter()
            .chain(nd_auth_struct_indices.iter())
            .filter(|idx| **idx != 1)
            .map(|idx| idx & (!1))
            .unique()
            .collect_vec();

        // Fill nd-indices with all non-root indices for node values that can be
        // derived from those from the auth-struct and the leafs.
        {
            let mut i = 0;
            loop {
                let parent = nd_left_indices[i] >> 1;
                if parent == 1 {
                    break;
                }

                let new_left_node = parent & (!1);
                if !nd_left_indices.contains(&new_left_node) {
                    nd_left_indices.push(new_left_node);
                }

                nd_left_indices.sort_unstable();
                nd_left_indices.reverse();

                i += 1;
            }
        }

        let nd_sibling_indices = nd_left_indices
            .into_iter()
            .map(|idx| (idx, idx + 1))
            .collect_vec();

        // Collect all node digests that can be calculated
        let peak = mmra.peaks()[peak_index];
        let mut node_digests: HashMap<u64, Digest> = HashMap::default();
        node_digests.insert(ROOT_MT_INDEX, peak);
        for ((_mmr_leaf_index, mut node, mmr_mp), mut mt_index) in indexed_mmr_mps
            .clone()
            .into_iter()
            .zip_eq(mt_indices.clone())
        {
            for ap_elem in mmr_mp.authentication_path.iter() {
                node_digests.insert(mt_index, node);
                node_digests.insert(mt_index ^ 1, *ap_elem);
                node = if mt_index & 1 == 0 {
                    Tip5::hash_pair(node, *ap_elem)
                } else {
                    Tip5::hash_pair(*ap_elem, node)
                };

                mt_index /= 2;
            }

            // Sanity check that MMR-MPs are valid
            assert_eq!(peak, node, "Derived peak must match provided peak");
        }

        let nd_siblings = nd_sibling_indices
            .iter()
            .map(|(left_idx, right_idx)| (node_digests[left_idx], node_digests[right_idx]))
            .collect_vec();
        let auth_struct = nd_auth_struct_indices
            .iter()
            .map(|idx| node_digests[idx])
            .collect_vec();
        let indexed_leafs = local_mt_leaf_indices
            .into_iter()
            .zip_eq(
                indexed_mmr_mps
                    .into_iter()
                    .map(|(_mmr_leaf_index, leaf, _mmr_mp)| leaf),
            )
            .collect_vec();

        let auth_struct_witness = Self {
            nd_auth_struct_indices,
            nd_sibling_indices,
            nd_siblings,
        };

        (auth_struct_witness, auth_struct, indexed_leafs)
    }

    /// Return the authentication structure witness, authentication structure,
    /// and the (leaf-index, leaf-digest) pairs.
    pub fn new_from_merkle_tree(
        tree: &MerkleTree<Tip5>,
        mut revealed_leaf_indices: Vec<u64>,
    ) -> (Self, Vec<Digest>, Vec<(u64, Digest)>) {
        fn nd_sibling_indices(
            revealed_leaf_indices: &[u64],
            nd_auth_struct_indices: &[u64],
            num_leafs: u64,
        ) -> Vec<(u64, u64)> {
            // TODO: For a better way finding all nd-sibling indices, see the
            // code for [`PartialMerkleTree`] in the `merkle_tree` module.
            let mut nd_sibling_indices = revealed_leaf_indices
                .iter()
                .map(|li| *li ^ num_leafs)
                .chain(nd_auth_struct_indices.iter().copied())
                .filter(|idx| *idx != 1)
                .map(|idx| (idx & (u64::MAX - 1), idx | 1u64))
                .unique()
                .collect_vec();

            if !nd_sibling_indices.is_empty() {
                let mut i = 0;
                loop {
                    let elm = nd_sibling_indices[i];
                    let parent = elm.0 >> 1;
                    if parent == 1 {
                        break;
                    }
                    let uncle = parent ^ 1;

                    let new_pair = if parent & 1 == 0 {
                        (parent, uncle)
                    } else {
                        (uncle, parent)
                    };
                    if !nd_sibling_indices.contains(&new_pair) {
                        nd_sibling_indices.push(new_pair);
                    }

                    nd_sibling_indices.sort_by_key(|(left_idx, _right_idx)| *left_idx);
                    nd_sibling_indices.reverse();

                    i += 1;
                }
            }

            nd_sibling_indices
        }

        revealed_leaf_indices.sort_unstable();
        revealed_leaf_indices.dedup();
        revealed_leaf_indices.reverse();
        let num_leafs: u64 = tree.num_leafs() as u64;

        let mut nd_auth_struct_indices =
            Self::authentication_structure_mt_indices(num_leafs, &revealed_leaf_indices)
                .collect_vec();
        if revealed_leaf_indices.is_empty() {
            nd_auth_struct_indices = vec![ROOT_MT_INDEX];
        }

        let nd_sibling_indices =
            nd_sibling_indices(&revealed_leaf_indices, &nd_auth_struct_indices, num_leafs);
        let nd_siblings = nd_sibling_indices
            .iter()
            .map(|&(l, r)| {
                let l: usize = l.try_into().unwrap();
                let r: usize = r.try_into().unwrap();
                (tree.node(l).unwrap(), tree.node(r).unwrap())
            })
            .collect_vec();

        let revealed_leafs = revealed_leaf_indices
            .iter()
            .map(|j| tree.node((*j + num_leafs) as usize).unwrap())
            .collect_vec();
        let indexed_leafs = revealed_leaf_indices
            .clone()
            .into_iter()
            .zip_eq(revealed_leafs)
            .collect_vec();

        let auth_struct = nd_auth_struct_indices
            .iter()
            .map(|node_index| tree.node(*node_index as usize).unwrap())
            .collect_vec();

        let auth_struct_witness = Self {
            nd_auth_struct_indices,
            nd_sibling_indices,
            nd_siblings,
        };

        (auth_struct_witness, auth_struct, indexed_leafs)
    }
}

#[cfg(test)]
mod tests {
    use crate::math::other::random_elements;
    use crate::prelude::CpuParallel;
    use crate::prelude::MerkleTree;
    use crate::util_types::mmr::mmr_accumulator::util::mmra_with_mps;
    use proptest::collection::vec;
    use proptest::prop_assert_eq;
    use rand::random;
    use test_strategy::proptest;

    use super::*;

    #[test]
    fn auth_struct_from_mmr_mps_test_height_5_9_indices() {
        let local_tree_height = 5;
        let mmr_leaf_indices = [0, 1, 2, 16, 17, 18, 27, 29, 31];
        let indexed_leafs_input: Vec<(u64, Digest)> = mmr_leaf_indices
            .iter()
            .map(|idx| (*idx, random()))
            .collect_vec();
        let (mmra, mmr_mps) = mmra_with_mps(1 << local_tree_height, indexed_leafs_input.clone());
        let indexed_mmr_mps = mmr_mps
            .into_iter()
            .zip_eq(indexed_leafs_input)
            .map(|(mmr_mp, (idx, leaf))| (idx, leaf, mmr_mp))
            .collect_vec();

        let (mmr_auth_struct, auth_struct, indexed_leafs) =
            MerkleAuthenticationStructAuthenticityWitness::new_from_mmr_membership_proofs(
                &mmra,
                indexed_mmr_mps,
            );

        let tree_height: u32 = local_tree_height.try_into().unwrap();
        let computed_root = mmr_auth_struct.root_from_authentication_struct(
            tree_height,
            auth_struct,
            indexed_leafs,
        );

        let peak_index = 0;
        let expected_root = mmra.peaks()[peak_index];
        assert_eq!(expected_root, computed_root);
    }

    #[test]
    fn auth_struct_from_mmr_mps_test_height_4_2_indices() {
        let local_tree_height = 4;
        let mmr_leaf_indices = [0, 1];
        let indexed_leafs_input: Vec<(u64, Digest)> = mmr_leaf_indices
            .iter()
            .map(|idx| (*idx, random()))
            .collect_vec();
        let (mmra, mmr_mps) = mmra_with_mps(1 << local_tree_height, indexed_leafs_input.clone());
        let indexed_mmr_mps = mmr_mps
            .into_iter()
            .zip_eq(indexed_leafs_input)
            .map(|(mmr_mp, (idx, leaf))| (idx, leaf, mmr_mp))
            .collect_vec();

        let (mmr_auth_struct, auth_struct, indexed_leafs) =
            MerkleAuthenticationStructAuthenticityWitness::new_from_mmr_membership_proofs(
                &mmra,
                indexed_mmr_mps,
            );

        let tree_height: u32 = local_tree_height.try_into().unwrap();
        let computed_root = mmr_auth_struct.root_from_authentication_struct(
            tree_height,
            auth_struct,
            indexed_leafs,
        );

        let peak_index = 0;
        let expected_root = mmra.peaks()[peak_index];
        assert_eq!(expected_root, computed_root);
    }

    #[proptest(cases = 20)]
    fn root_from_authentication_struct_prop_test(
        #[strategy(0..12u64)] tree_height: u64,
        #[strategy(0usize..100)] _num_revealed_leafs: usize,
        #[strategy(vec(0u64..1<<#tree_height, #_num_revealed_leafs))] revealed_leaf_indices: Vec<
            u64,
        >,
    ) {
        let num_leafs = 1u64 << tree_height;
        let leafs: Vec<Digest> = random_elements(num_leafs.try_into().unwrap());
        let tree = MerkleTree::<Tip5>::new::<CpuParallel>(&leafs).unwrap();

        let (mmr_auth_struct, auth_struct, indexed_leafs) =
            MerkleAuthenticationStructAuthenticityWitness::new_from_merkle_tree(
                &tree,
                revealed_leaf_indices,
            );

        let tree_height: u32 = tree_height.try_into().unwrap();
        let computed_root = mmr_auth_struct.root_from_authentication_struct(
            tree_height,
            auth_struct,
            indexed_leafs,
        );
        let expected_root = tree.root();
        prop_assert_eq!(expected_root, computed_root);
    }

    fn prop_from_merkle_tree(
        tree_height: usize,
        leaf_indices: Vec<u64>,
        nd_auth_struct_indices: Vec<u64>,
        nd_sibling_indices: Vec<(u64, u64)>,
    ) {
        let leafs: Vec<Digest> = random_elements(1 << tree_height);
        let tree = MerkleTree::<Tip5>::new::<CpuParallel>(&leafs).unwrap();

        let auth_struct = nd_auth_struct_indices
            .iter()
            .map(|i| tree.node(*i as usize).unwrap())
            .collect_vec();
        let revealed_leafs = leaf_indices
            .iter()
            .map(|i| tree.leaf(*i as usize).unwrap())
            .collect_vec();
        let revealed_leafs = leaf_indices
            .into_iter()
            .zip_eq(revealed_leafs)
            .collect_vec();
        let nd_siblings = nd_sibling_indices
            .iter()
            .map(|(left_idx, right_idx)| {
                (
                    tree.node(*left_idx as usize).unwrap(),
                    tree.node(*right_idx as usize).unwrap(),
                )
            })
            .collect_vec();

        let mmr_auth_struct = MerkleAuthenticationStructAuthenticityWitness {
            nd_auth_struct_indices,
            nd_sibling_indices,
            nd_siblings,
        };
        let tree_height: u32 = tree_height.try_into().unwrap();
        let calculated_root = mmr_auth_struct.root_from_authentication_struct(
            tree_height,
            auth_struct,
            revealed_leafs,
        );
        assert_eq!(tree.root(), calculated_root);
    }

    #[test]
    fn root_from_authentication_struct_tree_height_0_no_revealed_leafs() {
        let tree_height = 0;
        let leaf_indices = vec![];
        let nd_auth_struct_indices = vec![1];
        let nd_sibling_indices = vec![];
        prop_from_merkle_tree(
            tree_height,
            leaf_indices,
            nd_auth_struct_indices,
            nd_sibling_indices,
        )
    }

    #[test]
    fn root_from_authentication_struct_tree_height_0_1_revealed() {
        let tree_height = 0;
        let leaf_indices = vec![0];
        let nd_auth_struct_indices = vec![];
        let nd_sibling_indices = vec![];
        prop_from_merkle_tree(
            tree_height,
            leaf_indices,
            nd_auth_struct_indices,
            nd_sibling_indices,
        )
    }

    #[test]
    fn root_from_authentication_struct_tree_height_1_1_revealed() {
        let tree_height = 1;
        let leaf_indices = vec![0u64];
        let nd_auth_struct_indices = vec![3];
        let nd_sibling_indices = vec![(2u64, 3u64)];
        prop_from_merkle_tree(
            tree_height,
            leaf_indices,
            nd_auth_struct_indices,
            nd_sibling_indices,
        )
    }

    #[test]
    fn root_from_authentication_struct_tree_height_1_2_revealed() {
        let tree_height = 1;
        let leaf_indices = vec![0u64, 1];
        let nd_auth_struct_indices = vec![];
        let nd_sibling_indices = vec![(2u64, 3u64)];
        prop_from_merkle_tree(
            tree_height,
            leaf_indices,
            nd_auth_struct_indices,
            nd_sibling_indices,
        )
    }

    #[test]
    fn root_from_authentication_struct_tree_height_2_0_revealed() {
        let tree_height = 2;
        let leaf_indices = vec![];
        let auth_struct_indices = vec![1];
        let nd_sibling_indices = vec![];
        prop_from_merkle_tree(
            tree_height,
            leaf_indices,
            auth_struct_indices,
            nd_sibling_indices,
        )
    }

    #[test]
    fn root_from_authentication_struct_tree_height_2_2_revealed() {
        let tree_height = 2;
        let leaf_indices = vec![0u64, 1];
        let auth_struct_indices = vec![3];
        let nd_sibling_indices = vec![(4u64, 5u64), (2, 3)];
        prop_from_merkle_tree(
            tree_height,
            leaf_indices,
            auth_struct_indices,
            nd_sibling_indices,
        )
    }

    #[test]
    fn root_from_authentication_struct_tree_height_4_4_revealed() {
        let tree_height = 4;
        let leaf_indices = vec![14u64, 12, 10, 8];
        let num_leafs = 1 << tree_height;
        let auth_struct_indices = vec![
            num_leafs + 15,
            num_leafs + 13,
            num_leafs + 11,
            num_leafs + 9,
            2,
        ];
        let nd_sibling_indices_layer_height_0 = [(14u64, 15u64), (12, 13), (10, 11), (8, 9)]
            .map(|(l, r)| (l + num_leafs, r + num_leafs));
        let nd_sibling_indices_layer_height_1 = [(14u64, 15u64), (12u64, 13u64)];
        let nd_sibling_indices_layer_height_2 = [(6u64, 7u64)];
        let nd_sibling_indices_layer_height_3 = [(2u64, 3u64)];
        let nd_sibling_indices = [
            nd_sibling_indices_layer_height_0.to_vec(),
            nd_sibling_indices_layer_height_1.to_vec(),
            nd_sibling_indices_layer_height_2.to_vec(),
            nd_sibling_indices_layer_height_3.to_vec(),
        ]
        .concat();

        prop_from_merkle_tree(
            tree_height,
            leaf_indices,
            auth_struct_indices,
            nd_sibling_indices,
        )
    }
}
