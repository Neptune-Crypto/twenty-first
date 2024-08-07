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
use crate::prelude::Sponge;
use crate::prelude::Tip5;
use crate::prelude::XFieldElement;

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
    use proptest::collection::vec;
    use proptest::prop_assert_eq;
    use test_strategy::proptest;

    use super::*;

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
