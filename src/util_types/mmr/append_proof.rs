use crate::util_types::simple_hasher::{Hasher, ToDigest};
use std::{fmt::Debug, marker::PhantomData};

use super::shared::calculate_new_peaks_from_append;

#[derive(Debug, Clone)]
pub struct AppendProof<HashDigest, H>
where
    HashDigest: ToDigest<HashDigest> + Clone + Debug,
    H: Hasher<Digest = HashDigest> + Clone,
{
    pub old_leaf_count: u128,
    pub old_peaks: Vec<HashDigest>,
    pub new_peaks: Vec<HashDigest>,
    pub _hasher: PhantomData<H>,
}

impl<HashDigest: PartialEq, H> PartialEq for AppendProof<HashDigest, H>
where
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
    H: Hasher<Digest = HashDigest> + Clone,
{
    // Equality is tested on all fields but the phantom data
    fn eq(&self, other: &Self) -> bool {
        self.old_leaf_count == other.old_leaf_count
            && self.old_peaks == other.old_peaks
            && self.new_peaks == other.new_peaks
    }
}

impl<HashDigest, H> AppendProof<HashDigest, H>
where
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
    H: Hasher<Digest = HashDigest> + Clone,
{
    /// Verify a proof for integral append
    pub fn verify(&self, new_leaf: HashDigest) -> bool {
        let expected_new_peaks = self.new_peaks.clone();
        let new_peaks_calculated: Option<Vec<HashDigest>> =
            calculate_new_peaks_from_append::<H, HashDigest>(
                self.old_leaf_count,
                self.old_peaks.clone(),
                new_leaf,
            )
            .map(|x| x.0);

        match new_peaks_calculated {
            None => false,
            Some(peaks) => expected_new_peaks == peaks,
        }
    }
}

#[cfg(test)]
mod mrr_append_proof_tests {
    use crate::{
        shared_math::b_field_element::BFieldElement,
        util_types::simple_hasher::RescuePrimeProduction,
    };

    use super::*;

    #[test]
    fn equality_test() {
        let digest0 = RescuePrimeProduction::new().hash_one(
            &(10..15)
                .map(BFieldElement::new)
                .collect::<Vec<BFieldElement>>(),
        );
        let digest1 = RescuePrimeProduction::new().hash_one(
            &(11..16)
                .map(BFieldElement::new)
                .collect::<Vec<BFieldElement>>(),
        );
        let append_proof0: AppendProof<Vec<BFieldElement>, RescuePrimeProduction> = AppendProof {
            _hasher: PhantomData,
            new_peaks: vec![digest0.clone()],
            old_leaf_count: 0,
            old_peaks: vec![],
        };
        let append_proof1: AppendProof<Vec<BFieldElement>, RescuePrimeProduction> = AppendProof {
            _hasher: PhantomData,
            new_peaks: vec![digest0.clone()],
            old_leaf_count: 0,
            old_peaks: vec![],
        };
        let append_proof2: AppendProof<Vec<BFieldElement>, RescuePrimeProduction> = AppendProof {
            _hasher: PhantomData,
            new_peaks: vec![digest0.clone()],
            old_leaf_count: 1,
            old_peaks: vec![digest0.clone(), digest1.clone()],
        };
        let append_proof3: AppendProof<Vec<BFieldElement>, RescuePrimeProduction> = AppendProof {
            _hasher: PhantomData,
            new_peaks: vec![digest0.clone()],
            old_leaf_count: 1,
            old_peaks: vec![digest1.clone(), digest0.clone()],
        };
        let append_proof4: AppendProof<Vec<BFieldElement>, RescuePrimeProduction> = AppendProof {
            _hasher: PhantomData,
            new_peaks: vec![digest0.clone()],
            old_leaf_count: 2,
            old_peaks: vec![digest1.clone(), digest0.clone()],
        };
        let append_proof5: AppendProof<Vec<BFieldElement>, RescuePrimeProduction> = AppendProof {
            _hasher: PhantomData,
            new_peaks: vec![digest1.clone()],
            old_leaf_count: 2,
            old_peaks: vec![digest1.clone(), digest0.clone()],
        };
        assert_eq!(
            append_proof0, append_proof1,
            "Equal append proofs must evaluated to equal"
        );
        assert_ne!(
            append_proof1, append_proof2,
            "append proofs not equal when differing on two fields"
        );
        assert_ne!(
            append_proof2, append_proof3,
            "append proofs not equal when differing on order of old peaks"
        );
        assert_ne!(
            append_proof3, append_proof4,
            "append proofs not equal when differing on leaf count"
        );
        assert_ne!(
            append_proof4, append_proof5,
            "append proof not equal when differing on new peaks"
        );
    }
}
