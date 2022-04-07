use super::stark::TERMINAL_COUNT;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::x_field_element::XFieldElement;
use crate::util_types::proof_stream_typed::ProofStream;
use crate::util_types::simple_hasher::RescuePrimeProduction;
use itertools::Itertools;

pub type StarkProofStream = ProofStream<Item, RescuePrimeProduction>;

#[derive(Debug, Clone)]
pub enum Item {
    MerkleRoot(Vec<BFieldElement>),
    Terminals([XFieldElement; TERMINAL_COUNT]),
    TransposedBaseElements(Vec<BFieldElement>),
    TransposedExtensionElements(Vec<XFieldElement>),
    AuthenticationPath(Vec<Vec<BFieldElement>>),
    RevealedCombinationElement(XFieldElement),
}

impl Item {
    pub fn as_bs(&self) -> Option<Vec<BFieldElement>> {
        match self {
            Self::MerkleRoot(bs) => Some(bs.clone()),
            _ => None,
        }
    }
}

impl IntoIterator for Item {
    type Item = BFieldElement;

    type IntoIter = std::vec::IntoIter<BFieldElement>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Item::MerkleRoot(bs) => bs.into_iter(),
            Item::Terminals(xs) => xs_to_bs(&xs),
            Item::TransposedBaseElements(bs) => bs.into_iter(),
            Item::TransposedExtensionElements(xs) => xs_to_bs(&xs),
            Item::AuthenticationPath(bss) => bss.concat().into_iter(),
            Item::RevealedCombinationElement(x) => xs_to_bs(&[x]),
        }
    }
}

fn xs_to_bs(xs: &[XFieldElement]) -> std::vec::IntoIter<BFieldElement> {
    xs.into_iter()
        .map(|x| x.coefficients.to_vec())
        .concat()
        .into_iter()
}
