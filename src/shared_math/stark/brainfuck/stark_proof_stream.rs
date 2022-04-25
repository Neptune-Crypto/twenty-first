use super::stark::TERMINAL_COUNT;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::rescue_prime_xlix::{RescuePrimeXlix, RP_DEFAULT_WIDTH};
use crate::shared_math::x_field_element::XFieldElement;
use crate::util_types::merkle_tree::PartialAuthenticationPath;
use crate::util_types::proof_stream_typed::{ProofStream, ProofStreamError};
use itertools::Itertools;

pub type StarkProofStream = ProofStream<Item, RescuePrimeXlix<RP_DEFAULT_WIDTH>>;

type FriProof = Vec<(PartialAuthenticationPath<Vec<BFieldElement>>, XFieldElement)>;

#[derive(Debug, Clone)]
pub enum Item {
    MerkleRoot(Vec<BFieldElement>),
    Terminals([XFieldElement; TERMINAL_COUNT]),
    TransposedBaseElements(Vec<BFieldElement>),
    TransposedExtensionElements(Vec<XFieldElement>),
    AuthenticationPath(Vec<Vec<BFieldElement>>),
    RevealedCombinationElement(XFieldElement),
    FriCodeword(Vec<XFieldElement>),
    FriProof(FriProof),
}

impl Item {
    pub fn as_merkle_root(&self) -> Result<Vec<BFieldElement>, Box<dyn std::error::Error>> {
        match self {
            Self::MerkleRoot(bs) => Ok(bs.clone()),
            _ => Err(ProofStreamError::boxed(
                "expected merkle root, but got something else",
            )),
        }
    }

    pub fn as_terminals(
        &self,
    ) -> Result<[XFieldElement; TERMINAL_COUNT], Box<dyn std::error::Error>> {
        match self {
            Self::Terminals(xs) => Ok(xs.to_owned()),
            _ => Err(ProofStreamError::boxed(
                "expected terminals, but got something else",
            )),
        }
    }

    pub fn as_transposed_base_elements(
        &self,
    ) -> Result<Vec<BFieldElement>, Box<dyn std::error::Error>> {
        match self {
            Self::TransposedBaseElements(bs) => Ok(bs.to_owned()),
            _ => Err(ProofStreamError::boxed(
                "expected tranposed base elements, but got something else",
            )),
        }
    }

    pub fn as_transposed_extension_elements(
        &self,
    ) -> Result<Vec<XFieldElement>, Box<dyn std::error::Error>> {
        match self {
            Self::TransposedExtensionElements(xs) => Ok(xs.to_owned()),
            _ => Err(ProofStreamError::boxed(
                "expected tranposed extension elements, but got something else",
            )),
        }
    }

    pub fn as_authentication_path(
        &self,
    ) -> Result<Vec<Vec<BFieldElement>>, Box<dyn std::error::Error>> {
        match self {
            Self::AuthenticationPath(bss) => Ok(bss.to_owned()),
            _ => Err(ProofStreamError::boxed(
                "expected authentication path, but got something else",
            )),
        }
    }

    pub fn as_revealed_combination_element(
        &self,
    ) -> Result<XFieldElement, Box<dyn std::error::Error>> {
        match self {
            Self::RevealedCombinationElement(x) => Ok(x.to_owned()),
            _ => Err(ProofStreamError::boxed(
                "expected revealed combination element, but got something else",
            )),
        }
    }

    pub fn as_fri_codeword(&self) -> Result<Vec<XFieldElement>, Box<dyn std::error::Error>> {
        match self {
            Self::FriCodeword(xs) => Ok(xs.to_owned()),
            _ => Err(ProofStreamError::boxed(
                "expected FRI codeword, but got something else",
            )),
        }
    }

    pub fn as_fri_proof(&self) -> Result<FriProof, Box<dyn std::error::Error>> {
        match self {
            Self::FriProof(fri_proof) => Ok(fri_proof.to_owned()),
            _ => Err(ProofStreamError::boxed(
                "expected FRI proof, but got something else",
            )),
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
            Item::FriCodeword(xs) => xs_to_bs(&xs),
            Item::FriProof(fri_proof) => {
                // FIXME: An iterator can be derived without re-creating a &mut Vec<BFieldElement>.
                let mut bs: Vec<BFieldElement> = vec![];

                for (partial_auth_path, x) in fri_proof.iter() {
                    for bs_in_partial_auth_path in partial_auth_path.0.iter().flatten() {
                        bs.append(&mut bs_in_partial_auth_path.clone());
                    }
                    bs.append(&mut xs_to_bs(&[x.to_owned()]).collect());
                }

                bs.into_iter()
            }
        }
    }
}

fn xs_to_bs(xs: &[XFieldElement]) -> std::vec::IntoIter<BFieldElement> {
    xs.iter()
        .map(|x| x.coefficients.to_vec())
        .concat()
        .into_iter()
}
