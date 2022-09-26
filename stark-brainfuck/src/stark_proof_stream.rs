use itertools::Itertools;

use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::util_types::merkle_tree::PartialAuthenticationPath;
use twenty_first::util_types::proof_stream_typed::{ProofStream, ProofStreamError};
use twenty_first::util_types::simple_hasher::{Hashable, Hasher, ToVec};

use crate::stark::TERMINAL_COUNT;

pub type StarkProofStream<H> = ProofStream<ProofItem<H>, H>;

#[allow(type_alias_bounds)]
type FriProof<H: Hasher> = Vec<(PartialAuthenticationPath<H::Digest>, XFieldElement)>;
type CompressedAuthenticationPaths<Digest> = Vec<PartialAuthenticationPath<Vec<Digest>>>;

#[derive(Debug, Clone)]
pub enum ProofItem<H: Hasher> {
    CompressedAuthenticationPaths(CompressedAuthenticationPaths<H::Digest>),
    TransposedBaseElementVectors(Vec<Vec<BFieldElement>>),
    TransposedExtensionElementVectors(Vec<Vec<XFieldElement>>),
    MerkleRoot(H::Digest),
    Terminals([XFieldElement; TERMINAL_COUNT]),
    TransposedBaseElements(Vec<BFieldElement>),
    TransposedExtensionElements(Vec<XFieldElement>),
    AuthenticationPath(Vec<Vec<BFieldElement>>),
    RevealedCombinationElement(XFieldElement),
    RevealedCombinationElements(Vec<XFieldElement>),
    FriCodeword(Vec<XFieldElement>),
    FriProof(FriProof<H>),
}

impl<H: Hasher> ProofItem<H> {
    pub fn as_compressed_authentication_paths(
        &self,
    ) -> Result<CompressedAuthenticationPaths<H::Digest>, Box<dyn std::error::Error>> {
        match self {
            Self::CompressedAuthenticationPaths(caps) => Ok(caps.to_owned()),
            _ => Err(ProofStreamError::boxed(
                "expected compressed authentication paths, but got something else",
            )),
        }
    }

    pub fn as_transposed_base_element_vectors(
        &self,
    ) -> Result<Vec<Vec<BFieldElement>>, Box<dyn std::error::Error>> {
        match self {
            Self::TransposedBaseElementVectors(bss) => Ok(bss.to_owned()),
            _ => Err(ProofStreamError::boxed(
                "expected transposed base element vectors, but got something else",
            )),
        }
    }

    pub fn as_transposed_extension_element_vectors(
        &self,
    ) -> Result<Vec<Vec<XFieldElement>>, Box<dyn std::error::Error>> {
        match self {
            Self::TransposedExtensionElementVectors(xss) => Ok(xss.to_owned()),
            _ => Err(ProofStreamError::boxed(
                "expected transposed extension element vectors, but got something else",
            )),
        }
    }

    pub fn as_merkle_root(&self) -> Result<H::Digest, Box<dyn std::error::Error>> {
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

    pub fn as_revealed_combination_elements(
        &self,
    ) -> Result<Vec<XFieldElement>, Box<dyn std::error::Error>> {
        match self {
            Self::RevealedCombinationElements(xs) => Ok(xs.to_owned()),
            _ => Err(ProofStreamError::boxed(
                "expected revealed combination elements, but got something else",
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

    pub fn as_fri_proof(&self) -> Result<FriProof<H>, Box<dyn std::error::Error>> {
        match self {
            Self::FriProof(fri_proof) => Ok(fri_proof.to_owned()),
            _ => Err(ProofStreamError::boxed(
                "expected FRI proof, but got something else",
            )),
        }
    }
}

impl<H: Hasher> Default for ProofItem<H> {
    fn default() -> Self {
        panic!("Do not have a default method ProofItem")
    }
}

impl<H: Hasher> IntoIterator for ProofItem<H>
where
    BFieldElement: Hashable<H::T>,
    ProofItem<H>: Default,
{
    type Item = H::T;

    type IntoIter = std::vec::IntoIter<H::T>;

    // simulates serialization
    fn into_iter(self) -> Self::IntoIter {
        match self {
            ProofItem::MerkleRoot(digest) => digest.to_vec().into_iter(),
            ProofItem::Terminals(xs) => bs_to_ts::<H::T>(&xs_to_bs(&xs)).into_iter(),
            ProofItem::TransposedBaseElements(bs) => bs_to_ts(&bs).into_iter(),
            ProofItem::TransposedExtensionElements(xs) => bs_to_ts(&xs_to_bs(&xs)).into_iter(),
            ProofItem::AuthenticationPath(bss) => bs_to_ts(&bss.concat()).into_iter(),
            ProofItem::RevealedCombinationElement(x) => bs_to_ts(&xs_to_bs(&[x])).into_iter(),
            ProofItem::FriCodeword(xs) => bs_to_ts(&xs_to_bs(&xs)).into_iter(),
            ProofItem::RevealedCombinationElements(xs) => bs_to_ts(&xs_to_bs(&xs)).into_iter(),
            ProofItem::FriProof(fri_proof) => {
                let mut element_sequence: Vec<H::T> = vec![];

                for (partial_auth_path, xfield_element) in fri_proof.iter() {
                    for bs_in_partial_auth_path in partial_auth_path.0.iter().flatten() {
                        element_sequence.append(&mut bs_in_partial_auth_path.clone().to_vec());
                    }
                    element_sequence.append(&mut bs_to_ts(&xs_to_bs(&[xfield_element.to_owned()])));
                }

                element_sequence.into_iter()
            }
            ProofItem::CompressedAuthenticationPaths(partial_auth_paths) => {
                let mut bs: Vec<H::T> = vec![];

                for partial_auth_path in partial_auth_paths.iter() {
                    for bs_in_partial_auth_path in partial_auth_path.0.iter().flatten() {
                        let flattened = bs_in_partial_auth_path
                            .iter()
                            .flat_map(|d| d.to_vec())
                            .collect_vec();
                        bs.append(&mut flattened.clone());
                    }
                }

                bs.into_iter()
            }
            ProofItem::TransposedBaseElementVectors(bss) => {
                bs_to_ts::<H::T>(&bss.concat()).into_iter()
            }
            ProofItem::TransposedExtensionElementVectors(xss) => xss
                .into_iter()
                .map(|xs| bs_to_ts::<H::T>(&xs_to_bs(&xs)))
                .concat()
                .into_iter(),
        }
    }
}

// impl<H: Hasher> IntoIterator for ProofItem<H>
// where
//     Vec<BFieldElement>: From<Vec<H::T>> + Hashable<H::T>,
//     BFieldElement: Hashable<H::T>,
// {
//     type Item = H::T;

//     type IntoIter = std::vec::IntoIter<H::T>;
// }

fn xs_to_bs(xs: &[XFieldElement]) -> Vec<BFieldElement> {
    xs.iter().map(|x| x.coefficients.to_vec()).concat()
}

fn bs_to_ts<T>(bs: &[BFieldElement]) -> Vec<T>
where
    BFieldElement: Hashable<T>,
{
    bs.iter().flat_map(|b| b.to_sequence()).collect_vec()
}
