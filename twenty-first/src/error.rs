use std::str::FromStr;

use thiserror::Error;

pub use crate::math::bfield_codec::BFieldCodecError;
pub use crate::math::bfield_codec::PolynomialBFieldCodecError;
use crate::prelude::tip5::Digest;
use crate::prelude::x_field_element::EXTENSION_DEGREE;
use crate::prelude::BFieldElement;
pub use crate::util_types::merkle_tree::MerkleTreeError;

#[derive(Debug, Clone, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum ParseBFieldElementError {
    #[error("invalid `u64`")]
    ParseU64Error(#[source] <u64 as FromStr>::Err),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Error)]
#[non_exhaustive]
pub enum TryFromU32sError {
    #[error("U32s<N>: `N` not big enough to hold the value")]
    InsufficientSize,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Error)]
#[non_exhaustive]
pub enum TryFromXFieldElementError {
    #[error("expected {EXTENSION_DEGREE} elements for extension field element, but got {0}")]
    InvalidLength(usize),

    #[error("Digest is not an XFieldElement")]
    InvalidDigest,
}

#[derive(Debug, Clone, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum TryFromDigestError {
    #[error("expected {} elements for digest, but got {0}", Digest::LEN)]
    InvalidLength(usize),

    #[error("invalid `BFieldElement`")]
    InvalidBFieldElement(#[from] ParseBFieldElementError),

    #[error("non-canonical {0} >= {} == `BFieldElement::P`", BFieldElement::P)]
    NotCanonical(u64),

    #[error("overflow converting to Digest")]
    Overflow,
}

#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum TryFromHexDigestError {
    #[error("hex decoding error")]
    HexDecode(#[from] hex::FromHexError),

    #[error("digest error")]
    Digest(#[from] TryFromDigestError),
}
