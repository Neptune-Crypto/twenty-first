use std::str::FromStr;

use thiserror::Error;

pub use crate::math::bfield_codec::BFieldCodecError;
pub use crate::math::bfield_codec::PolynomialBFieldCodecError;
use crate::prelude::BFieldElement;
use crate::prelude::Digest;
use crate::prelude::x_field_element::EXTENSION_DEGREE;
pub use crate::util_types::merkle_tree::MerkleTreeError;

pub(crate) const USIZE_TO_U64_ERR: &str =
    "internal error: type `usize` should have at most 64 bits";
pub(crate) const U32_TO_USIZE_ERR: &str =
    "internal error: type `usize` should have at least 32 bits";

#[derive(Debug, Clone, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum ParseBFieldElementError {
    #[error("invalid `i128`")]
    ParseIntError(#[source] <i128 as FromStr>::Err),

    #[error("{0} must be in canonical (open) interval (-{p}, {p})", p = BFieldElement::P - 1)]
    NotCanonical(i128),

    #[error(
        "incorrect number of bytes: {0} != {bytes} == `BFieldElement::BYTES`",
        bytes = BFieldElement::BYTES
    )]
    InvalidNumBytes(usize),
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
    #[error("expected {len} elements for digest, but got {0}", len = Digest::LEN)]
    InvalidLength(usize),

    #[error("invalid `BFieldElement`")]
    InvalidBFieldElement(#[from] ParseBFieldElementError),

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
