use std::str::FromStr;
use thiserror::Error;

use crate::prelude::tip5::DIGEST_LENGTH;
use crate::prelude::x_field_element::EXTENSION_DEGREE;
pub use crate::shared_math::bfield_codec::BFieldCodecError;
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
    #[error("expected {DIGEST_LENGTH} elements for digest, but got {0}")]
    InvalidLength(usize),

    #[error("invalid `BFieldElement`")]
    InvalidBFieldElement(#[from] ParseBFieldElementError),

    #[error("overflow converting to Digest")]
    Overflow,
}
