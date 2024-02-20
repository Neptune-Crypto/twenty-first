use thiserror::Error;

pub use crate::shared_math::bfield_codec::BFieldCodecError;
pub use crate::util_types::merkle_tree::MerkleTreeError;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Error)]
#[non_exhaustive]
pub enum TryFromU32sError {
    #[error("U32s<N>: `N` not big enough to hold the value")]
    InsufficientSize,
}
