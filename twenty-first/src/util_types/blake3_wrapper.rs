use blake3::OUT_LEN;
use num_traits::Zero;

use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::rescue_prime_digest::Digest;
use crate::util_types::algebraic_hasher::AlgebraicHasher;

use super::algebraic_hasher::{INPUT_LENGTH, OUTPUT_LENGTH};

impl AlgebraicHasher for blake3::Hasher {
    fn hash_op(input: &[BFieldElement; INPUT_LENGTH]) -> [BFieldElement; OUTPUT_LENGTH] {
        let mut hasher = Self::new();
        for elem in input.iter() {
            hasher.update(&elem.value().to_be_bytes());
        }
        blake3_hash_op(&hasher.finalize())
    }

    fn hash_slice(elements: &[BFieldElement]) -> Digest {
        let mut hasher = Self::new();
        for elem in elements.iter() {
            hasher.update(&elem.value().to_be_bytes());
        }
        from_blake3_digest(&hasher.finalize())
    }
}

/// Convert a `blake3::Hash` to a `[BFieldElement; OUTPUT_LENGTH]`.
///
/// This is used by legacy STARKs as well as twenty-first unit tests.
pub fn blake3_hash_op(digest: &blake3::Hash) -> [BFieldElement; OUTPUT_LENGTH] {
    let bytes: &[u8; OUT_LEN] = digest.as_bytes();
    [
        BFieldElement::from_ne_bytes(&bytes[0..8]),
        BFieldElement::from_ne_bytes(&bytes[8..16]),
        BFieldElement::from_ne_bytes(&bytes[16..24]),
        BFieldElement::from_ne_bytes(&bytes[24..32]),
        BFieldElement::zero(),
    ]
}

/// Convert a `blake3::Hash` to a `rescue_prime_digest::Digest`.
///
/// This is used by legacy STARKs as well as twenty-first unit tests.
pub fn from_blake3_digest(digest: &blake3::Hash) -> Digest {
    Digest::new(blake3_hash_op(digest))
}
