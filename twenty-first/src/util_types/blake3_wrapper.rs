use blake3::OUT_LEN;
use num_traits::Zero;

use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::rescue_prime_digest::Digest;
use crate::util_types::algebraic_hasher::{AlgebraicHasher, Hashable};

impl AlgebraicHasher for blake3::Hasher {
    fn hash_slice(elements: &[BFieldElement]) -> Digest {
        let mut hasher = Self::new();
        for elem in elements.iter() {
            hasher.update(&elem.value().to_be_bytes());
        }
        from_blake3_digest(&hasher.finalize())
    }

    fn hash_pair(left: &Digest, right: &Digest) -> Digest {
        Self::hash_slice(&vec![left.to_sequence(), right.to_sequence()].concat())
    }
}

pub fn from_blake3_digest(digest: &blake3::Hash) -> Digest {
    let bytes: &[u8; OUT_LEN] = digest.as_bytes();
    let elements = [
        BFieldElement::from_ne_bytes(&bytes[0..8]),
        BFieldElement::from_ne_bytes(&bytes[8..16]),
        BFieldElement::from_ne_bytes(&bytes[16..24]),
        BFieldElement::from_ne_bytes(&bytes[24..32]),
        BFieldElement::zero(),
    ];
    Digest::new(elements)
}
