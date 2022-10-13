use itertools::Itertools;

use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::rescue_prime_digest::Digest;
use crate::shared_math::rescue_prime_regular::DIGEST_LENGTH;
use crate::util_types::algebraic_hasher::{AlgebraicHasher, Hashable};

impl AlgebraicHasher for blake3::Hasher {
    fn hash_slice(elements: &[BFieldElement]) -> Digest {
        let mut hasher = Self::new();
        for elem in elements.iter() {
            hasher.update(&elem.value().to_be_bytes());
        }
        let digest_elements: [BFieldElement; DIGEST_LENGTH] = hasher
            .finalize()
            .as_bytes()
            .chunks(std::mem::size_of::<u64>())
            .take(DIGEST_LENGTH)
            .map(|bytes: &[u8]| {
                let mut bytes_copied: [u8; 8] = [0; 8];
                bytes_copied.copy_from_slice(bytes);
                BFieldElement::new(u64::from_be_bytes(bytes_copied))
            })
            .collect_vec()
            .try_into()
            .expect("A BLAKE3 digest is larger than a Digest");

        Digest::new(digest_elements)
    }

    fn hash_pair(left: &Digest, right: &Digest) -> Digest {
        Self::hash_slice(&vec![left.to_sequence(), right.to_sequence()].concat())
    }
}
