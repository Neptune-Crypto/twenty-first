use blake3::OUT_LEN;
use num_traits::Zero;

use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::rescue_prime_digest::Digest;
use crate::util_types::algebraic_hasher::{AlgebraicHasher, Hashable};
use crate::util_types::algebraic_hasher::{AlgebraicHasherNew, SpongeHasher, RATE};

impl SpongeHasher for blake3::Hasher {
    type SpongeState = blake3::Hasher;

    fn absorb_init(input: &[BFieldElement; RATE]) -> Self::SpongeState {
        let mut sponge = blake3::Hasher::new();
        Self::absorb(&mut sponge, input);
        sponge
    }

    fn absorb(sponge: &mut Self::SpongeState, input: &[BFieldElement; RATE]) {
        for elem in input.iter() {
            sponge.update(&elem.value().to_be_bytes());
        }
    }

    fn squeeze(sponge: &mut Self::SpongeState) -> [BFieldElement; RATE] {
        let digest_a = from_blake3_digest(&sponge.finalize());

        // There's at most 256 bits of entropy in a blake3::Hash; we stretch
        // this to fit RATE elements, even though it does not provide more
        // security.
        sponge.update(&[1]);
        let digest_b = from_blake3_digest(&sponge.finalize());

        vec![digest_a.values(), digest_b.values()]
            .concat()
            .try_into()
            .unwrap()
    }
}

impl AlgebraicHasherNew for blake3::Hasher {
    fn hash_pair(left: &Digest, right: &Digest) -> Digest {
        let mut hasher = blake3::Hasher::new();
        for elem in left.values().iter().chain(right.values().iter()) {
            hasher.update(&elem.value().to_be_bytes());
        }
        from_blake3_digest(&hasher.finalize())
    }
}

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
