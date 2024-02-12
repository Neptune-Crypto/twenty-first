use blake3::OUT_LEN;
use num_traits::Zero;

use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::digest::Digest;
use crate::util_types::algebraic_hasher::AlgebraicHasher;
use crate::util_types::algebraic_hasher::Sponge;
use crate::util_types::algebraic_hasher::RATE;

impl Sponge for blake3::Hasher {
    const RATE: usize = RATE;

    fn init() -> Self {
        blake3::Hasher::new()
    }

    fn absorb(&mut self, input: [BFieldElement; RATE]) {
        for elem in input {
            self.update(&elem.value().to_be_bytes());
        }
    }

    fn squeeze(&mut self) -> [BFieldElement; RATE] {
        let digest_a = from_blake3_digest(&self.finalize());

        // There's at most 256 bits of entropy in a blake3::Hash; we stretch this to fit RATE
        // elements, even though it does not provide more security.
        let stretch_bytes = 1u64.to_be_bytes();
        self.update(&stretch_bytes);
        let digest_b = from_blake3_digest(&self.finalize());

        [digest_a.values(), digest_b.values()]
            .concat()
            .try_into()
            .unwrap()
    }
}

impl AlgebraicHasher for blake3::Hasher {
    fn hash_pair(left: Digest, right: Digest) -> Digest {
        let mut hasher = blake3::Hasher::new();
        for elem in left.values().iter().chain(right.values().iter()) {
            hasher.update(&elem.value().to_be_bytes());
        }
        from_blake3_digest(&hasher.finalize())
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
