use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use blake3::OUT_LEN;
use num_traits::Zero;
use twenty_first::shared_math::b_field_element::BFIELD_ONE;

use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::digest::Digest;
use crate::util_types::algebraic_hasher::{AlgebraicHasher, SpongeHasher, RATE};

#[derive(Clone, Debug, Default)]
pub struct Blake3State {
    hasher: blake3::Hasher,
}

impl<'a> Arbitrary<'a> for Blake3State {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let bytes: [u8; 64] = u.arbitrary()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(&bytes);
        Ok(Self { hasher })
    }
}

impl Blake3State {
    pub fn new() -> Self {
        let hasher = blake3::Hasher::new();
        Self { hasher }
    }

    pub fn update(&mut self, element: BFieldElement) {
        self.hasher.update(&element.value().to_be_bytes());
    }

    pub fn finalize(&mut self) -> blake3::Hash {
        self.hasher.finalize()
    }
}

impl SpongeHasher for blake3::Hasher {
    const RATE: usize = RATE;
    type SpongeState = Blake3State;

    fn init() -> Self::SpongeState {
        Self::SpongeState::new()
    }

    fn absorb(sponge: &mut Self::SpongeState, input: [BFieldElement; RATE]) {
        for elem in input {
            sponge.update(elem);
        }
    }

    fn squeeze(sponge: &mut Self::SpongeState) -> [BFieldElement; RATE] {
        let digest_a = from_blake3_digest(&sponge.finalize());

        // There's at most 256 bits of entropy in a blake3::Hash; we stretch
        // this to fit RATE elements, even though it does not provide more
        // security.
        sponge.update(BFIELD_ONE);
        let digest_b = from_blake3_digest(&sponge.finalize());

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
