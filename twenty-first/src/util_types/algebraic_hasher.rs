use std::fmt::Debug;
use std::iter;

use arbitrary::Arbitrary;
use itertools::Itertools;

use crate::shared_math::b_field_element::{BFieldElement, BFIELD_ONE, BFIELD_ZERO};
use crate::shared_math::bfield_codec::BFieldCodec;
use crate::shared_math::digest::{Digest, DIGEST_LENGTH};
use crate::shared_math::other::{is_power_of_two, roundup_nearest_multiple};
use crate::shared_math::x_field_element::{XFieldElement, EXTENSION_DEGREE};

pub const RATE: usize = 10;

/// The hasher [Domain] differentiates between the modes of hashing.
///
/// The main purpose of declaring the domain is to prevent collisions between
/// different types of hashing by introducing defining differences in the way
/// the hash function's internal state (e.g. a sponge state's capacity) is
/// initialized.
#[derive(Debug, PartialEq, Eq)]
pub enum Domain {
    /// The `VariableLength` domain is used for hashing objects that potentially
    /// serialize to more than [RATE] number of field elements.
    VariableLength,

    /// The `FixedLength` domain is used for hashing objects that always fit
    /// within [RATE] number of fields elements, e.g. a pair of [Digest].
    FixedLength,
}

pub trait SpongeHasher: Clone + Debug + Default + Send + Sync {
    const RATE: usize;
    type SpongeState: Clone + Debug + Default + for<'a> Arbitrary<'a>;

    /// Initialize a sponge state
    fn init() -> Self::SpongeState;

    /// Absorb an array of [RATE] field elements into the sponge's state, mutating it.
    fn absorb(sponge: &mut Self::SpongeState, input: [BFieldElement; RATE]);

    /// Squeeze an array of [RATE] field elements out from the sponge's state, mutating it.
    fn squeeze(sponge: &mut Self::SpongeState) -> [BFieldElement; RATE];

    /// Chunk `input` into arrays of [RATE] elements and repeatedly [SpongeHasher::absorb()].
    fn pad_and_absorb_all(sponge: &mut Self::SpongeState, input: &[BFieldElement]) {
        // pad input with [1, 0, 0, ...] – padding is at least one element
        let padded_length = roundup_nearest_multiple(input.len() + 1, RATE);
        let padding_iter = [BFIELD_ONE].iter().chain(iter::repeat(&BFIELD_ZERO));
        let padded_input = input.iter().chain(padding_iter).take(padded_length);
        for chunk in padded_input.chunks(RATE).into_iter() {
            let absorb_elems: [BFieldElement; RATE] = chunk
                .cloned()
                .collect::<Vec<_>>()
                .try_into()
                .expect("a multiple of RATE elements");
            Self::absorb(sponge, absorb_elems);
        }
    }
}

pub trait AlgebraicHasher: SpongeHasher {
    /// Hash two [Digest]s into one.
    fn hash_pair(left: Digest, right: Digest) -> Digest;

    /// Hash a `value: &T` to a [Digest].
    ///
    /// The `T` must implement BFieldCodec.
    fn hash<T: BFieldCodec>(value: &T) -> Digest {
        Self::hash_varlen(&value.encode())
    }

    /// Hash a variable-length sequence of [BFieldElement].
    ///
    /// - Apply the correct padding
    /// - [SpongeHasher::absorb_repeatedly()]
    /// - [SpongeHasher::squeeze()] once.
    fn hash_varlen(input: &[BFieldElement]) -> Digest {
        let mut sponge = Self::init();
        Self::pad_and_absorb_all(&mut sponge, input);
        let produce: [BFieldElement; RATE] = Self::squeeze(&mut sponge);

        Digest::new((&produce[..DIGEST_LENGTH]).try_into().unwrap())
    }

    /// Produce `num_indices` random integer values in the range `[0, upper_bound)`.
    ///
    /// - The randomness depends on `state`.
    /// - `upper_bound` must be a power of 2.
    ///
    /// This method uses von Neumann rejection sampling.
    /// Specifically, if the top 32 bits of a BFieldElement are all
    /// ones, then the bottom 32 bits are not uniformly distributed,
    /// and so they are dropped. This method invokes squeeze until
    /// enough uniform u32s have been sampled.
    fn sample_indices(
        state: &mut Self::SpongeState,
        upper_bound: u32,
        num_indices: usize,
    ) -> Vec<u32> {
        debug_assert!(is_power_of_two(upper_bound));
        let mut indices = vec![];
        let mut squeezed_elements = vec![];
        while indices.len() != num_indices {
            if squeezed_elements.is_empty() {
                squeezed_elements = Self::squeeze(state).into_iter().rev().collect_vec();
            }
            let element = squeezed_elements.pop().unwrap();
            if element != BFieldElement::new(BFieldElement::MAX) {
                indices.push(element.value() as u32 % upper_bound);
            }
        }
        indices
    }

    /// Produce `num_elements` random [XFieldElement] values.
    ///
    /// - The randomness depends on `state`.
    ///
    /// If `num_elements` is not divisible by `RATE`, spill the remaining elements of the last `squeeze`.
    fn sample_scalars(state: &mut Self::SpongeState, num_elements: usize) -> Vec<XFieldElement> {
        let num_squeezes = (num_elements * EXTENSION_DEGREE + Self::RATE - 1) / Self::RATE;
        debug_assert!(
            num_elements * EXTENSION_DEGREE <= num_squeezes * Self::RATE,
            "need {} elements but getting {}",
            num_elements * EXTENSION_DEGREE,
            num_squeezes * Self::RATE
        );
        (0..num_squeezes)
            .flat_map(|_| Self::squeeze(state))
            .collect_vec()
            .chunks(3)
            .take(num_elements)
            .map(|elem| XFieldElement::new([elem[0], elem[1], elem[2]]))
            .collect()
    }
}

#[cfg(test)]
mod algebraic_hasher_tests {
    use std::ops::Mul;

    use num_traits::One;
    use num_traits::Zero;
    use rand::Rng;
    use rand_distr::Distribution;
    use rand_distr::Standard;

    use crate::shared_math::digest::DIGEST_LENGTH;
    use crate::shared_math::tip5::Tip5;
    use crate::shared_math::x_field_element::EXTENSION_DEGREE;

    use super::*;

    fn encode_prop<T>(smallest: T, largest: T)
    where
        T: Eq + BFieldCodec,
        Standard: Distribution<T>,
    {
        let smallest_seq = smallest.encode();
        let largest_seq = largest.encode();
        assert_ne!(smallest_seq, largest_seq);
        assert_eq!(smallest_seq.len(), largest_seq.len());

        let mut rng = rand::thread_rng();
        let random_a: T = rng.gen();
        let random_b: T = rng.gen();

        if random_a != random_b {
            assert_ne!(random_a.encode(), random_b.encode());
        } else {
            assert_eq!(random_a.encode(), random_b.encode());
        }
    }

    #[test]
    fn to_sequence_test() {
        // bool
        encode_prop(false, true);

        // u32
        encode_prop(0u32, u32::MAX);

        // u64
        encode_prop(0u64, u64::MAX);

        // BFieldElement
        let bfe_max = BFieldElement::new(BFieldElement::MAX);
        encode_prop(BFIELD_ZERO, bfe_max);

        // XFieldElement
        let xfe_max = XFieldElement::new([bfe_max; EXTENSION_DEGREE]);
        encode_prop(XFieldElement::zero(), xfe_max);

        // Digest
        let digest_zero = Digest::new([BFIELD_ZERO; DIGEST_LENGTH]);
        let digest_max = Digest::new([bfe_max; DIGEST_LENGTH]);
        encode_prop(digest_zero, digest_max);

        // u128
        encode_prop(0u128, u128::MAX);
    }

    fn sample_indices_prop(max: u32, num_indices: usize) {
        let mut sponge = Tip5::randomly_seeded();
        let indices = Tip5::sample_indices(&mut sponge, max, num_indices);
        assert_eq!(num_indices, indices.len());
        assert!(indices.into_iter().all(|index| index < max));
    }

    #[test]
    fn sample_indices_test() {
        let cases = [
            (2, 0),
            (4, 1),
            (8, 9),
            (16, 10),
            (32, 11),
            (64, 19),
            (128, 20),
            (256, 21),
            (512, 65),
        ];

        for (upper_bound, num_indices) in cases {
            sample_indices_prop(upper_bound, num_indices);
        }
    }

    #[test]
    fn sample_scalars_test() {
        let amounts = [0, 1, 2, 3, 4];
        let mut sponge = Tip5::randomly_seeded();
        let mut product = XFieldElement::one();
        for amount in amounts {
            let scalars = Tip5::sample_scalars(&mut sponge, amount);
            assert_eq!(amount, scalars.len());
            product *= scalars
                .into_iter()
                .fold(XFieldElement::one(), XFieldElement::mul);
        }
        assert_ne!(product, XFieldElement::zero()); // false failure with prob ~2^{-192}
    }
}
