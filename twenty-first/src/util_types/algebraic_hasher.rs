use std::iter;

use itertools::Itertools;

use crate::shared_math::b_field_element::{BFieldElement, BFIELD_ONE, BFIELD_ZERO};
use crate::shared_math::other::{is_power_of_two, roundup_nearest_multiple};
use crate::shared_math::rescue_prime_digest::{Digest, DIGEST_LENGTH};
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

pub trait SpongeHasher: Clone + Send + Sync {
    const RATE: usize;
    type SpongeState: Clone;

    /// Initialize a sponge state
    fn init() -> Self::SpongeState;

    /// Absorb an array of [RATE] field elements into the sponge's state, mutating it.
    fn absorb(sponge: &mut Self::SpongeState, input: &[BFieldElement; RATE]);

    /// Squeeze an array of [RATE] field elements out from the sponge's state, mutating it.
    fn squeeze(sponge: &mut Self::SpongeState) -> [BFieldElement; RATE];

    /// Chunk `input` into arrays of [RATE] elements and repeatedly [SpongeHasher::absorb()].
    ///
    /// **Note:** This method panics if `input` does not contain a multiple of [RATE] elements.
    fn absorb_repeatedly<'a, I>(sponge: &mut Self::SpongeState, input: I)
    where
        I: Iterator<Item = &'a BFieldElement>,
    {
        for chunk in input.chunks(RATE).into_iter() {
            let absorb_elems: [BFieldElement; RATE] = chunk
                .cloned()
                .collect::<Vec<_>>()
                .try_into()
                .expect("a multiple of RATE elements");
            Self::absorb(sponge, &absorb_elems);
        }
    }
}

pub trait AlgebraicHasher: SpongeHasher {
    /// Hash two [Digest]s into one.
    fn hash_pair(left: &Digest, right: &Digest) -> Digest;

    /// Hash a `value: &T` to a [Digest].
    ///
    /// The `T` must implement [Hashable].
    fn hash<T: Hashable>(value: &T) -> Digest {
        Self::hash_varlen(&value.to_sequence())
    }

    /// Hash a variable-length sequence of [BFieldElement].
    ///
    /// - Apply the correct padding
    /// - [SpongeHasher::absorb_repeatedly()]
    /// - [SpongeHasher::squeeze()] once.
    fn hash_varlen(input: &[BFieldElement]) -> Digest {
        // calculate padded length; padding is at least one element
        let padded_length = roundup_nearest_multiple(input.len() + 1, RATE);

        // pad input with [1, 0, 0, ...]
        let input_iter = input.iter();
        let padding_iter = [&BFIELD_ONE].into_iter().chain(iter::repeat(&BFIELD_ZERO));
        let padded_input = input_iter.chain(padding_iter).take(padded_length);

        let mut sponge = Self::init();
        Self::absorb_repeatedly(&mut sponge, padded_input);
        let produce: [BFieldElement; RATE] = Self::squeeze(&mut sponge);

        Digest::new((&produce[..DIGEST_LENGTH]).try_into().unwrap())
    }

    /// Produce `num_indices` random integer values in the range `[0, upper_bound)`.
    ///
    /// - The randomness depends on `state`.
    /// - `upper_bound` must be a power of 2.
    ///
    /// If `num_indices` is not divisible by `RATE`, spill the remaining elements of the last `squeeze`.
    fn sample_indices(
        state: &mut Self::SpongeState,
        upper_bound: u32,
        num_indices: usize,
    ) -> Vec<u32> {
        assert!(is_power_of_two(upper_bound));
        let num_squeezes = (num_indices + Self::RATE - 1) / Self::RATE;
        (0..num_squeezes)
            .flat_map(|_| Self::squeeze(state))
            .take(num_indices)
            .map(|b| (b.value() % upper_bound as u64) as u32)
            .collect()
    }

    /// Produce `num_elements` random [XFieldElement] values.
    ///
    /// - The randomness depends on `state`.
    ///
    /// Since [RATE] is not divisible by [EXTENSION_DEGREE], produce as many [XFieldElement] per
    /// `squeeze` as possible, and spill the remaining element(s). This causes some internal
    /// fragmentation, but it greatly simplifies building [AlgebraicHasher::sample_xfield()] on
    /// Triton VM.
    fn sample_scalars(state: &mut Self::SpongeState, num_elements: usize) -> Vec<XFieldElement> {
        let num_squeezes = num_elements * EXTENSION_DEGREE / Self::RATE;
        (0..num_squeezes)
            .map(|_| Self::squeeze(state))
            .flat_map(|elems| {
                vec![
                    XFieldElement::new([elems[0], elems[1], elems[2]]),
                    XFieldElement::new([elems[3], elems[4], elems[5]]),
                    XFieldElement::new([elems[6], elems[7], elems[8]]),
                    // spill 1 element, elems[9], per squeeze
                ]
            })
            .collect()
    }
}

pub trait Hashable {
    fn to_sequence(&self) -> Vec<BFieldElement>;
}

impl Hashable for bool {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        vec![BFieldElement::new(*self as u64)]
    }
}

impl Hashable for u32 {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        vec![BFieldElement::new(*self as u64)]
    }
}

impl Hashable for BFieldElement {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        vec![*self]
    }
}

impl Hashable for u64 {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        let lo: u64 = self & 0xffff_ffff;
        let hi: u64 = self >> 32;

        vec![BFieldElement::new(lo), BFieldElement::new(hi)]
    }
}

impl Hashable for usize {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        assert_eq!(usize::BITS, u64::BITS);
        (*self as u64).to_sequence()
    }
}

impl Hashable for XFieldElement {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        self.coefficients.to_vec()
    }
}

impl Hashable for Digest {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        self.values().to_vec()
    }
}

impl Hashable for u128 {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        let lo: u64 = (self & 0xffff_ffff) as u64;
        let lo_mid: u64 = ((self >> 32) & 0xffff_ffff) as u64;
        let hi_mid: u64 = ((self >> 64) & 0xffff_ffff) as u64;
        let hi: u64 = ((self >> 96) & 0xffff_ffff) as u64;

        vec![
            BFieldElement::new(lo),
            BFieldElement::new(lo_mid),
            BFieldElement::new(hi_mid),
            BFieldElement::new(hi),
        ]
    }
}

#[cfg(test)]
mod algebraic_hasher_tests {
    use std::ops::Mul;

    use num_traits::{One, Zero};
    use rand::{thread_rng, Rng, RngCore};
    use rand_distr::{Distribution, Standard};

    use crate::shared_math::rescue_prime_digest::DIGEST_LENGTH;
    use crate::shared_math::tip5::{Tip5, Tip5State};
    use crate::shared_math::x_field_element::EXTENSION_DEGREE;

    use super::*;

    fn to_sequence_prop<T>(smallest: T, largest: T)
    where
        T: Eq + Hashable,
        Standard: Distribution<T>,
    {
        let smallest_seq = smallest.to_sequence();
        let largest_seq = largest.to_sequence();
        assert_ne!(smallest_seq, largest_seq);
        assert_eq!(smallest_seq.len(), largest_seq.len());

        let mut rng = rand::thread_rng();
        let random_a: T = rng.gen();
        let random_b: T = rng.gen();

        if random_a != random_b {
            assert_ne!(random_a.to_sequence(), random_b.to_sequence());
        } else {
            assert_eq!(random_a.to_sequence(), random_b.to_sequence());
        }
    }

    #[test]
    fn to_sequence_test() {
        // bool
        to_sequence_prop(false, true);

        // u32
        to_sequence_prop(0u32, u32::MAX);

        // u64
        to_sequence_prop(0u64, u64::MAX);

        // BFieldElement
        let bfe_max = BFieldElement::new(BFieldElement::MAX);
        to_sequence_prop(BFIELD_ZERO, bfe_max);

        // XFieldElement
        let xfe_max = XFieldElement::new([bfe_max; EXTENSION_DEGREE]);
        to_sequence_prop(XFieldElement::zero(), xfe_max);

        // Digest
        let digest_zero = Digest::new([BFIELD_ZERO; DIGEST_LENGTH]);
        let digest_max = Digest::new([bfe_max; DIGEST_LENGTH]);
        to_sequence_prop(digest_zero, digest_max);

        // u128
        to_sequence_prop(0u128, u128::MAX);
    }

    fn seed_tip5(sponge: &mut Tip5State) {
        let mut rng = thread_rng();
        Tip5::absorb(
            sponge,
            &(0..RATE)
                .map(|_| BFieldElement::new(rng.next_u64()))
                .collect_vec()
                .try_into()
                .unwrap(),
        );
    }

    fn sample_indices_prop(max: u32, num_indices: usize) {
        let mut sponge = Tip5::init();
        seed_tip5(&mut sponge);
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
        let mut sponge = Tip5::init();
        seed_tip5(&mut sponge);
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
