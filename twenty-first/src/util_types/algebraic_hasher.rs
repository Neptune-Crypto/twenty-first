use std::fmt::Debug;
use std::iter;

use itertools::Itertools;
use num_traits::ConstOne;
use num_traits::ConstZero;

use crate::math::b_field_element::BFieldElement;
use crate::math::bfield_codec::BFieldCodec;
use crate::math::digest::Digest;
use crate::math::x_field_element::XFieldElement;
use crate::math::x_field_element::EXTENSION_DEGREE;

pub const RATE: usize = 10;

/// The hasher [Domain] differentiates between the modes of hashing.
///
/// The main purpose of declaring the domain is to prevent collisions between different types of
/// hashing by introducing defining differences in the way the hash function's internal state
/// (e.g. a sponge state's capacity) is initialized.
#[derive(Debug, PartialEq, Eq)]
pub enum Domain {
    /// The `VariableLength` domain is used for hashing objects that potentially serialize to more
    /// than [`RATE`] number of field elements.
    VariableLength,

    /// The `FixedLength` domain is used for hashing objects that always fit within [RATE] number
    /// of fields elements, e.g. a pair of [Digest].
    FixedLength,
}

/// A [cryptographic sponge][sponge]. Should only be based on a cryptographic permutation, e.g.,
/// [`Tip5`][tip5].
///
/// [sponge]: https://keccak.team/files/CSF-0.1.pdf
/// [tip5]: crate::prelude::Tip5
pub trait Sponge: Clone + Debug + Default + Send + Sync {
    const RATE: usize;

    fn init() -> Self;

    fn absorb(&mut self, input: [BFieldElement; RATE]);

    fn squeeze(&mut self) -> [BFieldElement; RATE];

    fn pad_and_absorb_all(&mut self, input: &[BFieldElement]) {
        // pad input with [1, 0, 0, …] – padding is at least one element
        let padded_length = (input.len() + 1).next_multiple_of(RATE);
        let padding_iter =
            iter::once(&BFieldElement::ONE).chain(iter::repeat(&BFieldElement::ZERO));
        let padded_input = input.iter().chain(padding_iter).take(padded_length);

        for chunk in padded_input.chunks(RATE).into_iter() {
            // the padded input has length some multiple of `RATE`
            let absorb_elems = chunk.cloned().collect_vec().try_into().unwrap();
            self.absorb(absorb_elems);
        }
    }
}

pub trait AlgebraicHasher: Sponge {
    /// 2-to-1 hashing
    fn hash_pair(left: Digest, right: Digest) -> Digest;

    /// Thin wrapper around [`hash_varlen`](Self::hash_varlen).
    fn hash<T: BFieldCodec>(value: &T) -> Digest {
        Self::hash_varlen(&value.encode())
    }

    /// Hash a variable-length sequence of [`BFieldElement`].
    ///
    /// - Apply the correct padding
    /// - [Sponge::pad_and_absorb_all()]
    /// - [Sponge::squeeze()] once.
    fn hash_varlen(input: &[BFieldElement]) -> Digest {
        let mut sponge = Self::init();
        sponge.pad_and_absorb_all(input);
        let produce: [BFieldElement; RATE] = sponge.squeeze();

        Digest::new((&produce[..Digest::LEN]).try_into().unwrap())
    }

    /// Produce `num_indices` random integer values in the range `[0, upper_bound)`. The
    /// `upper_bound` must be a power of 2.
    ///
    /// This method uses von Neumann rejection sampling.
    /// Specifically, if the top 32 bits of a BFieldElement are all ones, then the bottom 32 bits
    /// are not uniformly distributed, and so they are dropped. This method invokes squeeze until
    /// enough uniform u32s have been sampled.
    fn sample_indices(&mut self, upper_bound: u32, num_indices: usize) -> Vec<u32> {
        debug_assert!(upper_bound.is_power_of_two());
        let mut indices = vec![];
        let mut squeezed_elements = vec![];
        while indices.len() != num_indices {
            if squeezed_elements.is_empty() {
                squeezed_elements = self.squeeze().into_iter().rev().collect_vec();
            }
            let element = squeezed_elements.pop().unwrap();
            if element != BFieldElement::new(BFieldElement::MAX) {
                indices.push(element.value() as u32 % upper_bound);
            }
        }
        indices
    }

    /// Produce `num_elements` random [`XFieldElement`] values.
    ///
    /// If `num_elements` is not divisible by [`RATE`][rate], spill the remaining elements of the
    /// last [`squeeze`][Sponge::squeeze].
    ///
    /// [rate]: Sponge::RATE
    fn sample_scalars(&mut self, num_elements: usize) -> Vec<XFieldElement> {
        let num_squeezes = (num_elements * EXTENSION_DEGREE + Self::RATE - 1) / Self::RATE;
        debug_assert!(
            num_elements * EXTENSION_DEGREE <= num_squeezes * Self::RATE,
            "need {} elements but getting {}",
            num_elements * EXTENSION_DEGREE,
            num_squeezes * Self::RATE
        );
        (0..num_squeezes)
            .flat_map(|_| self.squeeze())
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

    use rand::Rng;
    use rand_distr::Distribution;
    use rand_distr::Standard;

    use super::*;
    use crate::math::digest::Digest;
    use crate::math::tip5::Tip5;
    use crate::math::x_field_element::EXTENSION_DEGREE;

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
        encode_prop(BFieldElement::ZERO, bfe_max);

        // XFieldElement
        let xfe_max = XFieldElement::new([bfe_max; EXTENSION_DEGREE]);
        encode_prop(XFieldElement::ZERO, xfe_max);

        // Digest
        let digest_max = Digest::new([bfe_max; Digest::LEN]);
        encode_prop(Digest::ALL_ZERO, digest_max);

        // u128
        encode_prop(0u128, u128::MAX);
    }

    fn sample_indices_prop(max: u32, num_indices: usize) {
        let mut sponge = Tip5::randomly_seeded();
        let indices = sponge.sample_indices(max, num_indices);
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
        let mut product = XFieldElement::ONE;
        for amount in amounts {
            let scalars = sponge.sample_scalars(amount);
            assert_eq!(amount, scalars.len());
            product *= scalars
                .into_iter()
                .fold(XFieldElement::ONE, XFieldElement::mul);
        }
        assert_ne!(product, XFieldElement::ZERO); // false failure with prob ~2^{-192}
    }
}
