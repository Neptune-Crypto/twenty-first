use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Neg;
use std::ops::Sub;
use std::ops::SubAssign;

use num_traits::ConstOne;
use num_traits::ConstZero;
use num_traits::Zero;
use rayon::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;

pub trait CyclicGroupGenerator
where
    Self: Sized,
{
    fn get_cyclic_group_elements(&self, max: Option<usize>) -> Vec<Self>;
}

// TODO: Assert if we're risking inverting 0 at any point.
pub trait Inverse
where
    Self: Sized + Zero,
{
    /// The multiplicative inverse: `a * a.inverse() == 1`
    ///
    /// # Panics
    ///
    /// Panics if `self` does not have a multiplicative inverse, for example, when
    /// `self` is zero. (For fields, this is the only case.)
    fn inverse(&self) -> Self;

    fn inverse_or_zero(&self) -> Self {
        if self.is_zero() {
            Self::zero()
        } else {
            self.inverse()
        }
    }
}

pub trait PrimitiveRootOfUnity
where
    Self: Sized,
{
    fn primitive_root_of_unity(n: u64) -> Option<Self>;
}

pub trait ModPowU64 {
    #[must_use]
    fn mod_pow_u64(&self, pow: u64) -> Self;
}

pub trait ModPowU32 {
    #[must_use]
    fn mod_pow_u32(&self, exp: u32) -> Self;
}

pub trait FiniteField:
    Copy
    + Debug
    + Display
    + Eq
    + Serialize
    + DeserializeOwned
    + Hash
    + ConstZero
    + ConstOne
    + Add<Output = Self>
    + Mul<Output = Self>
    + Sub<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + MulAssign
    + SubAssign
    + CyclicGroupGenerator
    + PrimitiveRootOfUnity
    + Inverse
    + ModPowU32
    + From<u64>
    + Send
    + Sync
{
    /// If values of this type are laid out in memory as a run of
    /// [`BFieldElement`]s, the number of base field elements in that run.
    ///
    /// [`BFieldElement`]: crate::math::b_field_element::BFieldElement
    /// Types with any other layout must leave this at the default `None`.
    ///
    /// SIMD kernels, like the NTT's, use this to process the base field limbs
    /// of many elements at once. Setting it for a type whose layout does not
    /// match the description is unsound.
    const NUM_BFE_LIMBS: Option<usize> = None;

    /// Montgomery Batch Inversion
    // Adapted from https://paulmillr.com/posts/noble-secp256k1-fast-ecc/#batch-inversion
    fn batch_inversion(input: Vec<Self>) -> Vec<Self> {
        let input_length = input.len();
        if input_length == 0 {
            return Vec::<Self>::new();
        }

        let zero = Self::zero();
        let one = Self::one();
        let mut scratch: Vec<Self> = vec![zero; input_length];
        let mut acc = one;
        scratch[0] = input[0];

        for i in 0..input_length {
            assert!(!input[i].is_zero(), "Cannot do batch inversion on zero");
            scratch[i] = acc;
            acc *= input[i];
        }

        acc = acc.inverse();

        let mut res = input;
        for i in (0..input_length).rev() {
            let tmp = acc * res[i];
            res[i] = acc * scratch[i];
            acc = tmp;
        }

        res
    }

    /// Parallel version of [`batch_inversion`](Self::batch_inversion).
    ///
    /// # Panics
    ///
    /// Panics if any of the elements is zero.
    fn par_batch_inversion(mut input: Vec<Self>) -> Vec<Self> {
        // Large enough to amortize the one inversion per chunk, small enough
        // to keep all threads busy.
        const CHUNK_SIZE: usize = 1 << 12;

        input.par_chunks_mut(CHUNK_SIZE).for_each(|chunk| {
            let inverses = Self::batch_inversion(chunk.to_vec());
            chunk.copy_from_slice(&inverses);
        });
        input
    }

    #[inline(always)]
    fn square(self) -> Self {
        self * self
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::math::other::random_elements;
    use crate::prelude::*;
    use crate::tests::test;

    #[macro_rules_attr::apply(test)]
    fn parallel_and_serial_batch_inversion_agree() {
        // lengths around the chunk boundaries of `par_batch_inversion`
        for len in [
            0,
            1,
            7,
            (1 << 12) - 1,
            1 << 12,
            (1 << 12) + 1,
            (1 << 14) + 3,
        ] {
            let bfes: Vec<BFieldElement> = random_elements(len);
            let bfes: Vec<_> = bfes.into_iter().filter(|x| !x.is_zero()).collect();
            let serial = BFieldElement::batch_inversion(bfes.clone());
            let parallel = BFieldElement::par_batch_inversion(bfes.clone());
            assert_eq!(serial, parallel, "{len}");
            for (x, x_inv) in bfes.into_iter().zip(parallel) {
                assert_eq!(BFieldElement::ONE, x * x_inv);
            }

            let xfes: Vec<XFieldElement> = random_elements(len);
            let xfes: Vec<_> = xfes.into_iter().filter(|x| !x.is_zero()).collect();
            let serial_xfes = XFieldElement::batch_inversion(xfes.clone());
            let parallel_xfes = XFieldElement::par_batch_inversion(xfes);
            assert_eq!(serial_xfes, parallel_xfes, "{len}");
        }
    }
}
