use num_traits::{One, Zero};
use rand::Rng;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

pub trait CyclicGroupGenerator
where
    Self: Sized,
{
    fn get_cyclic_group_elements(&self, max: Option<usize>) -> Vec<Self>;
}

// TODO: Assert if we're risking inverting 0 at any point.
pub trait Inverse
where
    Self: Sized,
{
    fn inverse(&self) -> Self;
}

pub trait PrimitiveRootOfUnity
where
    Self: Sized,
{
    fn primitive_root_of_unity(n: u64) -> Option<Self>;
}

// Used for testing.
pub trait GetRandomElements
where
    Self: Sized,
{
    fn random_elements<R: Rng>(length: usize, rng: &mut R) -> Vec<Self>;
}

// TODO: Remove in favor of CyclicGroupGenerator
pub trait GetGeneratorDomain
where
    Self: Sized,
{
    fn get_generator_domain(&self) -> Vec<Self>;
}

pub trait ModPowU64 {
    #[must_use]
    fn mod_pow_u64(&self, pow: u64) -> Self;
}

pub trait ModPowU32 {
    #[must_use]
    fn mod_pow_u32(&self, exp: u32) -> Self;
}

// We **cannot** use the std library From/Into traits as they cannot
// capture which field the new element is a member of.
pub trait New {
    #[must_use]
    fn new_from_usize(&self, value: usize) -> Self;
}

pub trait FromVecu8 {
    #[must_use]
    #[allow(clippy::wrong_self_convention)]
    fn from_vecu8(bytes: Vec<u8>) -> Self;
}

pub trait FiniteField:
    Clone
    + Eq
    + Display
    + Serialize
    + DeserializeOwned
    + PartialEq
    + Debug
    + One
    + Zero
    + Add<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + FromVecu8
    + New
    + CyclicGroupGenerator
    + ModPowU32
    + PrimitiveRootOfUnity
    + Send
    + Sync
    + Copy
    + Hash
    + Inverse
    + Default
{
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

    #[inline(always)]
    fn square(self) -> Self {
        self * self
    }
}
