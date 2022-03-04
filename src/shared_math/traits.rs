use rand::prelude::ThreadRng;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

pub trait IdentityValues {
    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;

    #[must_use]
    fn ring_zero(&self) -> Self;

    #[must_use]
    fn ring_one(&self) -> Self;
}

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

pub trait GetPrimitiveRootOfUnity
where
    Self: Sized,
{
    fn get_primitive_root_of_unity(&self, n: u128) -> (Option<Self>, Vec<u128>);
}

// Used for testing.
pub trait GetRandomElements
where
    Self: Sized,
{
    fn random_elements(length: usize, rng: &mut ThreadRng) -> Vec<Self>;
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
    fn from_vecu8(&self, bytes: Vec<u8>) -> Self;
}

pub trait PrimeField:
    Clone
    + Eq
    + Display
    + Serialize
    + DeserializeOwned
    + PartialEq
    + Debug
    + IdentityValues
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
    + GetPrimitiveRootOfUnity
    + Send
    + Sync
    + Copy
    + Hash
    + Inverse
{
    // Adapted from https://paulmillr.com/posts/noble-secp256k1-fast-ecc/#batch-inversion
    fn batch_inversion(input: Vec<Self>) -> Vec<Self> {
        let input_length = input.len();
        if input_length == 0 {
            return Vec::<Self>::new();
        }

        let zero = input[0].ring_zero();
        let one = input[0].ring_one();
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
}
