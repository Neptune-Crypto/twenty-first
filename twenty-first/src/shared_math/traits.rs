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

use num_traits::One;
use num_traits::Pow;
use num_traits::Zero;
use serde::de::DeserializeOwned;
use serde::Serialize;

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
    + Zero
    + One
    + Add<Output = Self>
    + Mul<Output = Self>
    + Sub<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + MulAssign
    + SubAssign
    + FromVecu8
    + New
    + CyclicGroupGenerator
    + Pow<u8, Output = Self>
    + Pow<u32, Output = Self>
    + PrimitiveRootOfUnity
    + Send
    + Sync
    + Copy
    + Hash
    + Inverse
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
