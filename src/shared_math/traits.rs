use rand::prelude::ThreadRng;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

pub trait IdentityValues {
    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
    fn ring_zero(&self) -> Self;
    fn ring_one(&self) -> Self;
}

pub trait CyclicGroupGenerator
where
    Self: Sized,
{
    fn get_cyclic_group_elements(&self, max: Option<usize>) -> Vec<Self>;
}

pub trait FieldBatchInversion
where
    Self: Sized,
{
    fn batch_inversion(elements: Vec<Self>) -> Vec<Self>;
}

pub trait GetPrimitiveRootOfUnity
where
    Self: Sized,
{
    fn get_primitive_root_of_unity(&self, n: u128) -> (Option<Self>, Vec<u128>);
}

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
    fn mod_pow_u64(&self, pow: u64) -> Self;
}

pub trait ModPowU32 {
    fn mod_pow_u32(&self, exp: u32) -> Self;
}

// We **cannot** use the std library From/Into traits as they cannot
// capture which field the new element is a member of.
pub trait New {
    fn new_from_usize(&self, value: usize) -> Self;
}

pub trait FromVecu8 {
    fn from_vecu8(&self, bytes: Vec<u8>) -> Self;
}

pub trait PrimeFieldElement {
    type Elem: Clone
        + Eq
        + Display
        + Serialize
        + DeserializeOwned
        + PartialEq
        + Debug
        + IdentityValues
        + Add<Output = Self::Elem>
        + AddAssign
        + SubAssign
        + MulAssign
        + Sub<Output = Self::Elem>
        + Mul<Output = Self::Elem>
        + Div<Output = Self::Elem>
        + Neg<Output = Self::Elem>
        + FromVecu8 // TODO: Replace with From<Blake3Hash>
        + New
        + CyclicGroupGenerator
        + FieldBatchInversion
        + ModPowU32
        + GetPrimitiveRootOfUnity
        + Send
        + Sync;
}
