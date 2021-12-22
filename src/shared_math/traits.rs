use serde::de::DeserializeOwned;
use serde::Serialize;
use std::convert::From;
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};

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
    fn get_cyclic_group(&self) -> Vec<Self>;
}

pub trait FieldBatchInversion
where
    Self: Sized,
{
    fn batch_inversion(elements: Vec<Self>) -> Vec<Self>;
}

pub trait GetRandomElements
where
    Self: Sized,
{
    fn random_elements(length: u32) -> Vec<Self>;
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
        + Sub<Output = Self::Elem>
        + Mul<Output = Self::Elem>
        + Div<Output = Self::Elem>
        + Neg<Output = Self::Elem>
        + From<Vec<u8>> // TODO: Replace with From<Blake3Hash>
        + New
        + CyclicGroupGenerator
        + FieldBatchInversion
        + ModPowU32;
}
