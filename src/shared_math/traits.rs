use std::convert::{From, TryInto};
use std::fmt::{Debug, Display};
use std::hash::Hash;

pub trait IdentityValues {
    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
    fn ring_zero(&self) -> Self;
    fn ring_one(&self) -> Self;
}

pub trait FieldElement:
    num_traits::Num
    + Clone
    + Hash
    + Debug
    + Display
    + PartialEq
    + Eq
    + PartialOrd
    + Ord
    + From<i128>
    + TryInto<i128>
{
    fn is_power_of_2(&self) -> bool;
}
impl<
        T: num_traits::Num
            + Clone
            + Copy
            + Hash
            + Debug
            + Display
            + PartialEq
            + Eq
            + PartialOrd
            + Ord
            + From<i128>
            + TryInto<i128>,
    > FieldElement for T
{
}
