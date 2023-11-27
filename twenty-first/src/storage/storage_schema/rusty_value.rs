use std::fmt::Debug;

use crate::shared_math::b_field_element::BFieldElement;

/// Represents a database value as bytes and provides some conversions.
#[derive(Debug)]
pub struct RustyValue(pub Vec<u8>);

impl From<Vec<u8>> for RustyValue {
    #[inline]
    fn from(value: Vec<u8>) -> Self {
        RustyValue(value)
    }
}
impl From<RustyValue> for Vec<u8> {
    #[inline]
    fn from(value: RustyValue) -> Self {
        value.0
    }
}
impl From<RustyValue> for u64 {
    #[inline]
    fn from(value: RustyValue) -> Self {
        u64::from_be_bytes(
            value
                .0
                .try_into()
                .expect("should have deserialized bytes into u64"),
        )
    }
}
impl From<u64> for RustyValue {
    #[inline]
    fn from(value: u64) -> Self {
        RustyValue(value.to_be_bytes().to_vec())
    }
}
impl From<RustyValue> for crate::shared_math::tip5::Digest {
    #[inline]
    fn from(value: RustyValue) -> Self {
        crate::shared_math::tip5::Digest::new(
            value
                .0
                .chunks(8)
                .map(|ch| {
                    u64::from_be_bytes(ch.try_into().expect("Cannot cast RustyValue into Digest"))
                })
                .map(BFieldElement::new)
                .collect::<Vec<_>>()
                .try_into().expect("Can cast RustyValue into BFieldElements but number does not match that of Digest."),
        )
    }
}
impl From<crate::shared_math::tip5::Digest> for RustyValue {
    #[inline]
    fn from(value: crate::shared_math::tip5::Digest) -> Self {
        RustyValue(
            value
                .values()
                .map(|b| b.value())
                .map(u64::to_be_bytes)
                .concat(),
        )
    }
}
