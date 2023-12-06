use std::fmt::Debug;

use crate::shared_math::b_field_element::BFieldElement;

/// Represents a database value as bytes and provides some conversions.
#[derive(Debug, Clone)]
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

impl From<String> for RustyValue {
    #[inline]
    fn from(value: String) -> Self {
        RustyValue(value.as_bytes().to_vec())
    }
}
impl From<RustyValue> for String {
    #[inline]
    fn from(value: RustyValue) -> String {
        String::from_utf8(value.0).expect("should have deserialized utf8 bytes into String")
    }
}

impl From<RustyValue> for u128 {
    #[inline]
    fn from(value: RustyValue) -> Self {
        u128::from_be_bytes(
            value
                .0
                .try_into()
                .expect("should have deserialized bytes into u128"),
        )
    }
}
impl From<u128> for RustyValue {
    #[inline]
    fn from(value: u128) -> Self {
        RustyValue(value.to_be_bytes().to_vec())
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

impl From<RustyValue> for u32 {
    #[inline]
    fn from(value: RustyValue) -> Self {
        u32::from_be_bytes(
            value
                .0
                .try_into()
                .expect("should have deserialized bytes into u32"),
        )
    }
}
impl From<u32> for RustyValue {
    #[inline]
    fn from(value: u32) -> Self {
        RustyValue(value.to_be_bytes().to_vec())
    }
}

impl From<RustyValue> for u16 {
    #[inline]
    fn from(value: RustyValue) -> Self {
        u16::from_be_bytes(
            value
                .0
                .try_into()
                .expect("should have deserialized bytes into u16"),
        )
    }
}
impl From<u16> for RustyValue {
    #[inline]
    fn from(value: u16) -> Self {
        RustyValue(value.to_be_bytes().to_vec())
    }
}

impl From<RustyValue> for u8 {
    #[inline]
    fn from(value: RustyValue) -> Self {
        u8::from_be_bytes(
            value
                .0
                .try_into()
                .expect("should have deserialized bytes into u8"),
        )
    }
}
impl From<u8> for RustyValue {
    #[inline]
    fn from(value: u8) -> Self {
        RustyValue(value.to_be_bytes().to_vec())
    }
}

impl From<RustyValue> for bool {
    #[inline]
    fn from(value: RustyValue) -> Self {
        bincode::deserialize(&value.0).expect("should have deserialized bytes into bool")
    }
}
impl From<bool> for RustyValue {
    #[inline]
    fn from(value: bool) -> Self {
        RustyValue(bincode::serialize(&value).expect("should have serialized bool into bytes"))
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
