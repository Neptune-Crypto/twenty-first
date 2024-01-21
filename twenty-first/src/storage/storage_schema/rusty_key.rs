use crate::leveldb::database::key::IntoLevelDBKey;
use crate::leveldb::error::Error;
use serde::{de::DeserializeOwned, Serialize};

// Todo: consider making RustyKey a newtype for RustyValue and auto derive all its From impls
//       using either `derive_more` or `newtype_derive` crate

/// Represents a database key as bytes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RustyKey(pub Vec<u8>);

impl From<&[u8]> for RustyKey {
    #[inline]
    fn from(value: &[u8]) -> Self {
        Self(value.to_vec())
    }
}

impl From<u8> for RustyKey {
    #[inline]
    fn from(value: u8) -> Self {
        Self([value].to_vec())
    }
}
impl From<(RustyKey, RustyKey)> for RustyKey {
    #[inline]
    fn from(value: (RustyKey, RustyKey)) -> Self {
        let v0 = value.0 .0;
        let v1 = value.1 .0;
        RustyKey([v0, v1].concat())
    }
}
impl From<u64> for RustyKey {
    #[inline]
    fn from(value: u64) -> Self {
        RustyKey(value.to_be_bytes().to_vec())
    }
}

impl IntoLevelDBKey for RustyKey {
    fn as_u8_slice_for_write(&self, f: &dyn Fn(&[u8]) -> Result<(), Error>) -> Result<(), Error> {
        f(&self.0)
    }

    fn as_u8_slice_for_get(
        &self,
        f: &dyn Fn(&[u8]) -> Result<Option<Vec<u8>>, Error>,
    ) -> Result<Option<Vec<u8>>, Error> {
        f(&self.0)
    }
}

impl From<&dyn IntoLevelDBKey> for RustyKey {
    #[inline]
    fn from(value: &dyn IntoLevelDBKey) -> Self {
        let vec_u8 = value
            .as_u8_slice_for_get(&|k| Ok(Some(k.to_vec())))
            .unwrap()
            .unwrap();
        Self(vec_u8)
    }
}

impl RustyKey {
    /// serialize a value `T` that implements `serde::Serialize` into `RustyValue`
    ///
    /// This provides the serialization for `RustyValue` in the database
    /// (or anywhere else it is stored)
    ///
    /// At present `bincode` is used for de/serialization, but this could
    /// change in the future.  No guarantee is made as the to serialization format.
    #[inline]
    pub fn from_any<T: Serialize>(value: &T) -> Self {
        Self(super::rusty_value::serialize(value))
    }
    /// Deserialize `RustyValue` into a value `T` that implements `serde::de::DeserializeOwned`
    ///
    /// This provides the deserialization for `RustyValue` from the database
    /// (or anywhere else it is stored)
    ///
    /// At present `bincode` is used for de/serialization, but this could
    /// change in the future.  No guarantee is made as the to serialization format.
    #[inline]
    pub fn into_any<T: DeserializeOwned>(&self) -> T {
        super::rusty_value::deserialize(&self.0)
    }
}
