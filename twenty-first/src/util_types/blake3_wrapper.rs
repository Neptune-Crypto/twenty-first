use std::ops::Deref;

use rand::Rng;
use serde::ser::SerializeTuple;
use serde::{Deserialize, Serialize};

use crate::shared_math::traits::GetRandomElements;

/// A wrapper around `blake3::Hash` because it does not Serialize.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Blake3Hash(pub blake3::Hash);

impl From<blake3::Hash> for Blake3Hash {
    fn from(digest: blake3::Hash) -> Self {
        Blake3Hash(digest)
    }
}

impl From<[u8; 32]> for Blake3Hash {
    fn from(bytes: [u8; 32]) -> Self {
        Blake3Hash(bytes.into())
    }
}

impl From<u128> for Blake3Hash {
    fn from(n: u128) -> Self {
        Blake3Hash(blake3::Hash::from_hex(format!("{:064x}", n)).unwrap())
    }
}

/// Automatically dereference our wapper type into the underlying hash.
/// This allows us to pass a `&blake3_wrapper::Blake3Hash` everywhere we need a `&blake3::Hash`.
/// [Deref]: https://doc.rust-lang.org/book/ch15-02-deref.html#implicit-deref-coercions-with-functions-and-methods
impl Deref for Blake3Hash {
    type Target = blake3::Hash;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Serialize for Blake3Hash {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_tuple(32)?;
        let bytes: &[u8; 32] = self.0.as_bytes();
        for byte in bytes {
            seq.serialize_element(byte)?;
        }
        seq.end()
    }
}

impl<'de> Deserialize<'de> for Blake3Hash {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes: [u8; 32] = Deserialize::deserialize(deserializer)?;
        let digest: blake3::Hash = bytes.into();
        Ok(Blake3Hash(digest))
    }
}

pub fn hash(bytes: &[u8]) -> Blake3Hash {
    Blake3Hash(blake3::hash(bytes))
}

impl GetRandomElements for Blake3Hash {
    fn random_elements<R: Rng>(length: usize, prng: &mut R) -> Vec<Self> {
        let mut values: Vec<Blake3Hash> = Vec::with_capacity(length);

        while values.len() < length {
            let mut random_256bits = [0u8; 32];
            prng.fill_bytes(&mut random_256bits);

            values.push(Blake3Hash::from(random_256bits));
        }

        values
    }
}

#[cfg(test)]
mod blake3_wrapper_test {
    use super::*;

    #[test]
    fn serialize_deserialize_identity_test() {
        let before: Blake3Hash = blake3::hash(b"hello").into();

        let ser = bincode::serialize(&before);
        assert!(ser.is_ok());

        let de = bincode::deserialize(&ser.unwrap());
        assert!(de.is_ok());

        let after = de.unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn blake3_wrapper_uses_32_bytes_test() {
        let zero: Blake3Hash = [0; 32].into();

        let res_bytes = bincode::serialize(&zero);
        let bytes = res_bytes.unwrap();

        assert_eq!(std::mem::size_of::<Blake3Hash>(), 32);
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn generate_random_digests() {
        let n = 17;
        let mut rng = rand::thread_rng();
        let digests = Blake3Hash::random_elements(n, &mut rng);

        assert_eq!(n, digests.len())
    }

    #[test]
    fn deref_coercion() {
        let boxed: Blake3Hash = blake3::hash(b"hello").into();

        let exclicitly_derefed = &boxed.0;
        // Coerce a reference to our wrapper to be a reference to the wrapped type.
        let deref_coerced: &blake3::Hash = &boxed;

        assert_eq!(exclicitly_derefed, deref_coerced);
    }
}
