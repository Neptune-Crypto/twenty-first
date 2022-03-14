use serde::Deserialize;

/// A wrapper around `blake3::Hash` because it does not Serialize.
#[derive(Debug, Copy, Clone, PartialEq)]
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

impl serde::Serialize for Blake3Hash {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let bytes = self.0.as_bytes();
        serializer.serialize_bytes(bytes)
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
