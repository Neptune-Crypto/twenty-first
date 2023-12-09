use crate::shared_math::tip5;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, LinkedList, VecDeque};
use std::fmt::Debug;

/// Represents a database value as bytes and provides conversions for standard types
///
/// It is simple to extend RustyValue for use with any locally defined type
/// that implements `serde::Serialize` and `serde::Deserialize`.
///
/// ## Examples
///
/// ```
/// use serde::{Serialize, Deserialize};
/// use twenty_first::storage::storage_schema::RustyValue;
///
/// #[derive(Debug, Clone, Serialize, Deserialize)]
/// pub struct Person {
///     name: String,
///     age: u16,
/// }
///
/// impl From<RustyValue> for Person {
///    fn from(value: RustyValue) -> Self {
///        value.deserialize_from()
///    }
/// }
///
/// impl From<Person> for RustyValue {
///    fn from(value: Person) -> Self {
///        Self::serialize_into(&value)
///    }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustyValue(pub Vec<u8>);

impl RustyValue {
    /// serialize a value `T` that implements `serde::Serialize` into `RustyValue`
    ///
    /// This provides the serialization for `RustyValue` in the database
    /// (or anywhere else it is stored)
    ///
    /// At present `bincode` is used for de/serialization, but this could
    /// change in the future.  No guarantee is made as the to serialization format.
    #[inline]
    pub fn serialize_into<T: Serialize>(value: &T) -> Self {
        Self(serialize(value))
    }
    /// Deserialize `RustyValue` into a value `T` that implements `serde::de::DeserializeOwned`
    ///
    /// This provides the deserialization for `RustyValue` from the database
    /// (or anywhere else it is stored)
    ///
    /// At present `bincode` is used for de/serialization, but this could
    /// change in the future.  No guarantee is made as the to serialization format.
    #[inline]
    pub fn deserialize_from<T: DeserializeOwned>(&self) -> T {
        deserialize(&self.0)
    }
}

/// serialize a value T that implements serde::Serialize into bytes
///
/// At present `bincode` is used for de/serialization, but this could
/// change in the future.  No guarantee is made as the to serialization format.
#[inline]
pub fn serialize<T: Serialize>(value: &T) -> Vec<u8> {
    bincode::serialize(value).expect("should have serialized T into bytes")

    // for now, we use bincode.  but it would be so easy to switch to eg postcard or ron.
    // ron::to_string(value).unwrap().as_bytes().to_vec()
    // postcard::to_allocvec(value).unwrap()
}

/// serialize bytes into a value T that implements serde::de::SerializeOwned
///
/// At present `bincode` is used for de/serialization, but this could
/// change in the future.  No guarantee is made as the to serialization format.
#[inline]
pub fn deserialize<T: DeserializeOwned>(bytes: &[u8]) -> T {
    bincode::deserialize(bytes).expect("should have deserialized bytes")

    // for now, we use bincode.  but it would be so easy to switch to eg postcard or ron.
    // ron::from_str(String::from_utf8(bytes.to_vec()).unwrap().as_str()).unwrap()
    // postcard::from_bytes(bytes).unwrap()
}

impl<K, T> From<BTreeMap<K, T>> for RustyValue
where
    K: Serialize + std::cmp::Ord,
    T: Serialize,
{
    #[inline]
    fn from(value: BTreeMap<K, T>) -> Self {
        RustyValue(serialize(&value))
    }
}
impl<K, T> From<RustyValue> for BTreeMap<K, T>
where
    K: DeserializeOwned + std::cmp::Ord,
    T: DeserializeOwned,
{
    #[inline]
    fn from(value: RustyValue) -> Self {
        deserialize(&value.0)
    }
}

impl<T> From<BTreeSet<T>> for RustyValue
where
    T: Serialize + Ord,
{
    #[inline]
    fn from(value: BTreeSet<T>) -> Self {
        RustyValue(serialize(&value))
    }
}
impl<T> From<RustyValue> for BTreeSet<T>
where
    T: DeserializeOwned + Ord,
{
    #[inline]
    fn from(value: RustyValue) -> Self {
        deserialize(&value.0)
    }
}

impl<T> From<BinaryHeap<T>> for RustyValue
where
    T: Serialize + Ord,
{
    #[inline]
    fn from(value: BinaryHeap<T>) -> Self {
        RustyValue(serialize(&value))
    }
}
impl<T> From<RustyValue> for BinaryHeap<T>
where
    T: DeserializeOwned + Ord,
{
    #[inline]
    fn from(value: RustyValue) -> Self {
        deserialize(&value.0)
    }
}

impl<K, T> From<HashMap<K, T>> for RustyValue
where
    K: Serialize + std::cmp::Eq + std::hash::Hash,
    T: Serialize,
{
    #[inline]
    fn from(value: HashMap<K, T>) -> Self {
        RustyValue(serialize(&value))
    }
}
impl<K, T> From<RustyValue> for HashMap<K, T>
where
    K: DeserializeOwned + std::cmp::Eq + std::hash::Hash,
    T: DeserializeOwned,
{
    #[inline]
    fn from(value: RustyValue) -> Self {
        deserialize(&value.0)
    }
}

impl<T> From<HashSet<T>> for RustyValue
where
    T: Serialize + std::cmp::Eq + std::hash::Hash,
{
    #[inline]
    fn from(value: HashSet<T>) -> Self {
        RustyValue(serialize(&value))
    }
}
impl<T> From<RustyValue> for HashSet<T>
where
    T: DeserializeOwned + std::cmp::Eq + std::hash::Hash,
{
    #[inline]
    fn from(value: RustyValue) -> Self {
        deserialize(&value.0)
    }
}

macro_rules! impl_From_Generic_T_For_RV {
    ( $($type:ty),* $(,)? ) => {
        $(
            impl<T> From<$type> for RustyValue
            where
                T: Serialize,
            {
                #[inline]
                fn from(value: $type) -> Self {
                    RustyValue(serialize(&value))
                }
            }
        )*
    };
}

macro_rules! impl_From_RV_For_Generic_T {
    ( $($type:ty),* $(,)? ) => {
        $(
            impl<T> From<RustyValue> for $type
            where
                T: DeserializeOwned,
            {
                #[inline]
                fn from(value: RustyValue) -> Self {
                    deserialize(&value.0)
                }
            }

        )*
    };
}

impl_From_Generic_T_For_RV!(Vec<T>, VecDeque<T>, LinkedList<T>);
impl_From_RV_For_Generic_T!(Vec<T>, VecDeque<T>, LinkedList<T>);

macro_rules! impl_From_T_For_RV {
    ( $($type:ty),* $(,)? ) => {
        $(
            impl From<RustyValue> for $type {

                #[inline]
                fn from(value: RustyValue) -> Self {
                    deserialize(&value.0)
                }
            }
        )*
    };
}
macro_rules! impl_From_RV_For_T {
    ( $($type:ty),* $(,)? ) => {
        $(
            impl From<$type> for RustyValue {

                #[inline]
                fn from(value: $type) -> Self {
                    Self(serialize(&value))
                }
            }
        )*
    };
}

impl_From_T_For_RV!(u8, u16, u32, u64, u128, usize);
impl_From_T_For_RV!(i8, i16, i32, i64, i128, isize);
impl_From_T_For_RV!(bool);
impl_From_T_For_RV!(f32, f64);
impl_From_T_For_RV!(char, String);
impl_From_T_For_RV!(tip5::Digest);

impl_From_RV_For_T!(u8, u16, u32, u64, u128, usize);
impl_From_RV_For_T!(i8, i16, i32, i64, i128, isize);
impl_From_RV_For_T!(bool);
impl_From_RV_For_T!(f32, f64);
impl_From_RV_For_T!(char, String);
impl_From_RV_For_T!(tip5::Digest);
