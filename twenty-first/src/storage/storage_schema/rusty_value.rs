// RustyValue is an interface for serializing data as bytes for storing
// in the database.  See type description below.
//
// RustyValue now makes use of `bincode` for serializing data.
// There is a (temporary) caveat: see bottom of file.
// Any rust `serializer` crate that understands serde
// Serialize and Deserialize traits could be plugged in instead.
//
// Here is a good comparison of rust serializers and their performance/size.
//  https://github.com/djkoloski/rust_serialization_benchmark
// See also this blog:
//  https://david.kolo.ski/blog/rkyv-is-faster-than/
//
// An important consideration is that many of these crates are not
// compatible with `serde`.   of those that are, `bincode` and `postcard`
// appear to be the fastest options, and are pretty close.
// `postcard` seems to create smaller serialized data size though,
// so that could be a good reason to use it for a large DB like
// a blockchain.  `ron` appears too slow to be a good option.
//
// With regards to blockchain size, we should also consider
// compressing our serialized data with something like zlib.
// In the above benchmarks, all libraries are compared with
// both uncompressed size and zlib compressed size.
//
// The fn that does the compression in the benchmark is:
//  https://github.com/djkoloski/rust_serialization_benchmark/blob/e1f6a31a431d5e8c3889525696231acbb691cbd9/src/lib.rs#L169
//
// For RustyValue, we would do the compression/decompression in the
// `serialize` and `deserialize` functions.
//
// Before making final decision(s), we should benchmark our
// code with real data and try out different crates/options.
//
// todo: consider moving the serialization functions, or perhaps all
// of rusty_value.rs to a top level module, eg twenty-first::serialization.

use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::tip5;
use itertools::Itertools;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::any::{Any, TypeId};
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
///
// todo: consider compressing serialized bytes with zlib, or similar.
//       see comments at top of file.
// todo: consider moving this fn, or perhaps all of rusty_value.rs
//       to a top level module, eg twenty-first::serialization
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
///
// todo: consider decompressing serialized bytes with zlib, or similar.
//       see comments at top of file.
// todo: consider moving this fn, or perhaps all of rusty_value.rs
//       to a top level module, eg twenty-first::serialization
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

macro_rules! impl_From_Generic {
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

// impl_From_Generic!(Vec<T>, VecDeque<T>, LinkedList<T>);  // Vec<T> impl'd below, for now.
impl_From_Generic!(VecDeque<T>, LinkedList<T>);

macro_rules! impl_From {
    ( $($type:ty),* $(,)? ) => {
        $(
            impl From<RustyValue> for $type {

                #[inline]
                fn from(value: RustyValue) -> Self {
                    deserialize(&value.0)
                }
            }

            impl From<$type> for RustyValue {

                #[inline]
                fn from(value: $type) -> Self {
                    Self(serialize(&value))
                }
            }
        )*
    };
}

impl_From!(u8, u16, u32, u128, usize); // note: u64 is impl'd below (for now)
impl_From!(i8, i16, i32, i64, i128, isize);
impl_From!(bool);
impl_From!(f32, f64);
impl_From!(char, String);
// impl_From!(tip5::Digest);  // note: impl'd below (for now)

// ****** From impls without De/Serialize ******************
//
// These are the original RustyValue impls.
//
// All of these *can* be auto-impl'd above and code below disappears.
// However that breaks compatibility with existing databases.
//
// In particular, without the code below, the neptune-core test
// `can_initialize_mutator_set_database` would fail for a blockchain
// synced before the serialization code above was in place.
// That's because that test opens the user's "real" database and reads
// in values, rather than generating and using a test DB, as other
// tests do.
//
// For now we leave these in place until a decision is made if
// we want to use a single serialization for all data
// types.

// Vec<T> is special because we want the auto serialization
// for all types of T except Vec<u32>
//
// This is because neptune-core had a Vec<u32> impl for
// RustyMSValue, but that type has gone away with the removal
// of the ParentKey, ParentValue generics.  Since `From` and
// `RustyValue` and `Vec` are all foreign to neptune-core,
// it must be impl'd here, to get the same behavior.
//
// So we use some trickery from std::any::Any to detect and special
// case this type.
//
// Todo: Vec<T> can/should be auto-impld above if we decide we do not
// need to continue the original (manual) bytes serialization.
//
// Todo: If we want to keep the original (manual) bytes serialization,
// it would be much better if we can use `downcast()` here instead of
// `downcast_ref()` and avoid allocating with `to_vec()`.
// When I tried using `downcast()`, I was initially unable to get
// around some compiler errors, but I believe it should be possible.
impl<T> From<Vec<T>> for RustyValue
where
    T: Serialize + 'static,
{
    #[inline]
    fn from(value: Vec<T>) -> Self {
        if let Some(value_u32) = (&value as &dyn Any).downcast_ref::<Vec<u32>>() {
            Self(
                value_u32
                    .iter()
                    .map(|&i| i.to_be_bytes())
                    .collect_vec()
                    .concat(),
            )
        } else {
            RustyValue(serialize(&value))
        }
    }
}
impl<T> From<RustyValue> for Vec<T>
where
    T: DeserializeOwned + 'static + Clone,
{
    #[inline]
    fn from(value: RustyValue) -> Self {
        if TypeId::of::<Self>() == TypeId::of::<Vec<u32>>() {
            let vec_u32 = value
                .0
                .chunks(4)
                .map(|ch| {
                    u32::from_be_bytes(
                        ch.try_into()
                            .expect("Cannot unpack RustyMSValue as Vec<u32>s"),
                    )
                })
                .collect_vec();
            (&vec_u32 as &dyn Any)
                .downcast_ref::<Vec<T>>()
                .unwrap()
                .to_vec()
        } else {
            deserialize(&value.0)
        }
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

impl From<RustyValue> for tip5::Digest {
    #[inline]
    fn from(value: RustyValue) -> Self {
        tip5::Digest::new(
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
impl From<tip5::Digest> for RustyValue {
    #[inline]
    fn from(value: tip5::Digest) -> Self {
        RustyValue(
            value
                .values()
                .map(|b| b.value())
                .map(u64::to_be_bytes)
                .concat(),
        )
    }
}
