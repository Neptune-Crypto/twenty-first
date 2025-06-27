use core::fmt;
use std::str::FromStr;

use arbitrary::Arbitrary;
use bfieldcodec_derive::BFieldCodec;
use get_size2::GetSize;
use itertools::Itertools;
use num_bigint::BigUint;
use num_traits::ConstZero;
use num_traits::Zero;
use rand::Rng;
use rand::distr::Distribution;
use rand::distr::StandardUniform;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde::Serializer;

use crate::error::TryFromDigestError;
use crate::error::TryFromHexDigestError;
use crate::math::b_field_element::BFieldElement;
use crate::prelude::Tip5;

/// The result of hashing a sequence of elements, for example using [Tip5].
/// Sometimes called a “hash”.
// note: Serialize and Deserialize have custom implementations below
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, GetSize, BFieldCodec, Arbitrary)]
pub struct Digest(pub [BFieldElement; Digest::LEN]);

impl PartialOrd for Digest {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Digest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let Digest(self_inner) = self;
        let Digest(other_inner) = other;
        let self_as_u64s = self_inner.iter().rev().map(|bfe| bfe.value());
        let other_as_u64s = other_inner.iter().rev().map(|bfe| bfe.value());
        self_as_u64s.cmp(other_as_u64s)
    }
}

impl Digest {
    /// The number of [elements](BFieldElement) in a digest.
    pub const LEN: usize = 5;

    /// The number of bytes in a digest.
    pub const BYTES: usize = Self::LEN * BFieldElement::BYTES;

    /// The all-zero digest.
    pub(crate) const ALL_ZERO: Self = Self([BFieldElement::ZERO; Self::LEN]);

    pub const fn values(self) -> [BFieldElement; Self::LEN] {
        self.0
    }

    pub const fn new(digest: [BFieldElement; Self::LEN]) -> Self {
        Self(digest)
    }

    /// Returns a new digest but whose elements are reversed relative to self.
    /// This function is an involutive endomorphism.
    pub const fn reversed(self) -> Digest {
        let Digest([d0, d1, d2, d3, d4]) = self;
        Digest([d4, d3, d2, d1, d0])
    }
}

impl Default for Digest {
    fn default() -> Self {
        Self::ALL_ZERO
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.map(|elem| elem.to_string()).join(","))
    }
}

impl fmt::LowerHex for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = <[u8; Self::BYTES]>::from(*self);
        write!(f, "{}", hex::encode(bytes))
    }
}

impl fmt::UpperHex for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = <[u8; Self::BYTES]>::from(*self);
        write!(f, "{}", hex::encode_upper(bytes))
    }
}

impl Distribution<Digest> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Digest {
        Digest::new(rng.random())
    }
}

impl FromStr for Digest {
    type Err = TryFromDigestError;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        let bfes: Vec<_> = string
            .split(',')
            .map(str::parse::<BFieldElement>)
            .try_collect()?;
        let invalid_len_err = Self::Err::InvalidLength(bfes.len());
        let digest_innards = bfes.try_into().map_err(|_| invalid_len_err)?;

        Ok(Digest(digest_innards))
    }
}

impl TryFrom<&[BFieldElement]> for Digest {
    type Error = TryFromDigestError;

    fn try_from(value: &[BFieldElement]) -> Result<Self, Self::Error> {
        let len = value.len();
        let maybe_digest = value.try_into().map(Digest::new);
        maybe_digest.map_err(|_| Self::Error::InvalidLength(len))
    }
}

impl TryFrom<Vec<BFieldElement>> for Digest {
    type Error = TryFromDigestError;

    fn try_from(value: Vec<BFieldElement>) -> Result<Self, Self::Error> {
        Digest::try_from(&value as &[BFieldElement])
    }
}

impl From<Digest> for Vec<BFieldElement> {
    fn from(val: Digest) -> Self {
        val.0.to_vec()
    }
}

impl From<Digest> for [u8; Digest::BYTES] {
    fn from(Digest(innards): Digest) -> Self {
        innards
            .map(<[u8; BFieldElement::BYTES]>::from)
            .concat()
            .try_into()
            .unwrap()
    }
}

impl TryFrom<[u8; Digest::BYTES]> for Digest {
    type Error = TryFromDigestError;

    fn try_from(item: [u8; Self::BYTES]) -> Result<Self, Self::Error> {
        let digest_innards: Vec<_> = item
            .chunks_exact(BFieldElement::BYTES)
            .map(BFieldElement::try_from)
            .try_collect()?;

        Ok(Self(digest_innards.try_into().unwrap()))
    }
}

impl TryFrom<&[u8]> for Digest {
    type Error = TryFromDigestError;

    fn try_from(slice: &[u8]) -> Result<Self, Self::Error> {
        let array = <[u8; Self::BYTES]>::try_from(slice)
            .map_err(|_e| TryFromDigestError::InvalidLength(slice.len()))?;
        Self::try_from(array)
    }
}

impl TryFrom<BigUint> for Digest {
    type Error = TryFromDigestError;

    fn try_from(value: BigUint) -> Result<Self, Self::Error> {
        let mut remaining = value;
        let mut digest_innards = [BFieldElement::ZERO; Self::LEN];
        let modulus: BigUint = BFieldElement::P.into();
        for digest_element in digest_innards.iter_mut() {
            let element = u64::try_from(remaining.clone() % modulus.clone()).unwrap();
            *digest_element = BFieldElement::new(element);
            remaining /= modulus.clone();
        }

        if !remaining.is_zero() {
            return Err(Self::Error::Overflow);
        }

        Ok(Digest::new(digest_innards))
    }
}

impl From<Digest> for BigUint {
    fn from(digest: Digest) -> Self {
        let Digest(digest_innards) = digest;
        let mut ret = BigUint::zero();
        let modulus: BigUint = BFieldElement::P.into();
        for i in (0..Digest::LEN).rev() {
            ret *= modulus.clone();
            let digest_element: BigUint = digest_innards[i].value().into();
            ret += digest_element;
        }

        ret
    }
}

impl Digest {
    /// Hash this digest using [Tip5], producing a new digest.
    ///
    /// A digest can be used as a source of entropy. It can be beneficial or even
    /// necessary to not reveal the entropy itself, but use it as the seed for
    /// some deterministic computation. In such cases, hashing the digest using
    /// this method is probably the right thing to do.
    /// If the digest in question is used for its entropy only, there might not be a
    /// known meaningful pre-image for that digest.
    ///
    /// This method invokes [`Tip5::hash_pair`] with the right operand being the
    /// zero digest, agreeing with the standard way to hash a digest in the virtual
    /// machine.
    // todo: introduce a dedicated newtype for an entropy source
    pub fn hash(self) -> Digest {
        Tip5::hash_pair(self, Self::ALL_ZERO)
    }

    /// Encode digest as hex.
    ///
    /// Since `Digest` also implements [`LowerHex`][lo] and [`UpperHex`][up], it is
    /// possible to `{:x}`-format directly, _e.g._, `print!("{digest:x}")`.
    ///
    /// [lo]: fmt::LowerHex
    /// [up]: fmt::UpperHex
    pub fn to_hex(self) -> String {
        format!("{self:x}")
    }

    /// Decode hex string to [`Digest`]. Must not include leading “0x”.
    pub fn try_from_hex(data: impl AsRef<[u8]>) -> Result<Self, TryFromHexDigestError> {
        let slice = hex::decode(data)?;
        Ok(Self::try_from(&slice as &[u8])?)
    }
}

// we implement Serialize so that we can serialize as hex for human readable
// formats like JSON but use default serializer for other formats likes bincode
impl Serialize for Digest {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if serializer.is_human_readable() {
            self.to_hex().serialize(serializer)
        } else {
            self.0.serialize(serializer)
        }
    }
}

// we impl Deserialize so that we can deserialize as hex for human readable
// formats like JSON but use default deserializer for other formats like bincode
impl<'de> Deserialize<'de> for Digest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            let hex_string = String::deserialize(deserializer)?;
            Self::try_from_hex(hex_string).map_err(serde::de::Error::custom)
        } else {
            Ok(Self::new(<[BFieldElement; Self::LEN]>::deserialize(
                deserializer,
            )?))
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(crate) mod tests {
    use num_traits::One;
    use proptest::collection::vec;
    use proptest::prelude::Arbitrary as ProptestArbitrary;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use super::*;
    use crate::error::ParseBFieldElementError;
    use crate::prelude::*;

    impl ProptestArbitrary for Digest {
        type Parameters = ();
        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            arb().prop_map(|d| d).no_shrink().boxed()
        }

        type Strategy = BoxedStrategy<Self>;
    }

    /// Test helper struct for corrupting digests. Primarily used for negative tests.
    #[derive(Debug, Clone, PartialEq, Eq, test_strategy::Arbitrary)]
    pub(crate) struct DigestCorruptor {
        #[strategy(vec(0..Digest::LEN, 1..=Digest::LEN))]
        #[filter(#corrupt_indices.iter().all_unique())]
        corrupt_indices: Vec<usize>,

        #[strategy(vec(arb(), #corrupt_indices.len()))]
        corrupt_elements: Vec<BFieldElement>,
    }

    impl DigestCorruptor {
        pub fn corrupt_digest(&self, digest: Digest) -> Result<Digest, TestCaseError> {
            let mut corrupt_digest = digest;
            for (&i, &element) in self.corrupt_indices.iter().zip(&self.corrupt_elements) {
                corrupt_digest.0[i] = element;
            }
            if corrupt_digest == digest {
                let reject_reason = "corruption must change digest".into();
                return Err(TestCaseError::Reject(reject_reason));
            }

            Ok(corrupt_digest)
        }
    }

    #[test]
    fn digest_corruptor_rejects_uncorrupting_corruption() {
        let digest = Digest(bfe_array![1, 2, 3, 4, 5]);
        let corruptor = DigestCorruptor {
            corrupt_indices: vec![0],
            corrupt_elements: bfe_vec![1],
        };
        let err = corruptor.corrupt_digest(digest).unwrap_err();
        assert!(matches!(err, TestCaseError::Reject(_)));
    }

    #[test]
    fn get_size() {
        let stack = Digest::get_stack_size();

        let bfes = bfe_array![12, 24, 36, 48, 60];
        let tip5_digest_type_from_array: Digest = Digest::new(bfes);
        let heap = tip5_digest_type_from_array.get_heap_size();
        let total = tip5_digest_type_from_array.get_size();
        println!("stack: {stack} + heap: {heap} = {total}");

        assert_eq!(stack + heap, total)
    }

    #[test]
    fn digest_from_str() {
        let valid_digest_string = "12063201067205522823,\
            1529663126377206632,\
            2090171368883726200,\
            12975872837767296928,\
            11492877804687889759";
        let valid_digest = Digest::from_str(valid_digest_string);
        assert!(valid_digest.is_ok());

        let invalid_digest_string = "00059361073062755064,05168490802189810700";
        let invalid_digest = Digest::from_str(invalid_digest_string);
        assert!(invalid_digest.is_err());

        let second_invalid_digest_string = "this_is_not_a_bfield_element,05168490802189810700";
        let second_invalid_digest = Digest::from_str(second_invalid_digest_string);
        assert!(second_invalid_digest.is_err());
    }

    #[proptest]
    fn test_reversed_involution(digest: Digest) {
        prop_assert_eq!(digest, digest.reversed().reversed())
    }

    #[test]
    fn digest_biguint_conversion_simple_test() {
        let fourteen: BigUint = 14u128.into();
        let fourteen_converted_expected = Digest(bfe_array![14, 0, 0, 0, 0]);

        let bfe_max: BigUint = BFieldElement::MAX.into();
        let bfe_max_converted_expected = Digest(bfe_array![BFieldElement::MAX, 0, 0, 0, 0]);

        let bfe_max_plus_one: BigUint = BFieldElement::P.into();
        let bfe_max_plus_one_converted_expected = Digest(bfe_array![0, 1, 0, 0, 0]);

        let two_pow_64: BigUint = (1u128 << 64).into();
        let two_pow_64_converted_expected = Digest(bfe_array![(1u64 << 32) - 1, 1, 0, 0, 0]);

        let two_pow_123: BigUint = (1u128 << 123).into();
        let two_pow_123_converted_expected =
            Digest([18446744069280366593, 576460752437641215, 0, 0, 0].map(BFieldElement::new));

        let two_pow_315: BigUint = BigUint::from(2u128).pow(315);

        // Result calculated on Wolfram alpha
        let two_pow_315_converted_expected = Digest(bfe_array![
            18446744069280366593_u64,
            1729382257312923647_u64,
            13258597298683772929_u64,
            3458764513015234559_u64,
            576460752840294400_u64,
        ]);

        // Verify conversion from BigUint to Digest
        assert_eq!(
            fourteen_converted_expected,
            fourteen.clone().try_into().unwrap()
        );
        assert_eq!(
            bfe_max_converted_expected,
            bfe_max.clone().try_into().unwrap()
        );
        assert_eq!(
            bfe_max_plus_one_converted_expected,
            bfe_max_plus_one.clone().try_into().unwrap()
        );
        assert_eq!(
            two_pow_64_converted_expected,
            two_pow_64.clone().try_into().unwrap()
        );
        assert_eq!(
            two_pow_123_converted_expected,
            two_pow_123.clone().try_into().unwrap()
        );
        assert_eq!(
            two_pow_315_converted_expected,
            two_pow_315.clone().try_into().unwrap()
        );

        // Verify conversion from Digest to BigUint
        assert_eq!(fourteen, fourteen_converted_expected.into());
        assert_eq!(bfe_max, bfe_max_converted_expected.into());
        assert_eq!(bfe_max_plus_one, bfe_max_plus_one_converted_expected.into());
        assert_eq!(two_pow_64, two_pow_64_converted_expected.into());
        assert_eq!(two_pow_123, two_pow_123_converted_expected.into());
        assert_eq!(two_pow_315, two_pow_315_converted_expected.into());
    }

    #[proptest]
    fn digest_biguint_conversion_pbt(components_0: [u64; 4], component_1: u32) {
        let big_uint = components_0
            .into_iter()
            .fold(BigUint::one(), |acc, x| acc * x);
        let big_uint = big_uint * component_1;

        let as_digest: Digest = big_uint.clone().try_into().unwrap();
        let big_uint_again: BigUint = as_digest.into();
        prop_assert_eq!(big_uint, big_uint_again);
    }

    #[test]
    fn digest_ordering() {
        let val0 = Digest::new(bfe_array![0; Digest::LEN]);
        let val1 = Digest::new(bfe_array![14, 0, 0, 0, 0]);
        assert!(val1 > val0);

        let val2 = Digest::new(bfe_array![14; Digest::LEN]);
        assert!(val2 > val1);
        assert!(val2 > val0);

        let val3 = Digest::new(bfe_array![15, 14, 14, 14, 14]);
        assert!(val3 > val2);
        assert!(val3 > val1);
        assert!(val3 > val0);

        let val4 = Digest::new(bfe_array![14, 15, 14, 14, 14]);
        assert!(val4 > val3);
        assert!(val4 > val2);
        assert!(val4 > val1);
        assert!(val4 > val0);
    }

    #[test]
    fn digest_biguint_overflow_test() {
        let mut two_pow_384: BigUint = (1u128 << 96).into();
        two_pow_384 = two_pow_384.pow(4);
        let err = Digest::try_from(two_pow_384).unwrap_err();

        assert_eq!(TryFromDigestError::Overflow, err);
    }

    #[proptest]
    fn forty_bytes_can_be_converted_to_digest(bytes: [u8; Digest::BYTES]) {
        let digest = Digest::try_from(bytes).unwrap();
        let bytes_again: [u8; Digest::BYTES] = digest.into();
        prop_assert_eq!(bytes, bytes_again);
    }

    // note: for background on this test, see issue 195
    #[test]
    fn try_from_bytes_not_canonical() -> Result<(), TryFromDigestError> {
        let bytes: [u8; Digest::BYTES] = [255; Digest::BYTES];

        assert!(Digest::try_from(bytes).is_err_and(|e| matches!(
            e,
            TryFromDigestError::InvalidBFieldElement(ParseBFieldElementError::NotCanonical(_))
        )));

        Ok(())
    }

    // note: for background on this test, see issue 195
    #[test]
    fn from_str_not_canonical() -> Result<(), TryFromDigestError> {
        let str = format!("0,0,0,0,{}", u64::MAX);

        assert!(Digest::from_str(&str).is_err_and(|e| matches!(
            e,
            TryFromDigestError::InvalidBFieldElement(ParseBFieldElementError::NotCanonical(_))
        )));

        Ok(())
    }

    #[test]
    fn bytes_in_matches_bytes_out() -> Result<(), TryFromDigestError> {
        let bytes1: [u8; Digest::BYTES] = [254; Digest::BYTES];
        let d1 = Digest::try_from(bytes1)?;

        let bytes2: [u8; Digest::BYTES] = d1.into();
        let d2 = Digest::try_from(bytes2)?;

        assert_eq!(d1, d2);
        assert_eq!(bytes1, bytes2);

        Ok(())
    }

    mod hex_test {
        use super::*;

        pub(super) fn hex_examples() -> Vec<(Digest, &'static str)> {
            vec![
                (
                    Digest::default(),
                    concat!(
                        "0000000000000000000000000000000000000000",
                        "0000000000000000000000000000000000000000"
                    ),
                ),
                (
                    Digest::new(bfe_array![0, 1, 10, 15, 255]),
                    concat!(
                        "000000000000000001000000000000000a000000",
                        "000000000f00000000000000ff00000000000000"
                    ),
                ),
                // note: this would result in NotCanonical error. See issue 195
                // (
                //     Digest::new(bfe_array![0, 1, 10, 15, 255]),
                //     concat!("ffffffffffffffffffffffffffffffffffffffff",
                //             "ffffffffffffffffffffffffffffffffffffffff"),
                // ),
            ]
        }

        #[test]
        fn digest_to_hex() {
            for (digest, hex) in hex_examples() {
                assert_eq!(&digest.to_hex(), hex);
            }
        }

        #[proptest]
        fn to_hex_and_from_hex_are_reciprocal_proptest(bytes: [u8; Digest::BYTES]) {
            let digest = Digest::try_from(bytes).unwrap();
            let hex = digest.to_hex();
            let digest_again = Digest::try_from_hex(&hex).unwrap();
            let hex_again = digest_again.to_hex();
            prop_assert_eq!(digest, digest_again);
            prop_assert_eq!(hex, hex_again);

            let lower_hex = format!("{digest:x}");
            let digest_from_lower_hex = Digest::try_from_hex(lower_hex).unwrap();
            prop_assert_eq!(digest, digest_from_lower_hex);

            let upper_hex = format!("{digest:X}");
            let digest_from_upper_hex = Digest::try_from_hex(upper_hex).unwrap();
            prop_assert_eq!(digest, digest_from_upper_hex);
        }

        #[test]
        fn to_hex_and_from_hex_are_reciprocal() -> Result<(), TryFromHexDigestError> {
            let hex_vals = vec![
                "00000000000000000000000000000000000000000000000000000000000000000000000000000000",
                "10000000000000000000000000000000000000000000000000000000000000000000000000000000",
                "0000000000000000000000000000000000000000000000000000000000000000000000000000000f",
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                // note: all "ff…ff" would result in NotCanonical error. See issue 195
            ];
            for hex in hex_vals {
                let digest = Digest::try_from_hex(hex)?;
                assert_eq!(hex, &digest.to_hex())
            }
            Ok(())
        }

        #[test]
        fn digest_from_hex() -> Result<(), TryFromHexDigestError> {
            for (digest, hex) in hex_examples() {
                assert_eq!(digest, Digest::try_from_hex(hex)?);
            }

            Ok(())
        }

        #[test]
        fn digest_from_invalid_hex_errors() {
            use hex::FromHexError;

            assert!(Digest::try_from_hex("taco").is_err_and(|e| matches!(
                e,
                TryFromHexDigestError::HexDecode(FromHexError::InvalidHexCharacter { .. })
            )));

            assert!(Digest::try_from_hex("0").is_err_and(|e| matches!(
                e,
                TryFromHexDigestError::HexDecode(FromHexError::OddLength)
            )));

            assert!(Digest::try_from_hex("00").is_err_and(|e| matches!(
                e,
                TryFromHexDigestError::Digest(TryFromDigestError::InvalidLength(_))
            )));

            // NotCanonical error. See issue 195
            assert!(Digest::try_from_hex(
                "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
            )
            .is_err_and(|e| matches!(
                e,
                TryFromHexDigestError::Digest(TryFromDigestError::InvalidBFieldElement(
                    ParseBFieldElementError::NotCanonical(_)
                ))
            )));
        }
    }

    mod serde_test {
        use super::hex_test::hex_examples;
        use super::*;

        mod json_test {
            use super::*;

            #[test]
            fn serialize() -> Result<(), serde_json::Error> {
                for (digest, hex) in hex_examples() {
                    assert_eq!(serde_json::to_string(&digest)?, format!("\"{hex}\""));
                }
                Ok(())
            }

            #[test]
            fn deserialize() -> Result<(), serde_json::Error> {
                for (digest, hex) in hex_examples() {
                    let json_hex = format!("\"{hex}\"");
                    let digest_deserialized: Digest = serde_json::from_str::<Digest>(&json_hex)?;
                    assert_eq!(digest_deserialized, digest);
                }
                Ok(())
            }
        }

        mod bincode_test {
            use super::*;

            fn bincode_examples() -> Vec<(Digest, [u8; Digest::BYTES])> {
                vec![
                    (Digest::default(), [0u8; Digest::BYTES]),
                    (
                        Digest::new(bfe_array![0, 1, 10, 15, 255]),
                        [
                            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0,
                            0, 15, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0,
                        ],
                    ),
                ]
            }

            #[test]
            fn serialize() {
                for (digest, bytes) in bincode_examples() {
                    assert_eq!(bincode::serialize(&digest).unwrap(), bytes);
                }
            }

            #[test]
            fn deserialize() {
                for (digest, bytes) in bincode_examples() {
                    assert_eq!(bincode::deserialize::<Digest>(&bytes).unwrap(), digest);
                }
            }
        }
    }
}
