use core::fmt;
use std::str::FromStr;

use bfieldcodec_derive::BFieldCodec;
use get_size::GetSize;
use itertools::Itertools;
use num_bigint::{BigUint, TryFromBigIntError};
use num_traits::Zero;
use rand::Rng;
use rand_distr::{Distribution, Standard};
use serde::{Deserialize, Serialize};

use crate::shared_math::b_field_element::{BFieldElement, BFIELD_ZERO};
use crate::shared_math::traits::FromVecu8;
use crate::util_types::algebraic_hasher::AlgebraicHasher;
use crate::util_types::emojihash_trait::Emojihash;

pub const DIGEST_LENGTH: usize = 5;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash, BFieldCodec)]
pub struct Digest(pub [BFieldElement; DIGEST_LENGTH]);

impl GetSize for Digest {
    fn get_stack_size() -> usize {
        std::mem::size_of::<Self>()
    }

    fn get_heap_size(&self) -> usize {
        0
    }
}

impl PartialOrd for Digest {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        for i in (0..DIGEST_LENGTH).rev() {
            if self.0[i].value() != other.0[i].value() {
                return self.0[i].value().partial_cmp(&other.0[i].value());
            }
        }

        None
    }
}

impl Ord for Digest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Digest {
    pub const BYTES: usize = DIGEST_LENGTH * BFieldElement::BYTES;

    pub fn values(&self) -> [BFieldElement; DIGEST_LENGTH] {
        self.0
    }

    pub const fn new(digest: [BFieldElement; DIGEST_LENGTH]) -> Self {
        Self(digest)
    }

    /// Returns a new digest but whose elements are reversed relative to self.
    /// This function is an involutive endomorphism.
    pub const fn reversed(self) -> Digest {
        Digest([self.0[4], self.0[3], self.0[2], self.0[1], self.0[0]])
    }
}

impl Emojihash for Digest {
    fn emojihash(&self) -> String {
        self.0.emojihash()
    }
}

impl Default for Digest {
    fn default() -> Self {
        Self([BFIELD_ZERO; DIGEST_LENGTH])
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.map(|elem| elem.to_string()).join(","))
    }
}

impl Distribution<Digest> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Digest {
        // FIXME: impl Fill for [BFieldElement] to rng.fill() a [BFieldElement; DIGEST_LENGTH].
        let elements = rng
            .sample_iter(Standard)
            .take(DIGEST_LENGTH)
            .collect_vec()
            .try_into()
            .unwrap();
        Digest::new(elements)
    }
}

impl FromStr for Digest {
    type Err = String;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        let parsed_u64s: Vec<Result<u64, _>> = string
            .split(',')
            .map(|substring| substring.parse::<u64>())
            .collect();
        if parsed_u64s.len() != DIGEST_LENGTH {
            Err("Given invalid number of BFieldElements in string.".to_owned())
        } else {
            let mut bf_elms: Vec<BFieldElement> = Vec::with_capacity(DIGEST_LENGTH);
            for parse_result in parsed_u64s {
                if let Ok(content) = parse_result {
                    bf_elms.push(BFieldElement::new(content));
                } else {
                    return Err("Given invalid BFieldElement in string.".to_owned());
                }
            }
            Ok(bf_elms.try_into()?)
        }
    }
}

impl TryFrom<&[BFieldElement]> for Digest {
    type Error = String;

    fn try_from(value: &[BFieldElement]) -> Result<Self, Self::Error> {
        let len = value.len();
        value.try_into().map(Digest::new).map_err(|_| {
            format!("Expected {DIGEST_LENGTH} BFieldElements for digest, but got {len}")
        })
    }
}

impl TryFrom<Vec<BFieldElement>> for Digest {
    type Error = String;

    fn try_from(value: Vec<BFieldElement>) -> Result<Self, Self::Error> {
        Digest::try_from(value.as_ref())
    }
}

impl From<Digest> for Vec<BFieldElement> {
    fn from(val: Digest) -> Self {
        val.0.to_vec()
    }
}

impl From<Digest> for [u8; Digest::BYTES] {
    fn from(item: Digest) -> Self {
        let u64s = item.0.iter().map(|x| x.value());
        u64s.map(|x| x.to_ne_bytes())
            .collect::<Vec<_>>()
            .concat()
            .try_into()
            .unwrap()
    }
}

impl From<[u8; Digest::BYTES]> for Digest {
    fn from(item: [u8; Digest::BYTES]) -> Self {
        let mut bfes: [BFieldElement; DIGEST_LENGTH] = [BFieldElement::zero(); DIGEST_LENGTH];
        for (i, bfe) in bfes.iter_mut().enumerate() {
            let start_index = i * BFieldElement::BYTES;
            let end_index = (i + 1) * BFieldElement::BYTES;
            *bfe = BFieldElement::from_vecu8(item[start_index..end_index].to_vec())
        }

        Self(bfes)
    }
}

impl TryFrom<BigUint> for Digest {
    type Error = String;

    fn try_from(value: BigUint) -> Result<Self, Self::Error> {
        let mut remaining = value;
        let mut ret = Digest::default();
        let modulus: BigUint = BFieldElement::P.into();
        for i in 0..DIGEST_LENGTH {
            let resulting_u64: u64 = (remaining.clone() % modulus.clone()).try_into().map_err(
                |err: TryFromBigIntError<BigUint>| {
                    format!("Could not convert remainder back to u64: {:?}", err)
                },
            )?;
            ret.0[i] = BFieldElement::new(resulting_u64);
            remaining /= modulus.clone();
        }

        if !remaining.is_zero() {
            return Err("Overflow when converting from BigUint to Digest".to_string());
        }

        Ok(ret)
    }
}

impl From<Digest> for BigUint {
    fn from(digest: Digest) -> Self {
        let mut ret = BigUint::zero();
        let modulus: BigUint = BFieldElement::P.into();
        for i in (0..DIGEST_LENGTH).rev() {
            ret *= modulus.clone();
            let digest_element: BigUint = digest.0[i].value().into();
            ret += digest_element;
        }

        ret
    }
}

impl Digest {
    /// Simulates the VM as it hashes a digest. This
    /// method invokes hash_pair with the right operand being the zero
    /// digest, agreeing with the standard way to hash a digest in the
    /// virtual machine.
    pub fn hash<H: AlgebraicHasher>(&self) -> Digest {
        H::hash_pair(self, &Digest::new([BFieldElement::zero(); DIGEST_LENGTH]))
    }
}

#[cfg(test)]
mod digest_tests {
    use num_traits::One;
    use rand::{thread_rng, RngCore};

    use super::*;

    #[test]
    pub fn get_size() {
        let stack = Digest::get_stack_size();

        let bfe_vec = vec![
            BFieldElement::new(12),
            BFieldElement::new(24),
            BFieldElement::new(36),
            BFieldElement::new(48),
            BFieldElement::new(60),
        ];
        let tip5_digest_type_from_array: Digest = bfe_vec.try_into().unwrap();

        let heap = tip5_digest_type_from_array.get_heap_size();

        let total = tip5_digest_type_from_array.get_size();

        println!("stack: {stack} + heap: {heap} = {total}");

        assert_eq!(stack + heap, total)
    }

    #[test]
    pub fn digest_from_str() {
        // This tests a valid digest. It will fail if DIGEST_LENGTH is changed.
        let valid_digest_string = "12063201067205522823,1529663126377206632,2090171368883726200,12975872837767296928,11492877804687889759";
        let valid_digest = Digest::from_str(valid_digest_string);
        assert!(valid_digest.is_ok());

        // This ensures that it can fail when given a wrong number of BFieldElements.
        let invalid_digest_string = "00059361073062755064,05168490802189810700";
        let invalid_digest = Digest::from_str(invalid_digest_string);
        assert!(invalid_digest.is_err());

        // This ensures that it can fail if given something that isn't a valid string of a BFieldElement.
        let second_invalid_digest_string = "this_is_not_a_bfield_element,05168490802189810700";
        let second_invalid_digest = Digest::from_str(second_invalid_digest_string);
        assert!(second_invalid_digest.is_err());
    }

    #[test]
    pub fn test_reversed_involution() {
        let digest: Digest = thread_rng().gen();
        assert_eq!(digest, digest.reversed().reversed())
    }

    #[test]
    fn digest_biguint_conversion_simple_test() {
        let fourteen: BigUint = 14u128.into();
        let fourteen_converted_expected: Digest = Digest([
            BFieldElement::new(14),
            BFieldElement::zero(),
            BFieldElement::zero(),
            BFieldElement::zero(),
            BFieldElement::zero(),
        ]);

        let bfe_max: BigUint = BFieldElement::MAX.into();
        let bfe_max_converted_expected = Digest([
            BFieldElement::new(BFieldElement::MAX),
            BFieldElement::zero(),
            BFieldElement::zero(),
            BFieldElement::zero(),
            BFieldElement::zero(),
        ]);

        let bfe_max_plus_one: BigUint = BFieldElement::P.into();
        let bfe_max_plus_one_converted_expected = Digest([
            BFieldElement::zero(),
            BFieldElement::one(),
            BFieldElement::zero(),
            BFieldElement::zero(),
            BFieldElement::zero(),
        ]);

        let two_pow_64: BigUint = (1u128 << 64).into();
        let two_pow_64_converted_expected = Digest([
            BFieldElement::new((1u64 << 32) - 1),
            BFieldElement::one(),
            BFieldElement::zero(),
            BFieldElement::zero(),
            BFieldElement::zero(),
        ]);

        let two_pow_123: BigUint = (1u128 << 123).into();
        let two_pow_123_converted_expected = Digest([
            BFieldElement::new(18446744069280366593),
            BFieldElement::new(576460752437641215),
            BFieldElement::zero(),
            BFieldElement::zero(),
            BFieldElement::zero(),
        ]);

        let two_pow_315: BigUint = BigUint::from(2u128).pow(315);

        // Result calculated on Wolfram alpha
        let two_pow_315_converted_expected = Digest([
            BFieldElement::new(18446744069280366593),
            BFieldElement::new(1729382257312923647),
            BFieldElement::new(13258597298683772929),
            BFieldElement::new(3458764513015234559),
            BFieldElement::new(576460752840294400),
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
        assert_eq!(fourteen, fourteen_converted_expected.try_into().unwrap());
        assert_eq!(bfe_max, bfe_max_converted_expected.try_into().unwrap());
        assert_eq!(
            bfe_max_plus_one,
            bfe_max_plus_one_converted_expected.try_into().unwrap()
        );
        assert_eq!(
            two_pow_64,
            two_pow_64_converted_expected.try_into().unwrap()
        );
        assert_eq!(
            two_pow_123,
            two_pow_123_converted_expected.try_into().unwrap()
        );
        assert_eq!(
            two_pow_315,
            two_pow_315_converted_expected.try_into().unwrap()
        );
    }

    #[test]
    fn digest_biguint_conversion_pbt() {
        let count = 100;
        let mut rng = thread_rng();
        for _ in 0..count {
            // Generate a random BigUint that will fit into an ordered digest
            let mut biguint: BigUint = BigUint::one();
            for _ in 0..4 {
                biguint *= rng.next_u64();
            }
            biguint *= rng.next_u32();

            // Verify that conversion back and forth is the identity operator
            let as_digest: Digest = biguint.clone().try_into().unwrap();
            let converted_back: BigUint = as_digest.into();
            assert_eq!(biguint, converted_back);
        }
    }

    #[test]
    fn digest_ordering() {
        let val0 = Digest::new([BFieldElement::new(0); DIGEST_LENGTH]);
        let val1 = Digest::new([
            BFieldElement::new(14),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
        ]);
        assert!(val0 < val1);

        let val2 = Digest::new([BFieldElement::new(14); DIGEST_LENGTH]);
        assert!(val2 > val1);
        assert!(val2 > val0);

        let val3 = Digest::new([
            BFieldElement::new(15),
            BFieldElement::new(14),
            BFieldElement::new(14),
            BFieldElement::new(14),
            BFieldElement::new(14),
        ]);
        assert!(val3 > val2);
        assert!(val3 > val1);
        assert!(val3 > val0);

        let val4 = Digest::new([
            BFieldElement::new(14),
            BFieldElement::new(15),
            BFieldElement::new(14),
            BFieldElement::new(14),
            BFieldElement::new(14),
        ]);
        assert!(val4 > val3);
        assert!(val4 > val2);
        assert!(val4 > val1);
        assert!(val4 > val0);
    }

    #[test]
    #[should_panic(expected = "Overflow when converting from BigUint to Digest")]
    fn digest_biguint_overflow_test() {
        let mut two_pow_384: BigUint = (1u128 << 96).into();
        two_pow_384 = two_pow_384.pow(4);
        let _failing_conversion: Digest = two_pow_384.try_into().unwrap();
    }
}
