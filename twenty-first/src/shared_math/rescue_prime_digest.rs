use core::fmt;
use std::str::FromStr;

use get_size::GetSize;
use itertools::Itertools;
use num_traits::Zero;
use rand::Rng;
use rand_distr::{Distribution, Standard};
use serde::{Deserialize, Serialize};

use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::rescue_prime_regular::DIGEST_LENGTH;
use crate::shared_math::traits::FromVecu8;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Digest([BFieldElement; DIGEST_LENGTH]);
// FIXME: Make Digest a record instead of a tuple.

impl GetSize for Digest {
    fn get_stack_size() -> usize {
        std::mem::size_of::<Self>()
    }

    fn get_heap_size(&self) -> usize {
        0
    }

    fn get_size(&self) -> usize {
        Self::get_stack_size()
    }
}

impl Digest {
    pub const BYTES: usize = DIGEST_LENGTH * BFieldElement::BYTES;

    pub fn values(&self) -> [BFieldElement; DIGEST_LENGTH] {
        self.0
    }

    pub fn new(digest: [BFieldElement; DIGEST_LENGTH]) -> Self {
        Self(digest)
    }

    pub fn emojihash(&self) -> String {
        let [a, b, c, d, e] = self.0.map(|bfe| {
            emojihash::hash(&bfe.value().to_be_bytes())
                .chars()
                .take(DIGEST_LENGTH)
                .collect::<String>()
        });
        format!("[{a}|{b}|{c}|{d}|{e}]")
    }
}

impl Default for Digest {
    fn default() -> Self {
        Self([BFieldElement::zero(); DIGEST_LENGTH])
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
            format!(
                "Expected {} BFieldElements for digest, but got {}",
                DIGEST_LENGTH, len,
            )
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

#[cfg(test)]
mod digest_tests {
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
        let rescue_prime_digest_type_from_array: Digest = bfe_vec.try_into().unwrap();

        let heap = rescue_prime_digest_type_from_array.get_heap_size();

        let total = rescue_prime_digest_type_from_array.get_size();

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
}
