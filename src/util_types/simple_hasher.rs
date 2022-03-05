use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::rescue_prime::RescuePrime;
use crate::shared_math::rescue_prime_params;
use serde::{Deserialize, Serialize};
/// A simple `Hasher` trait that allows for hashing one, two or many values into one digest.
///
/// The type of digest is determined by the `impl` of a given `Hasher`, and it requires that
/// `Value` has a `ToDigest<Self::Digest>` instance. For hashing hash digests, this `impl`
/// is quite trivial. For non-trivial cases it may include byte-encoding or hashing.
pub trait Hasher {
    type Digest: Eq;

    fn new() -> Self;
    fn hash_one<Value: ToDigest<Self::Digest>>(&mut self, input: &Value) -> Self::Digest;
    fn hash_two<Value: ToDigest<Self::Digest>>(
        &mut self,
        left_input: &Value,
        right_input: &Value,
    ) -> Self::Digest;
    fn hash_many<Value: ToDigest<Self::Digest>>(&mut self, inputs: &[Value]) -> Self::Digest;
}

/// In order to hash arbitrary things using a `Hasher`, it must `impl ToDigest<Digest>`
/// where the concrete `Digest` is what's chosen for the `impl Hasher`. For example, in
/// order to
pub trait ToDigest<Digest> {
    fn to_digest(&self) -> Digest;
}

/// Trivial implementation when hashing `blake3::Hash` into `blake3::Hash`es.
impl ToDigest<blake3::Hash> for blake3::Hash {
    fn to_digest(&self) -> blake3::Hash {
        self.to_owned()
    }
}

// The specification for MMR from mimblewimble specifies that the
// node count is included in the hash preimage. Representing the
// node count as a u128 makes this possible
impl ToDigest<blake3::Hash> for u128 {
    fn to_digest(&self) -> blake3::Hash {
        blake3::Hash::from_hex(format!("{:064x}", self)).unwrap()
    }
}

impl ToDigest<Vec<BFieldElement>> for u128 {
    fn to_digest(&self) -> Vec<BFieldElement> {
        // Only shifting with 63 *should* prevent collissions for all
        // numbers below u64::MAX
        vec![
            BFieldElement::new((self >> 63) % u64::MAX as u128),
            BFieldElement::new(self % BFieldElement::MAX),
        ]
    }
}

/// Trivial implementation when hashing `Vec<BFieldElement>` into `Vec<BFieldElement>`s.
impl ToDigest<Vec<BFieldElement>> for Vec<BFieldElement> {
    fn to_digest(&self) -> Vec<BFieldElement> {
        self.to_owned()
    }
}

impl Hasher for blake3::Hasher {
    type Digest = blake3::Hash;

    fn new() -> Self {
        blake3::Hasher::new()
    }

    fn hash_one<Value: ToDigest<Self::Digest>>(&mut self, input: &Value) -> Self::Digest {
        self.reset();
        self.update(input.to_digest().as_bytes());
        self.finalize()
    }

    fn hash_two<Value: ToDigest<Self::Digest>>(
        &mut self,
        left: &Value,
        right: &Value,
    ) -> Self::Digest {
        self.reset();
        self.update(left.to_digest().as_bytes());
        self.update(right.to_digest().as_bytes());
        self.finalize()
    }

    fn hash_many<Value: ToDigest<Self::Digest>>(&mut self, input: &[Value]) -> blake3::Hash {
        self.reset();
        for value in input {
            self.update(value.to_digest().as_bytes());
        }
        self.finalize()
    }
}

/// Since each struct can only have one `impl Hasher`, and we have many sets
/// of RescuePrime parameters, we create one here to act as the canonical
/// production version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RescuePrimeProduction(pub RescuePrime);

impl Hasher for RescuePrimeProduction {
    type Digest = Vec<BFieldElement>;

    fn new() -> Self {
        RescuePrimeProduction(rescue_prime_params::rescue_prime_params_bfield_0())
    }

    fn hash_one<Value: ToDigest<Self::Digest>>(&mut self, input: &Value) -> Self::Digest {
        self.0.hash(&input.to_digest())
    }

    // TODO: Avoid list copying.
    fn hash_two<Value: ToDigest<Self::Digest>>(
        &mut self,
        left: &Value,
        right: &Value,
    ) -> Self::Digest {
        let input: Vec<BFieldElement> = vec![left.to_digest(), right.to_digest()].concat();
        self.0.hash(&input)
    }

    fn hash_many<Value: ToDigest<Self::Digest>>(&mut self, input: &[Value]) -> Self::Digest {
        let mut input_: Vec<BFieldElement> = vec![];
        for v in input.iter() {
            let mut digest = v.to_digest().to_owned();
            input_.append(&mut digest);
        }
        self.0.hash(&input_)
    }
}

#[cfg(test)]
pub mod test_simple_hasher {

    use super::*;

    #[test]
    fn u128_to_digest_test() {
        let one = 1u128;
        let bfields_one: Vec<BFieldElement> = one.to_digest();
        assert_eq!(2, bfields_one.len());
        assert_eq!(BFieldElement::ring_zero(), bfields_one[0]);
        assert_eq!(BFieldElement::ring_one(), bfields_one[1]);

        let beyond_bfield0 = u64::MAX as u128;
        let bfields: Vec<BFieldElement> = beyond_bfield0.to_digest();
        assert_eq!(2, bfields.len());
        assert_eq!(BFieldElement::ring_one(), bfields[0]);
        assert_eq!(BFieldElement::new(4294967295u128), bfields[1]);

        let beyond_bfield1 = BFieldElement::MAX + 1;
        let bfields: Vec<BFieldElement> = beyond_bfield1.to_digest();
        assert_eq!(2, bfields.len());
        assert_eq!(BFieldElement::ring_one(), bfields[0]);
        assert_eq!(BFieldElement::new(1u128), bfields[1]);
    }

    #[test]
    fn blake3_digest_from_u128_test() {
        // Verify that u128 values can be converted into Blake3 hash input digests
        let _128_val: blake3::Hash = 100u128.to_digest();
        assert_eq!(
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 100
            ],
            _128_val.as_bytes()
        );
    }

    #[test]
    fn rescue_prime_equivalence_test() {
        let mut rpp: RescuePrimeProduction = RescuePrimeProduction::new();
        let input0: Vec<BFieldElement> = vec![1u128, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            .into_iter()
            .map(BFieldElement::new)
            .collect();
        let expected_output0: Vec<BFieldElement> = vec![
            BFieldElement::new(16408223883448864076),
            BFieldElement::new(17937404513354951095),
            BFieldElement::new(17784658070603252681),
            BFieldElement::new(4690418723130302842),
            BFieldElement::new(3079713491308723285),
        ];
        assert_eq!(expected_output0, rpp.hash_one(&input0));

        let input2_left: Vec<BFieldElement> = vec![3, 1, 4, 1, 5]
            .into_iter()
            .map(BFieldElement::new)
            .collect();
        let input2_right: Vec<BFieldElement> = vec![9, 2, 6, 5, 3]
            .into_iter()
            .map(BFieldElement::new)
            .collect();
        let expected_output2: Vec<BFieldElement> = vec![
            8224332136734371881,
            8736343702647113032,
            9660176071866133892,
            575034608412522142,
            13216022346578371396,
        ]
        .into_iter()
        .map(BFieldElement::new)
        .collect();

        assert_eq!(expected_output2, rpp.hash_two(&input2_left, &input2_right));

        let inputs_2: Vec<Vec<BFieldElement>> = vec![
            vec![BFieldElement::new(3)],
            vec![BFieldElement::new(1)],
            vec![BFieldElement::new(4)],
            vec![BFieldElement::new(1)],
            vec![BFieldElement::new(5)],
            vec![BFieldElement::new(9)],
            vec![BFieldElement::new(2)],
            vec![BFieldElement::new(6)],
            vec![BFieldElement::new(5)],
            vec![BFieldElement::new(3)],
        ];
        assert_eq!(expected_output2, rpp.hash_many(&inputs_2));
    }
}
