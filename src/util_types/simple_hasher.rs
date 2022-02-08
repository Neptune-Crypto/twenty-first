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
    type Digest;

    fn new() -> Self;
    fn hash_one<Value: ToDigest<Self::Digest>>(&mut self, one: &Value) -> Self::Digest;
    fn hash_two<Value: ToDigest<Self::Digest>>(&mut self, one: &Value, two: &Value)
        -> Self::Digest;
    fn hash_many<Value: ToDigest<Self::Digest>>(&mut self, input: &[Value]) -> Self::Digest;
}

/// In order to hash arbitrary things using a `Hasher`, it must `impl ToDigest<Digest>`
/// where the concrete `Digest` is what's chosen for the `impl Hasher`. For example, in
/// order to
pub trait ToDigest<Digest> {
    fn to_digest(&self) -> &Digest;
}

/// Trivial implementation when hashing `blake3::Hash` into `blake3::Hash`es.
impl ToDigest<blake3::Hash> for blake3::Hash {
    fn to_digest(&self) -> &blake3::Hash {
        &self
    }
}

/// Trivial implementation when hashing `BFieldElement` into `BFieldElement`s.
impl ToDigest<BFieldElement> for BFieldElement {
    fn to_digest(&self) -> &BFieldElement {
        &self
    }
}

impl Hasher for blake3::Hasher {
    type Digest = blake3::Hash;

    fn new() -> Self {
        blake3::Hasher::new()
    }

    fn hash_one<Value: ToDigest<Self::Digest>>(&mut self, one: &Value) -> Self::Digest {
        self.reset();
        self.update(one.to_digest().as_bytes());
        self.finalize()
    }

    fn hash_two<Value: ToDigest<Self::Digest>>(
        &mut self,
        one: &Value,
        two: &Value,
    ) -> Self::Digest {
        self.reset();
        self.update(one.to_digest().as_bytes());
        self.update(two.to_digest().as_bytes());
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
pub struct RescuePrimeProduction(RescuePrime);

impl Hasher for RescuePrimeProduction {
    // TODO: For a sufficient security level, we require 5 `BFieldElement`s per hash digest.
    // We can ensure that by making the digest a [BFieldElement; 5] and by feeding those
    // into RescuePrime. As a gradual step, we pad single BFieldElements with enough zeros.
    type Digest = BFieldElement;

    fn new() -> Self {
        RescuePrimeProduction(rescue_prime_params::rescue_prime_params_bfield_0())
    }

    fn hash_one<Value: ToDigest<Self::Digest>>(&mut self, one: &Value) -> Self::Digest {
        let rescue_prime = &self.0;

        // TODO: Remove input padding once Self::Digest has required width.
        let padded_input = vec![*one.to_digest(); rescue_prime.input_length];

        // TODO: Use full output once Self::Digest has required width.
        let output = rescue_prime.hash(&padded_input);
        output[0]
    }

    fn hash_two<Value: ToDigest<Self::Digest>>(
        &mut self,
        one: &Value,
        two: &Value,
    ) -> Self::Digest {
        let rescue_prime = &self.0;

        // TODO: Remove input padding once Self::Digest has required width.
        let elem_width = rescue_prime.input_length / 2;
        let padded_one = vec![*one.to_digest(); elem_width];
        let padded_two = vec![*two.to_digest(); elem_width];
        let padded_input = [padded_one, padded_two].concat();

        // TODO: Use full output once Self::Digest has required width.
        let output = rescue_prime.hash(&padded_input);
        output[0]
    }

    // TODO: Implement me! (Or remove me from interface?)
    fn hash_many<Value: ToDigest<Self::Digest>>(&mut self, _input: &[Value]) -> Self::Digest {
        todo!()
    }
}
