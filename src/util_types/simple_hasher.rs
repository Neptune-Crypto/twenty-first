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
        self
    }
}

/// Trivial implementation when hashing `Vec<BFieldElement>` into `Vec<BFieldElement>`s.
impl ToDigest<Vec<BFieldElement>> for Vec<BFieldElement> {
    fn to_digest(&self) -> &Vec<BFieldElement> {
        self
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
        self.0.hash(input.to_digest())
    }

    // TODO: Avoid list copying.
    fn hash_two<Value: ToDigest<Self::Digest>>(
        &mut self,
        left: &Value,
        right: &Value,
    ) -> Self::Digest {
        let input: Vec<BFieldElement> =
            vec![left.to_digest().to_owned(), right.to_digest().to_owned()].concat();
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
