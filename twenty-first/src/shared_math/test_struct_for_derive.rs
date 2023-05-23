extern crate bfieldcodec_derive;

use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::bfield_codec::decode_field_length_prepended;
use crate::shared_math::bfield_codec::BFieldCodec;
use bfieldcodec_derive::BFieldCodec;

#[derive(BFieldCodec, PartialEq, Eq, Debug)]
pub struct TestStructA {
    field_a: u64,
    field_b: u64,
    field_c: u64,
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn deriving_bfieldcodec_impl_works_test() {
        let ts = TestStructA {
            field_a: 14,
            field_b: 555558,
            field_c: 1337,
        };
        let encoded = ts.encode();
        let decoded = *TestStructA::decode(&encoded).unwrap();
        assert_eq!(ts, decoded);

        let encoded_too_long = vec![encoded, vec![BFieldElement::new(5)]].concat();
        assert!(TestStructA::decode(&encoded_too_long).is_err());

        let encoded_too_short = encoded_too_long[..encoded_too_long.len() - 2].to_vec();
        assert!(TestStructA::decode(&encoded_too_short).is_err());
    }
}
