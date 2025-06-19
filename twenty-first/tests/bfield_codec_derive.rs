use arbitrary::Arbitrary;
use proptest::prelude::*;
use proptest_arbitrary_interop::arb;
use test_strategy::proptest;
// Required by the `BFieldCodec` derive macro. This is generally only needed
// once per crate, at the top-level `lib.rs`.
#[expect(clippy::single_component_path_imports)]
use twenty_first;
use twenty_first::math::b_field_element::BFieldElement;
use twenty_first::math::bfield_codec::BFieldCodec;
use twenty_first::math::digest::Digest;
use twenty_first::math::x_field_element::XFieldElement;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

#[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
struct BFieldCodecTestStructA {
    a: u32,
    b: BFieldElement,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[proptest]
fn integration_test_struct_a(#[strategy(arb())] test_struct: BFieldCodecTestStructA) {
    let encoding = test_struct.encode();
    let decoding = *BFieldCodecTestStructA::decode(&encoding).unwrap();
    prop_assert_eq!(test_struct, decoding);
}

#[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
struct BFieldCodecTestStructB {
    a: XFieldElement,
    b: Vec<(u64, Digest)>,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[proptest]
fn integration_test_struct_b(#[strategy(arb())] test_struct: BFieldCodecTestStructB) {
    let encoding = test_struct.encode();
    let decoding = *BFieldCodecTestStructB::decode(&encoding).unwrap();
    prop_assert_eq!(test_struct, decoding);
}

#[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
enum BFieldCodecTestEnumA {
    A,
    B,
    C,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[proptest]
fn integration_test_enum_a(#[strategy(arb())] test_enum: BFieldCodecTestEnumA) {
    let encoding = test_enum.encode();
    let decoding = *BFieldCodecTestEnumA::decode(&encoding).unwrap();
    prop_assert_eq!(test_enum, decoding);
}

#[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
enum BFieldCodecTestEnumB {
    A(u32),
    B(XFieldElement),
    C(Vec<(u64, Digest)>),
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[proptest]
fn integration_test_enum_b(#[strategy(arb())] test_enum: BFieldCodecTestEnumB) {
    let encoding = test_enum.encode();
    let decoding = *BFieldCodecTestEnumB::decode(&encoding).unwrap();
    prop_assert_eq!(test_enum, decoding);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[test]
fn try_build_various_failure_cases() {
    let trybuild = trybuild::TestCases::new();
    trybuild.compile_fail("trybuild/multiple_field_attributes.rs");
    trybuild.compile_fail("trybuild/incorrect_field_attribute.rs");
    trybuild.pass("trybuild/missing_field_attribute.rs");
}
