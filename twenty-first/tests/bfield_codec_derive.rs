use proptest::prelude::*;
use test_strategy::Arbitrary;
use test_strategy::proptest;
// Required by the `BFieldCodec` derive macro. This is generally only needed
// once per crate, at the top-level `lib.rs`.
#[expect(clippy::single_component_path_imports)]
use twenty_first;
use twenty_first::prelude::BFieldCodec;
use twenty_first::prelude::BFieldElement;
use twenty_first::prelude::Digest;
use twenty_first::prelude::XFieldElement;

#[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
struct BFieldCodecTestStructA {
    a: u32,

    #[strategy(bfe_strategy())]
    b: BFieldElement,
}

#[proptest]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn integration_test_struct_a(test_struct: BFieldCodecTestStructA) {
    let encoding = test_struct.encode();
    let decoding = *BFieldCodecTestStructA::decode(&encoding).unwrap();
    prop_assert_eq!(test_struct, decoding);
}

#[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
struct BFieldCodecTestStructB {
    #[strategy(xfe_strategy())]
    a: XFieldElement,

    #[strategy(prop::collection::vec((any::<u64>(), digest_strategy()), 0..50))]
    b: Vec<(u64, Digest)>,
}

#[proptest]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn integration_test_struct_b(test_struct: BFieldCodecTestStructB) {
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

#[proptest]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn integration_test_enum_a(test_enum: BFieldCodecTestEnumA) {
    let encoding = test_enum.encode();
    let decoding = *BFieldCodecTestEnumA::decode(&encoding).unwrap();
    prop_assert_eq!(test_enum, decoding);
}

#[derive(Debug, Clone, PartialEq, Eq, BFieldCodec)]
enum BFieldCodecTestEnumB {
    A(u32),
    B(XFieldElement),
    C(Vec<(u64, Digest)>),
}

impl Arbitrary for BFieldCodecTestEnumB {
    type Parameters = ();

    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        prop_oneof![
            any::<u32>().prop_map(BFieldCodecTestEnumB::A),
            xfe_strategy().prop_map(BFieldCodecTestEnumB::B),
            prop::collection::vec((any::<u64>(), digest_strategy()), 0..50)
                .prop_map(BFieldCodecTestEnumB::C),
        ]
        .boxed()
    }

    type Strategy = BoxedStrategy<Self>;
}

#[proptest]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn integration_test_enum_b(test_enum: BFieldCodecTestEnumB) {
    let encoding = test_enum.encode();
    let decoding = *BFieldCodecTestEnumB::decode(&encoding).unwrap();
    prop_assert_eq!(test_enum, decoding);
}

#[test]
fn try_build_various_failure_cases() {
    let trybuild = trybuild::TestCases::new();
    trybuild.compile_fail("trybuild/multiple_field_attributes.rs");
    trybuild.compile_fail("trybuild/incorrect_field_attribute.rs");
    trybuild.pass("trybuild/missing_field_attribute.rs");
}

fn bfe_strategy() -> impl Strategy<Value = BFieldElement> {
    (0..=BFieldElement::MAX).prop_map(BFieldElement::new)
}

fn xfe_strategy() -> impl Strategy<Value = XFieldElement> {
    let b = bfe_strategy;

    [b(), b(), b()].prop_map(XFieldElement::new)
}

fn digest_strategy() -> impl Strategy<Value = Digest> {
    let b = bfe_strategy;

    [b(), b(), b(), b(), b()].prop_map(Digest::new)
}
