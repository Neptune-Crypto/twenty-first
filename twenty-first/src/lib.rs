#![warn(clippy::shadow_unrelated)]
//
// If code coverage tool `cargo-llvm-cov` is running with the nightly toolchain,
// enable the unstable “coverage” attribute. This allows using the annotation
// `#[coverage(off)]` to explicitly exclude certain parts of the code from
// being considered as “code under test.” Most prominently, the annotation
// should be added to every `#[cfg(test)]` module. Since the “coverage”
// feature is enable only conditionally, the annotation to use is:
// #[cfg_attr(coverage_nightly, coverage(off))]
//
// See also:
// - https://github.com/taiki-e/cargo-llvm-cov#exclude-code-from-coverage
// - https://github.com/rust-lang/rust/issues/84605
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

pub mod config;
pub mod error;
pub mod math;
pub mod prelude;
pub mod tip5;
pub mod util_types;

// This is needed for `#[derive(BFieldCodec)]` macro to work consistently across crates.
// Specifically:
// From inside the `twenty-first` crate, we need to refer to `twenty-first` by `crate`.
// However, from outside the `twenty-first` crate, we need to refer to it by `twenty_first`.
// The re-export below allows using identifier `twenty_first` even from inside `twenty-first`.
//
// See also:
// https://github.com/bkchr/proc-macro-crate/issues/2#issuecomment-572914520
extern crate self as twenty_first;

// re-export crates used in our public API
pub use bfieldcodec_derive;

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(crate) mod tests {
    use prelude::*;

    use super::*;

    /// A crate-specific replacement of the `#[test]` attribute for tests that
    /// should also be executed on `wasm` targets (which is almost all tests).
    ///
    /// If you specifically want to exclude a test from `wasm` targets, use the
    /// usual `#[test]` attribute instead.
    ///
    /// # Usage
    ///
    /// ```
    /// #[macro_rules_attr::apply(test)]
    /// fn foo() {
    ///     assert_eq!(4, 2 + 2);
    /// }
    /// ```
    macro_rules! test {
        ($item:item) => {
            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
            $item
        };
    }
    pub(crate) use test;

    /// A crate-specific replacement of the `#[test_strategy::proptest]`
    /// attribute for tests that should also be executed on `wasm` targets
    /// (which is almost all tests).
    ///
    /// If you specifically want to exclude a test from `wasm` targets, use the
    /// usual `#[test_strategy::proptest]` attribute instead.
    ///
    /// # Usage
    ///
    /// ```
    /// # use proptest::prop_assert_eq;
    /// #[macro_rules_attr::apply(proptest)]
    /// fn foo(#[strategy(0..=42)] x: i32) {
    ///     prop_assert_eq!(2 * x, x + x);
    /// }
    /// ```
    ///
    /// If you want to configure the test, use the usual syntax defined by
    /// [`test_strategy`]:
    /// ```
    /// # use proptest::prop_assert_eq;
    /// #[macro_rules_attr::apply(proptest(cases = 10, max_local_rejects = 5))]
    /// fn foo(#[strategy(0..=42)] x: i32) {
    ///     prop_assert_eq!(2 * x, x + x);
    /// }
    /// ```
    macro_rules! proptest {
        ($item:item $(($($config:tt)*))?) => {
            #[test_strategy::proptest $(($($config)*))?]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
            $item
        };
    }
    pub(crate) use proptest;

    /// The compiler automatically adds any applicable auto trait (all of which are
    /// marker traits) to self-defined types. This implies that these trait bounds
    /// might vanish if the necessary pre-conditions are no longer met. That'd be a
    /// breaking API change!
    ///
    /// To prevent _accidental_ removal of auto trait implementations, this method
    /// tests for their presence. If you are re-designing any of the types below
    /// and a test fails as a result, that might be fine. You are now definitely
    /// aware of a consequence you might not have known about otherwise. (If you
    /// were already aware you know how subtle this stuff can be and are hopefully
    /// fine with reading this comment.)
    ///
    /// Inspired by “Rust for Rustaceans” by Jon Gjengset.
    pub fn implements_usual_auto_traits<T: Sized + Send + Sync + Unpin>() {}

    #[macro_rules_attr::apply(test)]
    fn types_in_prelude_implement_the_usual_auto_traits() {
        implements_usual_auto_traits::<BFieldElement>();
        implements_usual_auto_traits::<Polynomial<BFieldElement>>();
        implements_usual_auto_traits::<Polynomial<XFieldElement>>();
        implements_usual_auto_traits::<Digest>();
        implements_usual_auto_traits::<Tip5>();
        implements_usual_auto_traits::<XFieldElement>();
        implements_usual_auto_traits::<MerkleTree>();
        implements_usual_auto_traits::<MerkleTreeInclusionProof>();
        implements_usual_auto_traits::<MmrMembershipProof>();
    }

    #[macro_rules_attr::apply(test)]
    fn public_types_implement_the_usual_auto_traits() {
        implements_usual_auto_traits::<math::lattice::CyclotomicRingElement>();
        implements_usual_auto_traits::<math::lattice::ModuleElement<42>>();
        implements_usual_auto_traits::<math::lattice::kem::SecretKey>();
        implements_usual_auto_traits::<math::lattice::kem::PublicKey>();
        implements_usual_auto_traits::<math::lattice::kem::Ciphertext>();
        implements_usual_auto_traits::<util_types::sponge::Domain>();
        implements_usual_auto_traits::<util_types::mmr::mmr_accumulator::MmrAccumulator>();
        implements_usual_auto_traits::<math::zerofier_tree::Branch<BFieldElement>>();
        implements_usual_auto_traits::<math::zerofier_tree::Leaf<BFieldElement>>();
        implements_usual_auto_traits::<math::zerofier_tree::ZerofierTree<BFieldElement>>();
        implements_usual_auto_traits::<
            math::polynomial::ModularInterpolationPreprocessingData<BFieldElement>,
        >();
    }

    #[macro_rules_attr::apply(test)]
    fn errors_implement_the_usual_auto_traits() {
        implements_usual_auto_traits::<error::BFieldCodecError>();
        implements_usual_auto_traits::<error::PolynomialBFieldCodecError>();
        implements_usual_auto_traits::<error::MerkleTreeError>();
        implements_usual_auto_traits::<error::ParseBFieldElementError>();
        implements_usual_auto_traits::<error::TryFromDigestError>();
        implements_usual_auto_traits::<error::TryFromHexDigestError>();
        implements_usual_auto_traits::<error::TryFromU32sError>();
        implements_usual_auto_traits::<error::TryFromXFieldElementError>();
    }
}
