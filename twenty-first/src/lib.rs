#![deny(clippy::shadow_unrelated)]
pub mod amount;
pub mod error;
pub mod prelude;
pub mod shared_math;
pub mod util_types;

#[cfg(any(test, feature = "mock"))]
pub mod mock;

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
