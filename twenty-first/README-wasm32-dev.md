# `twenty-first` wasm32 integration

This document provides a summary of changes that were required to get
the twenty-first crate to build and run tests for wasm32 target.


## 1\. Summary of `Cargo.toml` Changes

To support both native and `wasm32` environments, the `Cargo.toml` file was restructured to use target-specific dependency sections. This allows us to have different configurations and dependencies for each compilation target.

### General Dependencies for Wasm

A new section, `[target.'cfg(target_arch = "wasm32")'.dependencies]`, was added to define dependencies that are only needed when compiling for WebAssembly:

- `getrandom = { ..., features = ["js"] }`: Provides an entropy source required by the `rand` crate by linking to JavaScript's Web Crypto API.
- `wasm-bindgen`: The core library enabling communication between Rust and JavaScript.

```toml
[target.'cfg(target_arch = "wasm32")'.dependencies]
getrandom = { version = "0.3", features = ["wasm_js"] }
wasm-bindgen = "=0.2.100"
```

### Target-Specific Development Dependencies

The `[dev-dependencies]` have been split into two target-specific sections to handle the different needs of testing in native vs. Wasm environments.

- When targeting `wasm32`, we use `[target.'cfg(target_arch = "wasm32")'.dev-dependencies]`:

  - `criterion` and `proptest` are included with `default-features = false` to ensure they are `no_std`-compatible.
  - `wasm-bindgen-test` is included to provide the test runner for the Wasm environment.

- For all other targets (i.e., native builds), we use `[target.'cfg(not(target_arch = "wasm32"))'.dev-dependencies]`:

  - This section contains the standard versions of `criterion` and `proptest` with all their default features enabled for native benchmarking and testing.

## 2\. Rationale for Forked `proptest-arbitrary-interop`

The `twenty-first` crate uses property-based testing via the `proptest` library. To generate arbitrary instances of external types (like `num_bigint::BigUint`), we rely on the `proptest-arbitrary-interop` crate.

The official version of this crate on `crates.io` lacks `wasm32` support. To enable our property tests to run in a WebAssembly environment, we use a forked version of the crate. This fork specifically adds conditional dependencies for the `wasm32` target.

As shown in the fork's `Cargo.toml`, when compiling for `wasm32` it correctly pulls in `proptest` with its `std` feature enabled and adds `getrandom` with the `js` feature, which is necessary for the tests to compile and run successfully.

```toml
# In the forked proptest-arbitrary-interop/Cargo.toml
[target.'cfg(target_arch = "wasm32")'.dependencies]
proptest = { version = "1.2.0", default-features = false, features = ["std"] }
# convince getrandom to build for wasm target.
# note that there is also a flag in .cargo/config.toml
getrandom = { version = "0.2", features = ["js"] }
```

The dependency is specified in our `Cargo.toml` to point directly to the git repository containing the fork:

```toml
# In twenty-first/Cargo.toml
[dev-dependencies]
proptest-arbitrary-interop = { git = "https://github.com/dan-da/proptest-arbitrary-interop" }
```

### PR Raised to `proptest-arbitrary-interop`

[PR #3](https://github.com/graydon/proptest-arbitrary-interop/pull/3) has been raised to the [proptest-arbitrary-interop repo
](https://github.com/graydon/proptest-arbitrary-interop).  So if/when it is included in a new release on crates.io
twenty-first can again use the official version.

## 3\. Code Changes for Wasm Testing

The WebAssembly environment uses its own test harness (`wasm-pack test`) which ignores `#[test]` and looks for tests marked with `#[wasm_bindgen_test]` instead.
Yet our test suite should run in both the regular `cargo test` harness and the `wasm-pack test` harness.

## Dual Test Harness Compatibility

The core change is to conditionally add the `#[wasm_bindgen_test]` attribute to our existing tests, which already have the `#[test]` or `#[proptest]` attributes. This is accomplished using the `#[cfg_attr]` attribute, which applies another attribute only when a certain configuration is met.

This pattern allows a single test function to be recognized by both test harnesses:

1. The native harness sees `#[test]` (or `#[proptest]`) and runs the test.
2. The Wasm harness sees `#[wasm_bindgen_test]` and runs the test.
3. Each harness ignores the attribute it doesn't recognize.

Here is the specific pattern used throughout the codebase:

```rust
// This 'use' statement is only included when compiling Wasm tests.
#[cfg(all(test, target_arch = "wasm32"))]
use wasm_bindgen_test::wasm_bindgen_test;

// The #[wasm_bindgen_test] attribute is conditionally added only when building
// for the wasm32 target. This prevents compilation errors on native builds.
#[cfg_attr(all(test, target_arch = "wasm32"), wasm_bindgen_test)]
// The standard #[test] attribute is always present for the native test runner.
# [test]
fn my_dual_target_test() {
    // ... test logic ...
}

// The same pattern applies to proptests:
# [cfg_attr(all(test, target_arch = "wasm32"), wasm_bindgen_test)]
# [proptest]
fn my_dual_target_proptest() {
    // ... test logic ...
}

By using `#[cfg_attr(...)]` and a conditional `use` statement, we make our test suite compatible with the Wasm environment without breaking the existing native test workflow.
