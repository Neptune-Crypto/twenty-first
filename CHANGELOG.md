# Changelog

All notable changes are documented in this file.
Lines marked “(!)” indicate a breaking change.

## [0.36.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.34.0..v0.36.0) – 2023-12-22

### ✨ Features

- (!) Overhaul `BFieldCodec` and derive macro ([ac53378c](https://github.com/Neptune-Crypto/twenty-first/commit/ac53378c), [405d90b8](https://github.com/Neptune-Crypto/twenty-first/commit/405d90b8))

### 🐛 Bug Fixes

- Fix edge case bug in extended euclidean algorithm ([321b1948](https://github.com/Neptune-Crypto/twenty-first/commit/321b1948))

### ⚡️ Performance

- Improve performance of DB access ([#152](https://github.com/Neptune-Crypto/twenty-first/issues/152), [#157](https://github.com/Neptune-Crypto/twenty-first/issues/157)) ([035888f8](https://github.com/Neptune-Crypto/twenty-first/commit/035888f8), [e0cf7e64](https://github.com/Neptune-Crypto/twenty-first/commit/e0cf7e64))

### ♻️ Refactor

- Switch `LevelDB` dependency, refactor storage API ([#167](https://github.com/Neptune-Crypto/twenty-first/issues/167)) ([cf0a2040](https://github.com/Neptune-Crypto/twenty-first/commit/cf0a2040))
- (!) (de)serialize `BFieldElement` in canonical representation ([4accb46a](https://github.com/Neptune-Crypto/twenty-first/commit/4accb46a))

### ✅ Testing

- Use property testing framework `proptest` more extensively ([78b6993d](https://github.com/Neptune-Crypto/twenty-first/commit/78b6993d))

## [0.34.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.33.0..v0.34.0) – 2023-10-05

### ✨ Features

- Implement & derive `Arbitrary` for various `struct`s ([811b5131](https://github.com/Neptune-Crypto/twenty-first/commit/811b5131))

### ⚡️ Performance

- Add batch getters for database wrappers ([2bde8567](https://github.com/Neptune-Crypto/twenty-first/commit/2bde8567))

### ⚙️ Miscellaneous

- Upgrade dependencies ([5fd0ca71](https://github.com/Neptune-Crypto/twenty-first/commit/5fd0ca71))

### ♻️ Refactor

- (!) Require trait bound `Debug` for `SpongeState` ([e8ee9036](https://github.com/Neptune-Crypto/twenty-first/commit/e8ee9036))

## [0.33.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.32.1..v0.33.0) – 2023-09-25

### ✨ Features

- Make Tip5 permutation public ([ce5aabe2](https://github.com/Neptune-Crypto/twenty-first/commit/ce5aabe2)) ([#137](https://github.com/Neptune-Crypto/twenty-first/issues/137))

### 🐛 Bug Fixes

- Fix edge case bugs in (i)NTT ([d1389a32](https://github.com/Neptune-Crypto/twenty-first/commit/d1389a32), [904339d2](https://github.com/Neptune-Crypto/twenty-first/commit/904339d2))
- Fix bug in `BFieldCodec` derive, allowing empty structs ([164bdcf1](https://github.com/Neptune-Crypto/twenty-first/commit/164bdcf1))

### ⚙️ Miscellaneous

- Upgrade to rust v1.72.0 ([3d2fed5b](https://github.com/Neptune-Crypto/twenty-first/commit/3d2fed5b), [a6184bf5](https://github.com/Neptune-Crypto/twenty-first/commit/a6184bf5))
- Update dependencies ([de5abbc6](https://github.com/Neptune-Crypto/twenty-first/commit/de5abbc6), [5751f1a4](https://github.com/Neptune-Crypto/twenty-first/commit/5751f1a4))

### ♻️ Refactor

- (!) Avoid immutable borrows of copy types ([6fe514f4](https://github.com/Neptune-Crypto/twenty-first/commit/6fe514f4), [59501914](https://github.com/Neptune-Crypto/twenty-first/commit/59501914))

### ✅ Testing

- Use `proptest` framework for testing `BFieldElement` ([5670887b](https://github.com/Neptune-Crypto/twenty-first/commit/5670887b))

## [0.32.1](https://github.com/Neptune-Crypto/twenty-first/compare/v0.32.0..v0.32.1) – 2023-08-02

### 🐛 Bug Fixes

- `BFieldCodec` derive: ignore types of ignored fields ([40e75f3e](https://github.com/Neptune-Crypto/twenty-first/commit/40e75f3e))

## [0.32.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.31.0..v0.32.0) – 2023-08-01

### ✨ Features

- Derive `BFieldCodec` for `MmrMembershipProof` ([c8c8992b](https://github.com/Neptune-Crypto/twenty-first/commit/c8c8992b))

## [0.31.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.30.0..v0.31.0) – 2023-08-01

### ⚙️ Miscellaneous

- New version of `BFieldCodec` and derive macro (v0.4.0)
- Depend on new version of RustyLevelDB

## [0.30.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.28.0..v0.30.0) – 2023-07-04

### ✨ Features

- Use compact Merkle tree authentication structure ([40b0925b](https://github.com/Neptune-Crypto/twenty-first/commit/40b0925b))
- Add method to construct Merkle authentication paths from auth structure ([a51eac2e](https://github.com/Neptune-Crypto/twenty-first/commit/a51eac2e))

## [0.28.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.27.0..v0.28.0) – 2023-06-13

### ✨ Features

- Improve on the derive macro for `BFieldCodec`

### 🐛 Bug Fixes

- Use von Neumann sampling for perfect uniformity in `sample_indices()`

## [0.27.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.26.0..v0.27.0) – 2023-06-05

### ⚙️ Miscellaneous

- Update dependency `BFieldCodec`

## [0.26.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.25.1..v0.26.0) – 2023-06-02

### ✨ Features

- Add derive macro for `BFieldCodec`

### ♻️ Refactor

- Use use overwrite mode in Tip5 absorbing

## [0.25.1](https://github.com/Neptune-Crypto/twenty-first/compare/v0.25.0..v0.25.1) – 2023-05-24

### ✨ Features

- `BFieldCodec` stuff
- Vector analogue of `decode_field_length_prepended`

## [0.25.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.24.0..v0.25.0) – 2023-05-23

### ✨ Features
- Implement `BFieldCodec` for various structs

### ♻️ Refactor
- (!) Kill Rescue-Prime (Regular and Optimized)

## [0.24.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.23.0..v0.24.0) – 2023-05-22

### ✨ Features

- Implement codecs and other conversions

### ♻️ Refactor

- (!) Kill `Hashable`
- (!) Remove old stark crates

## [0.23.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.22.0..v0.23.0) – 2023-05-22

### ✨ Features

- Implement `PartialOrd` and `Ord` for `Digest`
- Pull `bfield_codec` from Triton VM

### ⚙️ Miscellaneous

- (!) Rename `Digest::vmhash()` to `hash()`
- (!) Drop Poseidon

## [0.22.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.21.3..v0.22.0) – 2023-05-17

### ✨ Features

- Implement VM-Hashing for `Digest`s
- Derive convenient traits for structs

### ♻️ Refactor

- (!) Rename `Digest` namespace

## [0.21.3](https://github.com/Neptune-Crypto/twenty-first/compare/[807a403c](https://github.com/Neptune-Crypto/twenty-first/commit/807a403c)..v0.21.3) – 2023-05-03

### ✨ Features

- Make `Digest::new` const ([6c374ec6](https://github.com/Neptune-Crypto/twenty-first/commit/6c374ec6))

## [0.21.2](https://github.com/Neptune-Crypto/twenty-first/compare/v0.21.1..[807a403c](https://github.com/Neptune-Crypto/twenty-first/commit/807a403c)) – 2023-05-03

### ✨ Features

- Add `Debug`, `Serialize`, and `Deserialize` to lattice kem types ([64835126](https://github.com/Neptune-Crypto/twenty-first/commit/64835126))

## [0.21.1](https://github.com/Neptune-Crypto/twenty-first/compare/v0.21.0..v0.21.1) – 2023-05-02

### 🐛 Bug Fixes

- Tolerate duplicate indices in Merkle authentication paths ([6fc0a982](https://github.com/Neptune-Crypto/twenty-first/commit/6fc0a982))

## [0.21.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.20.1..v0.21.0) – 2023-05-02

### ✨ Features

- Implement zero trait for module element ([00d6aa9e](https://github.com/Neptune-Crypto/twenty-first/commit/00d6aa9e))

### 🐛 Bug Fixes

- Fix generation of partial authentication paths

### ⚡️ Performance

- (!) Speed up & simplify Merkle tree verification ([6575492f](https://github.com/Neptune-Crypto/twenty-first/commit/6575492f))
- Improve generation and verification of partial authentication paths.

### ⚙️ Miscellaneous

- Add benchmark for verifying partial Merkle authentication paths
- Make overflowing adds, subs, and muls explicit ([33eb39bd](https://github.com/Neptune-Crypto/twenty-first/commit/33eb39bd))
- Update dependencies

### ♻️ Refactor

- (!) Remove salted Merkle trees ([275f20b3](https://github.com/Neptune-Crypto/twenty-first/commit/275f20b3))
- (!) Drop generic type parameter maker from Merkle tree ([4ee8f753](https://github.com/Neptune-Crypto/twenty-first/commit/4ee8f753))

## [0.20.1](https://github.com/Neptune-Crypto/twenty-first/compare/v0.20.0..v0.20.1) – 2023-04-26

### ✨ Features

- Convert between arrays of `BFieldElement`s and ciphertexts ([2e6ce1b7](https://github.com/Neptune-Crypto/twenty-first/commit/2e6ce1b7))

## [v0.20.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.19.3..v0.20.0) – 2023-04-25

### ♻️ Refactor

- (!) Provide randomness explicitly for lattice kem ([667c88f2](https://github.com/Neptune-Crypto/twenty-first/commit/667c88f2))

## [0.19.3](https://github.com/Neptune-Crypto/twenty-first/compare/v0.19.2..v0.19.3) – 2023-04-13

### ✨ Features

- Make RustyReader struct and its DB public

## [0.19.2](https://github.com/Neptune-Crypto/twenty-first/compare/v0.19.1..v0.19.2) – 2023-04-11

### ✨ Features

- Add canonical conversion between `XFieldElement`s and `Digest`s ([3b9586d7](https://github.com/Neptune-Crypto/twenty-first/commit/3b9586d7))
- Make Tip5 serializable ([8495d36f](https://github.com/Neptune-Crypto/twenty-first/commit/8495d36f))
- Add simple storage reader for generic databases ([c43b0278](https://github.com/Neptune-Crypto/twenty-first/commit/c43b0278))
- Circuit with auto-generated mds matrix multiplication procedure ([#105](https://github.com/Neptune-Crypto/twenty-first/issues/105))

## [0.19.1](https://github.com/Neptune-Crypto/twenty-first/compare/v0.19.0..v0.19.1) – 2023-03-17

### ✨ Features

- Add abstract Storage schema functionality for simulating containers

## [0.19.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.18.0..v0.19.0) – 2023-03-10

### ✨ Features

- Abstract out the data structure where the archival MMR's list of digests are stored: `StorageVec`
- Implement `StorageVec` for rusty-leveldb

### 🐛 Bug Fixes

- Panic when trying to invert 0 in extension field

### ⚡️ Performance

- Optimize + restructure some MMR helper functions

### ♻️ Refactor

- (!) Change MMR indices from `u128` to `u64`

## [0.18.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.17.0..v0.18.0) – 2023-03-01

### ✨ Features

- New functionality for databse vector
- Scalar and index sampling from sponges

## [0.17.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.16.0..v0.17.0) – 2023-02-24

### ✨ Features

- Lattice-based crypto routines

### ⚡️ Performance

- New MDS matrix for Tip5

## [0.16.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.15.0..v0.16.0) – 2023-02-10

### ✨ Features

- Make `SpongeHasher`'s `SpongeState` `Clone` ([8c45dc2](https://github.com/Neptune-Crypto/twenty-first/commit/8c45dc2))
- Add `absorb_repeatedly()` to `SpongeHasher` ([#89](https://github.com/Neptune-Crypto/twenty-first/issues/89))

### 🐛 Bug Fixes

- (!) Fix `Tip5`'s split_and_lookup ([#88](https://github.com/Neptune-Crypto/twenty-first/issues/88))

### ⚙️ Miscellaneous

- Fix linter error: zero_prefixed_literal ([ce74205](https://github.com/Neptune-Crypto/twenty-first/commit/ce74205))
- Add fixme/type hint to prevent LSP from jerking out ([7ffc187](https://github.com/Neptune-Crypto/twenty-first/commit/7ffc187))

### ♻️ Refactor

- (!) Move `SpongeHasher`'s `sample_indices()` and `sample_weights()` to Triton VM ([8c45dc2](https://github.com/Neptune-Crypto/twenty-first/commit/8c45dc2))
- (!) Change `SpongeHasher`'s `absorb_init()` to `init()` ([#89](https://github.com/Neptune-Crypto/twenty-first/issues/89))

## [0.15.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.14.1..v0.15.0) – 2023-01-31

### ✨ Features

- Standardize `Hashable` for `bool`, `u32`, `u64`, `u128`, `BFE`, `XFE`, `Digest` ([ae1a837](https://github.com/Neptune-Crypto/twenty-first/commit/ae1a837))

### ♻️ Refactor

- (!) Make domain separation for hash10 apply to all capacity elements ([#86](https://github.com/Neptune-Crypto/twenty-first/issues/86), [7abb3f6](https://github.com/Neptune-Crypto/twenty-first/commit/7abb3f6))
- (!) Replace old `AlgebraicHasher` with `AlgebraicHasher: SpongeHasher` ([#84](https://github.com/Neptune-Crypto/twenty-first/issues/84), [#85](https://github.com/Neptune-Crypto/twenty-first/issues/85), [174a4da](https://github.com/Neptune-Crypto/twenty-first/commit/174a4da))
- (!) Remove references to `self` from Tip5's implementation ([518bd70](https://github.com/Neptune-Crypto/twenty-first/commit/518bd70))

## [0.14.1](https://github.com/Neptune-Crypto/twenty-first/compare/v0.14.0..v0.14.1) – 2023-01-20

### 🐛 Bug Fixes

- `RescuePrimeRegularState::new()` does not use `state` ([868854a](https://github.com/Neptune-Crypto/twenty-first/commit/868854a))

## [0.14.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.11.0..v0.14.0) – 2023-01-18

### ✨ Features

- Add `SpongeHasher` and `AlgebraicHasherNew` ([#83](https://github.com/Neptune-Crypto/twenty-first/issues/83))

## [0.11.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.10.0..v0.11.0) – 2023-01-12

### ✨ Features

- Tip5 hash function ([#72](https://github.com/Neptune-Crypto/twenty-first/issues/72))
- Allow for generating random U32s<N>s ([#80](https://github.com/Neptune-Crypto/twenty-first/issues/80))
- Add a leaf_index -> Merkle tree index function ([5e5b863](https://github.com/Neptune-Crypto/twenty-first/commit/5e5b863))
- Add Right Lineage Length function ([#76](https://github.com/Neptune-Crypto/twenty-first/issues/76))
- Add MMR helper function right_ancestor_count ([139e462](https://github.com/Neptune-Crypto/twenty-first/commit/139e462))

### ⚡️ Performance

- Inline tip5-related functions for a small speedup ([#79](https://github.com/Neptune-Crypto/twenty-first/issues/79))
- Optimize non_leaf_nodes_left ([4c0739d](https://github.com/Neptune-Crypto/twenty-first/commit/4c0739d), [77092c8](https://github.com/Neptune-Crypto/twenty-first/commit/77092c8))
- Reduce time it takes to run all benchmarks ([1c8a768](https://github.com/Neptune-Crypto/twenty-first/commit/1c8a768))

### ♻️ Refactor

- Rewrite some MMR functions to be mode TASM-friendly ([df81f4b](https://github.com/Neptune-Crypto/twenty-first/commit/df81f4b))
- (!) Update MMR's leaf index -> MT index conversion function ([b097ce2](https://github.com/Neptune-Crypto/twenty-first/commit/b097ce2))
- Change MMR functions to use Merkle tree index logic ([c4a1b8b](https://github.com/Neptune-Crypto/twenty-first/commit/c4a1b8b), [1651035](https://github.com/Neptune-Crypto/twenty-first/commit/1651035))
- Make non_leaf_nodes_left more friendly to a TASM compilation ([a31cc70](https://github.com/Neptune-Crypto/twenty-first/commit/a31cc70))
- (!) Delete deprecated right_child_and_height ([b55fcad](https://github.com/Neptune-Crypto/twenty-first/commit/b55fcad))
- Remove all use of deprecated is_right_child_and_height ([65318d0](https://github.com/Neptune-Crypto/twenty-first/commit/65318d0))
- Simplify node-traversal logic for MMR	([5b2e93e](https://github.com/Neptune-Crypto/twenty-first/commit/5b2e93e))
- (!) Change height of MMRs to be of type u32 ([8eeb800](https://github.com/Neptune-Crypto/twenty-first/commit/8eeb800))
- Adapt MMR function for TASM snippet implementation ([2fb0ec9](https://github.com/Neptune-Crypto/twenty-first/commit/2fb0ec9))
- (!) Move STARK tutorial-only code into 'stark-shared' crate ([#82](https://github.com/Neptune-Crypto/twenty-first/issues/82))
- (!) Minor improvements ([#77](https://github.com/Neptune-Crypto/twenty-first/issues/77), [e3a2855](https://github.com/Neptune-Crypto/twenty-first/commit/e3a2855), [87947f8](https://github.com/Neptune-Crypto/twenty-first/commit/87947f8), [9304558](https://github.com/Neptune-Crypto/twenty-first/commit/9304558))
- Remove custom `Hash` and `PartialEq` implementations ([5b6649e](https://github.com/Neptune-Crypto/twenty-first/commit/5b6649e))
- (!) Remove deprecated MMR helper function to find peak index ([71ec7a6](https://github.com/Neptune-Crypto/twenty-first/commit/71ec7a6))

## [0.10.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.9.0..v0.10.0) – 2022-12-22

### ⚡️ Performance

- Avoid all branching when adding `BFieldElement`s ([#70](https://github.com/Neptune-Crypto/twenty-first/issues/70))

### ♻️ Refactor

- (!) Make the Rescue Prime round constants and matrices `const BFieldElement`

## [0.9.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.8.0..v0.9.0) – 2022-12-20

### ✨ Features

- Add `raw` and `from_raw` methods to `BFieldElement`

### ♻️ Refactor

- Switch to Montgomery representation for `BFieldElement`

### ✅ Testing

- Add tests for `BFieldElement`

## [0.8.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.7.2..v0.8.0) – 2022-12-19

### ✨ Features

- Add `::sample_weights()` to AlgebraicHasher, implement with `::hash_pair()` ([#66](https://github.com/Neptune-Crypto/twenty-first/issues/66))

### ⚡️ Performance

- Make `sample_weights()` faster in tasm by changing `XFieldElement::sample()` ([#66](https://github.com/Neptune-Crypto/twenty-first/issues/66))

## [0.7.2](https://github.com/Neptune-Crypto/twenty-first/compare/v0.7.1..v0.7.2) – 2022-11-23

### ✨ Features

- Emojihash trait ([#64](https://github.com/Neptune-Crypto/twenty-first/issues/64))

## [0.7.1](https://github.com/Neptune-Crypto/twenty-first/compare/v0.7.0..v0.7.1) – 2022-11-22

### ✨ Features

- Add batch-version of fast_interpolate, `batch_fast_interpolate`

### ⚡️ Performance

- Make existing `fast_interpolate` slightly faster

## [0.7.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.6.0..v0.7.0) – 2022-11-22

### ✨ Features

- Add `impl AsRef<[u32; N]> for U32s<N>` ([#59](https://github.com/Neptune-Crypto/twenty-first/issues/59))
- Add `Digest::emojihash(&self)` ([#62](https://github.com/Neptune-Crypto/twenty-first/issues/62))
- Add `impl TryFrom<&[BFieldElement]> for Digest, XFieldElement` ([#61](https://github.com/Neptune-Crypto/twenty-first/issues/61))

### 📚 Documentation

- Add notes on how to get started working on repo ([97cb44f9](https://github.com/Neptune-Crypto/twenty-first/commit/97cb44f9))

### ⚙️ Miscellaneous

- (!) Remove `impl From<Digest> for [u8; MSG_DIGEST_SIZE_IN_BYTES]` ([#62](https://github.com/Neptune-Crypto/twenty-first/issues/62))
- (!) Remove `BFieldElement::from_byte_array()` ([#62](https://github.com/Neptune-Crypto/twenty-first/issues/62))
- (!) Remove `impl Default for XFieldElement` ([#62](https://github.com/Neptune-Crypto/twenty-first/issues/62))

## [0.6.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.5.0..v0.6.0) – 2022-11-17

### ✨ Features

- (!) Parameterise `MerkleTree` with `M: MerkleTreeMaker<H>` ([#57](https://github.com/Neptune-Crypto/twenty-first/issues/57))

### ♻️ Refactor

- (!) Remove `simple_hasher.rs` ([#58](https://github.com/Neptune-Crypto/twenty-first/issues/58))
- (!) Remove deprecated auxiliary functions

## [0.5.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.4.0..v0.5.0) – 2022-11-15

### ✨ Features

- Add `transpose()` ([#55](https://github.com/Neptune-Crypto/twenty-first/issues/55))

### ♻️ Refactor

- (!) Remove unused functions ([#55](https://github.com/Neptune-Crypto/twenty-first/issues/55))

## [0.4.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.3.2..v0.4.0) – 2022-11-14

### ✨ Features

- `XFieldElement`: Add `EXTENSION_DEGREE` constant ([#54](https://github.com/Neptune-Crypto/twenty-first/issues/54))
- `XFieldElement`, `BFieldElement`: Implement `Add`, `Mul`, `Sub` ([#54](https://github.com/Neptune-Crypto/twenty-first/issues/54))
- `TimingReporter`: Make measurements easier to read ([#49](https://github.com/Neptune-Crypto/twenty-first/issues/49))

## [0.3.2](https://github.com/Neptune-Crypto/twenty-first/compare/v0.3.1..v0.3.2) – 2022-11-09

### ✨ Features

- Implement `Hash` for `MPolynomial` structs to be hashed

## [0.3.1](https://github.com/Neptune-Crypto/twenty-first/compare/v0.3.0..v0.3.1) – 2022-10-21

### ✨ Features

- Implement `From<BFieldElement>` for `XFieldElement`

### ♻️ Refactor

- Rename `PFElem` type parameter into `FF`

## [0.3.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.2.0..v0.3.0) – 2022-10-20

### ✨ Features

- Add `.inverse_or_zero()` to {B, X}FieldElement ([#35](https://github.com/Neptune-Crypto/twenty-first/issues/35))
- Add `.emojihash()` for {B, X}FieldElement
- Implement `xgcd()` generically ([#40](https://github.com/Neptune-Crypto/twenty-first/issues/40))
- Add `Digest` struct ([#44](https://github.com/Neptune-Crypto/twenty-first/issues/44))
- Make `blake3::Hasher` an instance of `AlgebraicHasher` ([#44](https://github.com/Neptune-Crypto/twenty-first/issues/44))

### 🐛 Bug Fixes

- (!) Add `+ MulAssign<BFieldElement>` to `ntt()` / `intt()` ([#41](https://github.com/Neptune-Crypto/twenty-first/issues/41))

### ⚡️ Performance

- Speed up NTT for XFieldElements ([#41](https://github.com/Neptune-Crypto/twenty-first/issues/41))

### 📚 Documentation

- Add release protocol, library overview to README.md

### ⚙️ Miscellaneous

- Add `.mailmap` for cleaner git logs
- Remove `split-debuginfo = '...'`
- Lots of cleaning up code
- Add various trait instances ([#42](https://github.com/Neptune-Crypto/twenty-first/issues/42))

### ♻️ Refactor

- Move stark-brainfuck and stark-rescue-prime to separate crates ([#38](https://github.com/Neptune-Crypto/twenty-first/issues/38))
- (!) Remove `GetRandomElements` in favor of standard library ([#42](https://github.com/Neptune-Crypto/twenty-first/issues/42))
- (!) Remove `GetGeneratorDomain` trait (it was already unused) ([#42](https://github.com/Neptune-Crypto/twenty-first/issues/42))
- (!) Change some MMR batch functions to take `&mut` membership proofs ([#43](https://github.com/Neptune-Crypto/twenty-first/issues/43))
- (!) Replace `simple_hasher::Hasher` with `AlgebraicHasher` ([#44](https://github.com/Neptune-Crypto/twenty-first/issues/44))
- (!) Simplify `Hashable` so its parameters are more fixed ([#44](https://github.com/Neptune-Crypto/twenty-first/issues/44))


## [0.2.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.1.5..v0.2.0) – 2022-09-13

### ✨ Features

- (!) Fix Rescue-Prime ([#25](https://github.com/Neptune-Crypto/twenty-first/issues/25))

## [0.1.5](https://github.com/Neptune-Crypto/twenty-first/releases/tag/v0.1.5) – 2022-09-13

### ✨ Features

- (!) Simplify Lagrange interpolation function interface

### ⚡️ Performance

- Add faster Lagrange interpolation and benchmarks
