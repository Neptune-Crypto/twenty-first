# Changelog

All notable changes are documented in this file.
Lines marked “(!)” indicate a breaking change.

## [0.50.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.49.0..v0.50.0) – 2025-06-27

### ✨ Features

- Re-interpret xfe-slices as bfe-slices ([0870bf16](https://github.com/Neptune-Crypto/twenty-first/commit/0870bf16))

### ⚡️ Performance

- *(tip5)* Don't allocate in `pad_and_absorb_all` ([ecd5e369](https://github.com/Neptune-Crypto/twenty-first/commit/ecd5e369))

### ⚙️ Miscellaneous

- Add wasm feature flag ([1aeb2a21](https://github.com/Neptune-Crypto/twenty-first/commit/1aeb2a21))

### ♻️ Refactor

- (!) *(tip5)* Create dedicated module for Tip5 ([aedecf93](https://github.com/Neptune-Crypto/twenty-first/commit/aedecf93))

## [0.49.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.48.0..0.49.0) – 2025-06-18

### ✨ Features

- Impl Zeroize for SecretKey ([47853c00](https://github.com/Neptune-Crypto/twenty-first/commit/47853c00))
- (!) MerkleTree-specific Type Aliases ([fb6bb691](https://github.com/Neptune-Crypto/twenty-first/commit/fb6bb691))
- Compute Merkle root with minimal RAM usage ([ddc93cc7](https://github.com/Neptune-Crypto/twenty-first/commit/ddc93cc7))
- Build RAM-frugal Merkle root in parallel ([fbd308ae](https://github.com/Neptune-Crypto/twenty-first/commit/fbd308ae))
- Compute authentication structure from leafs ([4cfdf535](https://github.com/Neptune-Crypto/twenty-first/commit/4cfdf535))
- Compute Merkle root with little RAM ([360a2963](https://github.com/Neptune-Crypto/twenty-first/commit/360a2963))
- Set configuration programatically ([e9fae6aa](https://github.com/Neptune-Crypto/twenty-first/commit/e9fae6aa))

### 🐛 Bug Fixes

- (!) Correctly parse “negative” `BFieldElement`s ([85a6b307](https://github.com/Neptune-Crypto/twenty-first/commit/85a6b307))
- Avoid infinite iteration edge case ([27a4c826](https://github.com/Neptune-Crypto/twenty-first/commit/27a4c826))
- Validate auth struct node index parameters ([20a6827b](https://github.com/Neptune-Crypto/twenty-first/commit/20a6827b))
- Respect rayon's number of threads ([b4379e83](https://github.com/Neptune-Crypto/twenty-first/commit/b4379e83))

### ⚡️ Performance

- *(tip5)* Avoid one permutation in `hash_varlen` ([8c942b44](https://github.com/Neptune-Crypto/twenty-first/commit/8c942b44))
- *(MMR)* Use less RAM on MMR initialization ([048008fe](https://github.com/Neptune-Crypto/twenty-first/commit/048008fe))
- *(MerkleTre)* Parallelize over subtrees ([02720bef](https://github.com/Neptune-Crypto/twenty-first/commit/02720bef))
- *(MerkleTree)* Initialize in parallel ([9e96e848](https://github.com/Neptune-Crypto/twenty-first/commit/9e96e848))
- Speed up (i)NTT ([a72ff379](https://github.com/Neptune-Crypto/twenty-first/commit/a72ff379))

### 📚 Documentation

- Improve documentation for the various `hash_*` methods ([27a4325a](https://github.com/Neptune-Crypto/twenty-first/commit/27a4325a))
- *(polynomial)* Document public functions ([53c4b699](https://github.com/Neptune-Crypto/twenty-first/commit/53c4b699))

### ⚙️ Miscellaneous

- (!) Upgrade to rust edition 2024 ([23068289](https://github.com/Neptune-Crypto/twenty-first/commit/23068289))
- Upgrade dependencies ([cb2ff66c](https://github.com/Neptune-Crypto/twenty-first/commit/cb2ff66c))
- (!) *(ntt)* Delete unused code ([912d33ad](https://github.com/Neptune-Crypto/twenty-first/commit/912d33ad))
- (!) Remove feature `mock` ([4b5000ee](https://github.com/Neptune-Crypto/twenty-first/commit/4b5000ee))
- (!) Remove module `amounts` ([a5cbc605](https://github.com/Neptune-Crypto/twenty-first/commit/a5cbc605))
- (!) *(polynomial)* Remove fn `are_colinear_3` ([a39b6e67](https://github.com/Neptune-Crypto/twenty-first/commit/a39b6e67))

### ♻️ Refactor

- (!) *(merkle_tree)* Make helper method generic ([d5b94749](https://github.com/Neptune-Crypto/twenty-first/commit/d5b94749))

### ⏱️ Benchmark

- Bench all ways of computing a Merkle root ([29c312f5](https://github.com/Neptune-Crypto/twenty-first/commit/29c312f5))
- Bench authentication struct recomputation ([ee39add5](https://github.com/Neptune-Crypto/twenty-first/commit/ee39add5))
- *(ntt)* Refactor NTT benchmark ([11c0abf1](https://github.com/Neptune-Crypto/twenty-first/commit/11c0abf1))

## [0.48.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.45.0..v0.48.0) – 2025-02-11

### ✨ Features

- *(mmr)* Get auth-path node indices ([38102c84](https://github.com/Neptune-Crypto/twenty-first/commit/38102c84))
- *(Tip5)* Implement the Hasher trait ([9a6a74ee](https://github.com/Neptune-Crypto/twenty-first/commit/9a6a74ee))
- Implement `BFieldCodec` for `()` (“Unit”) ([9906d5ef](https://github.com/Neptune-Crypto/twenty-first/commit/9906d5ef))
- (!) Commit to leaf count when bagging peaks ([7bfc2eda](https://github.com/Neptune-Crypto/twenty-first/commit/7bfc2eda))

### 📚 Documentation

- Document limitation of `BFieldCodec` derive ([2dba3182](https://github.com/Neptune-Crypto/twenty-first/commit/2dba3182))

### ⚙️ Miscellaneous

- (!) *(deps)* Upgrade dependency `rand` to v0.9 ([6c598811](https://github.com/Neptune-Crypto/twenty-first/commit/6c598811))
- Drop “detailed build instructions” ([b65eb12e](https://github.com/Neptune-Crypto/twenty-first/commit/b65eb12e))

### ♻️ Refactor

- Don't require trait to invert element ([21562072](https://github.com/Neptune-Crypto/twenty-first/commit/21562072))

### ✅ Testing

- Add proptest for `auth_path_node_indices` ([437116f2](https://github.com/Neptune-Crypto/twenty-first/commit/437116f2))

## [0.45.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.44.1..v0.45.0) – 2025-01-13

### ✨ Features

- Provide sequential Merkle tree builder ([0a5a8b3c](https://github.com/Neptune-Crypto/twenty-first/commit/0a5a8b3c))

### 📚 Documentation

- *(BFieldCodec)* Mention dyn-incompatibility ([56dd14cf](https://github.com/Neptune-Crypto/twenty-first/commit/56dd14cf))

### ♻️ Refactor

- (!) Drop `MerkleTreeMaker` ([cf6a3983](https://github.com/Neptune-Crypto/twenty-first/commit/cf6a3983))

## [0.44.1](https://github.com/Neptune-Crypto/twenty-first/compare/v0.44.0..v0.44.1) – 2025-01-07

### ⚡️ Performance

- *(MmrSuccessorProof)* Use shorter auth paths ([b0e4e93b](https://github.com/Neptune-Crypto/twenty-first/commit/b0e4e93b))

### 📚 Documentation

- Describe Merkle Mountain Ranges ([9012fb07](https://github.com/Neptune-Crypto/twenty-first/commit/9012fb07))

### ⚙️ Miscellaneous

- Fix typos ([3e57923f](https://github.com/Neptune-Crypto/twenty-first/commit/3e57923f))

### ♻️ Refactor

- Simplify bagging of peaks ([ab4dc32e](https://github.com/Neptune-Crypto/twenty-first/commit/ab4dc32e))
- Clean up some MMR verification ([c1415cf3](https://github.com/Neptune-Crypto/twenty-first/commit/c1415cf3))

## [0.44.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.43.0..v0.44.0) – 2024-12-09

### ✨ Features

- Introduce `BFieldCodec` for 1-tuples ([3e048c6b](https://github.com/Neptune-Crypto/twenty-first/commit/3e048c6b))
- Introduce `BFieldCodec` for up to 12-tuples ([ec466e6d](https://github.com/Neptune-Crypto/twenty-first/commit/ec466e6d))
- Introduce `BFieldCodec` for signed integers ([34f37acb](https://github.com/Neptune-Crypto/twenty-first/commit/34f37acb))

### 🐛 Bug Fixes

- (!) *(BFieldCodec)* Avoid possibly wrong `as` casts ([1380e379](https://github.com/Neptune-Crypto/twenty-first/commit/1380e379))

### ♻️ Refactor

- (!) Remove deprecated functions ([f1e22eb1](https://github.com/Neptune-Crypto/twenty-first/commit/f1e22eb1))
- (!) Remove trait `AlgebraicHasher` ([78d42b51](https://github.com/Neptune-Crypto/twenty-first/commit/78d42b51))

## [0.43.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.41.0..v0.43.0) – 2024-10-29

### ✨ Features

- Generalize polynomial multiplication ([4235a802](https://github.com/neptune-crypto/twenty-first/commit/4235a802))
- Generalize polynomial scalar multiplication ([e71b2662](https://github.com/neptune-crypto/twenty-first/commit/e71b2662))
- Easily construct polynomials `x^n` ([2c1c08d8](https://github.com/neptune-crypto/twenty-first/commit/2c1c08d8))
- Allow evaluating polynomials in “value form” ([b1baee8a](https://github.com/neptune-crypto/twenty-first/commit/b1baee8a))
- Fast modular interpolate and extrapolate on coset ([2c491342](https://github.com/neptune-crypto/twenty-first/commit/2c491342))
- Batch and parallel versions of coset extrapolate ([9e7585d3](https://github.com/neptune-crypto/twenty-first/commit/9e7585d3))
- Convert BFieldElement from {u, i}size ([30c24c3b](https://github.com/neptune-crypto/twenty-first/commit/30c24c3b))
- Implement `BFieldCodec` for `u8` and `u16` ([8ec5f680](https://github.com/neptune-crypto/twenty-first/commit/8ec5f680))
- Add conversions from and to `BFieldElement` ([8e403080](https://github.com/neptune-crypto/twenty-first/commit/8e403080))
- Implement `Const{Zero, One}` for xfe ([50a9fb3a](https://github.com/neptune-crypto/twenty-first/commit/50a9fb3a))
- (!) Use `num_traits::Const{Zero, One}` ([60ed2b2b](https://github.com/neptune-crypto/twenty-first/commit/60ed2b2b))
- Introduce struct to help prove MMR succession ([0f521bb5](https://github.com/neptune-crypto/twenty-first/commit/0f521bb5))
- Implement `{Lower, Upper}Hex` for `Digest` ([65dc94d3](https://github.com/neptune-crypto/twenty-first/commit/65dc94d3))

### 🐛 Bug Fixes

- `Digest::try_from()` returns `NotCanonical` error ([a4daa23f](https://github.com/neptune-crypto/twenty-first/commit/a4daa23f))
- Fix edge-case failure in `fast_interpolate` ([04ff58a2](https://github.com/neptune-crypto/twenty-first/commit/04ff58a2))
- (!) Add field-length indicator to encoding of polynomial ([585b4a31](https://github.com/neptune-crypto/twenty-first/commit/585b4a31))
- *(mmr)* Don't panic on out-of-bounds MMR membership proof ([45dcedcb](https://github.com/neptune-crypto/twenty-first/commit/45dcedcb))
- (!) Fix platform-dependent digest encoding bug ([6e2c0127](https://github.com/neptune-crypto/twenty-first/commit/6e2c0127))
- Fix `structured_multiple_of_degree` ([4d867366](https://github.com/neptune-crypto/twenty-first/commit/4d867366))
- Fix MMR membership proof crash if peak list is too short ([52034e2e](https://github.com/neptune-crypto/twenty-first/commit/52034e2e))
- Various edge case bugs exposed ([5c30ef45](https://github.com/neptune-crypto/twenty-first/commit/5c30ef45))

### ⚡️ Performance

- Fast reduce with preprocessing ([10e763ec](https://github.com/neptune-crypto/twenty-first/commit/10e763ec))
- Integrate fast reduction into batch evaluate dispatcher ([7818ebe3](https://github.com/neptune-crypto/twenty-first/commit/7818ebe3))
- Use separate dispatcher threshold for `par_interpolate` ([3589b5e2](https://github.com/neptune-crypto/twenty-first/commit/3589b5e2))
- In `par_interpolate`, recurse to parallel version ([95a1f5a0](https://github.com/neptune-crypto/twenty-first/commit/95a1f5a0))
- (!) *(polynomial)* Optionally borrow coefficients ([8ed2445f](https://github.com/neptune-crypto/twenty-first/commit/8ed2445f))

### 📚 Documentation

- Add docstrings to some MMR methods ([b7244744](https://github.com/neptune-crypto/twenty-first/commit/b7244744))
- Improve some MMR-related documentation ([382fa32d](https://github.com/neptune-crypto/twenty-first/commit/382fa32d))
- Conform to clippy v1.80.0 indentation rules in doc strings ([eaa0a991](https://github.com/neptune-crypto/twenty-first/commit/eaa0a991))
- Drop instructions for installing leveldb ([938141e6](https://github.com/neptune-crypto/twenty-first/commit/938141e6))
- *(`MmrSuccessorProof`)* Add panics disclaimer ([9560c99a](https://github.com/neptune-crypto/twenty-first/commit/9560c99a))

### ⚙️ Miscellaneous

- Add ZerofierTree ([5be2c43a](https://github.com/neptune-crypto/twenty-first/commit/5be2c43a))
- Work around `nextest` bug ([0a71c3e7](https://github.com/neptune-crypto/twenty-first/commit/0a71c3e7))
- Conform to new rust version (v1.80.0) linting rules
- Update perflog ([4e0d872c](https://github.com/neptune-crypto/twenty-first/commit/4e0d872c))
- *(Makefile)* Use `nextest` for better timing results ([638b0ae3](https://github.com/neptune-crypto/twenty-first/commit/638b0ae3))
- Canonicalize `use` statements ([920c5db5](https://github.com/neptune-crypto/twenty-first/commit/920c5db5))
- Move blake3 crate to dev-dependencies ([d3d76616](https://github.com/neptune-crypto/twenty-first/commit/d3d76616))

### ♻️ Refactor

- (!) Remove `tree_m_ary.rs` ([b9264abb](https://github.com/neptune-crypto/twenty-first/commit/b9264abb))
- Separate parallel from sequential interpolate functions ([a0a4cc0e](https://github.com/neptune-crypto/twenty-first/commit/a0a4cc0e))
- Separate parallel from sequential `zerofier` methods ([881d4411](https://github.com/neptune-crypto/twenty-first/commit/881d4411))
- Drop `fast_divide` ([3e978b61](https://github.com/neptune-crypto/twenty-first/commit/3e978b61))
- (!) Make `Digest::LEN` an associated const ([cb53ad64](https://github.com/neptune-crypto/twenty-first/commit/cb53ad64))
- (!) Use `Const…` over `BFIELD_{ONE, ZERO}` ([6210f44c](https://github.com/neptune-crypto/twenty-first/commit/6210f44c))
- (!) Use `Const{Zero, One}` in `FiniteField` ([a84927d4](https://github.com/neptune-crypto/twenty-first/commit/a84927d4))
- (!) Drop generic type argument from MMR fn's and structs ([06f2c06d](https://github.com/neptune-crypto/twenty-first/commit/06f2c06d))
- (!) Copy (don't point to) auth path in `LeafMutation` ([07e423bc](https://github.com/neptune-crypto/twenty-first/commit/07e423bc))
- (!) Rename `MmrAccumulator`'s `new` to `new_from_leafs` ([750c057c](https://github.com/neptune-crypto/twenty-first/commit/750c057c))
- (!) Drop unused `get_height_from_leaf_index` ([38d78358](https://github.com/neptune-crypto/twenty-first/commit/38d78358))
- (!) Remove generic type from `MerkleTree` ([aa340ff6](https://github.com/neptune-crypto/twenty-first/commit/aa340ff6))
- *(polynomial)* Disallow negative exponents ([5f81f862](https://github.com/neptune-crypto/twenty-first/commit/5f81f862))
- (!) *(polynomial)* Generalize `evaluate()` ([8482f939](https://github.com/neptune-crypto/twenty-first/commit/8482f939))
- (!) Move canon check to `BFieldElement` ([7dafa32e](https://github.com/neptune-crypto/twenty-first/commit/7dafa32e))
- (!) Simplify interfaces of `ntt` and `intt` ([051fbfd0](https://github.com/neptune-crypto/twenty-first/commit/051fbfd0))
- (!) Simplify interfaces of `{i}ntt_noswap` ([4af2ffa3](https://github.com/neptune-crypto/twenty-first/commit/4af2ffa3))
- (!) *(polynomial)* Make `coefficients` private ([65914d5a](https://github.com/neptune-crypto/twenty-first/commit/65914d5a))

### ✅ Testing

- Ensure public types implement auto traits ([516ff9a7](https://github.com/neptune-crypto/twenty-first/commit/516ff9a7))
- Add tests for mismatching lengths in leaf index lists
- *(`MmrAccumulator`)* Make `Arbitrary` impl consistent ([18a2ff86](https://github.com/neptune-crypto/twenty-first/commit/18a2ff86))
- Add tests for edge cases ([5c30ef45](https://github.com/neptune-crypto/twenty-first/commit/5c30ef45))
- Add positive and negative tests for `MmrSuccessorProof`
- *(mmra_with_mps)* Replace slow sanity checks with a test ([eac5e7ac](https://github.com/neptune-crypto/twenty-first/commit/eac5e7ac))
- Reduce test runtimes for slow tests ([cbdaede2](https://github.com/neptune-crypto/twenty-first/commit/cbdaede2))

### ⏱️ Benchmark

- Add benchmark for polynomial modular reduction ([290a2f8a](https://github.com/neptune-crypto/twenty-first/commit/290a2f8a))
- Benchmark `par_interpolate`
- Formal power series inverse ([83a35ff6](https://github.com/neptune-crypto/twenty-first/commit/83a35ff6))
- Benchmark coset extrapolate ([3b909945](https://github.com/neptune-crypto/twenty-first/commit/3b909945))
- Reintroduce perflog-bench for evaluation ([80a5bec4](https://github.com/neptune-crypto/twenty-first/commit/80a5bec4))

### 🎨 Styling

- Harmonize divide interface ([8afef541](https://github.com/neptune-crypto/twenty-first/commit/8afef541))
- Idiomatic padding with zeros ([cb892a36](https://github.com/neptune-crypto/twenty-first/commit/cb892a36))

## [0.41.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.40.0..v0.41.0) – 2024-04-24

### ✨ Features

- Add hex encoding and decoding for Digest ([0f649786](https://github.com/Neptune-Crypto/twenty-first/commit/0f649786))
- Implement `BFieldCodec` for `Polynomial<FF>` ([3d994271](https://github.com/Neptune-Crypto/twenty-first/commit/3d994271))

### 📚 Documentation

- Provide example for `Polynomial::zerofier` ([0278f560](https://github.com/Neptune-Crypto/twenty-first/commit/0278f560))
- Document coefficient order of `Polynomial` ([c237e859](https://github.com/Neptune-Crypto/twenty-first/commit/c237e859))
- Motivate Merkle tree's constants ([53522191](https://github.com/Neptune-Crypto/twenty-first/commit/53522191))

### ♻️ Refactor

- (!) Don't return root hash on MMR's verify ([2a331dd3](https://github.com/Neptune-Crypto/twenty-first/commit/2a331dd3))
- Avoid environment variable collisions ([0340bf54](https://github.com/Neptune-Crypto/twenty-first/commit/0340bf54))
- (!) Make Merkle tree's `ROOT_INDEX` private ([d0dddeb5](https://github.com/Neptune-Crypto/twenty-first/commit/d0dddeb5))

### 🎨 Styling

- *(test)* Refactor `fast_evaluate` unit test ([182abc05](https://github.com/Neptune-Crypto/twenty-first/commit/182abc05))
- Avoid `match`ing on `bool` or `Option` ([729adc91](https://github.com/Neptune-Crypto/twenty-first/commit/729adc91))

## [0.40.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.39.0..v0.40.0) – 2024-04-16

### ✨ Features

- Make fast polynomial division complete ([0edb2902](https://github.com/Neptune-Crypto/twenty-first/commit/0edb2902))
- Add fast polynomial modulo for x^n ([21411be3](https://github.com/Neptune-Crypto/twenty-first/commit/21411be3))
- Add fast method for clean divisions ([f4a43450](https://github.com/Neptune-Crypto/twenty-first/commit/f4a43450))

### 🐛 Bug Fixes

- *(test)* Avoid test failure in edge case ([c7f753f0](https://github.com/Neptune-Crypto/twenty-first/commit/c7f753f0))

### ⚡️ Performance

- Parallelize `fast_evaluate` ([601ba840](https://github.com/Neptune-Crypto/twenty-first/commit/601ba840))
- Parallelize `fast_zerofier_inner` ([f0fdf92f](https://github.com/Neptune-Crypto/twenty-first/commit/f0fdf92f))
- Use `zerofier` without `fast` ([a9823c97](https://github.com/Neptune-Crypto/twenty-first/commit/a9823c97))
- (!) Remove special cases ([20242688](https://github.com/Neptune-Crypto/twenty-first/commit/20242688))
- Use fastest interpolation technique ([90c9cc89](https://github.com/Neptune-Crypto/twenty-first/commit/90c9cc89))
- Use fastest evaluation technique ([7a91f776](https://github.com/Neptune-Crypto/twenty-first/commit/7a91f776))
- Speed up `fast_evaluate` by smarter division ([d7edc17c](https://github.com/Neptune-Crypto/twenty-first/commit/d7edc17c))
- Use fastest field mul for polynomial “scale” ([d89da0da](https://github.com/Neptune-Crypto/twenty-first/commit/d89da0da))
- Reduce polynomial long divisions in xgcd ([ca4995fe](https://github.com/Neptune-Crypto/twenty-first/commit/ca4995fe))
- Make `Polynomial::scale` faster ([408838e7](https://github.com/Neptune-Crypto/twenty-first/commit/408838e7))
- Benchmark coset evaluate and interpolate ([070e6133](https://github.com/Neptune-Crypto/twenty-first/commit/070e6133))
- Collapse loops in Lagrange interpolation ([71c892f2](https://github.com/Neptune-Crypto/twenty-first/commit/71c892f2))
- Parallelize `Polynomial::fast_interpolate` ([504633c4](https://github.com/Neptune-Crypto/twenty-first/commit/504633c4))
- Add makefile `bench` target for publishing ([43c4d645](https://github.com/Neptune-Crypto/twenty-first/commit/43c4d645))
- Avoid allocation for scalar·polynomial ([87a21ee7](https://github.com/Neptune-Crypto/twenty-first/commit/87a21ee7))
- Speed up polynomial operations ([69a55edb](https://github.com/Neptune-Crypto/twenty-first/commit/69a55edb))
- *(bench)* Reduce sample size of some benchmarks ([bafe3620](https://github.com/Neptune-Crypto/twenty-first/commit/bafe3620))

### 📚 Documentation

- Document polynomial's `leading_coefficient()` ([0124d0bd](https://github.com/Neptune-Crypto/twenty-first/commit/0124d0bd))
- Explain extension field's “Shah polynomial” ([e9813ea4](https://github.com/Neptune-Crypto/twenty-first/commit/e9813ea4))
- Describe performance critical arguments ([9be21d6f](https://github.com/Neptune-Crypto/twenty-first/commit/9be21d6f))
- Add publishing instructions to README ([d8cd0465](https://github.com/Neptune-Crypto/twenty-first/commit/d8cd0465))

### ⚙️ Miscellaneous

- *(bench)* Remove oranges from comparison ([4f2fbd49](https://github.com/Neptune-Crypto/twenty-first/commit/4f2fbd49))

### ♻️ Refactor

- (!) Remove deprecated functions ([0ec7453a](https://github.com/Neptune-Crypto/twenty-first/commit/0ec7453a))
- *(bench)* Simplify zerofier cutoff search ([aaefc79f](https://github.com/Neptune-Crypto/twenty-first/commit/aaefc79f))
- (!) *(bench)* Add “smart” zerofier ([c56eb5e8](https://github.com/Neptune-Crypto/twenty-first/commit/c56eb5e8))
- (!) Improve polynomial multiplication names ([48370ec6](https://github.com/Neptune-Crypto/twenty-first/commit/48370ec6))
- *(test)* Use property test framework more ([bde30c79](https://github.com/Neptune-Crypto/twenty-first/commit/bde30c79))
- Remove superfluous threshold constants ([00be8309](https://github.com/Neptune-Crypto/twenty-first/commit/00be8309))
- (!) Don't “ref” `Copy` arg in `evaluate` ([5700e50b](https://github.com/Neptune-Crypto/twenty-first/commit/5700e50b))

### ✅ Testing

- Add “evaluation” benchmark ([478d4c5b](https://github.com/Neptune-Crypto/twenty-first/commit/478d4c5b))
- Benchmark polynomial multiplication ([37902ebc](https://github.com/Neptune-Crypto/twenty-first/commit/37902ebc))
- Other `evaluate` parallelization to bench ([e6454ed4](https://github.com/Neptune-Crypto/twenty-first/commit/e6454ed4))
- Add property test for polynomial division ([524d11b7](https://github.com/Neptune-Crypto/twenty-first/commit/524d11b7))
- Benchmark polynomial “scale” ([4a7f89c0](https://github.com/Neptune-Crypto/twenty-first/commit/4a7f89c0))
- Poly scale is equivalent in extension field ([d738d9b5](https://github.com/Neptune-Crypto/twenty-first/commit/d738d9b5))
- Un-ignore ignored tests ([ac9e5b96](https://github.com/Neptune-Crypto/twenty-first/commit/ac9e5b96))
- Use `proptest` framework more ([ec5b4943](https://github.com/Neptune-Crypto/twenty-first/commit/ec5b4943))

### 🎨 Styling

- Improve code style of polynomial division ([9110c732](https://github.com/Neptune-Crypto/twenty-first/commit/9110c732))
- Improve code to convert to `XFieldElement` ([9af85baf](https://github.com/Neptune-Crypto/twenty-first/commit/9af85baf))

## [0.39.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.38.0..v0.39.0) – 2024-04-08

### ✨ Features

- Add method `new` for `MerkleTree` ([50c634ef](https://github.com/Neptune-Crypto/twenty-first/commit/50c634ef))
- Make get_direct_path_indices() pub ([4f5f46b3](https://github.com/Neptune-Crypto/twenty-first/commit/4f5f46b3))
- Simplify construction of polynomials ([065dd753](https://github.com/Neptune-Crypto/twenty-first/commit/065dd753))
- Add `bfe_vec!` and `bfe_array!` macros ([d6bb353d](https://github.com/Neptune-Crypto/twenty-first/commit/d6bb353d))
- Add `xfe_vec!` and `xfe_array!` macros ([578b6472](https://github.com/Neptune-Crypto/twenty-first/commit/578b6472))
- Generalize `Polynomial::scale`'s offset ([ce4047b8](https://github.com/Neptune-Crypto/twenty-first/commit/ce4047b8))
- Make `{bfe, xfe}_{vec, array}` more powerful ([33a2003c](https://github.com/Neptune-Crypto/twenty-first/commit/33a2003c))

### ⚡️ Performance

- Remove `.clone()` in extended gcd ([b9a9e4af](https://github.com/Neptune-Crypto/twenty-first/commit/b9a9e4af))

### 📚 Documentation

- Add example to extended Euclidean algorithm ([5936c57b](https://github.com/Neptune-Crypto/twenty-first/commit/5936c57b))

### ⚙️ Miscellaneous

- Remove unused rusty_leveldb_vec* files ([890c451e](https://github.com/Neptune-Crypto/twenty-first/commit/890c451e))
- Remove `storage` and `sync` modules ([3775c51b](https://github.com/Neptune-Crypto/twenty-first/commit/3775c51b))
- Removed benches and unused dependencies ([57ce4535](https://github.com/Neptune-Crypto/twenty-first/commit/57ce4535))
- Clean up `polynomial.rs` ([8ca79022](https://github.com/Neptune-Crypto/twenty-first/commit/8ca79022))
- Simplify case handling for root of unity ([cca92f74](https://github.com/Neptune-Crypto/twenty-first/commit/cca92f74))
- Collect code coverage ([e0f32e0e](https://github.com/Neptune-Crypto/twenty-first/commit/e0f32e0e))
- Use `nextest` to test more things ([62088fda](https://github.com/Neptune-Crypto/twenty-first/commit/62088fda))
- Add code coverage badge ([6b33124e](https://github.com/Neptune-Crypto/twenty-first/commit/6b33124e))
- Clean up `XFieldElement` ([84a4176f](https://github.com/Neptune-Crypto/twenty-first/commit/84a4176f))
- (!) Remove constraint circuits ([0c757258](https://github.com/Neptune-Crypto/twenty-first/commit/0c757258))
- Remove unused dependencies ([0e3b8c98](https://github.com/Neptune-Crypto/twenty-first/commit/0e3b8c98))

### ♻️ Refactor

- Mmr tests no longer depend on storage layer ([#189](https://github.com/Neptune-Crypto/twenty-first/issues/189)) ([c4c0502f](https://github.com/Neptune-Crypto/twenty-first/commit/c4c0502f))
- Move mmra_with_mps into mmr_accumulator::util ([81573735](https://github.com/Neptune-Crypto/twenty-first/commit/81573735))
- (!) Simplify interface for evaluation ([a4641d35](https://github.com/Neptune-Crypto/twenty-first/commit/a4641d35))
- (!) Simplify interface for interpolation ([a321587e](https://github.com/Neptune-Crypto/twenty-first/commit/a321587e))
- (!) Simplify interface of fast division ([6113875a](https://github.com/Neptune-Crypto/twenty-first/commit/6113875a))
- (!) Simplify interface for fast poly mul ([5fde6ccd](https://github.com/Neptune-Crypto/twenty-first/commit/5fde6ccd))
- (!) Use `prelude` in `BFieldCodec` derive ([a3109fda](https://github.com/Neptune-Crypto/twenty-first/commit/a3109fda))
- (!) Remove multivariate polynomial support ([89c7cf54](https://github.com/Neptune-Crypto/twenty-first/commit/89c7cf54))
- (!) Remove timing reporter ([84b6e29c](https://github.com/Neptune-Crypto/twenty-first/commit/84b6e29c))
- Remove `utils` module ([e701395e](https://github.com/Neptune-Crypto/twenty-first/commit/e701395e))
- (!) Remove Blake3 ([e51c4afd](https://github.com/Neptune-Crypto/twenty-first/commit/e51c4afd))
- (!) Remove emojihash implementations ([94162f5f](https://github.com/Neptune-Crypto/twenty-first/commit/94162f5f))
- (!) Move de-facto MMR functionality there ([46531c4a](https://github.com/Neptune-Crypto/twenty-first/commit/46531c4a))
- (!) Remove unused helper methods ([4c22abee](https://github.com/Neptune-Crypto/twenty-first/commit/4c22abee))
- Deprecated methods mimicking built-ins ([e8c5b528](https://github.com/Neptune-Crypto/twenty-first/commit/e8c5b528))
- (!) Remove unused or shallow rng-methods ([547b7e51](https://github.com/Neptune-Crypto/twenty-first/commit/547b7e51))
- (!) Remove dyadic rationals ([c3584153](https://github.com/Neptune-Crypto/twenty-first/commit/c3584153))
- (!) Remove trait `FromVecu8` ([d041424f](https://github.com/Neptune-Crypto/twenty-first/commit/d041424f))
- (!) Rename `shared_math` into `math` ([ec87ee5e](https://github.com/Neptune-Crypto/twenty-first/commit/ec87ee5e))
- (!) Remove trait `New` ([e8d0c7e4](https://github.com/Neptune-Crypto/twenty-first/commit/e8d0c7e4))
- (!) Remove Sha3 implementation ([7175f7dd](https://github.com/Neptune-Crypto/twenty-first/commit/7175f7dd))
- Deprecate `log_2_ceil` ([42b2ca10](https://github.com/Neptune-Crypto/twenty-first/commit/42b2ca10))
- De-duplicate `BFieldCodec` test setup ([0dba559e](https://github.com/Neptune-Crypto/twenty-first/commit/0dba559e))

### ✅ Testing

- Add tests for `BFieldCodec` derive macro ([56cc6ce3](https://github.com/Neptune-Crypto/twenty-first/commit/56cc6ce3))
- Corrupting digest always corrupts digest ([db86606f](https://github.com/Neptune-Crypto/twenty-first/commit/db86606f))
- Tip5's trace is equivalent to permutation ([b548159d](https://github.com/Neptune-Crypto/twenty-first/commit/b548159d))

### 🎨 Styling

- Remove superfluous parentheses ([13feb388](https://github.com/Neptune-Crypto/twenty-first/commit/13feb388))

## [0.38.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.37.0..v0.38.0) – 2024-02-28

### ✨ Features

- (!) Use custom errors for fallible
  conversions ([8be60133](https://github.com/Neptune-Crypto/twenty-first/commit/8be60133))
- Simplify field element
  construction ([b974d1e3](https://github.com/Neptune-Crypto/twenty-first/commit/b974d1e3))

## [0.37.0](https://github.com/Neptune-Crypto/twenty-first/compare/v0.36.0..v0.37.0) – 2024-02-15

### ✨ Features

- Allow (read-only) access to nodes of Merkle tree ([61e697c7](https://github.com/Neptune-Crypto/twenty-first/commit/61e697c7))
- Derive `Eq` for `MerkleTree` ([1010d29e](https://github.com/Neptune-Crypto/twenty-first/commit/1010d29e))
- Optionally set parallelization cutoff through environment variable ([350cd8f5](https://github.com/Neptune-Crypto/twenty-first/commit/350cd8f5))
- Atomic lock event notification ([df8e99e8](https://github.com/Neptune-Crypto/twenty-first/commit/df8e99e8))
- Make use of lock events in storage ([7e64e05b](https://github.com/Neptune-Crypto/twenty-first/commit/7e64e05b))
- Implement `BFieldCodec` for `Box<_>` ([8bee834e](https://github.com/Neptune-Crypto/twenty-first/commit/8bee834e))
- Remove dependency `anyhow`, use custom errors instead ([01ddae66](https://github.com/Neptune-Crypto/twenty-first/commit/01ddae66))

### 🐛 Bug Fixes

- Add StorageSetter::index() method ([#170](https://github.com/Neptune-Crypto/twenty-first/issues/170)) ([5797095d](https://github.com/Neptune-Crypto/twenty-first/commit/5797095d))
- Make clippy v1.75 happy ([0374ce54](https://github.com/Neptune-Crypto/twenty-first/commit/0374ce54))
- Implement, don't derive `Arbitrary` for Merkle tree ([6c3f7be7](https://github.com/Neptune-Crypto/twenty-first/commit/6c3f7be7))
- Don't panic, return error if specified tree height is too large ([2bcdaf7a](https://github.com/Neptune-Crypto/twenty-first/commit/2bcdaf7a))
- Don't panic constructing Merkle tree with wrong number of leaves ([f584a611](https://github.com/Neptune-Crypto/twenty-first/commit/f584a611))
- Remove DB file on Drop, for windows. ([#176](https://github.com/Neptune-Crypto/twenty-first/issues/176)) ([d13af81e](https://github.com/Neptune-Crypto/twenty-first/commit/d13af81e))

### 📚 Documentation

- Document method `inclusion_proof_for_leaf_indices` ([dd7af2b9](https://github.com/Neptune-Crypto/twenty-first/commit/dd7af2b9))
- Document new structs and constants ([035f08a7](https://github.com/Neptune-Crypto/twenty-first/commit/035f08a7))
- Fix restore_or_new() placement in doctests ([#178](https://github.com/Neptune-Crypto/twenty-first/issues/178)) ([892015c2](https://github.com/Neptune-Crypto/twenty-first/commit/892015c2))
- Document `MerkleTree` building errors ([9f7ad17c](https://github.com/Neptune-Crypto/twenty-first/commit/9f7ad17c))
- Add changelog ([b765d144](https://github.com/Neptune-Crypto/twenty-first/commit/b765d144))
- Add `git-cliff` config file ([544ffae0](https://github.com/Neptune-Crypto/twenty-first/commit/544ffae0))

### ⚙️ Miscellaneous

- *(doc)* Make doc-strings use and adhere to max line length ([5bd3dd36](https://github.com/Neptune-Crypto/twenty-first/commit/5bd3dd36))
- Introduce struct `MerkleTreeInclusionProof` ([c4bd2bf0](https://github.com/Neptune-Crypto/twenty-first/commit/c4bd2bf0))
- Add windows, mac to github CI tasks ([03cde76c](https://github.com/Neptune-Crypto/twenty-first/commit/03cde76c))
- Re-export bfieldcodec_derive ([10afd069](https://github.com/Neptune-Crypto/twenty-first/commit/10afd069))
- Bump `bfieldcodec_derive` version ([a97464b8](https://github.com/Neptune-Crypto/twenty-first/commit/a97464b8))
- (!) Improve `SpongeHasher` trait ([5eb5dd39](https://github.com/Neptune-Crypto/twenty-first/commit/5eb5dd39))

### ♻️ Refactor

- Remove prefix `get_` from method's names ([cbf8ae5b](https://github.com/Neptune-Crypto/twenty-first/commit/cbf8ae5b))
- Move test-helper method to corresponding test module ([51c75f94](https://github.com/Neptune-Crypto/twenty-first/commit/51c75f94))
- Don't panic when requesting out-of-bounds leaf ([4e60a245](https://github.com/Neptune-Crypto/twenty-first/commit/4e60a245))
- *(test)* Turn ad-hoc property tests into `proptest` ([4bfdb2e9](https://github.com/Neptune-Crypto/twenty-first/commit/4bfdb2e9), [46b30c67](https://github.com/Neptune-Crypto/twenty-first/commit/46b30c67), [b546dffc](https://github.com/Neptune-Crypto/twenty-first/commit/b546dffc))
- *(test)* Translate more tests into `proptest`s ([b46307d2](https://github.com/Neptune-Crypto/twenty-first/commit/b46307d2), [f403dade](https://github.com/Neptune-Crypto/twenty-first/commit/f403dade))
- Avoid collecting into vector where only iterator is needed ([26a2dd35](https://github.com/Neptune-Crypto/twenty-first/commit/26a2dd35))
- Check all authentication paths for small trees ([cee2a9bf](https://github.com/Neptune-Crypto/twenty-first/commit/cee2a9bf))
- *(test)* Remove randomness from static tests ([ad75bab3](https://github.com/Neptune-Crypto/twenty-first/commit/ad75bab3))
- De-duplicate generation of static test Merkle trees ([d0e382f4](https://github.com/Neptune-Crypto/twenty-first/commit/d0e382f4))
- *(test)* Root from arbitrary number of digests belongs to MMR ([7d4121dc](https://github.com/Neptune-Crypto/twenty-first/commit/7d4121dc))
- (!) Improve `MerkleTree` implementation ([844f8723](https://github.com/Neptune-Crypto/twenty-first/commit/844f8723))
- Make leveldb::DB take &mut self ([2e984fe2](https://github.com/Neptune-Crypto/twenty-first/commit/2e984fe2))
- Add `prelude` to simplify `use`s downstream ([95d52acc](https://github.com/Neptune-Crypto/twenty-first/commit/95d52acc))
- Allow repeated sponge-absorption without mutating a list ([59944288](https://github.com/Neptune-Crypto/twenty-first/commit/59944288))
- Use `thiserror` to derive `BFieldCodecError` ([f0c5bf13](https://github.com/Neptune-Crypto/twenty-first/commit/f0c5bf13))
- (!) Integrate `Tip5State` into `Tip5` ([8300e6af](https://github.com/Neptune-Crypto/twenty-first/commit/8300e6af))

### 🛠 Build

- Prefix macro use statements with crate:: ([#180](https://github.com/Neptune-Crypto/twenty-first/issues/180)) ([7c7a61f2](https://github.com/Neptune-Crypto/twenty-first/commit/7c7a61f2))

### ✅ Testing

- Add `proptest` for verifying honestly generated inclusion proof ([54e606c7](https://github.com/Neptune-Crypto/twenty-first/commit/54e606c7))
- Add `proptest` for potential panics in `.height()`, `.num_leafs()` ([b61a1007](https://github.com/Neptune-Crypto/twenty-first/commit/b61a1007))
- Add `proptest` for verification failure against incorrect root ([e69d74f5](https://github.com/Neptune-Crypto/twenty-first/commit/e69d74f5))
- Add `proptest` verifying failure when supplying too many indices ([86238f47](https://github.com/Neptune-Crypto/twenty-first/commit/86238f47))
- Add `proptest` verifying failure when supplying too few indices ([3b01c4a0](https://github.com/Neptune-Crypto/twenty-first/commit/3b01c4a0))
- Test height and number of leaves for tree with one leaf ([55ef8463](https://github.com/Neptune-Crypto/twenty-first/commit/55ef8463))

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
