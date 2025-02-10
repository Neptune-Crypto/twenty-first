# twenty-first

[![GitHub CI](https://github.com/Neptune-Crypto/twenty-first/actions/workflows/main.yml/badge.svg)](https://github.com/Neptune-Crypto/twenty-first/actions)
[![crates.io](https://img.shields.io/crates/v/twenty-first.svg)](https://crates.io/crates/twenty-first)
[![Coverage Status](https://coveralls.io/repos/github/Neptune-Crypto/twenty-first/badge.svg?branch=master)](https://coveralls.io/github/Neptune-Crypto/twenty-first?branch=master)

A collection of cryptography primitives written in Rust.

## Content of this library

This library contains primarily the following cryptographic primitives:

- The Tip5 hash function
  - [The Tip5 Hash Function for Recursive STARKs](https://eprint.iacr.org/2023/107)
- Lattice-crypto
  - arithmetic for the quotient ring $\mathbb{F}_ p[X] / \langle X^{64} + 1 \rangle$
  - arithmetic for modules over this quotient ring
  - a IND-CCA2-secure key encapsulation mechanism
  - [Lattice-Based Cryptography in Miden VM](https://eprint.iacr.org/2022/1041)
- `BFieldElement`, `XFieldElement`
  - The prime-field type $\mathbb{F}_p$ where $p = 2^{64} - 2^{32} + 1$
  - The extension field $\mathbb{F}_p[x]/(x^3 - x + 1)$
  - A codec trait for encoding and decoding structs as `Vec`s of `BFieldElement`
  - [An efficient prime for number-theoretic transforms](https://cp4space.hatsya.com/2021/09/01/an-efficient-prime-for-number-theoretic-transforms/)
- NTT
  - Number Theoretic Transform (discrete Fast Fourier Transform)
  - [Anatomy of a STARK, Part 6: Speeding Things Up](https://neptune.cash/learn/stark-anatomy/faster/)
- Univariate and multivariate polynomials
- Merkle Trees
- Merkle Mountain Ranges

## Release protocol

While twenty-first's version is `0.x.y`, releasing a new version:

1. Is the release backwards-compatible?
   Then the new version is `0.x.y+1`. Otherwise the new version is `0.x+1.0`.
2. Checkout the last commit on Mjolnir, and run `make bench-publish`. Save the benchmark's result
   and verify that there is no performance degradation.
3. Create a commit that increases `version = "0.x.y"` in twenty-first/Cargo.toml.
   The commit message should give a one-line summary of each release change. Include the benchmark
   result at the bottom.
4. Have a `v0.x.y` [git tag][tag] on this commit created. (`git tag v0.x.y [sha]`, `git push upstream --tags`)
5. Have this commit `cargo publish`ed on [crates.io][crates] and in GitHub [tags][tags].

[tag]: https://git-scm.com/book/en/v2/Git-Basics-Tagging
[tags]: https://github.com/Neptune-Crypto/twenty-first/tags
[crates]: https://crates.io/crates/twenty-first/versions

If you do not have the privilege to create git tags or run `cargo publish`, submit a PR and the merger will take care of these.
