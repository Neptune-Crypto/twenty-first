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
- Univariate polynomials
- Merkle Trees
- Merkle Mountain Ranges

## Wasm support

The `twenty-first` library can be built for WebAssembly. See the [dedicated readme](twenty-first/README-wasm32.md) for
further information.