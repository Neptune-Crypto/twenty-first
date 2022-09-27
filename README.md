# twenty-first

![GitHub CI](https://github.com/Neptune-Crypto/twenty-first/actions/workflows/main.yml/badge.svg)
![crates.io](https://img.shields.io/crates/v/twenty-first.svg)

A collection of cryptography primitives written in Rust.

## Content of this library

This library contains primarily the following cryptographic primitives:

- The Rescue-Prime hash function
  - An arithmetization-oriented hash function with a compact description in terms of AIR
  - [Rescue-Prime: a Standard Specification (SoK)](https://eprint.iacr.org/2020/1143.pdf)
  - [Anatomy of a STARK, Part 5: A Rescue-Prime STARK](https://neptune.cash/learn/stark-anatomy/rescue-prime/)
- FRI
  - Fast Reed-Solomon IOP of Proximity
  - [Anatomy of a STARK, Part 3: FRI](https://neptune.cash/learn/stark-anatomy/fri/)
- BFieldElement, XFieldElement
  - The prime-field type $\mathbb{F}_p$ where $p = 2^{64} - 2^{32} + 1$
  - The extension field $\mathbb{F}_p[x]/(x^3 - x + 1)$
  - [An efficient prime for number-theoretic transforms](https://cp4space.hatsya.com/2021/09/01/an-efficient-prime-for-number-theoretic-transforms/)
- NTT
  - Number Theoretic Transform (discrete Fast Fourier Transform)
  - [Anatomy of a STARK, Part 6: Speeding Things Up](https://neptune.cash/learn/stark-anatomy/faster/)
- Univariate and multivariate polynomials
- Merkle Trees, Merkle Mountain Ranges

This library also contains some proof-of-concept STARK implementations:

- [Rescue-Prime](https://neptune.cash/learn/stark-anatomy/) ([source code](./stark-rescue-prime))
- [Brainfuck](https://aszepieniec.github.io/stark-brainfuck/) ([source code](./stark-brainfuck))

## Release protocol

While twenty-first's version is `0.x.y`, releasing a new version:

1. Is the release backwards-compatible?
   Then the new version is `0.x.y+1`. Otherwise the new version is `0.x+1.0`.
2. Create a commit that increases `version = "0.x.y"` in twenty-first/Cargo.toml.
   The commit message should give a one-line summary of each release change.
3. Have a `v0.x.y` [git tag][tag] on this commit created. (`git tag v0.x.y [sha]`, `git push upstream --tags`)
4. Have this commit `cargo publish`ed on [crates.io][crates] and in GitHub [tags][tags].

[tag]: https://git-scm.com/book/en/v2/Git-Basics-Tagging
[tags]: https://github.com/Neptune-Crypto/twenty-first/tags
[crates]: https://crates.io/crates/twenty-first/versions

If you do not have the privilege to create git tags or run `cargo publish`, submit a PR and the merger will take care of these.
