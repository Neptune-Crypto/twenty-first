# twenty-first

A collection of cryptography functions written in Rust.

## Setup
### Ubuntu
 - rustup -- `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` (installs rustup, cargo, rustc etc.)
 - gnuplot -- `apt install gnuplot`
 - build-essential (for `make`) -- `apt install build-essential`

## Cheatsheet

 - To test, use `cargo test [start_of_test_name]`. Or, for a complete and much slower build, run `make test`.
 - To run, use `make run`.
 - To lint, use `make lint`.
 - To format, use `make format`.
 - To build, use `make build`.
 - To install, use `make install`.
 - To run lint, compile, run tests use `make all`. Note that this does *not* run install.
 - To run the benchmarks and generate the benchmark report, use `make bench`.

## Notes

The `Makefile` recipes set the flag `RUSTFLAGS=-Dwarnings` and this makes the recompilation **much** slower than without this flag, as `cargo` for some reason rebuilds the entire crate when this flag is set and a minor change is made in a test. So it is much faster to run the tests using cargo and then use the `make test` command before e.g. committing to ensure that the test build does not produce any warnings.
