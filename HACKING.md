# Hacking on twenty-first

To compile code locally and run tests, the instructions below should help you get started.

## Setup on Ubuntu
 - curl -- `sudo apt install curl`
 - rustup -- `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` (installs `rustup`, `cargo`, `rustc` etc.)
 - source the rust environment `source $HOME/.cargo/env`
 - gnuplot -- `sudo apt install gnuplot`
 - build-essential (for `make`) -- `sudo apt install build-essential`
 - clone this repository
 - install `vscode`
 - in `vscode` install the plugin `rust-analyzer`
 - in `vscode` activate format-on-save via `File` > `Preferences` > `Settings` then check the box for "Format on Save"
 - install Criterion with `cargo install cargo-criterion`

## Setup on Windows
With a functioning version of `cargo`, compilation on Windows should just work out-of-the-box with `cargo build` etc. Installing `cargo`
might require you to install Visual Studio with some C++ support but the cargo installer for Windows should handle that.

## Cheatsheet
 - To test, use `cargo test [start_of_test_name]`. Or, for a complete and much slower build, run `make test`.
 - To generate and view API documentation, use `make doc`.
 - To run, use `make run`.
 - To lint, use `make lint`.
 - To format, use `make format`.
 - To check your code for errors, but skip code generation, use `make check`.  This should be faster than `make build`.
 - To build, use `make build`.
 - To install, use `make install`.
 - To run lint, compile, and run tests use `make all`.
 - To run the benchmarks and generate the benchmark report, use `make bench`, or run `cargo criterion --bench <specific-benchmark>`.

## Notes

The `Makefile` recipes set the flag `RUSTFLAGS=-Dwarnings` and this makes `cargo` recompile everything. For this reason, commands that are run through `make`/`Makefile` will compile to another target directory to prevent a complete recompilation in case you switch between using `cargo` and `make`. You can run `make all` for a comprehensive check that the code is in a good state.
