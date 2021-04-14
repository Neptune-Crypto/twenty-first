# rust-template-project

A template project for Rust applications. Suggestions for additions are very welcome.

## Features

 - Uses [structopt](https://docs.rs/structopt/0.3.21/structopt/) and
   [paw](https://docs.rs/paw/1.0.0/paw/) for handling command line arguments.
 - GitHub actions.
 - Uses [anyhow](https://docs.rs/anyhow/1.0.40/anyhow/index.html) for
   error-handling.

## Usage notes

 - To run, use `make run`.
 - To test, use `make test`.
 - To lint, use `make lint`.
 - To format, use `make format`.
 - To build, use `make build`.
 - To install, use `make install`.
 - To run lint, compile, run tests use `make all`. Note that this does *not* run install.
