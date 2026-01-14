# Building and Testing `twenty-first` for wasm32

This document provides instructions on how to build and test the `twenty-first` crate for the `wasm32-unknown-unknown`
target, which allows the library to run in WebAssembly environments.

## What is wasm32?

WebAssembly (Wasm) is a binary instruction format for a stack-based virtual machine. The `wasm32-unknown-unknown` target
allows Rust code to be compiled into Wasm, enabling high-performance applications to run directly in web browsers and
other Wasm-compatible environments. This is ideal for bringing computationally intensive tasks, like the cryptographic
operations in `twenty-first`, to the web without sacrificing performance.

For more detailed information, see the official [WebAssembly website](https://webassembly.org/).

## Required Tools and Setup

To build and test for `wasm32`, you need to set up your Rust environment with the correct target and tooling.

### Add the `wasm32` Target

First, add the `wasm32-unknown-unknown` target to your Rust toolchain using `rustup`:

```shell
rustup target add wasm32-unknown-unknown
```

### Install `wasm-pack`

`wasm-pack` is the primary tool for building, testing, and publishing Rust-generated WebAssembly. It coordinates the
build process and handles the interaction with other tools like `wasm-bindgen`.

Install `wasm-pack` using `cargo`:

```shell
cargo install wasm-pack
```

### Install Node.js (for Testing)

Running the `wasm32` test suite requires a JavaScript runtime. `wasm-pack` uses Node.js for this purpose.

You must have Node.js v20 (LTS) or later installed. The `getrandom` crate, a dependency for our tests, requires the Web
Crypto API, which is stable and fully supported in all modern LTS releases of Node.js.

You can download Node.js from the [official Node.js website](https://nodejs.org/) or install it using a version manager
like `nvm`.

## Build and Test Commands

With the environment configured, you can now build and test the crate. Make sure your current working directory is
`twenty-first/twenty-first` before executing any of the commands below.

### Build the Crate

To compile the `twenty-first` crate for WebAssembly, run the following command:

```shell
wasm-pack build --target nodejs
```

This command compiles the crate and generates the necessary JavaScript bindings, placing the output in a `pkg/`
directory.

### Run Tests

To run the test suite for the `wasm32` target, use the `test` command from `wasm-pack`. This command will compile the
tests and execute them using your installed Node.js runtime.

```shell
wasm-pack test --node
```
