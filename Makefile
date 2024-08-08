
prog :=twenty-first

debug ?=

$(info debug is $(debug))
# Treat all warnings as errors
export RUSTFLAGS = -Dwarnings

# Set another target dir than default to avoid builds from `make`
# to invalidate cache from barebones use of `cargo` commands.
# The cache is cleared when a new `RUSTFLAGS` value is encountered,
# so to prevent the two builds from interfering, we use two dirs.
export CARGO_TARGET_DIR=./makefile-target

ifdef debug
  release :=
  target :=debug
  extension :=-debug
else
  release :=--release
  target :=release
  extension :=
endif

build:
	$(info RUSTFLAGS is $(RUSTFLAGS))
	cargo build $(release)
	rustup check
	@echo "Update with \`rustup install stable\` if needed."

doc:
	cargo doc --no-deps
	xdg-open "target/doc/twenty_first/index.html"

check:
	cargo check

ctags:
	# Do `cargo install rusty-tags`
	# See https://github.com/dan-t/rusty-tags
	rusty-tags vi

format:
	cargo fmt

install:
	cp target/$(target)/$(prog) ~/bin/$(prog)$(extension)

lint:
	cargo clippy --all-targets

# Get a stack trace upon kernel panic (may slow down implementation)
run: export RUST_BACKTRACE = 1
run:
	$(info RUSTFLAGS is $(RUSTFLAGS))
	cargo run

# Get a stack trace upon kernel panic (may slow down implementation)
test: export RUST_BACKTRACE = 1
test:
	$(info RUSTFLAGS is $(RUSTFLAGS))
	cargo nextest r
	cargo test --doc

bench:
	$(info RUSTFLAGS is $(RUSTFLAGS))
	cargo bench

bench-no-run:
	$(info RUSTFLAGS is $(RUSTFLAGS))
	cargo bench --no-run

bench-publish:
	cargo criterion --bench evaluation
	cargo criterion --bench interpolation
	cargo criterion --bench poly_mul
	cargo criterion --bench polynomial_coset

all: lint format build test bench-no-run

help:
	@echo "usage: make [debug=1]"

clean:
	@echo "      ._.  ██    ██  ███  ██ ██ █████    ████ ██    █████  ███  ██  ██"
	@echo "    c/-|   ███  ███ ██ ██ ████  ██      ██    ██    ██    ██ ██ ███ ██"
	@echo "   c/--|   ████████ █████ ███   ███     ██    ██    ███   █████ ██████"
	@echo "   /  /|   ██ ██ ██ ██ ██ ████  ██      ██    ██    ██    ██ ██ ██ ███"
	@echo " mmm ' '   ██    ██ ██ ██ ██ ██ █████    ████ █████ █████ ██ ██ ██  ██"
	@rm -rf target
