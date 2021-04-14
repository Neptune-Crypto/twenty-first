
prog :=rust-tutorial

debug ?=

$(info debug is $(debug))
export RUSTFLAGS = -Dwarnings

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

format:
	cargo fmt

install:
	cp target/$(target)/$(prog) ~/bin/$(prog)$(extension)

lint:
	cargo clippy

run:
	$(info RUSTFLAGS is $(RUSTFLAGS))
	cargo run

test:
	$(info RUSTFLAGS is $(RUSTFLAGS))
	cargo test

all: lint build test

help:
	@echo "usage: make [debug=1]"

clean:
	@rm -rf target
