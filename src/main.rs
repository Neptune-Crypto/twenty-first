use anyhow::Result;
use structopt::StructOpt;

/// See the [structopt
/// documentation](https://docs.rs/structopt/0.3.21/structopt) for more
/// information.
#[derive(StructOpt)]
#[structopt(
    name = "rust-template-project",
    about = "A template you can use for your own projects."
)]
struct Args {}

#[paw::main]
fn main(_args: Args) -> Result<()> {
    rust_tutorial::my_library_function()
}
