use twenty_first;
use twenty_first::math::bfield_codec::BFieldCodec;

#[derive(BFieldCodec)]
struct MyStruct {
    #[bfield_codec()]
    a: u32,
}

fn main() {}
