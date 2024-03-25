use twenty_first::math::bfield_codec::BFieldCodec;

#[derive(BFieldCodec)]
struct MyStruct {
    #[bfield_codec(ignore)]
    #[bfield_codec(ignore)]
    a: u32,
}

fn main() {}
