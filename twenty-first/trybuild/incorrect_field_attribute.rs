use twenty_first::shared_math::bfield_codec::BFieldCodec;

#[derive(BFieldCodec)]
struct MyStruct {
    #[bfield_codec(skip)]
    a: u32,
}

fn main() {}
