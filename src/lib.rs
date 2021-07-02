use anyhow::{bail, Result};

mod conversion;
mod custom_types;
mod expressions;
pub mod fft;
mod flow_of_control;
mod formatted_print;
mod functions;
mod hello_world;
pub mod homomorphic_encryption;
mod primitives;
pub mod shared_math;
mod types;
pub mod util_types;
mod utils;

pub fn my_library_function() -> Result<()> {
    println!("\n\n*****Running my_private_function*****");
    my_private_function(42)?;
    println!("\n\n*****Running hello_world*****");
    hello_world::main::hello_world();
    println!("\n\n*****Running formatted_print*****");
    formatted_print::main::formatted_print();
    println!("\n\n*****Running primitives*****");
    println!("\n**primitives**");
    primitives::main::primitives();
    println!("\n**literals_and_operators**");
    primitives::literals_and_operators::literals_and_operators();
    println!("\n**tuples**");
    primitives::tuples::tuples();
    println!("\n**arrays_and_slices**");
    primitives::arrays_and_slices::arrays_and_slices();
    println!("\n\n*****Running custom_types*****");
    custom_types::structures::structures();
    println!("\n**enums**");
    custom_types::enums::enums();
    println!("\n**linked_list**");
    custom_types::linked_list::linked_list();
    println!("\n**constants**");
    custom_types::constants::constants();
    println!("\n\n*****Running types*****");
    types::casting::casting();
    types::literals::literals();
    types::inference::inference();
    types::aliasing::aliasing();
    println!("\n\n*****Running conversion*****");
    conversion::from_and_into::from_and_into();
    conversion::tryfrom_and_tryinto::tryfrom_and_tryinto();
    conversion::to_and_from_strings::to_and_from_strings();
    println!("\n\n*****Running expressions*****");
    expressions::expressions();
    println!("\n\n*****Running flow_of_controls*****");
    flow_of_control::if_else::if_else();
    flow_of_control::loop_example::loop_example();
    flow_of_control::fizz_buzz::fizz_buzz();
    flow_of_control::for_and_range::for_and_range();
    flow_of_control::match_example::match_example();
    println!("\n\n*****Running functions*****");
    functions::methods::methods();
    functions::closures::closures();
    println!("\n\n*****Homomorphic encryption*****");
    // homomorphic_encryption::test(); // Takes long
    println!("\n\n*****FFT*****");
    // fft::test(); // Takes long
    println!("\n\n*****blake3*****");
    println!("blake3(\"foobarbaz\") = {:?}", blake3::hash(b"foobarbaz"));
    println!("\n\n*****Merkle trees*****");
    let mt_input: Vec<&str> = vec!["Block 1", "Block 2"];
    // let mt: util_types::merkle_tree::MerkleTree<&str> =
    //     util_types::merkle_tree::MerkleTree::new_sha256_merkle_tree(mt_input);
    let mt: util_types::merkle_tree_vector::MerkleTreeVector<&str> =
        util_types::merkle_tree_vector::MerkleTreeVector::from_vec(&mt_input);
    println!("{:?}", mt);
    println!("root_hash = {:x?}", mt.get_root());
    Ok(())
}

fn my_private_function(n: i32) -> Result<()> {
    if n > 1000 {
        bail!("`n` cannot be larger than 1000!")
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal() {
        assert!(my_private_function(42).is_ok());
        assert!(my_private_function(9001).is_err());
    }
}
