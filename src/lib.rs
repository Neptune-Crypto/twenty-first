use anyhow::{bail, Result};

mod custom_types;
mod formatted_print;
mod hello_world;
mod primitives;

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
