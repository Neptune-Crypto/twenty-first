// Primitive types can be converted to each other through casting

// Rust addresses conversion between custom types (struct and enum) by
// the use of "traits".

use std::convert::From;

// derive(Debug) gives us a std way of printing for debugging purposes
// This printing is performed with "{:?}". The compiler can always(?)
// derive Debug for any type.
#[derive(Debug)]
struct Number {
    value: i32,
}

#[derive(Debug, Copy, Clone)]
struct Another {
    another: i128,
}

// static method for creating an instance of an object from another type
// This implements the trait (similar to interface) for a specific type
// Traits are like header files in C or like interfaces in C#. They contain
// the function signature (name and types) for a specific type.
// The trait "From" only defines *one* function: "from" but the trait takes
// a type parameter such that it can be implemented from and to any type.
impl From<i32> for Number {
    fn from(item: i32) -> Self {
        println!("PMD0");
        Number { value: item }
    }
}

impl From<i64> for Number {
    fn from(item: i64) -> Self {
        println!("PMD1");
        Number { value: item as i32 }
    }
}

impl From<i32> for Another {
    fn from(item: i32) -> Self {
        println!("PMD2");
        Another {
            another: item as i128,
        }
    }
}

impl From<Another> for Number {
    fn from(item: Another) -> Self {
        println!("PMD3");
        Number {
            value: item.another as i32,
        }
    }
}

pub fn from_and_into() {
    // static method call into method defined above
    let num = Number::from(30);
    println!("My number is {:?}", num);
    let num2 = Number::from(30i64);
    println!("My number is {:?}", num2);

    let int = 5;

    // also a static method call into the method defined above
    // The below is simply syntactic sugar for a call into the
    // static From function.
    // So `instance.into(): Type` := `Type::From(instance)`
    let num: Number = int.into();
    println!("My number is {:?}", num);

    // Create an Another object
    let another: Another = Another::from(31);
    println!("My another instance is {:?}", another);

    // We can only call `from` for types for which the `From` trait is declared
    // let another: Another = Another::from(30i64);
    // println!("My another instance is {:?}", another);

    // Create a Number object with the From<Another>
    let num3: Number = Number::from(another);
    println!("My number instance made from the From<Another>: {:?}", num3);

    // Create a Number object with the `into` syntax
    let num4: Number = another.into();
    println!(
        "My number instance made from the From<Another> with the into syntactic sugar: {:?}",
        num4
    );
}
