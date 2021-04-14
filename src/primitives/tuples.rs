#![allow(unused_variables)]
use std::fmt;

fn reverse(pair: (i32, bool)) -> (bool, i32) {
    let (integer, boolean) = pair;

    (boolean, integer)
}

fn transpose(input: Matrix) -> Matrix {
    // First we destructure the tuple struct, this is done by declaring
    // the value constructor on the LHS of the declaration equality sign
    let Matrix(a, b, c, d) = input;
    Matrix(a, c, b, d)
}

// printing and format string is handled by a series of macros in std::fmt
// fmt::Debug uses the {:?} marker, formats text for debugging purposes, debug can be derived.
// fmt::Display uses the {} marker, formats text in a more user friendly fashion, Display cannot be derived.
#[derive(Debug)]
struct Matrix(f32, f32, f32, f32); // This is called a tuple struct

// Implements fmt::Display for the Matrix datatype defined above
impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "( {} {} )\n( {} {} )", self.0, self.1, self.2, self.3)
    }
}

pub fn tuples() {
    let long_tuple = (
        1u8, 2u16, 3u32, 4u64, -1i8, -2i16, -3i32, -4i64, 0.1f32, 0.2f64, 'a', true,
    );

    // Values can be extracted from the tuple using tuple indexing
    println!("long tuple first value: {}", long_tuple.0);
    println!("long tuple second value: {}", long_tuple.1);
    println!("long tuple third value: {}", long_tuple.2);
    println!("long tuple fourth value: {}", long_tuple.3);
    println!("long tuple fifth value: {}", long_tuple.4);
    println!("{}", long_tuple.5);
    println!("{}", long_tuple.6);
    println!("{}", long_tuple.7);
    println!("{}", long_tuple.8);
    println!("{}", long_tuple.9);
    println!("{}", long_tuple.10);
    println!("{}", long_tuple.11);
    println!("*** End tuple values ***");

    // Tuple can have tuple members
    let tuple_of_tuples = ((1u8, 2u16, 2u32), (4u64, -1i8), -2i16);

    // But long Tuples cannot be printed
    // Typle need to implement std::fmt::Debug to be formatted using `{:?}`.
    // let too_long_tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
    // println!("too long tuple: {:?}", too_long_tuple);
    // TODO ^ Uncomment the above 2 lines to see the compiler error

    let pair = (1, true);
    println!("pair is {:?}", pair);

    println!("the reversed pair is {:?}", reverse(pair));

    // To create one element tuples, the comma is required to tell them apart
    // from a literal surrounded by parentheses
    println!("one element tuple: {:?}", (5u32,));
    println!("just an integer: {:?}", (5u32));

    // Tuples can be destuctured to create bindings, just like in SML and in Haskell
    let tuple = (1, "hello", 4.5, true);
    let (a, b, c, d) = tuple;
    println!("{:?}, {:?}, {:?}, {:?}", a, b, c, d);

    let matrix = Matrix(1.1, 1.2, 2.1, 2.2);
    println!("{:?}", matrix); // print using Debug
    println!("Matrix: {}", matrix); // print using Display
    println!("Transposed: {}", transpose(matrix)); // print using Display
}
