#![allow(clippy::no_effect)]
#![allow(unused_must_use)]
#![allow(path_statements)]

pub fn expressions() {
    // variable binding
    let x = 5;

    // expression;
    x;
    x + 1;
    15;

    // In Rust blocks are also expressions. So they can be used as values in assignments.
    // The last value in a block expression is the value of the LHS of the assignment
    // If the block ends with a ';', the block evaluates to the unit value, meaning that
    // assignments in Rust returns the unit value.
    let y = {
        let x_squared = x * x;
        let x_cube = x_squared * x;

        // This expression will be assigned to `y`
        x_cube + x_squared + x
    };

    let z = {
        2 * x;
    };

    println!("x is {:?}", x);
    println!("y is {:?}", y);
    println!("z is {:?}", z);
}
