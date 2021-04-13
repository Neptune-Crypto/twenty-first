fn main() {
    let logical: bool = true;

    let a_float: f64 = 1.0;
    let an_integer = 5i32;

    let default_float = 3.0; // `f64`
    let default_integer = 7; // `i32`

    // A type can also be inferred from context
    let mut inferred_type = 12;
    inferred_type = 4294967296i64;

    // A mutable variables value can be changed.
    let mut mutable = 12;
    mutable = 21;

    // But its type cannot be changed.
    // mutable = true;

    // Variables can be overwritten with shadowing.
    let mutable = true;
}