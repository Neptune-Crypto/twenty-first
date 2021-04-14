pub fn hello_world() {
    // println! is a macro that prints to stdout
    // Macros can be recognized by their names ending with an exclamation mark.
    // Macros are expanded to ASTs and not through string preprocessing as in C,
    // this avoids weird precedence behavior.
    // `println!` is defined in the module std::fmt which contains utilities for
    // formatting and printing strings.
    println!("Hello World!");
}
