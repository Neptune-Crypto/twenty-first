use std::cmp::Ordering;

pub fn if_else() {
    // In Rust if-else confitionals are expressions, and all branches must return the same type.
    let n = 5;

    // Clippy prefers to rewrite if-elseif-else to a `match` expression
    // where x.cmp is used. But Clippy accepts simple if-else conditionals.
    #[allow(clippy::comparison_chain)]
    if n < 0 {
        print!("{} is negative", n);
    } else if n > 0 {
        print!("{} is positive", n);
    } else {
        print!("{} is zero", n);
    }

    // Notice that a normal if-else is accepted. Only if-elseif produce warnings
    if n < 0 {
        println!();
    } else {
        println!("hi");
    }

    match 0.cmp(&n) {
        Ordering::Greater => print!(" negative"),
        Ordering::Less => print!(" positive"),
        Ordering::Equal => print!(" neither"),
    }

    let big_n = if n < 10 && n > -10 {
        println!(", and is a small number, increase ten-fold");

        10 * n
    } else {
        println!(", and is a big number, halve the number");
        n / 2
    }; // the `let` binding needs this semicolon

    // let big_n2 = match 10.cmp(&n)

    println!("{} -> {}", n, big_n);
}
