#![allow(clippy::blacklisted_name)]
#![allow(clippy::match_ref_pats)]
#![allow(clippy::match_single_binding)]
#![allow(clippy::toplevel_ref_arg)]
#![allow(clippy::upper_case_acronyms)]
#![allow(dead_code)]
enum Color {
    // These 3 are specified solely by their name.
    Red,
    Blue,
    Green,
    // These likewise tie `u32` tuples to different names: color models.
    RGB(u32, u32, u32),
    HSV(u32, u32, u32),
    HSL(u32, u32, u32),
    CMY(u32, u32, u32),
    CMYK(u32, u32, u32, u32),
}

struct Foo {
    x: (u32, u32),
    y: u32,
}

pub fn match_example() {
    let number = 11;

    println!("Tell me about {}", number);
    match number {
        1 => println!("One!"),
        2 | 3 | 5 | 7 | 11 => println!("This is a prime"),
        13..=19 => println!("A teen"),
        _ => println!("Ain't special"),
    }

    let boolean = true;
    let binary = match boolean {
        false => 0,
        true => 1,
    };

    println!("{} -> {}", boolean, binary);

    // Destructuring tuples
    let triple = (0, -2, 3);
    // TODO ^ Try different values for `triple`

    println!("Tell me about {:?}", triple);
    // Match can be used to destructure a tuple
    match triple {
        // Destructure the second and third elements
        (0, y, z) => println!("First is `0`, `y` is {:?}, and `z` is {:?}", y, z),
        (1, ..) => println!("First is `1` and the rest doesn't matter"),
        // `..` can be the used ignore the rest of the tuple
        _ => println!("It doesn't matter what they are"),
        // `_` means don't bind the value to a variable
    }

    let color = Color::RGB(122, 17, 40);
    // TODO ^ Try different variants for `color`

    println!("What color is it?");
    // An `enum` can be destructured using a `match`.
    match color {
        Color::Red => println!("The color is Red!"),
        Color::Blue => println!("The color is Blue!"),
        Color::Green => println!("The color is Green!"),
        Color::RGB(r, g, b) => println!("Red: {}, green: {}, and blue: {}!", r, g, b),
        Color::HSV(h, s, v) => println!("Hue: {}, saturation: {}, value: {}!", h, s, v),
        Color::HSL(h, s, l) => println!("Hue: {}, saturation: {}, lightness: {}!", h, s, l),
        Color::CMY(c, m, y) => println!("Cyan: {}, magenta: {}, yellow: {}!", c, m, y),
        Color::CMYK(c, m, y, k) => println!(
            "Cyan: {}, magenta: {}, yellow: {}, key (black): {}!",
            c, m, y, k
        ),
        // Don't need another arm because all variants have been examined
    }

    // Create a pointer to the i32 4.
    let reference = &4;

    match reference {
        &val => println!("Got a value via destructuring: {:?}", val),
    }

    // To avoid the `&`, you dereference before matching.
    match *reference {
        val => println!("Got a value via dereferencing: {:?}", val),
    }

    // Create a value type binding
    let _not_a_reference = 3;

    // Create a reference type binding
    let ref _is_a_reference = 3;

    let value = 5;
    let mut mut_value = 6;

    // We use the `ref` keyword to create a reference.
    match value {
        ref r => println!("Got a reference to a value: {:?}", r),
    }

    // And we use `ref mut` to get a reference to a mutable value
    match mut_value {
        ref mut m => {
            *m += 10; // dereference and bind new value
            println!("We added 10 to mut_value. New value is: {:?}", m)
        }
    }

    // Similarly, a struct can be destructured as below
    let foo = Foo { x: (1, 2), y: 3 };

    match foo {
        Foo { x: (1, b), y } => println!("First of x is 1, b = {}, y = {} ", b, y),

        Foo { y: 2, x: i } => println!("y is 2, i = {:?}", i),

        Foo { y, .. } => println!("y = {}, and we don't care about x", y),
    }

    // Guards can be used instead of or as a supplement to pattern matching
    // A guard is an if-expression after the pattern matching binding.
    let pair = (-2, 2);

    println!("Tell me about {:?}", pair);
    match pair {
        (1, y) if y == 2 => println!("This is (1, 2)"),
        (x, y) if x == y => println!("These are twins"),
        (x, y) if x + y == 0 => println!("These are inverse"),
        (x, _) if x % 2 == 1 => println!("The first one is odd"),
        _ => println!("Nothing to say"),
    }

    // The compiler cannot verify that all cases have been covered for arbitrary expressions
    // Since the number is unsigned, all cases are covered by the two first guards
    let number: u8 = 4;
    match number {
        i if i == 0 => println!("Zero"),
        i if i > 0 => println!("Greater than zero"),
        _ => println!("Fell through"), // This should not be possible to reach
    }

    // The sigil '@' can be used to make a binding in match expression if
    // a match is made over e.g. a range.
    fn age() -> u32 {
        15
    }
    println!("Tell me what type of person you are");
    match age() {
        0 => println!("Zero years old"),
        n @ 1..=12 => println!("I am a child of age {}", n),
        n @ 13..=19 => println!("I am a teenager of age {}", n),
        n => println!("I am a grown-up of age {}", n),
    }

    // Match can be done on ranges of values inside nested structures
    #[allow(clippy::unnecessary_wraps)]
    fn some_age() -> Option<u32> {
        Some(43)
    }
    match some_age() {
        Some(n @ 42..=45) => println!("{}", n),
        Some(n) => println!("Another number: {}", n),
        None => println!("No number returned!"),
    }

    // if-let constructions
    let number = Some(7);
    let letter: Option<i32> = None;
    let emoticon: Option<i32> = None;

    // The if-let construct means:
    // if the binding destructures to the LHS, run this block of code
    if let Some(i) = number {
        println!("Matched {:?}!", i);
    }

    // The if-let construct can also be used with an else block
    if let Some(i) = letter {
        println!("Matched: {:?}", i);
    } else {
        println!("Didn't match a number.");
    }

    // else-if can also be used in if-let constructs
    let i_like_letters = false;
    if let Some(i) = emoticon {
        println!("Matched {:?}!", i);
    } else if i_like_letters {
        println!("Didn't match number, but I like letters");
    } else {
        println!("I don't like letters. Let's go with an emoticon")
    }

    // if-let can also be used for enums
    enum FooEnum {
        Bar,
        Baz,
        Qux(u32),
    }

    let a = FooEnum::Bar;
    let b = FooEnum::Baz;
    let c = FooEnum::Qux(150);

    // If a is of type FooEnum::Bar, then run this
    // block of code with this binding.
    if let FooEnum::Bar = a {
        println!("a is foobar");
    }

    if let FooEnum::Bar = b {
        println!("b is foobar");
    }

    if let FooEnum::Qux(value) = c {
        println!("c is Qux({})", value);
    }

    if let FooEnum::Qux(value @ 100..=200) = c {
        println!("c is Qux({})", value);
    }

    #[derive(Debug)]
    enum FooEnum2 {
        Bar,
    }

    // By default, Rust allows shadowing!!
    // Even though Eq or PartialEq is not implemented
    // here, we can use the if-let construct, even though the
    // equality operator cannot be used.
    let a = FooEnum2::Bar;
    #[allow(irrefutable_let_patterns)]
    if let FooEnum2::Bar = a {
        println!("a is foobar");
    }
    println!("{:?}", a);

    // **while-let**
    let mut optional = Some(0);

    // The following loop can be written better as a while-let loop
    // The loop keyword defines an infinite loop that can only be escaped
    // with a break statement or a process termination.
    #[allow(clippy::while_let_loop)]
    loop {
        match optional {
            Some(i) => {
                if i > 9 {
                    println!("Greater than 9, quit!");
                    optional = None;
                } else {
                    println!("`i` is `{:?}`. Try again.", i);
                    optional = Some(i + 1);
                }
            }
            None => break,
        }
    }

    // This does the same as the above loop
    // while-let runs as long as the pattern matching (typically type dependant)
    // is fulfilled. This way 14 lines are reduced to 9 lines. clippy should demand
    // the loop-match to be rewritten as a while-let structure.
    optional = Some(0);
    while let Some(i) = optional {
        if i > 9 {
            println!("Greater than 9, quit!");
            optional = None;
        } else {
            println!("`i` is `{:?}`. Try again.", i);
            optional = Some(i + 1);
        }
    }
}
