#![allow(unreachable_code)]
pub fn loop_example() {
    let mut count = 0u32;

    println!("Let's count until infinity!");

    // Infinite loop
    loop {
        count += 1;

        if count == 3 {
            println!("three");

            // Skip the rest of this iteration
            continue;
        }

        println!("{}", count);

        if count == 5 {
            println!("OK, that's enough");

            // Exit this loop
            break;
        }
    }

    // Loops can be annotated, just like in Perl
    'outer: loop {
        println!("Entered the outer loop");

        #[allow(clippy::never_loop)]
        #[allow(unused_labels)]
        'inner: loop {
            println!("Entered the inner loop");

            // This would break only the inner loop
            //break;

            // This breaks the outer loop
            break 'outer;
        }

        println!("This point will never be reached");
    }

    println!("Exited the outer loop");

    // Loops can return values
    let mut i = 0;

    let result = loop {
        i += 1;

        if i == 10 {
            break i * 2;
        }
    };

    assert_eq!(result, 20);
}
