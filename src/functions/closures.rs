// Closures are functions that can capture the enclosing environment.
// Rust's closures are anonymous functions that can capture values from
// the scope in which they're defined.
// - The input variables of closures are delimited by || instead of ().
// - Body delimination "{}" is optional for single expressions, otherwise mandatory.
// - Out environment variables can be captured
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::unused_unit)]
#![allow(unused_assignments)]

pub fn closures() {
    // local function
    fn function(i: i32) -> i32 {
        i + 1
    }

    // closures are anonymous.
    let closure_annotated = |i: i32| -> i32 { i + 1 };
    let closure_inferred = |i: i32| i + 1;

    let i = 1;
    println!("function {}", function(i));
    println!("closure_annotated: {}", closure_annotated(i));
    println!("closure_inferred: {}", closure_inferred(i));

    // A closure taking no arguments, returning an `i32`.
    // The return type is inferred.
    let one = || 1;
    println!("closure returning one: {}", one());

    use std::mem;
    let color: String = String::from("green");

    // `println!` only requires arguments by immutable reference,
    // so it doesn't impose anything more restrictive.
    let print = || println!("`color`: {}", color);

    // call the closure using the borrow.
    print();

    let _reborrow = &color;
    print();

    // After this move/reborrow, print cannot be called again
    let _color_moved = color;

    // print();

    // A closure to increment `count` could take either `&mut count` or `count`
    // but &mut count is less restrictive
    // A `mut` is needed on `inc` since a `mut` is stored inside
    let mut count = 0;
    let mut inc = || {
        count += 1;
        println!("count: {}", count);
    };

    inc();

    // let _reborrow = &count;
    inc();

    // The closure no longer needs to borrow `&mut count`. So it can be reborrowed.
    let _count_reborrowed = &mut count;

    // A non-copy type
    let movable = Box::new(3);

    let consume = || {
        println!("`movable`: {:?}", movable);
        mem::drop(movable);
    };

    consume();

    // Using `move` before vertical pipes forces closure to take ownership of captured variables
    let haystack = vec![1, 2, 3];

    let contains = move |needle| haystack.contains(needle);

    println!("{}", contains(&1));
    println!("{}", contains(&4));
    // haystack.contains(&4); // Illegal since ownership of `haystack` was _moved_ to the closure `contains`.
    // Rust's borrow checker will not allow the reuse of a binding after it has been moved.

    // Removing the `move` keyword from the closure definition will make the closure use an immutably borrowed
    // binding instead
    let haystack2 = vec![1, 2, 3];
    let contains_no_move = |needle| haystack2.contains(needle);
    println!("{}", contains_no_move(&1));
    println!("{}", contains_no_move(&4));

    println!("Length of haystack2 is: {}", haystack2.len());

    // When taking a closure as an argument, the closure's type
    // must be annotated using one of these traits: Fn, FnMut, FnOnce.
    // Fn: The closure captures by reference (&T)
    // FnMut: The closure captures by mutable reference (&mut T)
    // FnOnce: The closure captures by value (T)
    fn apply<F>(f: F)
    where
        F: FnOnce(),
    {
        f()
    }

    fn apply_to_3<F>(f: F) -> i32
    where
        F: Fn(i32) -> i32,
    {
        f(3)
    }

    let greeting = "hello";
    let mut farewell = "goodbye".to_owned();

    let diary = || {
        println!("I said {}.", greeting);

        // Mutation forces `farewell` to be captured by this closure
        farewell.push_str("!!!");
        println!("Then I screamed {}", farewell);
        println!("Now I can sleep. Zzz...");

        // Manually calling drop forces `farewell` to be captured by value.
        mem::drop(farewell);
    };

    apply(diary);

    let double = |x: i32| 2 * x;

    println!("3 doubled: {}", apply_to_3(double));
    println!("3 doubled: {}", apply_to_3(double));
    println!("3 doubled: {}", apply_to_3(double));

    // The Fn annotations indicate how the closures capture its arguments
    // It's however not clear to me what happens if the closure takes multiple arguments.
    // Or perhaps it is the environment bindings that should be considered, and not the
    // input arguments?
    // The Fn annotation means that everything in it is captured by reference, so all symbol
    // table ownerships are unchanged by a call to the closure, the `FnOnce` notation means that
    // the closure _may_ capture with borrowing, *and* moving.
    fn apply_two_args<F>(f: F, x: i32)
    where
        F: FnOnce(i32) -> (),
    {
        f(x)
    }
    let another_closure = |mut x: i32| {
        x = 2 * x;
    };
    apply_two_args(another_closure, 0);

    fn apply2<F>(f: F)
    where
        F: Fn(),
    {
        f();
    }
    let x = 7;

    // Capture `x` into an anonymous type and implement
    // `Fn` for it. Store it in `print`.
    let print = || println!("{}", x);

    apply2(print);

    // Function can return closures
}
