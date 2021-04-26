pub fn for_and_range() {
    // `n` will take the values: 1, 2, ..., 100 in each iteration
    for n in 1..101 {
        if n % 15 == 0 {
            println!("fizzbuzz");
        } else if n % 3 == 0 {
            println!("fizz");
        } else if n % 5 == 0 {
            println!("buzz");
        } else {
            println!("{}", n);
        }
    }

    // Below three different versions of an iterator method is used

    // The iter method leaves the collection untouched
    let names = vec!["Bob", "Frank", "Ferris"];

    // here, iter :: vec<&str> -> &&str, so it gives a pointer to each element
    // iter :: &self -> Iter<T>
    for name in names.iter() {
        match *name {
            "Ferris" => println!("There is a rustacean among us!"),
            // TODO ^ Try deleting the & and matching just "Ferris"
            _ => println!("Hello {}", name),
        }
    }
    println!("names: {:?}", names);

    // The into_iter method consumes the collection
    let names2 = vec!["Bob", "Frank", "Ferris"];

    // here, into_iter :: vec<&str> -> &str, so it returns each element, not a pointer to it
    // into_iter :: self -> Iter<T>. Since into_iter takes `self` as input, `self` cannot be
    // used after a call to `into_iter` unless `self` implements clone/copy
    for name2 in names2.into_iter() {
        match name2 {
            "Ferris" => println!("There is a rustacean among us!"),
            _ => println!("Hello {}", name2),
        }
    }

    // iter_mut allows us to iterate through the iterable while changing its content.
    // It creates a iterator over the iterable with which the original can be mutated
    let mut names3 = vec!["Bob", "Frank", "Ferris"];
    for name in names3.iter_mut() {
        *name = match *name {
            "Ferris" => "There is a rustacean among us!",
            _ => "Hello",
        }
    }

    println!("names: {:?}", names);
}
