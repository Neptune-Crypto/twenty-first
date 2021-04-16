use List::*;

// A linked list of u32 numbers
enum List {
    // Cons: Tuple struct that wraps an element and a pointer to the next node
    Cons(u32, Box<List>),
    // Nil: A node that signifies the end of the linked list
    Nil,
}

// Code block for implementations for the List enum
impl List {
    fn new() -> List {
        Nil
    }

    fn prepend(self, element: u32) -> List {
        // Box is a pointer type for heap allocation
        Cons(element, Box::new(self))
    }

    fn len(&self) -> u32 {
        // The '*' character dereferences the reference to self
        // Matching on a concrete type `T` is preferred over a match on a
        // reference `&T`.
        match *self {
            Cons(_, ref rest) => 1 + rest.len(),
            Nil => 0,
        }
    }

    fn stringify(&self) -> String {
        match *self {
            Cons(head, ref tail) => {
                format!("{}, {}", head, tail.stringify())
            }
            Nil => String::from("Nil"),
        }
    }
}

pub fn linked_list() {
    let mut list = List::new();
    list = list.prepend(1);
    list = list.prepend(2);
    list = list.prepend(3);

    // Show the final state of the list
    println!("linked list has length: {}", list.len());
    println!("{}", list.stringify());
}
