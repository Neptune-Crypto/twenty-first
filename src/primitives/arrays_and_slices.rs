use std::mem;

// This function borrows a slice
// This means that the slice cannot be destroyed.
// Potentially also that it cannot have its elements altered.
fn analyze_slice(slice: &[i32]) {
    println!("first element of the slice: {}", slice[0]);
    println!("the slice has {} elements", slice.len());
}

pub fn arrays_and_slices() {
    // Fixed-size array, type signature is not needed
    let xs: [i32; 5] = [1, 2, 3, 4, 5];

    // All elements can be initialized to the same value
    let ys: [i32; 500] = [0; 500];

    println!("first element of the array: {}", xs[0]);
    println!("second element of the array: {}", xs[1]);
    println!("first element of the array: {:?}", xs[0]);
    println!("second element of the array: {:?}", xs[1]);

    // `len` returns the count of elements in the array
    println!("number of elements in array: {}", xs.len());

    // Arrays are allocated on the stack
    println!(
        "array occupies {} bytes, size_of_val",
        mem::size_of_val(&xs)
    );

    // Arrays can be automatically borrowed as slices
    println!("borrow the whole array as a slice");
    analyze_slice(&xs);

    // Slices can point to a section of an array
    // This is done on the form [start_index..end_index]
    // where the start index is included in the slice, and the
    // end index is **not** includede
    println!("borrow a section of the array as a slice");
    analyze_slice(&ys[1..4]);

    // Out of bound indexing causes compile error
    // println!("{}", xs[5]);
}
