use std::fmt;

struct Circle {
    radius: i32,
}

// Implement the trait (~ interface) `fmt::Display` for `Circle`.
// This trait contains *one* function fmt :: (&self, &mut fmt::Formatter) -> Result
impl fmt::Display for Circle {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Circle of radius {}", self.radius)
    }
}

pub fn to_and_from_strings() {
    // Default integer type is i32, and default double type i f64
    let circle = Circle { radius: 6 };

    // Implementing the trait fmt::Display automatically provides the
    // to_string
    println!("{}", circle.to_string());

    // The parse method on strings can be used to convert strings to numbers
    // This code will panic if `unwrap` fails but that might be better than
    // handling all errors explicitly? unwrap is a partial implementation
    let parsed: i32 = "5".parse().unwrap();
    let turbo_parsed = "10".parse::<i32>().unwrap();

    let sum = parsed + turbo_parsed;
    println!("Sum: {:?}", sum);

    let parsed_result = "5".parse::<i32>();
    let parsed_result2: Result<i32, std::num::ParseIntError> = "10a".parse::<i32>();
    match parsed_result {
        Ok(i) => println!("Succesfully parsed the number: {}", i),
        Err(_) => println!("Failed to parse number"),
    }
    match parsed_result2 {
        Ok(i) => println!("Succesfully parsed the number: {}", i),
        Err(_) => println!("Failed to parse number"),
    }
}
