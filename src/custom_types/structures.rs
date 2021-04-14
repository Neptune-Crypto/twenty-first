use std::fmt;

#[derive(Debug)]
struct Person {
    name: String,
    age: u8,
}

impl fmt::Display for Person {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}, {} years old", self.name, self.age)
    }
}

// A unit struct
struct Unit;

// A tuple struct
struct Pair(i32, f32);

// A strict with two fields
#[derive(Debug, Copy, Clone)]
struct Point {
    x: f32,
    y: f32,
}

// Structs can be nested
#[derive(Debug)]
#[allow(dead_code)]
struct Rectangle {
    top_left: Point,
    bottom_right: Point,
}

// The implementation block contains all methods for this struct
impl Rectangle {
    fn rect_area(&self) -> f32 {
        // `self` gives access to the struct fields via the dot operator
        let Point { x: x1, y: y1 } = self.top_left;
        let Point { x: x2, y: y2 } = self.bottom_right;

        // `abs` is a `f64` method that returns the absolute value of the
        // caller
        ((x1 - x2) * (y1 - y2)).abs()
    }
}

fn square(bottom_left: Point, side_length: f32) -> Rectangle {
    let Point { x: left, y: bottom } = bottom_left;
    Rectangle {
        bottom_right: Point {
            x: left + side_length,
            y: bottom,
        },
        top_left: Point {
            x: left,
            y: bottom + side_length,
        },
    }
}

pub fn structures() {
    // Create a struct with field init shorthand, like JS and C#
    let name = String::from("Peter");
    let age = 27;
    let peter = Person { name, age };

    // Print debug of struct
    println!("{:?}", peter);

    // Print Display of struct
    println!("{}", peter);

    // Instantiate a `Point`
    let point = Point { x: 10.3, y: 0.4 };

    // Access the fields of the point
    println!("point coordinates: ({}, {})", point.x, point.y);

    // Make a new point by using the struct update syntax to use
    // the fields of our other one
    // We only update the fields that we declare, the rest are
    // inherited. This is a very interesting syntax imo.
    let bottom_right = Point { x: 5.2, ..point };

    // `bottom_right.y` will be the same as `point.y` because we used that field
    // from `point`
    println!("second point: ({}, {})", bottom_right.x, bottom_right.y);

    // Destructure point using a let binding
    // Here, the declarations for the bindings `top_edge`
    // and `left_edge` are made.
    let Point {
        x: top_edge,
        y: left_edge,
    } = point;

    let rectangle = Rectangle {
        top_left: Point {
            x: left_edge,
            y: top_edge,
        },
        bottom_right: bottom_right,
    };

    // Instantiate a unit struct
    let _unit = Unit;

    // Instantiate a tuple struct
    let pair = Pair(1, 0.1);

    // Access the fields of a tuple struct
    println!("pair contains {:?} and {:?}", pair.0, pair.1);

    // Destructure a tuple struct
    let Pair(integer, decimal) = pair;

    println!("pair contains {:?} and {:?}", integer, decimal);
    println!("{:?}", rectangle);
    println!("The area of this rectangle is: {}", rectangle.rect_area());

    println!(
        "A square with side length 7 has an area of {}",
        square(point, 7.0).rect_area()
    );
    println!("A square with side length 7: {:?}", square(point, 7.0));
    println!(
        "A square with side length 2 has an area of {}",
        square(point, 2.0).rect_area()
    );
    println!("A square with side length 2: {:?}", square(point, 2.0));
    println!(
        "A square with side length 9 has an area of {}",
        square(point, 9.0).rect_area()
    );
    println!("A square with side length 9: {:?}", square(point, 9.0));
    println!(
        "A square with side length 11 has an area of {}",
        square(point, 11.0).rect_area()
    );
    println!("A square with side length 11: {:?}", square(point, 11.0));
}
