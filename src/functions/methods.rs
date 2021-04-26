// Methods are functions attached to objects
// Methods have access to the data of the object and its other methods via the `self` keyword.
// Methods are defined in an `impl` block.

struct Point {
    x: f64,
    y: f64,
}

impl Point {
    // A static method is defined in without a `self` keyword.
    // A static method does not need to be called on an instance.
    fn origin() -> Point {
        Point { x: 0.0, y: 0.0 }
    }

    fn new(x: f64, y: f64) -> Point {
        Point { x, y }
    }
}

struct Rectangle {
    p1: Point,
    p2: Point,
}

impl Rectangle {
    // `&self` is sugar for `self: &Self`
    // The type &Self means that the variable is borrowed and cannot thus be mutated
    fn area(&self) -> f64 {
        let width: f64 = (self.p1.x - self.p2.x).abs();
        let length: f64 = (self.p1.y - self.p2.y).abs();
        // self is a reference, so the data it refers to cannot be written/mutated.
        // self.p1 = Point::origin();
        width * length
    }

    // return the length of the perimeter
    #[allow(clippy::needless_arbitrary_self_type)]
    fn perimeter(self: &Self) -> f64 {
        // Bindings can be made with pattern matching instead of the dot operator
        let Point { x: x1, y: y1 } = self.p1;
        let Point { x: x2, y: y2 } = self.p2;

        2.0 * ((x1 - x2).abs() + (y1 - y2).abs())
    }

    // `&mut self` is syntactic sugar for `self: &mut Self`
    fn translate(&mut self, x: f64, y: f64) {
        self.p1.x += x;
        self.p1.y += y;
        self.p2.x += x;
        self.p2.y += y;
    }
}

// Box means that this is a pointer to a heap-allocated resource
struct Pair(Box<i32>, Box<i32>);

impl Pair {
    fn destroy(self) {
        let Pair(first, second) = self;

        println!("Destroying Pair({}, {})", first, second);

        // First and second go out of scope and get freed
    }
}

pub fn methods() {
    let rectangle = Rectangle {
        p1: Point::origin(),
        p2: Point::new(3.0, 4.0),
    };

    println!("Rectangle perimeter: {}", rectangle.perimeter());
    println!("Rectangle area: {}", rectangle.area());

    let mut square = Rectangle {
        p1: Point::origin(),
        p2: Point::new(1.0, 1.0),
    };

    // `rectangle` is immutable, and `translate` requires a mutable object
    // rectangle.translate(1.0, 1.0);

    square.translate(1.0, 1.0);

    let pair = Pair(Box::new(1), Box::new(2));

    // This moves the ownership of `pair` to the callee as `Pair` does not implement the trait `Copy`.
    pair.destroy();

    // So `pair` cannot be used again.
    // pair.destroy();
}
