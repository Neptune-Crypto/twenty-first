// Type definition using enum which is a sum type
// enums in Rust are like enums in C where each enumeration
// can have additional data associated with it.
enum WebEvent {
    PageLoad,
    PageUnload,
    KeyPress(char),
    Paste(String),
    Click { x: i64, y: i64 },
}

fn inspect(event: WebEvent) {
    // `match` is like `switch` in C, but we can use "pattern matching"
    // to declare variables upon an enum match, as we do with `c`, `s`, `x`,
    // and `y` below.
    // The interesting thing when using `enum` + `match` in Rust is that the
    // compiler will tell you if you forgot to cover a case, if I remove
    // the case for WebEvent::Click, the compiler will tell me that there is
    // an enum case I haven't covered.
    match event {
        WebEvent::PageLoad => println!("page loaded"),
        WebEvent::PageUnload => println!("page unloaded"),
        WebEvent::KeyPress(c) => println!("pressed '{}'", c),
        WebEvent::Paste(s) => println!("pasted \"{}\".", s),
        WebEvent::Click { x, y } => {
            println!("clicked at x={}, y={}", x, y);
        }
    }
}

pub fn enums() {
    let pressed = WebEvent::KeyPress('x');
    let pasted = WebEvent::Paste("my text".to_owned());
    let click = WebEvent::Click { x: 20, y: 80 };
    let load = WebEvent::PageLoad;
    let unload = WebEvent::PageUnload;

    inspect(pressed);
    inspect(pasted);
    inspect(click);
    inspect(load);
    inspect(unload);
}
