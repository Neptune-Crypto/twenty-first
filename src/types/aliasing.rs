#![allow(dead_code)]
#![allow(clippy::unnecessary_cast)]
// The type statement can be used to give a new name to an existing type.
// Types must have UpperCamelCase names, or the compiler will raise a warning.
// The exception to this rule are the primitive types: usize, f32, etc.

// `NanoSecond` is a new name for `u64`.
type NanoSecond = u64;
type Inch = u64;

// Use an attribute to silence warning.
#[allow(non_camel_case_types)]
type u64_t = u64;
// TODO ^ Try removing the attribute

pub fn aliasing() {
    // `NanoSecond` = `Inch` = `u64_t` = `u64`.
    let nanoseconds: NanoSecond = 5 as u64_t;
    let inches: Inch = 2 as u64_t;

    // Note that type aliases *don't* provide any extra type safety, because
    // aliases are *not* new types
    println!(
        "{} nanoseconds + {} inches = {} unit?",
        nanoseconds,
        inches,
        nanoseconds + inches
    );
}
