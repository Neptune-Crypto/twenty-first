use std::convert::TryFrom;
use std::convert::TryInto;

// PartialEq provides an implementation of the equality operator
// Two structs are equal when all their fields are equal
// EvenNumber is a tuple struct where its fields have no names
#[derive(Debug, PartialEq)]
struct EvenNumber(i32);

#[derive(Debug, PartialEq)]
struct TupleStructEvenNumber(i32, i32);

impl TryFrom<i32> for EvenNumber {
    type Error = ();

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        if value % 2 == 0 {
            Ok(EvenNumber(value))
        } else {
            Err(())
        }
    }
}

impl TryFrom<i32> for TupleStructEvenNumber {
    type Error = ();

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        if value % 2 == 0 {
            Ok(TupleStructEvenNumber(value, value))
        } else {
            Err(())
        }
    }
}

pub fn tryfrom_and_tryinto() {
    // assert_eq calls `panic` if the equality is not fulfilled.
    // panic stops the thread and presents an error message to the
    // user.

    // TryFrom
    assert_eq!(EvenNumber::try_from(8), Ok(EvenNumber(8)));
    // assert_eq!(EvenNumber::try_from(0), Ok(EvenNumber(8))); // Will invoke kernel panic
    assert_eq!(EvenNumber::try_from(5), Err(()));

    // TryInto, EvenNumber
    let result: Result<EvenNumber, ()> = 8i32.try_into();
    assert_eq!(result, Ok(EvenNumber(8)));
    let result: Result<EvenNumber, ()> = 5i32.try_into();
    assert_eq!(result, Err(()));

    // TryInto, TupleStructEvenNumber
    let result: Result<TupleStructEvenNumber, ()> = 8i32.try_into();
    assert_eq!(result, Ok(TupleStructEvenNumber(8, 8)));
    let result: Result<TupleStructEvenNumber, ()> = 5i32.try_into();
    assert_eq!(result, Err(()));
}
