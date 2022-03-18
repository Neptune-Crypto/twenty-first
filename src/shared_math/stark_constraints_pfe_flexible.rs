use super::prime_field_element_flexible::PrimeFieldElementFlexible;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct BoundaryConstraint {
    pub cycle: usize,
    pub register: usize,
    pub value: PrimeFieldElementFlexible,
}

// A hashmap from register value to (x, y) value of boundary constraint
pub type BoundaryConstraintsMap =
    HashMap<usize, (PrimeFieldElementFlexible, PrimeFieldElementFlexible)>;
