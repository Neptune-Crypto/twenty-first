use super::b_field_element::BFieldElement;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct BoundaryConstraint {
    pub cycle: usize,
    pub register: usize,
    pub value: BFieldElement,
}

// A hashmap from register value to (x, y) value of boundary constraint
pub type BoundaryConstraintsMap = HashMap<usize, (BFieldElement, BFieldElement)>;
