use crate::shared_math::{b_field_element::BFieldElement, mpolynomial::MPolynomial};

use super::table::{Table, TableMoreTrait};

pub struct MemoryTable(Table<MemoryTableMore>);

pub struct MemoryTableMore(());

impl TableMoreTrait for MemoryTableMore {
    fn new_more() -> Self {
        MemoryTableMore(())
    }

    fn base_transition_constraints() -> Vec<MPolynomial<BFieldElement>> {
        todo!()
    }

    fn base_boundary_constraints() -> Vec<MPolynomial<BFieldElement>> {
        todo!()
    }
}

impl MemoryTable {
    // named indices for base columns
    pub const CYCLE: usize = 0;
    pub const MEMORY_POINTER: usize = 1;
    pub const MEMORY_VALUE: usize = 2;

    // named indices for extension columns
    pub const PERMUTATION: usize = 3;

    // base and extension table width
    pub const BASE_WIDTH: usize = 3;
    pub const FULL_WIDTH: usize = 4;

    pub fn new(
        length: usize,
        num_randomizers: usize,
        generator: BFieldElement,
        order: usize,
    ) -> Self {
        let table = Table::<MemoryTableMore>::new(
            Self::BASE_WIDTH,
            Self::FULL_WIDTH,
            length,
            num_randomizers,
            generator,
            order,
        );

        Self(table)
    }
}
