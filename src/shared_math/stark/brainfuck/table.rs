use crate::shared_math::{b_field_element::BFieldElement, mpolynomial::MPolynomial, other};

pub const PROCESSOR_TABLE: usize = 0;
pub const INSTRUCTION_TABLE: usize = 1;
pub const MEMORY_TABLE: usize = 2;

pub struct Table<T> {
    base_width: usize,
    full_width: usize,
    length: usize,
    num_randomizers: usize,
    height: usize,
    omicron: BFieldElement,
    generator: BFieldElement,
    order: usize,
    matrix: Vec<Vec<BFieldElement>>,
    pub more: T,
}

impl<T: TableMoreTrait> Table<T> {
    pub fn new(
        base_width: usize,
        full_width: usize,
        length: usize,
        num_randomizers: usize,
        generator: BFieldElement,
        order: usize,
    ) -> Self {
        let height = other::roundup_npo2(length as u64) as usize;
        let omicron = Self::derive_omicron(generator, order, height);
        let matrix = vec![];
        let more = T::new_more();

        Self {
            base_width,
            full_width,
            length,
            num_randomizers,
            height,
            omicron,
            generator,
            order,
            matrix,
            more,
        }
    }

    fn derive_omicron(generator: BFieldElement, order: usize, height: usize) -> BFieldElement {
        todo!()
    }
}

pub trait TableMoreTrait {
    fn new_more() -> Self;
    fn base_transition_constraints() -> Vec<MPolynomial<BFieldElement>>;
    fn base_boundary_constraints() -> Vec<MPolynomial<BFieldElement>>;
}
