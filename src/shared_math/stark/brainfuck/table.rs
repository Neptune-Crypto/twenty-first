use crate::shared_math::{
    b_field_element::BFieldElement, mpolynomial::MPolynomial, other,
    traits::GetPrimitiveRootOfUnity,
};

pub const PROCESSOR_TABLE: usize = 0;
pub const INSTRUCTION_TABLE: usize = 1;
pub const MEMORY_TABLE: usize = 2;

pub struct Table<T> {
    pub base_width: usize,
    pub full_width: usize,
    pub length: usize,
    pub num_randomizers: usize,
    pub height: usize,
    pub omicron: BFieldElement,
    pub generator: BFieldElement,
    pub order: usize,
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
        let height = if length != 0 {
            other::roundup_npo2(length as u64) as usize
        } else {
            0
        };
        let omicron = Self::derive_omicron(height);
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

    /// derive a generator with degree og height
    fn derive_omicron(height: usize) -> BFieldElement {
        if height == 0 {
            return BFieldElement::ring_one();
        }

        BFieldElement::ring_zero()
            .get_primitive_root_of_unity(height as u128)
            .0
            .unwrap()
    }
}

pub trait TableTrait {
    fn base_width(&self) -> usize;
    fn full_width(&self) -> usize;
    fn length(&self) -> usize;
    fn num_randomizers(&self) -> usize;
    fn height(&self) -> usize;
    fn omicron(&self) -> BFieldElement;
    fn generator(&self) -> BFieldElement;
    fn order(&self) -> usize;

    fn interpolant_degree(&self) -> usize {
        self.height() + self.num_randomizers() - 1
    }

    fn base_transition_constraints(&self) -> Vec<MPolynomial<BFieldElement>>;
    fn base_boundary_constraints(&self) -> Vec<MPolynomial<BFieldElement>>;
}

pub trait TableMoreTrait {
    fn new_more() -> Self;
}
