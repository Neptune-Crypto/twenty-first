use crate::shared_math::{b_field_element::BFieldElement, other, x_field_element::XFieldElement};

struct Table<T> {
    base_width: usize,
    full_width: usize,
    length: usize,
    num_randomizers: usize,
    height: usize,
    omicron: BFieldElement,
    generator: BFieldElement,
    order: usize,
    matrix: Vec<Vec<BFieldElement>>,
    more: T,
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
}

struct ProcessorTableMore {
    codewords: Vec<Vec<XFieldElement>>,
}

impl TableMoreTrait for ProcessorTableMore {
    fn new_more() -> Self {
        ProcessorTableMore { codewords: vec![] }
    }
}

pub struct ProcessorTable(Table<ProcessorTableMore>);

impl ProcessorTable {
    pub fn new(
        length: usize,
        num_randomizers: usize,
        generator: BFieldElement,
        order: usize,
    ) -> Self {
        let base_width = 7;
        let full_width = 11;
        let height = other::roundup_npo2(length as u64) as usize;

        let table = Table::<ProcessorTableMore>::new(
            base_width,
            full_width,
            length,
            num_randomizers,
            generator,
            order,
        );

        Self(table)
    }
}
