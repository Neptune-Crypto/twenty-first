use crate::shared_math::{b_field_element::BFieldElement, mpolynomial::MPolynomial};

use super::table::{Table, TableMoreTrait, TableTrait};

pub struct InstructionTable(Table<InstructionTableMore>);

pub struct InstructionTableMore(());

impl TableMoreTrait for InstructionTableMore {
    fn new_more() -> Self {
        InstructionTableMore(())
    }
}

impl InstructionTable {
    // named indices for base columns
    pub const ADDRESS: usize = 0;
    pub const CURRENT_INSTRUCTION: usize = 1;
    pub const NEXT_INSTRUCTION: usize = 2;

    // named indices for extension columns
    pub const PERMUTATION: usize = 3;
    pub const EVALUATION: usize = 4;

    // base and extension table width
    pub const BASE_WIDTH: usize = 3;
    pub const FULL_WIDTH: usize = 5;

    pub fn new(
        length: usize,
        num_randomizers: usize,
        generator: BFieldElement,
        order: usize,
    ) -> Self {
        let table = Table::<InstructionTableMore>::new(
            Self::BASE_WIDTH,
            Self::FULL_WIDTH,
            length,
            num_randomizers,
            generator,
            order,
        );

        Self(table)
    }

    fn transition_constraints_afo_named_variables(
        address: MPolynomial<BFieldElement>,
        current_instruction: MPolynomial<BFieldElement>,
        next_instruction: MPolynomial<BFieldElement>,
        address_next: MPolynomial<BFieldElement>,
        current_instruction_next: MPolynomial<BFieldElement>,
        next_instruction_next: MPolynomial<BFieldElement>,
    ) -> Vec<MPolynomial<BFieldElement>> {
        let mut polynomials: Vec<MPolynomial<BFieldElement>> = vec![];
        let one = MPolynomial::<BFieldElement>::from_constant(BFieldElement::ring_one(), 14);

        // instruction pointer increases by 0 or 1
        polynomials.push(
            (address_next.clone() - address.clone() - one.clone())
                * (address_next.clone() - address.clone()),
        );

        // if address is the same, then current instruction is also
        polynomials.push(
            (address_next.clone() - address.clone() - one.clone())
                * (current_instruction_next - current_instruction),
        );

        // if address is the same, then next instruction is also
        polynomials
            .push((address_next - address - one) * (next_instruction_next - next_instruction));

        polynomials
    }
}

impl TableTrait for InstructionTable {
    fn base_width(&self) -> usize {
        self.0.base_width
    }

    fn full_width(&self) -> usize {
        self.0.full_width
    }

    fn length(&self) -> usize {
        self.0.length
    }

    fn num_randomizers(&self) -> usize {
        self.0.num_randomizers
    }

    fn height(&self) -> usize {
        self.0.height
    }

    fn omicron(&self) -> BFieldElement {
        self.0.omicron
    }

    fn generator(&self) -> BFieldElement {
        self.0.generator
    }

    fn order(&self) -> usize {
        self.0.order
    }

    fn base_transition_constraints(&self) -> Vec<MPolynomial<BFieldElement>> {
        let vars = MPolynomial::<BFieldElement>::variables(6, BFieldElement::ring_one());
        let address = vars[0].clone();
        let current_instruction = vars[1].clone();
        let next_instruction = vars[2].clone();
        let address_next = vars[3].clone();
        let current_instruction_next = vars[4].clone();
        let next_instruction_next = vars[5].clone();

        InstructionTable::transition_constraints_afo_named_variables(
            address,
            current_instruction,
            next_instruction,
            address_next,
            current_instruction_next,
            next_instruction_next,
        )
    }

    // def base_boundary_constraints(self):
    //     # format: mpolynomial
    //     x = MPolynomial.variables(self.width, self.field)
    //     zero = MPolynomial.zero()
    //     return [x[InstructionTable.address]-zero]
    fn base_boundary_constraints(&self) -> Vec<MPolynomial<BFieldElement>> {
        let x = MPolynomial::<BFieldElement>::variables(
            InstructionTable::FULL_WIDTH,
            BFieldElement::ring_one(),
        );

        // Why create 'x' and then throw all but 'address' away?
        let address = x[InstructionTable::ADDRESS].clone();
        let zero = MPolynomial::<BFieldElement>::zero(InstructionTable::FULL_WIDTH);

        vec![address - zero]
    }
}
