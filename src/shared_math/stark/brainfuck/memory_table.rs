use crate::shared_math::{
    b_field_element::BFieldElement, mpolynomial::MPolynomial,
    stark::brainfuck::processor_table::ProcessorTable, x_field_element::XFieldElement,
};

use super::{
    stark::{EXTENSION_CHALLENGE_COUNT, PERMUTATION_ARGUMENTS_COUNT},
    table::{Table, TableMoreTrait, TableTrait},
};

#[derive(Debug, Clone)]
pub struct MemoryTable(pub Table<MemoryTableMore>);

#[derive(Debug, Clone)]
pub struct MemoryTableMore(());

impl TableMoreTrait for MemoryTableMore {
    fn new_more() -> Self {
        MemoryTableMore(())
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

    // @staticmethod
    // def derive_matrix(processor_matrix, num_randomizers):
    //     matrix = [[pt[ProcessorTable.cycle], pt[ProcessorTable.memory_pointer],
    //               pt[ProcessorTable.memory_value]] for pt in processor_matrix]
    //     matrix.sort(key=lambda mt: mt[MemoryTable.memory_pointer].value)
    //     return matrix
    pub fn derive_matrix(processor_matrix: Vec<Vec<BFieldElement>>) -> Vec<Vec<BFieldElement>> {
        let mut matrix = vec![];
        for pt in processor_matrix.iter() {
            matrix.push(vec![
                pt[ProcessorTable::CYCLE],
                pt[ProcessorTable::MEMORY_POINTER],
                pt[ProcessorTable::MEMORY_VALUE],
            ]);
        }

        matrix.sort_by_key(|k| k[MemoryTable::MEMORY_POINTER].value());
        matrix
    }

    fn transition_constraints_afo_named_variables(
        cycle: MPolynomial<BFieldElement>,
        address: MPolynomial<BFieldElement>,
        value: MPolynomial<BFieldElement>,
        cycle_next: MPolynomial<BFieldElement>,
        address_next: MPolynomial<BFieldElement>,
        value_next: MPolynomial<BFieldElement>,
    ) -> Vec<MPolynomial<BFieldElement>> {
        let mut polynomials: Vec<MPolynomial<BFieldElement>> = vec![];

        let variable_count = Self::BASE_WIDTH * 2;
        let one = MPolynomial::from_constant(BFieldElement::ring_one(), variable_count);

        // 1. memory pointer increases by one, zero, or minus one

        // # <=>. (MP*=MP+1) \/ (MP*=MP)
        // polynomials += [(address_next - address - one)
        //                 * (address_next - address)]
        polynomials.push(
            (address_next.clone() - address.clone() - one.clone())
                * (address_next.clone() - address.clone()),
        );

        // 2. if memory pointer does not increase, then memory value can change only if cycle counter increases by one

        // #        <=>. MP*=MP => (MV*=/=MV => CLK*=CLK+1)
        // #        <=>. MP*=/=MP \/ (MV*=/=MV => CLK*=CLK+1)
        // # (DNF:) <=>. MP*=/=MP \/ MV*=MV \/ CLK*=CLK+1
        // polynomials += [(address_next - address - one) *
        //                 (value_next - value) * (cycle_next - cycle - one)]
        polynomials.push(
            (address_next.clone() - address.clone() - one.clone())
                * (value_next.clone() - value)
                * (cycle_next - cycle - one),
        );

        // 3. if memory pointer increases by one, then memory value must be set to zero

        // #        <=>. MP*=MP+1 => MV* = 0
        // # (DNF:) <=>. MP*=/=MP+1 \/ MV*=0
        // polynomials += [(address_next - address)
        //                 * value_next]
        polynomials.push((address_next - address) * value_next);

        polynomials
    }
}

impl TableTrait for MemoryTable {
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
        let variable_count = Self::BASE_WIDTH * 2;
        let vars =
            MPolynomial::<BFieldElement>::variables(variable_count, BFieldElement::ring_one());

        let cycle = vars[0].clone();
        let address = vars[1].clone();
        let value = vars[2].clone();
        let cycle_next = vars[3].clone();
        let address_next = vars[4].clone();
        let value_next = vars[5].clone();

        MemoryTable::transition_constraints_afo_named_variables(
            cycle,
            address,
            value,
            cycle_next,
            address_next,
            value_next,
        )
    }

    fn base_boundary_constraints(&self) -> Vec<MPolynomial<BFieldElement>> {
        todo!()
    }

    fn extend(
        &mut self,
        all_challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT as usize],
        all_initials: [XFieldElement; PERMUTATION_ARGUMENTS_COUNT],
    ) {
        todo!()
    }
}

#[cfg(test)]
mod memory_table_tests {
    use super::*;
    use crate::shared_math::stark::brainfuck::vm::sample_programs;
    use crate::shared_math::stark::brainfuck;
    use crate::shared_math::stark::brainfuck::vm::BaseMatrices;
    use crate::shared_math::traits::{GetPrimitiveRootOfUnity, IdentityValues};

    // When we simulate a program, this generates a collection of matrices that contain
    // "abstract" execution traces. When we evaluate the base transition constraints on
    // the rows (points) from the InstructionTable matrix, these should evaluate to zero.
    #[test]
    fn memory_base_table_evaluate_to_zero_on_execution_trace_test() {
        for source_code in sample_programs::get_all_sample_programs().iter() {
            let actual_program = brainfuck::vm::compile(source_code).unwrap();
            let input_data = vec![BFieldElement::new(97)];
            let base_matrices: BaseMatrices =
                brainfuck::vm::simulate(&actual_program, &input_data).unwrap();

            let processor_matrix_from_simulate: Vec<Vec<BFieldElement>> = base_matrices
                .processor_matrix
                .into_iter()
                .map(|register| {
                    let row: Vec<BFieldElement> = register.into();
                    row
                })
                .collect();
            let derived_memory_matrix = MemoryTable::derive_matrix(processor_matrix_from_simulate);

            assert!(
                !derived_memory_matrix.is_empty(),
                "All tested programs update memory"
            );

            let number_of_randomizers = 2;
            let order = 1 << 32;
            let smooth_generator = BFieldElement::ring_zero()
                .get_primitive_root_of_unity(order)
                .0
                .unwrap();

            // instantiate table objects
            let memory_table: MemoryTable = MemoryTable::new(
                derived_memory_matrix.len(),
                number_of_randomizers,
                smooth_generator,
                order as usize,
            );

            let air_constraints = memory_table.base_transition_constraints();

            let step_count = std::cmp::max(0, derived_memory_matrix.len() as isize - 1) as usize;
            for step in 0..step_count {
                let row: Vec<BFieldElement> = derived_memory_matrix[step].clone();
                let next_row: Vec<BFieldElement> = derived_memory_matrix[step + 1].clone();
                let point: Vec<BFieldElement> = vec![row, next_row].concat();

                for air_constraint in air_constraints.iter() {
                    assert!(air_constraint.evaluate(&point).is_zero());
                }
            }
        }
    }
}
