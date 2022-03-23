use super::{
    instruction_table::InstructionTable, io_table::IOTable, memory_table::MemoryTable,
    processor_table::ProcessorTable,
};

pub struct TableCollection {
    pub processor_table: ProcessorTable,
    pub instruction_table: InstructionTable,
    pub memory_table: MemoryTable,
    pub input_table: IOTable,
    pub output_table: IOTable,
}

impl TableCollection {
    pub fn new(
        processor_table: ProcessorTable,
        instruction_table: InstructionTable,
        memory_table: MemoryTable,
        input_table: IOTable,
        output_table: IOTable,
    ) -> Self {
        Self {
            processor_table,
            instruction_table,
            memory_table,
            input_table,
            output_table,
        }
    }

    pub fn get_max_degree(&self) -> usize {
        let mut max_degree = 1;
        for air in self.processor_table.base_transition_constraints() {
            let degree_bounds: Vec<i64> = vec![
                self.processor_table.0.interpolant_degree() as i64;
                self.processor_table.0.base_width * 2
            ];
            let degree = air.symbolic_degree_bound(&degree_bounds)
                - (self.processor_table.0.height - 1) as i64;
            if max_degree < degree {
                max_degree = degree;
            }
        }

        // TODO: Add the other tables here to ensure that we calculate max degree correctly

        max_degree as usize
    }
}
