use super::processor_table::{IOTable, InstructionTable, MemoryTable, ProcessorTable};

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
}
