use super::base_table::Table;
use super::hash_coprocessor_table::HashCoprocessorTable;
use super::instruction_table::InstructionTable;
use super::io_table::InputTable;
use super::jump_stack_table::JumpStackTable;
use super::op_stack_table::OpStackTable;
use super::processor_table::ProcessorTable;
use super::program_table::ProgramTable;
use super::ram_table::RAMTable;
use super::u32_op_table::U32OpTable;
use crate::shared_math::b_field_element::BFieldElement;

type BWord = BFieldElement;

#[derive(Debug, Clone)]
pub struct BaseTableCollection {
    pub program_table: ProgramTable,
    pub instruction_table: InstructionTable,
    pub processor_table: ProcessorTable,
    pub input_table: InputTable,
    pub output_table: InputTable,
    pub op_stack_table: OpStackTable,
    pub ram_table: RAMTable,
    pub jump_stack_table: JumpStackTable,
    pub hash_coprocessor_table: HashCoprocessorTable,
    pub u32_op_table: U32OpTable,
}

impl BaseTableCollection {
    pub fn empty() -> Self {
        let program_table = todo!(); // ProgramTable,
        let instruction_table = todo!(); // InstructionTable,
        let processor_table = todo!(); // ProcessorTable,
        let input_table = todo!(); // InputTable,
        let output_table = todo!(); // InputTable,
        let op_stack_table = todo!(); // OpStackTable,
        let ram_table = todo!(); // RAMTable,
        let jump_stack_table = todo!(); // JumpStackTable,
        let hash_coprocessor_table = todo!(); // HashCoprocessorTable,
        let u32_op_table = todo!(); // U32OpTable,

        BaseTableCollection {
            program_table,
            instruction_table,
            processor_table,
            input_table,
            output_table,
            op_stack_table,
            ram_table,
            jump_stack_table,
            hash_coprocessor_table,
            u32_op_table,
        }
    }

    pub fn max_degree(&self) -> u64 {
        self.into_iter()
            .map(|table| table.max_degree())
            .max()
            .unwrap_or(1) as u64
    }
}

impl<'a> IntoIterator for &'a BaseTableCollection {
    type Item = &'a dyn Table<BWord>;

    type IntoIter = std::array::IntoIter<&'a dyn Table<BWord>, 10>;

    fn into_iter(self) -> Self::IntoIter {
        [
            &self.program_table as &'a dyn Table<BWord>,
            &self.instruction_table as &'a dyn Table<BWord>,
            &self.processor_table as &'a dyn Table<BWord>,
            &self.input_table as &'a dyn Table<BWord>,
            &self.output_table as &'a dyn Table<BWord>,
            &self.op_stack_table as &'a dyn Table<BWord>,
            &self.ram_table as &'a dyn Table<BWord>,
            &self.jump_stack_table as &'a dyn Table<BWord>,
            &self.hash_coprocessor_table as &'a dyn Table<BWord>,
            &self.u32_op_table as &'a dyn Table<BWord>,
        ]
        .into_iter()
    }
}
