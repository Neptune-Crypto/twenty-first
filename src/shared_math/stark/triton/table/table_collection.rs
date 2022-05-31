use super::base_table::Table;
use super::instruction_table::InstructionTable;
use super::io_table::InputTable;
use super::processor_table::ProcessorTable;
use super::ram_table::RAMTable;
use crate::shared_math::b_field_element::BFieldElement;

type BWord = BFieldElement;

#[derive(Debug, Clone)]
pub struct BaseTableCollection {
    pub processor_table: ProcessorTable,
    pub instruction_table: InstructionTable,
    pub memory_table: RAMTable,
    pub input_table: InputTable,
    pub output_table: InputTable,
}

impl<'a> IntoIterator for &'a BaseTableCollection {
    type Item = &'a dyn Table<BWord>;

    type IntoIter = std::array::IntoIter<&'a dyn Table<BWord>, 3>;

    fn into_iter(self) -> Self::IntoIter {
        [
            &self.processor_table as &'a dyn Table<BWord>,
            // &self.instruction_table as &'a dyn Table,
            // &self.memory_table as &'a dyn Table,
            &self.input_table as &'a dyn Table<BWord>,
            &self.output_table as &'a dyn Table<BWord>,
        ]
        .into_iter()
    }
}
