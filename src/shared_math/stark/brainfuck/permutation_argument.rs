use super::table_collection::TableCollection;
use std::rc::Rc;

pub struct PermutationArgument {
    tables: Rc<TableCollection>,
    lhs: (usize, usize),
    rhs: (usize, usize),
}

impl PermutationArgument {
    // FIXME: Change (usize, usize) into something readable
    pub fn new(tables: Rc<TableCollection>, lhs: (usize, usize), rhs: (usize, usize)) -> Self {
        PermutationArgument { tables, lhs, rhs }
    }
}
