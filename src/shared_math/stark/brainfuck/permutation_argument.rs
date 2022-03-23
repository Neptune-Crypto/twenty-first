use super::table_collection::TableCollection;

pub struct PermutationArgument<'a> {
    tables: &'a TableCollection,
    lhs: (usize, usize),
    rhs: (usize, usize),
}

impl<'a> PermutationArgument<'a> {
    // FIXME: Change (usize, usize) into something readable
    pub fn new(tables: &'a TableCollection, lhs: (usize, usize), rhs: (usize, usize)) -> Self {
        PermutationArgument { tables, lhs, rhs }
    }
}
