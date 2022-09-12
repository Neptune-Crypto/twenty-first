use num_traits::One;

use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::fri::FriDomain;
use crate::shared_math::mpolynomial::Degree;
use crate::shared_math::traits::FiniteField;
use crate::shared_math::x_field_element::XFieldElement;

use super::table_collection::TableCollection;
use std::cell::RefCell;
use std::cmp;
use std::rc::Rc;

pub struct PermutationArgument {
    tables: Rc<RefCell<TableCollection>>,
    lhs: ColumnPointer,
    rhs: ColumnPointer,
}

struct ColumnPointer {
    // todo: Change (usize, usize) into something less fickle
    // table_index could be replaced with an actual reference to a table
    // column_index might have to stay usize
    table_index: usize,
    column_index: usize,
}

impl PermutationArgument {
    pub fn new(
        tables: Rc<RefCell<TableCollection>>,
        lhs: (usize, usize),
        rhs: (usize, usize),
    ) -> Self {
        PermutationArgument {
            tables,
            lhs: ColumnPointer {
                table_index: lhs.0,
                column_index: lhs.1,
            },
            rhs: ColumnPointer {
                table_index: rhs.0,
                column_index: rhs.1,
            },
        }
    }

    // The linter seems to mistakenly think that a collect is not needed here
    #[allow(clippy::needless_collect)]
    pub fn quotient(&self, fri_domain: &FriDomain<XFieldElement>) -> Vec<XFieldElement> {
        let one: BFieldElement = BFieldElement::one();
        let zerofier = fri_domain
            .b_domain_values()
            .into_iter()
            .map(|x| x - one)
            .collect();
        let zerofier_inverse = BFieldElement::batch_inversion(zerofier);
        let difference_codeword: Vec<_> = self
            .tables
            .borrow()
            .get_table_codeword_by_index(self.lhs.table_index)[self.lhs.column_index]
            .iter()
            .zip(
                self.tables
                    .borrow()
                    .get_table_codeword_by_index(self.rhs.table_index)[self.rhs.column_index]
                    .iter(),
            )
            .map(|(l, r)| *l - *r)
            .collect();
        difference_codeword
            .into_iter()
            .zip(zerofier_inverse.into_iter())
            .map(|(d, z)| d * z.lift())
            .collect()
    }

    pub fn quotient_degree_bound(&self) -> Degree {
        let lhs_interpolant_degree = self
            .tables
            .borrow()
            .get_table_interpolant_degree_by_index(self.lhs.table_index);
        let rhs_interpolant_degree = self
            .tables
            .borrow()
            .get_table_interpolant_degree_by_index(self.rhs.table_index);
        let degree = cmp::max(lhs_interpolant_degree, rhs_interpolant_degree);
        degree - 1
    }

    pub fn evaluate_difference(&self, points: &[Vec<XFieldElement>]) -> XFieldElement {
        points[self.lhs.table_index][self.lhs.column_index]
            - points[self.rhs.table_index][self.rhs.column_index]
    }
}
