use crate::shared_math::{
    b_field_element::BFieldElement, mpolynomial::Degree, traits::PrimeField,
    x_field_element::XFieldElement, xfri::FriDomain,
};

use super::table_collection::TableCollection;
use std::{cell::RefCell, cmp, rc::Rc};

pub struct PermutationArgument {
    tables: Rc<RefCell<TableCollection>>,
    lhs: (usize, usize),
    rhs: (usize, usize),
}

impl PermutationArgument {
    // TOOD: Change (usize, usize) into something readable.
    // The 1st element of the tuple could be replaced with a pointer
    // to a table, the 2nd can probably stay a usize.
    pub fn new(
        tables: Rc<RefCell<TableCollection>>,
        lhs: (usize, usize),
        rhs: (usize, usize),
    ) -> Self {
        PermutationArgument { tables, lhs, rhs }
    }

    // The linter seems to mistakenly think that a collect is not needed here
    #[allow(clippy::needless_collect)]
    pub fn quotient(&self, fri_domain: &FriDomain) -> Vec<XFieldElement> {
        let difference_codeword: Vec<XFieldElement> = self
            .tables
            .borrow()
            .get_table_codeword_by_index(self.lhs.0)[self.lhs.1]
            .iter()
            .zip(self.tables.borrow().get_table_codeword_by_index(self.rhs.0)[self.rhs.1].iter())
            .map(|(l, r)| *l - *r)
            .collect();
        let one: BFieldElement = BFieldElement::ring_one();
        let zerofier: Vec<BFieldElement> = fri_domain
            .b_domain_values()
            .into_iter()
            .map(|x| x - one)
            .collect();
        let zerofier_inverse = BFieldElement::batch_inversion(zerofier);
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
            .get_table_interpolant_degree_by_index(self.lhs.0);
        let rhs_interpolant_degree = self
            .tables
            .borrow()
            .get_table_interpolant_degree_by_index(self.rhs.0);
        let degree = cmp::max(lhs_interpolant_degree, rhs_interpolant_degree);
        degree - 1
    }

    pub fn evaluate_difference(&self, points: &[Vec<XFieldElement>]) -> XFieldElement {
        points[self.lhs.0][self.lhs.1] - points[self.rhs.0][self.rhs.1]
    }
}
