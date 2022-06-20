use crate::shared_math::mpolynomial::Degree;
use crate::shared_math::stark::triton::table::table_collection::ExtTableCollection;
use crate::shared_math::x_field_element::XFieldElement;
use crate::shared_math::xfri::FriDomain;

pub fn quotient(
    _ext_tables: &ExtTableCollection,
    _lhs_codeword: &[XFieldElement],
    _rhs_codeword: &[XFieldElement],
    _fri_domain: &FriDomain,
) -> Vec<XFieldElement> {
    // let difference_codeword = lhs_codeword
    //     .iter()
    //     .zip(rhs_codeword.iter())
    //     .map(|(l, r)| *l - *r)
    //     .collect();

    // let ext_table_index = self.lhs.0;
    // let difference_codeword: Vec<XFieldElement> = ext_tables
    //     .get_table_codeword_by_index(self.lhs.0)[self.lhs.1]
    //     .iter()
    //     .zip(self.tables.borrow().get_table_codeword_by_index(self.rhs.0)[self.rhs.1].iter())
    //     .map(|(l, r)| *l - *r)
    //     .collect();

    // let one: BFieldElement = BFieldElement::ring_one();
    // let zerofier: Vec<BFieldElement> = fri_domain
    //     .b_domain_values()
    //     .into_iter()
    //     .map(|x| x - one)
    //     .collect();

    // let zerofier_inverse = BFieldElement::batch_inversion(zerofier);
    // difference_codeword
    //     .into_iter()
    //     .zip(zerofier_inverse.into_iter())
    //     .map(|(d, z)| d * z.lift())
    //     .collect()

    todo!()
}

pub fn quotient_degree_bound(_ext_tables: &ExtTableCollection) -> Degree {
    // let lhs_interpolant_degree = self
    //     .tables
    //     .borrow()
    //     .get_table_interpolant_degree_by_index(self.lhs.0);
    // let rhs_interpolant_degree = self
    //     .tables
    //     .borrow()
    //     .get_table_interpolant_degree_by_index(self.rhs.0);
    // let degree = std::cmp::max(lhs_interpolant_degree, rhs_interpolant_degree);
    // degree - 1

    todo!()
}

pub fn evaluate_difference(
    _ext_tables: &ExtTableCollection,
    _points: &[Vec<XFieldElement>],
) -> XFieldElement {
    // points[ext_tables.lhs.0][ext_tables.lhs.1] - points[ext_tables.rhs.0][ext_tables.rhs.1]

    todo!()
}
