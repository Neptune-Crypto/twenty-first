use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::Degree;
use crate::shared_math::traits::PrimeField;
use crate::shared_math::x_field_element::XFieldElement;
use crate::shared_math::xfri::FriDomain;
use std::cmp;

// The linter seems to mistakenly think that a collect is not needed here
#[allow(clippy::needless_collect)]
pub fn quotient(
    lhs_codeword: &[XFieldElement],
    rhs_codeword: &[XFieldElement],
    fri_domain: &FriDomain,
) -> Vec<XFieldElement> {
    todo!()
}

pub fn quotient_degree_bound() -> Degree {
    todo!()
}

pub fn evaluate_difference(points: &[Vec<XFieldElement>]) -> XFieldElement {
    todo!()
}
