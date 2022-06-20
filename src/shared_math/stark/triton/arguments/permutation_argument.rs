use crate::shared_math::mpolynomial::Degree;
use crate::shared_math::x_field_element::XFieldElement;
use crate::shared_math::xfri::FriDomain;

// The linter seems to mistakenly think that a collect is not needed here
#[allow(clippy::needless_collect)]
pub fn quotient(
    _lhs_codeword: &[XFieldElement],
    _rhs_codeword: &[XFieldElement],
    _fri_domain: &FriDomain,
) -> Vec<XFieldElement> {
    todo!()
}

pub fn quotient_degree_bound() -> Degree {
    todo!()
}

pub fn evaluate_difference(_points: &[Vec<XFieldElement>]) -> XFieldElement {
    todo!()
}
