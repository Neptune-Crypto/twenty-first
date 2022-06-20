use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::x_field_element::XFieldElement;

pub const PROGRAM_EVALUATION_CHALLENGE_INDICES_COUNT: usize = 4;

// This was: EvaluationArgument::compute_terminal()
pub fn compute_terminal(symbols: &[BFieldElement], challenge: XFieldElement) -> XFieldElement {
    let mut acc = XFieldElement::ring_zero();

    for s in symbols.iter() {
        acc = challenge * acc + s.lift();
    }

    acc
}
