use crate::shared_math::traits::PrimeFieldElement;

pub struct PrimeFieldPolynomial<F: PrimeFieldElement> {
    pub dummy: F::Elem,
}
