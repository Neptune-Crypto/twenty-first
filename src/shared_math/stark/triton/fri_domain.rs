use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::polynomial::Polynomial;
use crate::shared_math::traits::PrimeField;
use crate::shared_math::x_field_element::XFieldElement;

#[derive(Debug, Clone)]
pub struct FriDomain<PF>
where
    PF: PrimeField,
{
    pub offset: PF,
    pub omega: PF,
    pub length: usize,
}

impl<PF> FriDomain<PF>
where
    PF: PrimeField,
{
    pub fn evaluate(&self, polynomial: &Polynomial<PF>) -> Vec<PF> {
        polynomial.fast_coset_evaluate(&self.offset, self.omega, self.length)
    }

    pub fn interpolate(&self, values: &[PF]) -> Polynomial<PF> {
        Polynomial::<PF>::fast_coset_interpolate(&self.offset, self.omega, values)
    }

    pub fn domain_value(&self, index: u32) -> PF {
        self.omega.mod_pow_u32(index) * self.offset
    }

    pub fn domain_values(&self) -> Vec<PF> {
        let mut res = Vec::new();
        let mut acc = self.omega.ring_one();

        for _ in 0..self.length {
            acc *= self.omega;
            res.push(acc * self.offset)
        }

        res
    }
}

pub fn lift_domain(domain: &FriDomain<BFieldElement>) -> FriDomain<XFieldElement> {
    FriDomain {
        offset: domain.offset.lift(),
        omega: domain.omega.lift(),
        length: domain.length,
    }
}
