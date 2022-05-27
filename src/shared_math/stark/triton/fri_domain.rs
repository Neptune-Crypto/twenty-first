use crate::shared_math::polynomial::Polynomial;
use crate::shared_math::traits::PrimeField;

#[derive(Debug, Clone)]
pub struct FriDomain<PF>
where
    PF: PrimeField,
{
    offset: PF,
    omega: PF,
    length: u32,
}

impl<PF> FriDomain<PF>
where
    PF: PrimeField,
{
    pub fn evaluate(&self, polynomial: &Polynomial<PF>) -> Vec<PF> {
        polynomial.fast_coset_evaluate(&self.offset, self.omega, self.length as usize)
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
