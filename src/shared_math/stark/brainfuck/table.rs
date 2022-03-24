use rand::thread_rng;

use crate::shared_math::fri::FriDomain;
use crate::shared_math::mpolynomial::Degree;
use crate::shared_math::polynomial::Polynomial;
use crate::shared_math::traits::{GetRandomElements, IdentityValues};
use crate::shared_math::x_field_element::XFieldElement;
use crate::shared_math::{
    b_field_element::BFieldElement, mpolynomial::MPolynomial, other,
    traits::GetPrimitiveRootOfUnity,
};

pub const PROCESSOR_TABLE: usize = 0;
pub const INSTRUCTION_TABLE: usize = 1;
pub const MEMORY_TABLE: usize = 2;

#[derive(Debug, Clone)]
pub struct Table<T> {
    pub base_width: usize,
    pub full_width: usize,
    pub length: usize,
    pub num_randomizers: usize,
    pub height: usize,
    pub omicron: BFieldElement,
    pub generator: BFieldElement,
    pub order: usize,
    pub matrix: Vec<Vec<BFieldElement>>,
    pub codewords: Vec<Vec<BFieldElement>>,
    pub more: T,
}

impl<T: TableMoreTrait> Table<T> {
    pub fn new(
        base_width: usize,
        full_width: usize,
        length: usize,
        num_randomizers: usize,
        generator: BFieldElement,
        order: usize,
    ) -> Self {
        let height = if length != 0 {
            other::roundup_npo2(length as u64) as usize
        } else {
            0
        };
        let omicron = Self::derive_omicron(height);
        let matrix = vec![];
        let more = T::new_more();
        let codewords = vec![];

        Self {
            base_width,
            full_width,
            length,
            num_randomizers,
            height,
            omicron,
            generator,
            order,
            matrix,
            more,
            codewords,
        }
    }

    /// derive a generator with degree og height
    fn derive_omicron(height: usize) -> BFieldElement {
        if height == 0 {
            return BFieldElement::ring_one();
        }

        BFieldElement::ring_zero()
            .get_primitive_root_of_unity(height as u128)
            .0
            .unwrap()
    }

    /// Return the interpolation of columns. The `column_indices` variable
    /// must be called with *all* the column indices for this particular table,
    /// if it is called with a subset, it *will* fail.
    pub fn interpolate_columns(
        &self,
        omega: BFieldElement,
        omega_order: usize,
        column_indices: Vec<usize>,
    ) -> Vec<Polynomial<BFieldElement>> {
        assert!(
            omega.mod_pow(omega_order as u64).is_one(),
            "omega must have indicated order"
        );
        assert!(
            !omega.mod_pow(omega_order as u64 / 2).is_one(),
            "omega must be a primitive root of indicated order"
        );

        if self.height == 0 {
            return vec![Polynomial::ring_zero(); column_indices.len()];
        }

        assert!(
            self.height >= self.num_randomizers,
            "Temporary restriction that number of randomizers must not exceed table height"
        );

        let mut polynomials: Vec<Polynomial<BFieldElement>> = vec![];
        let omicron_domain: Vec<BFieldElement> = (0..self.height)
            .map(|i| self.omicron.mod_pow(i as u64))
            .collect();
        let randomizer_domain: Vec<BFieldElement> = (0..self.num_randomizers)
            .map(|i| omega * omicron_domain[i])
            .collect();
        let domain = vec![omicron_domain, randomizer_domain].concat();
        let mut rng = thread_rng();
        for c in column_indices {
            let trace: Vec<BFieldElement> = self.matrix.iter().map(|row| row[c]).collect();
            let randomizers: Vec<BFieldElement> =
                BFieldElement::random_elements(self.num_randomizers, &mut rng);
            let values = vec![trace, randomizers].concat();
            assert_eq!(
                values.len(),
                domain.len(),
                "Length of x values and y values must match"
            );
            polynomials.push(Polynomial::fast_interpolate(
                &domain,
                &values,
                &omega,
                omega_order,
            ));
        }

        polynomials
    }

    /// Evaluate the base table
    pub fn lde(&mut self, domain: &FriDomain<BFieldElement>) -> Vec<Vec<BFieldElement>> {
        let polynomials =
            self.interpolate_columns(domain.omega, domain.length, (0..self.base_width).collect());
        self.codewords = polynomials
            .iter()
            .map(|p| domain.evaluate(p, BFieldElement::ring_zero()))
            .collect();
        self.codewords.clone()
    }

    pub fn ldex(&self, domain: FriDomain<BFieldElement>) -> Vec<Vec<XFieldElement>> {
        // TODO: See `lde` and pattern match from that
        todo!()
    }
}

pub trait TableTrait {
    fn base_width(&self) -> usize;
    fn full_width(&self) -> usize;
    fn length(&self) -> usize;
    fn num_randomizers(&self) -> usize;
    fn height(&self) -> usize;
    fn omicron(&self) -> BFieldElement;
    fn generator(&self) -> BFieldElement;
    fn order(&self) -> usize;

    fn interpolant_degree(&self) -> Degree {
        self.height() as Degree + self.num_randomizers() as Degree - 1
    }

    fn max_degree(&self) -> Degree {
        let degree_bounds: Vec<Degree> = vec![self.interpolant_degree(); self.base_width() * 2];

        self.base_transition_constraints()
            .iter()
            .map(|air| air.symbolic_degree_bound(&degree_bounds) - (self.height() as Degree - 1))
            .max()
            .unwrap_or(-1)
    }

    fn base_transition_constraints(&self) -> Vec<MPolynomial<BFieldElement>>;
    fn base_boundary_constraints(&self) -> Vec<MPolynomial<BFieldElement>>;
}

pub trait TableMoreTrait {
    fn new_more() -> Self;
}

#[cfg(test)]
mod table_tests {
    use crate::shared_math::stark::brainfuck::instruction_table::InstructionTable;

    use super::*;

    #[test]
    fn table_matrix_interpolate_simple_test() {
        let order: usize = 1 << 32;
        let smooth_generator = BFieldElement::ring_zero()
            .get_primitive_root_of_unity(order as u128)
            .0
            .unwrap();
        let omega = BFieldElement::ring_zero()
            .get_primitive_root_of_unity(16)
            .0
            .unwrap();

        let trace_length = 5;
        let num_randomizers = 2;
        let mut instruction_table =
            InstructionTable::new(trace_length, num_randomizers, smooth_generator, order);
        instruction_table.0.matrix = vec![
            vec![
                BFieldElement::new(42),
                BFieldElement::new(42),
                BFieldElement::new(42),
            ],
            vec![
                BFieldElement::new(2),
                BFieldElement::new(3),
                BFieldElement::new(4),
            ],
            vec![
                BFieldElement::new(55500),
                BFieldElement::new(5550055500),
                BFieldElement::new(55500555005550055500),
            ],
            vec![
                BFieldElement::new(8989),
                BFieldElement::new(8),
                BFieldElement::new(466),
            ],
            vec![
                BFieldElement::new(100),
                BFieldElement::new(50),
                BFieldElement::new(0),
            ],
            vec![
                BFieldElement::new(100),
                BFieldElement::new(50),
                BFieldElement::new(0),
            ],
            vec![
                BFieldElement::new(100),
                BFieldElement::new(50),
                BFieldElement::new(0),
            ],
            vec![
                BFieldElement::new(100),
                BFieldElement::new(50),
                BFieldElement::new(0),
            ],
        ];
        let interpolants: Vec<Polynomial<BFieldElement>> =
            instruction_table
                .0
                .interpolate_columns(omega, 16, vec![0, 1, 2]);

        // Verify that when we evaluate the interpolants in the omicron domain, we get the
        // values that we defined for the matrix values
        let omicron = BFieldElement::ring_zero()
            .get_primitive_root_of_unity(8)
            .0
            .unwrap();
        assert_eq!(
            omicron, instruction_table.0.omicron,
            "omicron must match expected value"
        );
        assert_eq!(
            BFieldElement::new(42),
            interpolants[0].evaluate(&BFieldElement::ring_one())
        );
        assert_eq!(
            BFieldElement::new(42),
            interpolants[1].evaluate(&BFieldElement::ring_one())
        );
        assert_eq!(
            BFieldElement::new(42),
            interpolants[2].evaluate(&BFieldElement::ring_one())
        );
        assert_eq!(BFieldElement::new(2), interpolants[0].evaluate(&omicron));
        assert_eq!(BFieldElement::new(3), interpolants[1].evaluate(&omicron));
        assert_eq!(BFieldElement::new(4), interpolants[2].evaluate(&omicron));
        assert_eq!(
            BFieldElement::new(55500),
            interpolants[0].evaluate(&omicron.mod_pow(2))
        );
        assert_eq!(
            BFieldElement::new(5550055500),
            interpolants[1].evaluate(&omicron.mod_pow(2))
        );
        assert_eq!(
            BFieldElement::new(55500555005550055500),
            interpolants[2].evaluate(&omicron.mod_pow(2))
        );
        assert_eq!(
            BFieldElement::new(100),
            interpolants[0].evaluate(&omicron.mod_pow(7))
        );
        assert_eq!(
            BFieldElement::new(50),
            interpolants[1].evaluate(&omicron.mod_pow(7))
        );
        assert_eq!(
            BFieldElement::new(0),
            interpolants[2].evaluate(&omicron.mod_pow(7))
        );

        // Verify that some random values have been set
        // In all likelyhood, a random B field element will not be zero or
        // repeated when queried three times
        assert!(!interpolants[0].evaluate(&omega).is_zero());
        assert!(!interpolants[1].evaluate(&omega).is_zero());
        assert!(!interpolants[2].evaluate(&omega).is_zero());
        assert_ne!(
            interpolants[0].evaluate(&omega),
            interpolants[1].evaluate(&omega),
        );
        assert_ne!(
            interpolants[1].evaluate(&omega),
            interpolants[2].evaluate(&omega),
        );
        assert_ne!(
            interpolants[2].evaluate(&omega),
            interpolants[0].evaluate(&omega),
        );
    }
}
