use super::super::fri_domain::FriDomain;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::{Degree, MPolynomial};
use crate::shared_math::other::{is_power_of_two, roundup_npo2};
use crate::shared_math::polynomial::Polynomial;
use crate::shared_math::traits::{GetRandomElements, PrimeField};
use crate::shared_math::x_field_element::XFieldElement;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

type BWord = BFieldElement;
type XWord = XFieldElement;

#[derive(Debug, Clone)]
pub struct BaseTable<DataPF> {
    // The width of each `data` row
    width: usize,

    // The number of `data` rows after padding
    padded_height: usize,

    // The number of randomizers...?
    num_randomizers: usize,

    // The omicron...?
    omicron: DataPF,

    // The generator...?
    generator: DataPF,

    // The order...?
    order: usize,

    // The table data (trace data)
    matrix: Vec<Vec<DataPF>>,
}

impl<DataPF: PrimeField> BaseTable<DataPF> {
    pub fn new(
        width: usize,
        padded_height: usize,
        num_randomizers: usize,
        omicron: DataPF,
        generator: DataPF,
        order: usize,
        matrix: Vec<Vec<DataPF>>,
    ) -> Self {
        if !matrix.is_empty() {
            let actual_padded_height = roundup_npo2(matrix.len() as u64) as usize;
            assert_eq!(
                padded_height, actual_padded_height,
                "Expected padded_height {}, but data pads to {}",
                padded_height, actual_padded_height,
            );
        }

        BaseTable {
            width,
            padded_height,
            num_randomizers,
            omicron,
            generator,
            order,
            matrix,
        }
    }

    /// Create a `BaseTable<DataPF>` with the same parameters, but new `matrix` data.
    pub fn with_data(&self, matrix: Vec<Vec<DataPF>>) -> Self {
        BaseTable::new(
            self.width,
            self.padded_height,
            self.num_randomizers,
            self.omicron,
            self.generator,
            self.order,
            matrix,
        )
    }
}

impl BaseTable<BWord> {
    pub fn with_lifted_data(&self, matrix: Vec<Vec<XWord>>) -> BaseTable<XWord> {
        BaseTable::new(
            self.width,
            self.padded_height,
            self.num_randomizers,
            self.omicron.lift(),
            self.generator.lift(),
            self.order,
            matrix,
        )
    }
}

pub trait HasBaseTable<DataPF: PrimeField> {
    fn to_base(&self) -> &BaseTable<DataPF>;
    fn to_mut_base(&mut self) -> &mut BaseTable<DataPF>;

    fn width(&self) -> usize {
        self.to_base().width
    }

    fn padded_height(&self) -> usize {
        self.to_base().padded_height
    }

    fn num_randomizers(&self) -> usize {
        self.to_base().num_randomizers
    }

    fn omicron(&self) -> DataPF {
        self.to_base().omicron
    }

    fn generator(&self) -> DataPF {
        self.to_base().generator
    }

    fn order(&self) -> usize {
        self.to_base().order
    }

    fn data(&self) -> &Vec<Vec<DataPF>> {
        &self.to_base().matrix
    }

    fn mut_data(&mut self) -> &mut Vec<Vec<DataPF>> {
        &mut self.to_mut_base().matrix
    }
}

pub fn derive_omicron<DataPF: PrimeField>(padded_height: u64, dummy: DataPF) -> DataPF {
    debug_assert!(is_power_of_two(padded_height));
    dummy.get_primitive_root_of_unity(padded_height).0.unwrap()
}

pub fn pad_height(height: usize) -> usize {
    if height == 0 {
        0
    } else {
        roundup_npo2(height as u64) as usize
    }
}

pub trait Table<DataPF>: HasBaseTable<DataPF>
where
    // Self: Sized,
    DataPF: PrimeField + GetRandomElements,
{
    // Abstract functions that individual structs implement

    fn name(&self) -> String;

    fn pad(&mut self);

    fn base_transition_constraints(&self) -> Vec<MPolynomial<DataPF>>;

    // Generic functions common to all tables

    fn interpolant_degree(&self) -> Degree {
        let height: Degree = self.padded_height().try_into().unwrap_or(0);
        let num_randomizers: Degree = self.num_randomizers().try_into().unwrap_or(0);

        height + num_randomizers - 1
    }

    /// Returns the relation between the FRI domain and the omicron domain
    fn unit_distance(&self, omega_order: usize) -> usize {
        let height = self.padded_height();
        if height == 0 {
            0
        } else {
            omega_order / height
        }
    }

    fn max_degree(&self) -> Degree {
        let degree_bounds: Vec<Degree> = vec![self.interpolant_degree(); self.width() * 2];

        self.base_transition_constraints()
            .iter()
            .map(|air| {
                let symbolic_degree_bound: Degree = air.symbolic_degree_bound(&degree_bounds);
                let padded_height: Degree = self.padded_height() as Degree;

                symbolic_degree_bound - padded_height + 1
            })
            .max()
            .unwrap_or(-1)
    }

    fn low_degree_extension(&self, fri_domain: &FriDomain<DataPF>) -> Vec<Vec<DataPF>> {
        // FIXME: Table<> supports Vec<[DataPF; WIDTH]>, but FriDomain does not (yet).
        self.interpolate_columns(fri_domain.omega, fri_domain.length)
            .par_iter()
            .map(|polynomial| fri_domain.evaluate(polynomial))
            .collect()
    }

    /// Return the interpolation of columns. The `column_indices` variable
    /// must be called with *all* the column indices for this particular table,
    /// if it is called with a subset, it *will* fail.
    fn interpolate_columns(&self, omega: DataPF, omega_order: usize) -> Vec<Polynomial<DataPF>> {
        // FIXME: Inject `rng` instead.
        let mut rng = rand::thread_rng();

        // Ensure that `matrix` is set and padded before running this function
        assert_eq!(
            self.padded_height(),
            self.data().len(),
            "{}: Table data must be padded before interpolation",
            self.name()
        );

        assert!(
            omega.mod_pow_u32(omega_order as u32).is_one(),
            "omega must have indicated order"
        );
        assert!(
            !omega.mod_pow_u32(omega_order as u32 / 2).is_one(),
            "omega must be a primitive root of indicated order"
        );

        if self.padded_height() == 0 {
            return vec![Polynomial::ring_zero(); self.width()];
        }

        assert!(
            self.padded_height() >= self.num_randomizers(),
            "Temporary restriction that number of randomizers must not exceed table height"
        );

        // FIXME: Unfold with multiplication instead of mapping with power.
        let omicron_domain: Vec<DataPF> = (0..self.padded_height())
            .map(|i| self.omicron().mod_pow_u32(i as u32))
            .collect();

        let randomizer_domain: Vec<DataPF> = (0..self.num_randomizers())
            .map(|i| omega * omicron_domain[i])
            .collect();

        let domain: Vec<DataPF> = vec![omicron_domain, randomizer_domain].concat();

        let mut valuess: Vec<Vec<DataPF>> = vec![];

        let data = self.data();
        for c in 0..self.width() {
            let trace: Vec<DataPF> = data.iter().map(|row| row[c]).collect();
            let randomizers: Vec<DataPF> =
                DataPF::random_elements(self.num_randomizers(), &mut rng);
            let values = vec![trace, randomizers].concat();
            assert_eq!(
                values.len(),
                domain.len(),
                "Length of x values and y values must match"
            );
            valuess.push(values);
        }

        valuess
            .par_iter()
            .map(|values| {
                Polynomial::<DataPF>::fast_interpolate(&domain, values, &omega, omega_order)
            })
            .collect()
    }
}

#[cfg(test)]
mod test_base_table {
    use crate::shared_math::other;
    use crate::shared_math::stark::triton::table::base_table::pad_height;

    #[ignore]
    #[test]
    /// padding should be idempotent.
    fn pad_height_test() {
        assert_eq!(0, pad_height(0));
        for x in 1..=1025 {
            let padded_x = pad_height(x);
            assert_eq!(other::roundup_npo2(x as u64) as usize, padded_x);
            assert_eq!(padded_x, pad_height(padded_x))
        }
    }
}
