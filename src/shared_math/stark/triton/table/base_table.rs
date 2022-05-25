use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::{Degree, MPolynomial};
use crate::shared_math::other;
use crate::shared_math::traits::{GetPrimitiveRootOfUnity, PrimeField};

type BWord = BFieldElement;

#[derive(Debug, Clone)]
pub struct BaseTable<DataPF, const WIDTH: usize> {
    // The name of the table (for error reporting)
    name: String,

    // The number of `data` rows
    unpadded_height: usize,

    // The number of randomizers...?
    num_randomizers: usize,

    // The omicron...?
    omicron: BWord,

    // The generator...?
    generator: BWord,

    // The order...?
    order: usize,

    // The table data (trace data)
    matrix: Vec<[DataPF; WIDTH]>,
}

impl<DataPF, const WIDTH: usize> BaseTable<DataPF, WIDTH> {
    pub fn new(
        name: &str,
        unpadded_height: usize,
        num_randomizers: usize,
        omicron: BWord,
        generator: BWord,
        order: usize,
        matrix: Vec<[DataPF; WIDTH]>,
    ) -> Self {
        BaseTable::<DataPF, WIDTH> {
            name: name.to_string(),
            unpadded_height,
            num_randomizers,
            omicron,
            generator,
            order,
            matrix,
        }
    }
}

pub trait HasBaseTable<DataPF, const WIDTH: usize> {
    fn new(base: BaseTable<DataPF, WIDTH>) -> Self;
    fn base(&self) -> &BaseTable<DataPF, WIDTH>;
}

pub trait Table<DataPF, const WIDTH: usize>: HasBaseTable<DataPF, WIDTH>
where
    DataPF: PrimeField,
{
    // BaseTable getters

    fn name(&self) -> String {
        self.base().name.clone()
    }

    fn width(&self) -> usize {
        WIDTH
    }

    fn unpadded_height(&self) -> usize {
        self.base().unpadded_height
    }

    fn padded_height(&self) -> usize {
        other::roundup_npo2(self.unpadded_height() as u64) as usize
    }

    fn num_randomizers(&self) -> usize {
        self.base().num_randomizers
    }

    fn omicron(&self) -> BWord {
        self.base().omicron
    }

    fn generator(&self) -> BWord {
        self.base().generator
    }

    fn order(&self) -> usize {
        self.base().order
    }

    fn data(&self) -> &Vec<[DataPF; WIDTH]> {
        &self.base().matrix
    }

    // Abstract functions that individual structs implement

    fn pad(matrix: &mut Vec<[DataPF; WIDTH]>);

    fn boundary_constraints(&self, challenges: &[DataPF]) -> Vec<MPolynomial<DataPF>>;

    fn transition_constraints(&self, challenges: &[DataPF]) -> Vec<MPolynomial<DataPF>>;

    fn terminal_constraints(
        &self,
        challenges: &[DataPF],
        terminals: &[DataPF],
    ) -> Vec<MPolynomial<DataPF>>;

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

    fn derive_omicron(&self) -> BFieldElement {
        if self.unpadded_height() == 0 {
            return BWord::ring_one();
        }

        BWord::ring_zero()
            .get_primitive_root_of_unity(self.padded_height() as u64)
            .0
            .unwrap()
    }
}
