use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::{Degree, MPolynomial};
use crate::shared_math::other;
use crate::shared_math::traits::{GetPrimitiveRootOfUnity, PrimeField};

type BWord = BFieldElement;

#[derive(Debug, Clone)]
pub struct BaseTable<DataPF, const WIDTH: usize> {
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

pub trait HasBaseTable<DataPF, const WIDTH: usize> {
    fn to_base(&self) -> &BaseTable<DataPF, WIDTH>;
    fn to_mut_base(&mut self) -> &mut BaseTable<DataPF, WIDTH>;
    fn from_base(base: BaseTable<DataPF, WIDTH>) -> Self;

    fn new(
        unpadded_height: usize,
        num_randomizers: usize,
        generator: BWord,
        order: usize,
        matrix: Vec<[DataPF; WIDTH]>,
    ) -> Self
    where
        Self: Sized,
    {
        let omicron = derive_omicron(unpadded_height as u64);
        let base = BaseTable::<DataPF, WIDTH> {
            unpadded_height,
            num_randomizers,
            omicron,
            generator,
            order,
            matrix,
        };

        Self::from_base(base)
    }

    fn width(&self) -> usize {
        WIDTH
    }

    fn unpadded_height(&self) -> usize {
        self.to_base().unpadded_height
    }

    fn padded_height(&self) -> usize {
        other::roundup_npo2(self.unpadded_height() as u64) as usize
    }

    fn num_randomizers(&self) -> usize {
        self.to_base().num_randomizers
    }

    fn omicron(&self) -> BWord {
        self.to_base().omicron
    }

    fn generator(&self) -> BWord {
        self.to_base().generator
    }

    fn order(&self) -> usize {
        self.to_base().order
    }

    fn data(&mut self) -> &mut Vec<[DataPF; WIDTH]> {
        &mut self.to_mut_base().matrix
    }
}

fn derive_omicron(unpadded_height: u64) -> BFieldElement {
    if unpadded_height == 0 {
        return BWord::ring_one();
    }

    let padded_height = other::roundup_npo2(unpadded_height);
    BWord::ring_zero()
        .get_primitive_root_of_unity(padded_height)
        .0
        .unwrap()
}

pub trait Table<DataPF, const WIDTH: usize>: HasBaseTable<DataPF, WIDTH>
where
    DataPF: PrimeField,
{
    // Abstract functions that individual structs implement

    fn name(&self) -> String;

    fn pad(&mut self);

    fn codewords(&self) -> Self;

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
}
