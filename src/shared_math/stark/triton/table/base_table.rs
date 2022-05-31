use super::super::fri_domain::FriDomain;
use super::processor_table::ProcessorTable;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::{Degree, MPolynomial};
use crate::shared_math::other;
use crate::shared_math::polynomial::Polynomial;
use crate::shared_math::traits::{
    GetPrimitiveRootOfUnity, GetRandomElements, IdentityValues, ModPowU32, PrimeField,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

type BWord = BFieldElement;

#[derive(Debug, Clone)]
pub struct BaseTable<DataPF> {
    // The width of each `data` row
    width: usize,

    // The number of `data` rows
    unpadded_height: usize,

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

pub trait HasBaseTable<DataPF: PrimeField> {
    fn to_base(&self) -> &BaseTable<DataPF>;
    fn to_mut_base(&mut self) -> &mut BaseTable<DataPF>;
    // fn from_base(base: BaseTable<DataPF>) -> Self;
    // fn new(
    //     width: usize,
    //     unpadded_height: usize,
    //     num_randomizers: usize,
    //     generator: DataPF,
    //     order: usize,
    //     matrix: Vec<Vec<DataPF>>,
    // ) -> Self
    // where
    //     Self: Sized,
    // {
    //     let dummy = generator;
    //     let omicron = derive_omicron::<DataPF>(unpadded_height as u64, dummy);
    //     let base = BaseTable::<DataPF> {
    //         width,
    //         unpadded_height,
    //         num_randomizers,
    //         omicron,
    //         generator,
    //         order,
    //         matrix,
    //     };

    //     Self::from_base(base)
    // }

    fn width(&self) -> usize {
        self.to_base().width
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

fn derive_omicron<DataPF: PrimeField>(unpadded_height: u64, dummy: DataPF) -> DataPF {
    if unpadded_height == 0 {
        // FIXME: Cannot return 1 because of IdentityValues.
        return dummy.ring_one();
    }

    let padded_height = other::roundup_npo2(unpadded_height);
    dummy.get_primitive_root_of_unity(padded_height).0.unwrap()
}

pub trait Table<DataPF>: HasBaseTable<DataPF>
where
    // Self: Sized,
    DataPF: PrimeField + GetRandomElements,
{
    // Abstract functions that individual structs implement

    fn name(&self) -> String;

    fn pad(&mut self);

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

    // fn codewords(&self, fri_domain: &FriDomain<DataPF>) -> Self {
    //     let codewords = self.low_degree_extension(fri_domain);
    //     Self::new(
    //         self.width(),
    //         self.unpadded_height(),
    //         self.num_randomizers(),
    //         self.generator(),
    //         self.order(),
    //         codewords,
    //     )
    // }

    fn low_degree_extension(&self, fri_domain: &FriDomain<DataPF>) -> Vec<Vec<DataPF>> {
        // FIXME: Table<> supports Vec<[DataPF; WIDTH]>, but FriDomain does not (yet).
        self.interpolate_columns(fri_domain.omega, fri_domain.length)
            .par_iter()
            .map(|polynomial| {
                fri_domain
                    .evaluate(polynomial)
                    .try_into()
                    .expect("FriDomain.evaluate: Could not convert Vec<DataPF> til [DataPF; WIDTH]")
            })
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
