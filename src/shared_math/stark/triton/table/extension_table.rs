use super::base_table::Table;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::{Degree, MPolynomial};
use crate::shared_math::polynomial::Polynomial;
use crate::shared_math::traits::{Inverse, ModPowU32, PrimeField};
use crate::shared_math::x_field_element::XFieldElement;
use crate::shared_math::xfri::FriDomain;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

// Generic methods specifically for tables that have been extended

type XWord = XFieldElement;

pub trait ExtensionTable<const FULL_WIDTH: usize>: Table<XWord, FULL_WIDTH> {
    fn all_quotient_degree_bounds(&self, challenges: &[XWord], terminals: &[XWord]) -> Vec<Degree> {
        vec![
            self.boundary_quotient_degree_bounds(challenges),
            self.transition_quotient_degree_bounds(challenges),
            self.terminal_quotient_degree_bounds(challenges, terminals),
        ]
        .concat()
    }

    fn boundary_quotient_degree_bounds(&self, challenges: &[XWord]) -> Vec<Degree> {
        let max_degrees: Vec<Degree> = vec![self.interpolant_degree(); FULL_WIDTH];

        let degree_bounds: Vec<Degree> = self
            .boundary_constraints(challenges)
            .iter()
            .map(|mpo| mpo.symbolic_degree_bound(&max_degrees) - 1)
            .collect();

        degree_bounds
    }

    fn transition_quotient_degree_bounds(&self, challenges: &[XWord]) -> Vec<Degree> {
        let max_degrees: Vec<Degree> = vec![self.interpolant_degree(); 2 * FULL_WIDTH];

        let transition_constraints = self.transition_constraints(challenges);

        // Safe because padded height is at most 2^30.
        let padded_height: Degree = self.padded_height().try_into().unwrap();

        transition_constraints
            .iter()
            .map(|mpo| mpo.symbolic_degree_bound(&max_degrees) - padded_height + 1)
            .collect::<Vec<Degree>>()
    }

    fn terminal_quotient_degree_bounds(
        &self,
        challenges: &[XWord],
        terminals: &[XWord],
    ) -> Vec<Degree> {
        let max_degrees: Vec<Degree> = vec![self.interpolant_degree(); FULL_WIDTH];
        self.terminal_constraints(challenges, terminals)
            .iter()
            .map(|mpo| mpo.symbolic_degree_bound(&max_degrees) - 1)
            .collect::<Vec<Degree>>()
    }

    fn all_quotients(
        &self,
        fri_domain: &FriDomain,
        codewords: &[Vec<XWord>],
        challenges: &[XWord],
        terminals: &[XWord],
    ) -> Vec<Vec<XWord>> {
        let boundary_quotients = self.boundary_quotients(fri_domain, codewords, challenges);
        let transition_quotients = self.transition_quotients(fri_domain, codewords, challenges);
        let terminal_quotients =
            self.terminal_quotients(fri_domain, codewords, challenges, terminals);

        vec![boundary_quotients, transition_quotients, terminal_quotients].concat()
    }

    fn transition_quotients(
        &self,
        fri_domain: &FriDomain,
        codewords: &[Vec<XWord>],
        challenges: &[XWord],
    ) -> Vec<Vec<XWord>> {
        let one = BFieldElement::ring_one();
        let x_values: Vec<BFieldElement> = fri_domain.b_domain_values();
        let subgroup_zerofier: Vec<BFieldElement> = x_values
            .iter()
            .map(|x| x.mod_pow_u32(self.padded_height() as u32) - one)
            .collect();
        let subgroup_zerofier_inverse = if self.padded_height() == 0 {
            subgroup_zerofier
        } else {
            BFieldElement::batch_inversion(subgroup_zerofier)
        };

        let omicron_inverse = self.omicron().inverse();
        let zerofier_inverse: Vec<BFieldElement> = x_values
            .into_iter()
            .enumerate()
            .map(|(i, x)| subgroup_zerofier_inverse[i] * (x - omicron_inverse))
            .collect();

        let transition_constraints = self.transition_constraints(challenges);

        let mut quotients: Vec<Vec<XWord>> = vec![];
        let unit_distance = self.unit_distance(fri_domain.length);
        for tc in transition_constraints.iter() {
            let quotient_codeword: Vec<XWord> = zerofier_inverse
                .par_iter()
                .enumerate()
                .map(|(i, z_inverse)| {
                    let current: Vec<XWord> = (0..FULL_WIDTH).map(|j| codewords[j][i]).collect();
                    let next: Vec<XWord> = (0..FULL_WIDTH)
                        .map(|j| codewords[j][(i + unit_distance) % fri_domain.length])
                        .collect();
                    let point = vec![current, next].concat();
                    let composition_evaluation = tc.evaluate(&point);
                    composition_evaluation * z_inverse.lift()
                })
                .collect();

            quotients.push(quotient_codeword);
        }
        // If the `DEBUG` environment variable is set, interpolate the quotient and check the degree

        if std::env::var("DEBUG").is_ok() {
            for (i, qc) in quotients.iter().enumerate() {
                let interpolated: Polynomial<XWord> = fri_domain.x_interpolate(qc);
                assert!(
                    interpolated.degree() < fri_domain.length as isize - 1,
                    "Degree of transition quotient number {} in {} must not be maximal. Got degree {}, and FRI domain length was {}. Unsatisfied constraint: {}", i, self.name(), interpolated.degree(), fri_domain.length, transition_constraints[i]
                );
            }
        }

        quotients
    }

    fn terminal_quotients(
        &self,
        fri_domain: &FriDomain,
        codewords: &[Vec<XWord>],
        challenges: &[XWord],
        terminals: &[XWord],
    ) -> Vec<Vec<XWord>> {
        let omicron_inverse = self.omicron().inverse();

        // The zerofier for the terminal quotient has a root in the last
        // value in the cyclical group generated from omicron.
        let zerofier_codeword: Vec<BFieldElement> = fri_domain
            .b_domain_values()
            .into_iter()
            .map(|x| x - omicron_inverse)
            .collect();
        let zerofier_inverse = BFieldElement::batch_inversion(zerofier_codeword);
        let terminal_constraints = self.terminal_constraints(challenges, terminals);
        let mut quotient_codewords: Vec<Vec<XWord>> = vec![];
        for termc in terminal_constraints.iter() {
            let quotient_codeword: Vec<XWord> = (0..fri_domain.length)
                .into_par_iter()
                .map(|i| {
                    let point: Vec<XWord> = (0..FULL_WIDTH).map(|j| codewords[j][i]).collect();
                    termc.evaluate(&point) * zerofier_inverse[i].lift()
                })
                .collect();
            quotient_codewords.push(quotient_codeword);
        }

        if std::env::var("DEBUG").is_ok() {
            for (i, qc) in quotient_codewords.iter().enumerate() {
                let interpolated = fri_domain.x_interpolate(qc);
                assert!(
                    interpolated.degree() < fri_domain.length as isize - 1,
                    "Degree of terminal quotient number {} in {} must not be maximal. Got degree {}, and FRI domain length was {}. Unsatisfied constraint: {}", i, self.name(), interpolated.degree(), fri_domain.length, terminal_constraints[i]
                );
            }
        }

        quotient_codewords
    }

    fn boundary_quotients(
        &self,
        fri_domain: &FriDomain,
        codewords: &[Vec<XWord>],
        challenges: &[XWord],
    ) -> Vec<Vec<XWord>> {
        assert!(!codewords.is_empty(), "Codewords must be non-empty");
        let mut quotient_codewords: Vec<Vec<XWord>> = vec![];
        let boundary_constraints: Vec<MPolynomial<XWord>> = self.boundary_constraints(challenges);
        let one = BFieldElement::ring_one();
        let zerofier: Vec<BFieldElement> = (0..fri_domain.length)
            .map(|i| fri_domain.b_domain_value(i as u32) - one)
            .collect();
        let zerofier_inverse = BFieldElement::batch_inversion(zerofier);
        for bc in boundary_constraints {
            let quotient_codeword: Vec<XWord> = (0..fri_domain.length)
                .into_par_iter()
                .map(|i| {
                    let point: Vec<XWord> = (0..FULL_WIDTH).map(|j| codewords[j][i]).collect();
                    bc.evaluate(&point) * zerofier_inverse[i].lift()
                })
                .collect();
            quotient_codewords.push(quotient_codeword);
        }

        // If the `DEBUG` environment variable is set, run this extra validity check
        if std::env::var("DEBUG").is_ok() {
            for (i, qc) in quotient_codewords.iter().enumerate() {
                let interpolated = fri_domain.x_interpolate(qc);
                assert!(
                    interpolated.degree() < fri_domain.length as isize - 1,
                    "Degree of boundary quotient number {} in {} must not be maximal. Got degree {}, and FRI domain length was {}", i, self.name(), interpolated.degree(), fri_domain.length
                );
            }
        }

        quotient_codewords
    }
}
