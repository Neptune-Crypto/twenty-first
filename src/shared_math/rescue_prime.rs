use std::fmt::Display;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::stark_constraints::BoundaryConstraint;
use crate::shared_math::traits::IdentityValues;

use super::mpolynomial::MPolynomial;
use super::polynomial::Polynomial;
use super::traits::{CyclicGroupGenerator, ModPowU64};

// TODO: Make this work for XFieldElement via trait.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RescuePrime {
    pub m: usize,
    // rate: usize,
    // capacity: usize,
    pub steps_count: usize,
    pub alpha: u64,
    pub alpha_inv: u64,
    pub max_input_length: usize,
    pub output_length: usize,
    pub mds: Vec<Vec<BFieldElement>>,
    pub mds_inv: Vec<Vec<BFieldElement>>,
    pub round_constants: Vec<BFieldElement>,
}

impl RescuePrime {
    fn hash_round(
        &self,
        input_state: Vec<BFieldElement>,
        round_number: usize,
    ) -> Vec<BFieldElement> {
        // S-box
        let mut state: Vec<BFieldElement> = input_state
            .iter()
            .map(|&v| v.mod_pow_u64(self.alpha))
            .collect();

        // Matrix
        let mut temp: Vec<BFieldElement> = vec![input_state[0].ring_zero(); self.m];
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.m {
            for j in 0..self.m {
                temp[i] += self.mds[i][j] * state[j];
            }
        }

        // Add rounding constants
        state = temp
            .into_iter()
            .enumerate()
            .map(|(i, val)| val + self.round_constants[2 * round_number * self.m + i])
            .collect();

        // Backward half-round
        // S-box
        state = state.iter().map(|v| v.mod_pow(self.alpha_inv)).collect();

        // Matrix
        temp = vec![input_state[0].ring_zero(); self.m];
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.m {
            for j in 0..self.m {
                temp[i] += self.mds[i][j] * state[j];
            }
        }

        // Add rounding constants
        state = temp
            .into_iter()
            .enumerate()
            .map(|(i, val)| val + self.round_constants[2 * round_number * self.m + self.m + i])
            .collect();

        state
    }

    pub fn hash(&self, input: &[BFieldElement]) -> Vec<BFieldElement> {
        assert!(
            input.len() <= self.max_input_length,
            "Input length may not exceed expected length. Got length {}, max length is {}",
            input.len(),
            self.max_input_length
        );
        let mut state = input.to_vec();
        state.resize(self.m, BFieldElement::ring_zero());

        state = (0..self.steps_count).fold(state, |state, i| self.hash_round(state, i));

        state[0..self.output_length as usize].to_vec()
    }

    pub fn trace(&self, input: &[BFieldElement]) -> Vec<Vec<BFieldElement>> {
        assert!(
            input.len() <= self.max_input_length,
            "Input length may not exceed expected length. Got length {}, max length is {}",
            input.len(),
            self.max_input_length
        );
        let mut trace: Vec<Vec<BFieldElement>> = Vec::with_capacity(self.steps_count + 1);
        let mut state = input.to_vec();
        state.resize(self.m, BFieldElement::ring_zero());
        trace.push(state.clone());
        for i in 0..self.steps_count {
            let next_state = self.hash_round(state, i);
            trace.push(next_state.clone());
            state = next_state;
        }

        trace
    }

    pub fn eval_and_trace(
        &self,
        input: &[BFieldElement],
    ) -> (Vec<BFieldElement>, Vec<Vec<BFieldElement>>) {
        let trace = self.trace(input);
        let output: Vec<BFieldElement> =
            trace.last().unwrap()[0..self.output_length as usize].to_vec();

        (output, trace)
    }

    /// Return a pair of a list of polynomials, first element in the pair,
    /// (first_round_constants[register], second_round_constants[register])
    pub fn get_round_constant_polynomials(
        &self,
        omicron: BFieldElement,
    ) -> (
        Vec<MPolynomial<BFieldElement>>,
        Vec<MPolynomial<BFieldElement>>,
    ) {
        let domain = omicron.get_cyclic_group_elements(None);
        let variable_count = 2 * self.m + 1;
        let mut first_round_constants: Vec<MPolynomial<BFieldElement>> = vec![];
        for i in 0..self.m {
            let values: Vec<BFieldElement> = self
                .round_constants
                .clone()
                .into_iter()
                .skip(i)
                .step_by(2 * self.m)
                .collect();
            // let coefficients = intt(&values, omicron);
            let points: Vec<(BFieldElement, BFieldElement)> = domain
                .clone()
                .iter()
                .zip(values.iter())
                .map(|(x, y)| (x.to_owned(), y.to_owned()))
                .collect();
            let coefficients =
                Polynomial::<BFieldElement>::slow_lagrange_interpolation(&points).coefficients;
            first_round_constants.push(MPolynomial::lift(
                Polynomial { coefficients },
                0,
                variable_count,
            ));
        }

        let mut second_round_constants: Vec<MPolynomial<BFieldElement>> = vec![];
        for i in 0..self.m {
            let values: Vec<BFieldElement> = self
                .round_constants
                .clone()
                .into_iter()
                .skip(i + self.m)
                .step_by(2 * self.m)
                .collect();
            // let coefficients = intt(&values, omicron);
            let points: Vec<(BFieldElement, BFieldElement)> = domain
                .clone()
                .iter()
                .zip(values.iter())
                .map(|(x, y)| (x.to_owned(), y.to_owned()))
                .collect();
            let coefficients =
                Polynomial::<BFieldElement>::slow_lagrange_interpolation(&points).coefficients;
            second_round_constants.push(MPolynomial::lift(
                Polynomial { coefficients },
                0,
                variable_count,
            ));
        }

        (first_round_constants, second_round_constants)
    }

    // Returns the multivariate polynomial which takes the triplet (domain, trace, next_trace) and
    // returns composition polynomial, which is the evaluation of the air for a specific trace.
    // AIR: [F_p x F_p^m x F_p^m] --> F_p^m
    // The composition polynomial values are low-degree polynomial combinations
    // (as opposed to linear combinations) of the values:
    // `domain` (scalar), `trace` (vector), `next_trace` (vector).
    pub fn get_air_constraints(&self, omicron: BFieldElement) -> Vec<MPolynomial<BFieldElement>> {
        let (first_step_constants, second_step_constants) =
            self.get_round_constant_polynomials(omicron);

        let variable_count = 1 + 2 * self.m;
        let variables = MPolynomial::variables(1 + 2 * self.m, omicron.ring_one());
        let previous_state = &variables[1..(self.m + 1)];
        let next_state = &variables[(self.m + 1)..(2 * self.m + 1)];
        let one = omicron.ring_one();

        let previous_state_pow_alpha = previous_state
            .iter()
            .map(|poly| poly.mod_pow(self.alpha.into(), one))
            .collect::<Vec<MPolynomial<BFieldElement>>>();

        // TODO: Consider refactoring MPolynomial<BFieldElement>
        // ::mod_pow(exp: BigInt, one: BFieldElement) into
        // ::mod_pow_u64(exp: u64)
        let air: Vec<MPolynomial<BFieldElement>> = self
            .mds
            .par_iter()
            .zip(self.mds_inv.par_iter())
            .zip(first_step_constants.into_par_iter())
            .map(|((mds, mds_inv), fsc)| {
                let mut lhs = MPolynomial::from_constant(omicron.ring_zero(), variable_count);
                for k in 0..self.m {
                    lhs += previous_state_pow_alpha[k].scalar_mul(mds[k]);
                }
                lhs += fsc;

                let mut rhs = MPolynomial::from_constant(omicron.ring_zero(), variable_count);
                for k in 0..self.m {
                    rhs += (next_state[k].clone() - second_step_constants[k].clone())
                        .scalar_mul(mds_inv[k]);
                }
                rhs = rhs.mod_pow(self.alpha.into(), one);

                lhs - rhs
            })
            .collect();

        air
    }

    pub fn get_boundary_constraints(
        &self,
        output_elements: &[BFieldElement],
    ) -> Vec<BoundaryConstraint> {
        let mut bcs = vec![];

        // All registers not set by the (padded) input must be zero.
        // If the input is padded, i.e. an input shorter than
        // `max_input_length` is given, this padding does *not*
        // give rise to boundary conditions. If it did, information
        // about the input to the hash would be revealed. In other
        // words: the STARK must not reveal anything about the input,
        // including whether it is shorter than max input length
        // or not.
        for i in self.max_input_length..self.m {
            let bc = BoundaryConstraint {
                cycle: 0,
                register: i,
                value: BFieldElement::ring_zero(),
            };
            bcs.push(bc);
        }

        // The output of the hash function puts a constraint on the trace
        // in the form of boundary conditions
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.output_length {
            let end_constraint = BoundaryConstraint {
                cycle: self.steps_count,
                register: i,
                value: output_elements[i].to_owned(),
            };
            bcs.push(end_constraint);
        }

        bcs
    }
}

impl Display for RescuePrime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "m = {}, N = {}, alpha = {}",
            self.m, self.steps_count, self.alpha
        )
    }
}

#[cfg(test)]
mod rescue_prime_test {
    use itertools::izip;

    use super::*;
    use crate::shared_math::{rescue_prime_params as params, traits::GetPrimitiveRootOfUnity};

    #[test]
    #[should_panic]
    fn disallow_too_long_input_hash_test() {
        // Give a RP hasher with max input length 10 an input of length 11
        let rp = params::rescue_prime_params_bfield_0();
        rp.hash(&vec![
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
        ]);
    }

    #[test]
    #[should_panic]
    fn disallow_too_long_input_trace_test() {
        // Give a RP hasher with max input length 10 an input of length 11
        let rp = params::rescue_prime_params_bfield_0();
        rp.trace(&vec![
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
        ]);
    }

    #[test]
    fn hash_test_new() {
        let input0: Vec<u128> = vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let input1: Vec<u128> = vec![
            16408223883448864076,
            17937404513354951095,
            17784658070603252681,
            4690418723130302842,
            3079713491308723285,
            0,
            0,
            0,
            0,
            0,
        ];
        // Verify that input length does not have to be 10
        let input1_alt: Vec<u128> = vec![
            16408223883448864076,
            17937404513354951095,
            17784658070603252681,
            4690418723130302842,
            3079713491308723285,
        ];
        let input2: Vec<u128> = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
        let output0: Vec<u128> = vec![
            16408223883448864076,
            17937404513354951095,
            17784658070603252681,
            4690418723130302842,
            3079713491308723285,
        ];
        let output1: Vec<u128> = vec![
            15817975225520290566,
            15291972182281732842,
            8434682293988037518,
            16088630906125642382,
            2049996104833593705,
        ];
        let output2: Vec<u128> = vec![
            8224332136734371881,
            8736343702647113032,
            9660176071866133892,
            575034608412522142,
            13216022346578371396,
        ];

        let rp = params::rescue_prime_params_bfield_0();
        for (input_ints, output_ints) in izip!(
            vec![input0, input1, input1_alt, input2],
            vec![output0, output1.clone(), output1, output2]
        ) {
            let input: Vec<BFieldElement> =
                input_ints.into_iter().map(BFieldElement::new).collect();
            let output: Vec<BFieldElement> =
                output_ints.into_iter().map(BFieldElement::new).collect();
            assert_eq!(output, rp.hash(&input));
            assert_eq!(
                &output,
                &rp.trace(&input).last().unwrap().clone()[0..rp.output_length]
            );
        }
    }

    #[test]
    fn air_is_zero_on_execution_trace_test() {
        let rp = params::rescue_prime_small_test_params();
        // let rp = params::rescue_prime_medium_test_params();
        // let rp = params::rescue_prime_params_bfield_0();

        // rescue prime test vector 1
        let omicron_res = BFieldElement::ring_zero().get_primitive_root_of_unity(1 << 5);
        let omicron = omicron_res.0.unwrap();

        // Verify that the round constants polynomials are correct
        let (fst_rc_pol, snd_rc_pol) = rp.get_round_constant_polynomials(omicron);
        for step in 0..rp.steps_count {
            let mut point = vec![omicron.mod_pow(step as u64)];
            point.append(&mut vec![BFieldElement::ring_zero(); 2 * rp.m]);

            for (register, item) in fst_rc_pol.iter().enumerate().take(rp.m) {
                let fst_eval = item.evaluate(&point);
                assert_eq!(rp.round_constants[2 * step * rp.m + register], fst_eval);
            }
            for (register, item) in snd_rc_pol.iter().enumerate().take(rp.m) {
                let snd_eval = item.evaluate(&point);
                assert_eq!(
                    rp.round_constants[2 * step * rp.m + rp.m + register],
                    snd_eval
                );
            }
        }

        // Verify that the AIR constraints evaluation over the trace is zero along the trace
        let input_2 = [BFieldElement::new(42)];
        let trace = rp.trace(&input_2);
        println!("Computing get_air_constraints(omicron)...");
        let now = std::time::Instant::now();
        let air_constraints = rp.get_air_constraints(omicron);
        let elapsed = now.elapsed();
        println!("Completed get_air_constraints(omicron) in {:?}!", elapsed);

        for step in 0..rp.steps_count - 1 {
            println!("Step {}", step);
            for air_constraint in air_constraints.iter() {
                let mut point = vec![omicron.mod_pow(step as u64)];
                for i in 0..rp.m {
                    point.push(trace[step][i]);
                }
                for i in 0..rp.m {
                    point.push(trace[step + 1][i]);
                }
                let eval = air_constraint.evaluate(&point);
                assert!(eval.is_zero());
            }
        }
    }
}
