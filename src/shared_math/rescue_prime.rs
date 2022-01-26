use std::fmt::Display;

use serde::{Deserialize, Serialize};

use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::stark::BoundaryConstraint;
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

    /// Return the Rescue-Prime hash value
    pub fn hash(&self, input: &BFieldElement) -> BFieldElement {
        let mut state = vec![input.ring_zero(); self.m];
        state[0] = input.to_owned();

        state = (0..self.steps_count).fold(state, |state, i| self.hash_round(state, i));

        state[0]
    }

    pub fn trace(&self, input: &BFieldElement) -> Vec<Vec<BFieldElement>> {
        let mut trace: Vec<Vec<BFieldElement>> = vec![];
        let mut state = vec![input.ring_zero(); self.m];
        state[0] = input.to_owned();
        trace.push(state.clone());

        // It could be cool to write this with `scan` instead of a for-loop, but I couldn't get that to work
        for i in 0..self.steps_count {
            let next_state = self.hash_round(state, i);
            trace.push(next_state.clone());
            state = next_state;
        }

        trace
    }

    pub fn eval_and_trace(
        &self,
        input: &BFieldElement,
    ) -> (BFieldElement, Vec<Vec<BFieldElement>>) {
        let trace = self.trace(input);
        let output = trace.last().unwrap()[0];

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
            let coefficients = Polynomial::slow_lagrange_interpolation(&points).coefficients;
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
            let coefficients = Polynomial::slow_lagrange_interpolation(&points).coefficients;
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
        let mut air: Vec<MPolynomial<BFieldElement>> = vec![];

        let previous_state_pow_alpha = previous_state
            .iter()
            .map(|poly| poly.mod_pow(self.alpha.into(), one))
            .collect::<Vec<MPolynomial<BFieldElement>>>();

        // TODO: Consider refactoring MPolynomial<BFieldElement>
        // ::mod_pow(exp: BigInt, one: BFieldElement) into
        // ::mod_pow_u64(exp: u64)
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.m {
            let mut lhs = MPolynomial::from_constant(omicron.ring_zero(), variable_count);
            for k in 0..self.m {
                lhs += previous_state_pow_alpha[k].scalar_mul(self.mds[i][k]);
            }
            lhs += first_step_constants[i].clone();

            let mut rhs = MPolynomial::from_constant(omicron.ring_zero(), variable_count);
            for k in 0..self.m {
                rhs += (next_state[k].clone() - second_step_constants[k].clone())
                    .scalar_mul(self.mds_inv[i][k]);
            }
            // println!("rhs.variable_count = {}, rhs.degree() = {}", rhs.variable_count, rhs.degree());
            rhs = rhs.mod_pow(self.alpha.into(), one);
            // println!("done mod_pow'ing {}", i);

            air.push(lhs - rhs);
        }

        air
    }

    pub fn get_boundary_constraints(
        &self,
        output_element: BFieldElement,
    ) -> Vec<BoundaryConstraint> {
        let mut bcs = vec![];

        // All but the first registers should be 0 in the first cycle.
        for i in 1..self.m {
            let bc = BoundaryConstraint {
                cycle: 0,
                register: i,
                value: BFieldElement::ring_zero(),
            };
            bcs.push(bc);
        }

        // Register 0 should have output_element as value in last cycle.
        let end_constraint = BoundaryConstraint {
            cycle: self.steps_count,
            register: 0,
            value: output_element.to_owned(),
        };
        bcs.push(end_constraint);

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
    use super::*;
    use crate::shared_math::{rescue_prime_params as params, traits::GetPrimitiveRootOfUnity};

    #[test]
    fn hash_test() {
        let rp = params::rescue_prime_params_bfield_0();

        // Calculated with stark-anatomy tutorial implementation, starting with hash(1)
        let one = BFieldElement::new(1);
        let expected_sequence: Vec<BFieldElement> = vec![
            16408223883448864076,
            14851226605068667585,
            2638999062907144857,
            11729682885064735215,
            18241842748565968364,
            12761136320817622587,
            6569784252060404379,
            7456670293305349839,
            12092401435052133560,
        ]
        .iter()
        .map(|elem| BFieldElement::new(*elem))
        .collect();

        let mut actual = rp.hash(&one);
        for expected in expected_sequence {
            assert_eq!(expected, actual);
            actual = rp.hash(&expected);
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

            for register in 0..rp.m {
                let fst_eval = fst_rc_pol[register].evaluate(&point);
                assert_eq!(rp.round_constants[2 * step * rp.m + register], fst_eval);
            }
            for register in 0..rp.m {
                let snd_eval = snd_rc_pol[register].evaluate(&point);
                assert_eq!(
                    rp.round_constants[2 * step * rp.m + rp.m + register],
                    snd_eval
                );
            }
        }

        // Verify that the AIR constraints evaluation over the trace is zero along the trace
        let input_2 = BFieldElement::new(42);
        let trace = rp.trace(&input_2);
        println!("Computing get_air_constraints(omicron)...");
        let now = std::time::Instant::now();
        let air_constraints = rp.get_air_constraints(omicron);
        let elapsed = now.elapsed();
        println!("Completed get_air_constraints(omicron) in {:?}!", elapsed);

        for step in 0..rp.steps_count - 1 {
            println!("Step {}", step);
            for air_constraint in air_constraints.iter() {
                let mut point = vec![];
                point.push(omicron.mod_pow(step as u64));
                for i in 0..rp.m {
                    point.push(trace[step][i].clone());
                }
                for i in 0..rp.m {
                    point.push(trace[step + 1][i].clone());
                }
                let eval = air_constraint.evaluate(&point);
                assert!(eval.is_zero());
            }
        }
    }
}
