use crate::shared_math::low_degree_test;
use crate::shared_math::low_degree_test::LowDegreeProof;
use crate::shared_math::ntt::{intt, ntt};
use crate::shared_math::polynomial::Polynomial;
use crate::shared_math::prime_field_element::{PrimeField, PrimeFieldElement};
use crate::shared_math::stark::{StarkProofError, StarkVerifyError};
use crate::shared_math::traits::IdentityValues;
use crate::util_types::merkle_tree::{CompressedAuthenticationPath, MerkleTree};
use crate::utils;
use serde::Serialize;
use std::error::Error;

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct CollatzStarkProof {
    bq_merkle_root: [u8; 32],
    composition_polynomial_merkle_root: [u8; 32],
    boundary_quotient_authentication_paths: Vec<CompressedAuthenticationPath<i128>>,
    boundary_quotient_authentication_paths_next: Vec<CompressedAuthenticationPath<i128>>,
    composite_polynomial_fri: LowDegreeProof<i128>,
}

// Alan: Should num_registers be part of this struct?
pub struct CollatzClaim<'a> {
    start_value: PrimeFieldElement<'a>,
    end_value: PrimeFieldElement<'a>,
}

// Outputs values in the form of extended_trace[cycle][register]
pub fn get_extended_computational_traces<'a>(
    omega: &'a PrimeFieldElement,
    omicron: &'a PrimeFieldElement,
    expansion_factor: usize,
    value_trace: Vec<i128>,
) -> (
    Vec<Vec<PrimeFieldElement<'a>>>,
    Vec<Polynomial<PrimeFieldElement<'a>>>,
) {
    // We do a lot of indexing where we iterate over i in a[i][c] which is
    // very slow because of cache misses. We don't care as this is only a
    // toy example for building a multi-dimensional (multi-register) STARK.
    let binary_trace: Vec<Vec<PrimeFieldElement>> = convert_trace(omega.field, value_trace);

    let num_trace_elements = binary_trace.len();
    let extended_length = num_trace_elements * expansion_factor;
    let num_registers = binary_trace[0].len();

    // Holds extended trace values in the form of `trace_value[cycle][register]`
    let mut extended_traces: Vec<Vec<PrimeFieldElement>> =
        vec![vec![omega.ring_zero(); num_registers]; extended_length];
    let mut trace_interpolants: Vec<Polynomial<PrimeFieldElement>> = vec![];

    // Interpolate the polynomium for each bit-register value
    for i in 0..num_registers {
        let bit_values: Vec<PrimeFieldElement> = binary_trace.iter().map(|x| x[i]).collect();
        let coefficients = intt(&bit_values, omicron);
        trace_interpolants.push(Polynomial {
            coefficients: coefficients.clone(),
        });

        let mut padded_trace_interpolant_coefficients = coefficients;
        padded_trace_interpolant_coefficients.append(&mut vec![
            omega.ring_zero();
            extended_length - num_trace_elements
        ]);

        let extended_trace = ntt(&padded_trace_interpolant_coefficients, omega);
        for j in 0..extended_length {
            extended_traces[j][i] = extended_trace[j];
        }
    }

    (extended_traces, trace_interpolants)
}

fn i128_to_binary_elements(field: &PrimeField, input: i128) -> Vec<PrimeFieldElement> {
    // This is horribly inefficient, as the length of the output vector could be smaller than
    // 128 but we are not interesting in optimizing this toy example.
    let mut output: Vec<PrimeFieldElement> = vec![PrimeFieldElement::new(0, field); 128];
    let mut mask = 0x01i128;
    for elem in output.iter_mut() {
        elem.value = if mask & input == 0 { 0 } else { 1 };
        mask <<= 1;
    }

    output
}

// outputs trace in the form of `output[step][bit_count]`
fn convert_trace(field: &PrimeField, trace: Vec<i128>) -> Vec<Vec<PrimeFieldElement>> {
    trace
        .into_iter()
        .map(|x| i128_to_binary_elements(field, x))
        .collect()
}

fn get_collatz_trace(input: i128) -> Vec<i128> {
    let mut value = input;
    let mut trace: Vec<i128> = vec![input];
    while value != 1i128 {
        if value % 2 == 0 {
            value /= 2;
        } else {
            value = 3 * value + 1;
        }
        trace.push(value);
    }

    // Ensure that trace has a length which is a power of 2, otherwise we can't do NTT!
    while trace.len() & (trace.len() - 1) != 0 {
        if value % 2 == 0 {
            value /= 2;
        } else {
            value = 3 * value + 1;
        }
        trace.push(value);
    }

    trace
}

fn build_bit_airs<'a>(
    omega: &PrimeFieldElement<'a>,
    extended_computational_traces: Vec<Vec<PrimeFieldElement<'a>>>,
) -> Vec<Polynomial<PrimeFieldElement<'a>>> {
    let num_registers = extended_computational_traces[0].len();

    let air_codewords = extended_computational_traces
        .iter()
        .map(|x| {
            x.iter()
                .map(|&y| y * y - y)
                .collect::<Vec<PrimeFieldElement>>()
        })
        .collect::<Vec<Vec<PrimeFieldElement>>>();

    // Build the monomial representation of the polynomials
    let mut air_polynomials: Vec<Polynomial<PrimeFieldElement>> = vec![];
    for register in 0..num_registers {
        // Probably quite slow, because of cache misses
        let register_values: Vec<PrimeFieldElement> =
            air_codewords.iter().map(|x| x[register]).collect();
        air_polynomials.push(Polynomial {
            coefficients: intt(&register_values, omega),
        });
    }

    air_polynomials
}

fn build_sum_air<'a>(
    omega: &'a PrimeFieldElement,
    expansion_factor: usize,
    extended_computational_traces: Vec<Vec<PrimeFieldElement<'a>>>,
) -> Polynomial<PrimeFieldElement<'a>> {
    // Build the arithmetic intermediate representation (AIR). This is a multivariate function
    // built from the extended trace that evaluates to zero along the original (non-extended)
    // computational trace.
    let extended_domain_length = extended_computational_traces.len();
    let mut air_codeword = vec![omega.ring_zero(); extended_domain_length];
    let zero = omega.ring_zero();
    let one = omega.ring_one();
    let two = PrimeFieldElement::new(2, omega.field);
    let three = PrimeFieldElement::new(3, omega.field);

    // Build AIR codeword
    for i in 0..extended_domain_length {
        let current_sum = extended_computational_traces[i]
            .iter()
            .enumerate()
            .fold(zero, |sum, (i, val)| *val * two.mod_pow(i as i128) + sum);
        let next_sum = extended_computational_traces
            [(i + expansion_factor) % extended_domain_length]
            .iter()
            .enumerate()
            .fold(zero, |sum, (i, val)| *val * two.mod_pow(i as i128) + sum);
        air_codeword[i] = extended_computational_traces[i][0]
            * (three * current_sum + one - next_sum)
            + (one - extended_computational_traces[i][0]) * (current_sum - two * next_sum);
    }

    // Build and return AIR polynomial
    Polynomial {
        coefficients: intt(&air_codeword, omega),
    }
}

// Returns a polynomial that is zero along the trace and non-zero
// all other places, with a leading coefficient of 1.
// Returns P(x) = Π(x-ο^i), over the entire trace domain, the
// domain generated by omicron.
fn get_trace_zerofier<'a>(
    trace_length: usize,
    primefield_element: &PrimeFieldElement<'a>,
) -> Polynomial<PrimeFieldElement<'a>> {
    let mut trace_zerofier_coefficients = vec![primefield_element.ring_zero(); trace_length + 1];
    trace_zerofier_coefficients[0] = -primefield_element.ring_one();
    trace_zerofier_coefficients[trace_length] = primefield_element.ring_one();

    Polynomial {
        coefficients: trace_zerofier_coefficients,
    }
}

// The zerofier for the transition constraint has a root along the entire trace,
// except for the last point on the trace. So we need to divide out the factor
// `(x - last_x_value_of_trace)`.
fn get_transition_zerofier_polynomials_sum<'a>(
    trace_length: usize,
    last_x_value_of_trace: &PrimeFieldElement<'a>,
) -> (
    Polynomial<PrimeFieldElement<'a>>,
    Polynomial<PrimeFieldElement<'a>>,
) {
    let num = get_trace_zerofier(trace_length, last_x_value_of_trace);
    let extra_root = Polynomial {
        coefficients: vec![-*last_x_value_of_trace, last_x_value_of_trace.ring_one()],
    };

    (num, extra_root)
}

// Transform the collatz-sequence boundary conditions into a representation
// appropriate to get the boundary interpolants and zerofier polynomials.
fn format_boundary_values<'a>(
    start_value: i128,
    end_value: i128,
    omicron: &'a PrimeFieldElement,
) -> (
    Vec<Vec<(PrimeFieldElement<'a>, PrimeFieldElement<'a>)>>,
    Vec<PrimeFieldElement<'a>>,
) {
    let start_y_value_as_bits = i128_to_binary_elements(omicron.field, start_value);
    let start_points: std::iter::Map<_, _> = start_y_value_as_bits
        .iter()
        .map(|x| (omicron.ring_one(), *x));
    let end_y_value_as_bits = i128_to_binary_elements(omicron.field, end_value);
    let end_points: std::iter::Map<_, _> = end_y_value_as_bits.iter().map(|x| (omicron.inv(), *x));
    let boundary_points: Vec<Vec<(PrimeFieldElement, PrimeFieldElement)>> = start_points
        .zip(end_points)
        .map(|(start, end)| vec![start, end])
        .collect();

    // Here we assume that the boundary points are defined for the same x-values over all registers
    let boundary_roots: Vec<PrimeFieldElement> = boundary_points[0]
        .iter()
        .map(|points| points.0)
        .collect::<Vec<PrimeFieldElement>>();

    (boundary_points, boundary_roots)
}

// Return the interpolants for the provided points. This is the `L(x)` in the equation
// to derive the boundary quotient: `Q_B(x) = (ECT(x) - L(x)) / Z_B(x)`.
fn get_boundary_interpolants<'a>(
    pointss: &'a [Vec<(PrimeFieldElement, PrimeFieldElement)>],
) -> Vec<Polynomial<PrimeFieldElement<'a>>> {
    pointss
        .iter()
        .map(|points| Polynomial::slow_lagrange_interpolation(points))
        .collect::<Vec<Polynomial<PrimeFieldElement>>>()
}

fn get_composition_polynomial_evaluation<'a>(
    num_registers: usize,
    num_steps: usize,
    x: &'a PrimeFieldElement,
    composition_polynomial_weights: &'a [PrimeFieldElement],
    sum_transition_quotient_evaluation: &'a PrimeFieldElement,
    bit_transition_quotient_evaluations: &'a [PrimeFieldElement],
    boundary_quotient_evaluations: &'a [PrimeFieldElement],
) -> PrimeFieldElement<'a> {
    let x_to_ns = x.mod_pow(num_steps as i128);
    let x_to_ns_plus_2 = x_to_ns * *x * *x;
    let x_to_ns_plus_3 = x_to_ns_plus_2 * *x;
    let mut ret: PrimeFieldElement = *sum_transition_quotient_evaluation;
    ret = ret + x_to_ns * *sum_transition_quotient_evaluation * composition_polynomial_weights[0];
    for i in 0..num_registers {
        let scaled_btq =
            bit_transition_quotient_evaluations[i] * composition_polynomial_weights[1 + i];
        let scaled_bq = boundary_quotient_evaluations[i]
            * composition_polynomial_weights[1 + i + num_registers];
        let scaled_shifted_btq = bit_transition_quotient_evaluations[i]
            * composition_polynomial_weights[1 + i + 2 * num_registers]
            * x_to_ns_plus_2;
        let scaled_shifted_bq = boundary_quotient_evaluations[i]
            * composition_polynomial_weights[1 + i + 3 * num_registers]
            * x_to_ns_plus_3;

        ret = ret + scaled_btq + scaled_bq + scaled_shifted_btq + scaled_shifted_bq;
    }

    ret
}

#[allow(clippy::too_many_arguments)]
fn get_composition_polynomial_codeword<'a>(
    omega: PrimeFieldElement<'a>,
    extended_domain_length: usize,
    num_registers: usize,
    num_steps: usize,
    composition_polynomial_coefficients: &'a [PrimeFieldElement],
    sum_transition_quotient_polynomial: Polynomial<PrimeFieldElement<'a>>,
    bit_transition_quotient_polynomials: Vec<Polynomial<PrimeFieldElement<'a>>>,
    boundary_quotient_polynomials: Vec<Polynomial<PrimeFieldElement<'a>>>,
) -> Vec<PrimeFieldElement<'a>> {
    // Get an analytical representation of the composition polynomial
    let mut composition_polynomial = sum_transition_quotient_polynomial.clone();
    composition_polynomial = composition_polynomial
        + sum_transition_quotient_polynomial
            .scalar_mul(composition_polynomial_coefficients[0])
            .shift_coefficients(num_steps, omega.ring_zero());

    for i in 0..num_registers {
        let scaled_btq = bit_transition_quotient_polynomials[i]
            .scalar_mul(composition_polynomial_coefficients[1 + i]);
        let scaled_bq = boundary_quotient_polynomials[i]
            .scalar_mul(composition_polynomial_coefficients[1 + i + num_registers]);
        let scaled_shifted_btq = bit_transition_quotient_polynomials[i]
            .scalar_mul(composition_polynomial_coefficients[1 + i + 2 * num_registers])
            .shift_coefficients(num_steps + 2, omega.ring_zero());
        let scaled_shifted_bq = boundary_quotient_polynomials[i]
            .scalar_mul(composition_polynomial_coefficients[1 + i + 3 * num_registers])
            .shift_coefficients(num_steps + 3, omega.ring_zero());

        // TODO: We could add a degree check here, ensuring that all polynomials have
        // the expected degree.

        composition_polynomial = composition_polynomial.clone()
            + scaled_btq
            + scaled_bq
            + scaled_shifted_btq
            + scaled_shifted_bq;
    }

    // Evaluate the composition polynomial in the omega domain
    let mut coefficients = composition_polynomial.coefficients;
    coefficients.append(&mut vec![
        omega.ring_zero();
        extended_domain_length - coefficients.len()
    ]);

    // Evaluate the composition polynomial over the omega domain, and return this codeword
    ntt(&coefficients, &omega)
}

fn get_composition_polynomial_weights<'a>(
    num_registers: usize,
    field: &'a PrimeField,
    root_hash: &[u8],
) -> Vec<PrimeFieldElement<'a>> {
    let k_seeds = utils::get_n_hash_rounds(root_hash, (4 * num_registers + 1) as u32);
    k_seeds
        .iter()
        .map(|seed| PrimeFieldElement::from_bytes(field, seed))
        .collect::<Vec<PrimeFieldElement>>()
}

// Map from a set of indices to a set of indices that matches the flattening of
// the codeword over the index for the register.
fn get_indices_over_all_registers(indices: &[usize], num_registers: usize) -> Vec<usize> {
    indices
        .iter()
        .map(|x| (0..(num_registers)).map(move |y| y + *x * num_registers))
        .flatten()
        .collect()
}

impl CollatzStarkProof {
    pub fn from_serialization(
        transcript: &mut [u8],
        start_index: usize,
    ) -> Result<(CollatzStarkProof, usize), Box<dyn Error>> {
        // We could use the reader trait on transcript. But then the type of transcript should be
        // a byte slice, not a byte array. And we would have to modify the FRI deserializer to
        // handle the byte slice as a reader as well
        let mut index = start_index;
        let bq_merkle_root: [u8; 32] = bincode::deserialize(&transcript[index..index + 32])?;
        index += 32;
        let composition_polynomial_merkle_root: [u8; 32] =
            bincode::deserialize(&transcript[index..index + 32])?;
        index += 32;

        let (composite_polynomial_fri, new_index) =
            LowDegreeProof::<i128>::from_serialization(transcript.to_vec(), index)?;
        index = new_index;

        let mut auth_path_size: u32 = bincode::deserialize(&transcript[index..index + 4])?;
        index += 4;
        let boundary_quotient_authentication_paths: Vec<CompressedAuthenticationPath<i128>> =
            bincode::deserialize(&transcript[index..index + auth_path_size as usize])?;
        index += auth_path_size as usize;
        auth_path_size = bincode::deserialize(&transcript[index..index + 4])?;
        index += 4;
        let boundary_quotient_authentication_paths_next: Vec<CompressedAuthenticationPath<i128>> =
            bincode::deserialize(&transcript[index..index + auth_path_size as usize])?;

        Ok((
            CollatzStarkProof {
                bq_merkle_root,
                composition_polynomial_merkle_root,
                boundary_quotient_authentication_paths,
                boundary_quotient_authentication_paths_next,
                composite_polynomial_fri,
            },
            index + auth_path_size as usize,
        ))
    }

    pub fn verify(
        &self,
        claim: &CollatzClaim,
        omega: PrimeFieldElement,
        num_steps: usize,
        expansion_factor: usize,
        num_registers: usize,
    ) -> Result<(), Box<dyn Error>> {
        let omicron = omega.mod_pow(expansion_factor as i128);
        let trace_length = num_steps + 1;
        let extended_domain_length = trace_length * expansion_factor;
        let xlast = &omicron.inv();

        // Verify low-degreeness of linear combination
        low_degree_test::verify_i128(self.composite_polynomial_fri.clone(), omega.field.q)?;

        // Verify the authentication paths
        // let top_level_abc_indices = self.composite_polynomial_fri.get_abc_indices(0);
        let top_level_ab_indices: Vec<usize> =
            self.composite_polynomial_fri.get_ab_indices(0).unwrap();
        let ab_indices_over_registers =
            get_indices_over_all_registers(&top_level_ab_indices, num_registers);
        if !MerkleTree::verify_multi_proof(
            self.bq_merkle_root,
            &ab_indices_over_registers,
            &self.boundary_quotient_authentication_paths,
        ) {
            return Err(Box::new(
                StarkVerifyError::BadBoundaryConditionAuthenticationPaths,
            ));
        }

        // Verify the authentication paths of the shifted indices into the boundary
        // quotient codeword.
        let ab_indices_over_registers_next: Vec<usize> = ab_indices_over_registers
            .iter()
            .map(|x| {
                (x + num_registers * expansion_factor) % (extended_domain_length * num_registers)
            })
            .collect();
        if !MerkleTree::verify_multi_proof(
            self.bq_merkle_root,
            &ab_indices_over_registers_next,
            &self.boundary_quotient_authentication_paths_next,
        ) {
            return Err(Box::new(
                StarkVerifyError::BadBoundaryConditionAuthenticationPaths,
            ));
        }

        // Get composition polynomial weights from boundary quotient Merkle root
        let cp_weights =
            get_composition_polynomial_weights(num_registers, omega.field, &self.bq_merkle_root);

        // Get polynomials needed for the calculating the trace, the AIRs, and the transition
        // quotients from the boundary quotient evaluation in a specific point.
        let (boundary_points, boundary_roots) =
            format_boundary_values(claim.start_value.value, claim.end_value.value, &omicron);
        let boundary_interpolants: Vec<Polynomial<PrimeFieldElement>> =
            get_boundary_interpolants(&boundary_points);
        let boundary_zerofier: Polynomial<PrimeFieldElement> =
            Polynomial::get_polynomial_with_roots(&boundary_roots);
        let bit_zerofier: Polynomial<PrimeFieldElement> = get_trace_zerofier(trace_length, xlast);
        let (sum_zerofier_num, sum_zerofier_den): (
            Polynomial<PrimeFieldElement>,
            Polynomial<PrimeFieldElement>,
        ) = get_transition_zerofier_polynomials_sum(trace_length, xlast);

        // Extract the values from the authentication path data structure
        let boundary_quotient_values_flat: Vec<PrimeFieldElement> = self
            .boundary_quotient_authentication_paths
            .iter()
            .map(|x| PrimeFieldElement::new(x.get_value(), omega.field))
            .collect();
        let boundary_quotient_values_next_flat: Vec<PrimeFieldElement> = self
            .boundary_quotient_authentication_paths_next
            .iter()
            .map(|x| PrimeFieldElement::new(x.get_value(), omega.field))
            .collect();

        // Define some constants
        let one = omega.ring_one();
        let two = PrimeFieldElement::new(2, omega.field);
        let three = PrimeFieldElement::new(3, omega.field);

        // Loop over all indices indicated by the Merkle root of the boundary quotient codeword
        for (i, index) in top_level_ab_indices.iter().enumerate() {
            let mut boundary_quotients: Vec<PrimeFieldElement> = vec![];
            let mut bit_transition_quotients = vec![];
            let x = omega.mod_pow(*index as i128);
            let x_next = x * omicron;
            let bit_zerofier_eval = bit_zerofier.evaluate(&x);
            let boundary_zerofier_eval_x = boundary_zerofier.evaluate(&x);
            let boundary_zerofier_eval_x_next = boundary_zerofier.evaluate(&x_next);
            let mut current_sum_acc = omega.ring_zero();
            let mut next_sum_acc = omega.ring_zero();
            let mut power_of_two = omega.ring_one();
            let mut ect_current_0 = omega.ring_zero();
            for j in 0..num_registers {
                let boundary_quotient_current =
                    boundary_quotient_values_flat[i * num_registers + j];
                boundary_quotients.push(boundary_quotient_current);
                let ect_current: PrimeFieldElement = boundary_quotient_current
                    * boundary_zerofier_eval_x
                    + boundary_interpolants[j].evaluate(&x);
                let ect_next: PrimeFieldElement = boundary_quotient_values_next_flat
                    [i * num_registers + j]
                    * boundary_zerofier_eval_x_next
                    + boundary_interpolants[j].evaluate(&x_next);
                if j == 0 {
                    ect_current_0 = ect_current;
                }
                let bit_air_value: PrimeFieldElement = ect_current * ect_current - ect_current;
                let bit_transition_quotient: PrimeFieldElement = bit_air_value / bit_zerofier_eval;
                bit_transition_quotients.push(bit_transition_quotient);
                current_sum_acc = current_sum_acc + power_of_two * ect_current;
                next_sum_acc = next_sum_acc + power_of_two * ect_next;

                power_of_two = power_of_two * two;
            }
            let sum_air: PrimeFieldElement = ect_current_0
                * (three * current_sum_acc + one - next_sum_acc)
                + (one - ect_current_0) * (current_sum_acc - two * next_sum_acc);
            let sum_transition_quotient =
                sum_air * sum_zerofier_den.evaluate(&x) / sum_zerofier_num.evaluate(&x);
            let cp_calculated = get_composition_polynomial_evaluation(
                num_registers,
                num_steps,
                &x,
                &cp_weights,
                &sum_transition_quotient,
                &bit_transition_quotients,
                &boundary_quotients,
            );

            // Get the CP values from the FRI
            let cp_reported: i128 = self.composite_polynomial_fri.ab_proofs[0][i].get_value();
            if cp_reported != cp_calculated.value {
                return Err(Box::new(StarkVerifyError::LinearCombinationMismatch(
                    *index,
                )));
            }
        }

        Ok(())
    }
}

pub fn stark_of_collatz_sequence_prove(
    collatz_input: i128,
    num_colinearity_checks: usize,
    expansion_factor: usize,
    omega: PrimeFieldElement,
    transcript: &mut Vec<u8>,
) -> Result<CollatzStarkProof, StarkProofError> {
    let omicron: PrimeFieldElement = omega.mod_pow(expansion_factor as i128);

    // Calculate the trace
    let value_trace: Vec<i128> = get_collatz_trace(collatz_input);
    let end_y_value: i128 = value_trace.last().to_owned().unwrap().to_owned();
    let omicron_domain: Vec<PrimeFieldElement> = omicron.get_generator_domain();
    let xlast: &PrimeFieldElement = omicron_domain.last().unwrap();

    // Get the extended computational traces over all registers, where each register along
    // the trace represents a binary value in base 2.
    // This is done by computing the low-degree extension of computational trace,
    // indexed by [pseudo-cycle][register]
    let trace_length = value_trace.len();
    let (extended_computational_traces, trace_interpolants): (
        Vec<Vec<PrimeFieldElement>>,
        Vec<Polynomial<PrimeFieldElement>>,
    ) = get_extended_computational_traces(&omega, &omicron, expansion_factor, value_trace);
    let num_steps = trace_length - 1;
    let num_registers = extended_computational_traces[0].len();
    let extended_domain_length = expansion_factor * trace_length;

    // Calculate the AIR polynomial for the transition relation
    // `s_{i+1} = s_i / 2` if `s_i` even; `s_{i+1} = 3*s_i + 1` is `s_i` odd
    // We only need the polynomial, not the evaluations.
    // This is also called the "transition AIR".
    let sum_air_polynomial: Polynomial<PrimeFieldElement> = build_sum_air(
        &omega,
        expansion_factor,
        extended_computational_traces.clone(),
    );

    // compute transition-zerofier polynomial for the sum relations, and compute the
    // transition quotient polynomial for the constraint.
    let (transition_zerofier_sum_numerator, transition_zerofier_sum_denominator) =
        get_transition_zerofier_polynomials_sum(trace_length, xlast);
    let (sum_transition_quotient_polynomial, sum_air_rem) = sum_air_polynomial
        .multiply(transition_zerofier_sum_denominator)
        .divide(transition_zerofier_sum_numerator);
    if !sum_air_rem.is_zero() {
        panic!("Transition remainder for sum air is not zero!");
    }

    // Calculate the AIR to guarantee that all register values are bits (s_i^2 - s_i = 0)
    // Codeword is indexed with [pseudo-cycle][register].
    // This is also called the "consistency AIR".
    // And compute the transition-quotient polynomials for the bit-restriction,
    // one for each register. AKA the transition quotient for the consistency constraint.
    // The consistency constraint must be true along the entire trace, so the divisor to the
    // quotient is a polynomial that is zero along the entire trace.
    let bit_air_polynomials: Vec<Polynomial<PrimeFieldElement>> =
        build_bit_airs(&omega, extended_computational_traces.clone());
    let transition_zerofier_bit = get_trace_zerofier(trace_length, xlast);
    let mut bit_transition_quotient_polynomials: Vec<Polynomial<PrimeFieldElement>> = vec![];
    for (i, bit_air_polynomial) in bit_air_polynomials.iter().enumerate() {
        let (bit_transition_quotient_polynomial, bit_air_rem) = bit_air_polynomial
            .clone()
            .divide(transition_zerofier_bit.clone());
        if !bit_air_rem.is_zero() {
            panic!(
                "Transition remainder for bit air is not zero for i = {}!. Got: ({}) % ({}) = {}",
                i, bit_air_polynomial, transition_zerofier_bit, bit_air_rem
            );
        }

        bit_transition_quotient_polynomials.push(bit_transition_quotient_polynomial.clone());
    }

    // Get the boundary-zerofier and boundary interpolants.
    // There is one interpolant for each register.
    let (boundary_points, boundary_roots): (
        Vec<Vec<(PrimeFieldElement, PrimeFieldElement)>>,
        Vec<PrimeFieldElement>,
    ) = format_boundary_values(collatz_input, end_y_value, &omicron);
    let boundary_constraint_interpolants: Vec<Polynomial<PrimeFieldElement>> =
        get_boundary_interpolants(&boundary_points);
    let boundary_zerofier_polynomial: Polynomial<PrimeFieldElement> =
        Polynomial::get_polynomial_with_roots(&boundary_roots);

    // Compute the boundary-quotient polynomial and codeword. The codeword is needed since
    // an oracle commitment to it is part of the STARK proof.
    // The produced boundary quotient codeword is here indexed as [sub-cycle][register].
    let mut boundary_quotient_codewords: Vec<Vec<PrimeFieldElement>> =
        vec![vec![omega.ring_zero(); num_registers]; extended_domain_length];
    let mut boundary_quotient_polynomials: Vec<Polynomial<PrimeFieldElement>> = vec![];
    for i in 0..trace_interpolants.len() {
        let (boundary_quotient_polynomial, bq_rem) = (trace_interpolants[i].clone()
            - boundary_constraint_interpolants[i].clone())
        .divide(boundary_zerofier_polynomial.clone());
        if !bq_rem.is_zero() {
            panic!(
                "Boundary remainder is not zero! i = {}. Got: (({}) - ({})) % ({}) = {}",
                i,
                trace_interpolants[i],
                boundary_constraint_interpolants[i],
                boundary_zerofier_polynomial,
                bq_rem
            );
        }
        let num_coefficients = boundary_quotient_polynomial.coefficients.len();
        let mut boundary_constraint_coefficients_padded =
            boundary_quotient_polynomial.coefficients.clone();
        boundary_constraint_coefficients_padded.append(&mut vec![
            omega.ring_zero();
            extended_domain_length
                - num_coefficients
        ]);
        let boundary_quotient_codeword = ntt(&boundary_constraint_coefficients_padded, &omega);
        boundary_quotient_polynomials.push(boundary_quotient_polynomial);
        for j in 0..extended_domain_length {
            boundary_quotient_codewords[j][i] = boundary_quotient_codeword[j];
        }
    }

    // Flatten the boundary quotient codeword, so we can put it into a Merkle tree.
    // bq_codeword[pseudo-cycle][register] -> bq_codeword[pseudo-cycle * num_registers + register]
    let flattened_boundary_quotient_codewords = boundary_quotient_codewords
        .into_iter()
        .flatten()
        .map(|x| x.value)
        .collect::<Vec<i128>>();
    let boundary_quotient_mt = MerkleTree::from_vec(&flattened_boundary_quotient_codewords);
    transcript.append(&mut boundary_quotient_mt.get_root().to_vec());

    // Compute composition polynomial and commit to the Merkle root of its
    // evaluation over the omega domain.
    let composition_polynomial_weights: Vec<PrimeFieldElement> =
        get_composition_polynomial_weights(num_registers, omega.field, transcript);
    let composition_polynomial_codeword: Vec<PrimeFieldElement> =
        get_composition_polynomial_codeword(
            omega,
            extended_domain_length,
            num_registers,
            num_steps,
            &composition_polynomial_weights,
            sum_transition_quotient_polynomial,
            bit_transition_quotient_polynomials,
            boundary_quotient_polynomials,
        );
    let composition_polynomial_codeword_values = composition_polynomial_codeword
        .iter()
        .map(|x| x.value)
        .collect::<Vec<i128>>();
    let composition_polynomial_mt = MerkleTree::from_vec(&composition_polynomial_codeword_values);
    transcript.append(&mut composition_polynomial_mt.get_root().to_vec());

    // Create low-degree proof for composition polynomial
    let low_degree_proof_result = low_degree_test::prover_i128(
        &composition_polynomial_codeword_values,
        omega.field.q,
        (2 * trace_length - 1) as u32,
        num_colinearity_checks,
        transcript,
        omega.value,
    );
    let composite_polynomial_fri = match low_degree_proof_result {
        Ok(proof) => proof,
        Err(err) => panic!("FRI failed with error: {}", err),
    };

    // Sanity check that the FRI proof works
    if low_degree_test::verify_i128(composite_polynomial_fri.clone(), omega.field.q).is_err() {
        panic!("FRI proof did not pass correctness test");
    }

    // Produce authentication paths for the relevant codewords
    let ab_indices = composite_polynomial_fri.get_ab_indices(0).unwrap();

    // Get ab-indices over all registers, corresponding to the flattened structure from which
    // the Merkle tree was built.
    let ab_indices_over_registers: Vec<usize> =
        get_indices_over_all_registers(&ab_indices, num_registers);

    // Get authentication paths
    // We *only* need to commit to the boundary quotient values here since the extended
    // trace, air codewords, and transition quotient can be calculated from the boundary
    // quotient codeword.
    let boundary_quotient_authentication_paths: Vec<CompressedAuthenticationPath<i128>> =
        boundary_quotient_mt.get_multi_proof(&ab_indices_over_registers);
    let ab_indices_over_registers_next_step: Vec<usize> = ab_indices_over_registers
        .iter()
        .map(|x| (x + num_registers * expansion_factor) % (extended_domain_length * num_registers))
        .collect();
    let boundary_quotient_authentication_paths_next: Vec<CompressedAuthenticationPath<i128>> =
        boundary_quotient_mt.get_multi_proof(&ab_indices_over_registers_next_step);

    // Serialize authentication paths
    let mut bqap_serialized = bincode::serialize(&boundary_quotient_authentication_paths).unwrap();
    let mut bqap_next_serialized =
        bincode::serialize(&boundary_quotient_authentication_paths_next).unwrap();

    // Write size of serialization and actual serialization to transcript
    transcript.append(&mut bincode::serialize(&(bqap_serialized.len() as u32)).unwrap());
    transcript.append(&mut bqap_serialized);
    transcript.append(&mut bincode::serialize(&(bqap_next_serialized.len() as u32)).unwrap());
    transcript.append(&mut bqap_next_serialized);

    // Return proof object. Proof is also contained in the transcript in serialized form
    Ok(CollatzStarkProof {
        bq_merkle_root: boundary_quotient_mt.get_root(),
        composition_polynomial_merkle_root: composition_polynomial_mt.get_root(),
        boundary_quotient_authentication_paths,
        boundary_quotient_authentication_paths_next,
        composite_polynomial_fri,
    })
}

#[cfg(test)]
mod collatz_sequence_test {
    use super::*;
    use crate::shared_math::prime_field_element::PrimeField;

    #[test]
    fn small_numbers() {
        let expansion_factor = 8;
        let num_steps = 16;
        let extended_trace_length = num_steps * expansion_factor;
        let field = PrimeField::new(5 * 2i128.pow(25) + 1);
        println!("prime modulus = {}", field.q);
        let (omega_option, _) = field.get_primitive_root_of_unity(extended_trace_length);
        let omega = omega_option.unwrap();
        println!(
            "omega = {}, omega^{} = {}",
            omega,
            extended_trace_length,
            omega.mod_pow(extended_trace_length)
        );
        let mut transcript: Vec<u8> = vec![];
        let res = stark_of_collatz_sequence_prove(
            832,
            10,
            expansion_factor as usize,
            omega,
            &mut transcript,
        );
        let stark_proof = match res {
            Err(err) => panic!("{:?}", err),
            Ok(proof) => proof,
        };

        let claim = CollatzClaim {
            start_value: PrimeFieldElement::new(832, omega.field),
            end_value: PrimeFieldElement::new(1, omega.field),
        };

        match stark_proof.verify(&claim, omega, 15, 8, 128) {
            Err(err) => panic!("{:?}", err),
            Ok(_proof) => (),
        }

        // Verify that deserialization of transcript returns the same STARK proof as the function call
        let stark_proof_deserialized_res =
            CollatzStarkProof::from_serialization(&mut transcript, 0);
        let stark_proof_deserialized = stark_proof_deserialized_res.unwrap().0;
        match stark_proof_deserialized.verify(&claim, omega, 15, 8, 128) {
            Err(err) => panic!("{:?}", err),
            Ok(_proof) => (),
        }
        assert_eq!(stark_proof, stark_proof_deserialized);
    }
}
