use crate::shared_math::low_degree_test;
use crate::shared_math::low_degree_test::LowDegreeProof;
use crate::shared_math::ntt::{intt, ntt};
use crate::shared_math::polynomial::Polynomial;
use crate::shared_math::prime_field_element::PrimeFieldElement;
use crate::shared_math::prime_field_element_big::{PrimeFieldBig, PrimeFieldElementBig};
use crate::shared_math::traits::{IdentityValues, New};
use crate::util_types::merkle_tree::{CompressedAuthenticationPath, MerkleTree};
use crate::utils;
use num_bigint::BigInt;
use serde::Serialize;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Rem;
use std::ops::Sub;

pub enum StarkProofError {
    InputOutputMismatch,
    HighDegreeExtendedComputationalTrace,
    HighDegreeBoundaryQuotient,
    HighDegreeTransitionQuotient,
    HighDegreeLinearCombination,
    NonZeroBoundaryRemainder,
    NonZeroTransitionRemainder,
}

#[derive(Clone, Debug, Serialize)]
pub struct StarkProof<T: Clone + Debug + Serialize + PartialEq> {
    tuple_merkle_root: [u8; 32],
    linear_combination_merkle_root: [u8; 32],
    shifted_tuple_authentication_paths: Vec<CompressedAuthenticationPath<(T, T, T)>>,
    // TODO: Change this to three Merkle trees instead, do not store all triplets in a single
    // Merkle tree!
    tuple_authentication_paths: Vec<CompressedAuthenticationPath<(T, T, T)>>,
    linear_combination_fri: LowDegreeProof<T>,
}

// Returns a pair of polynomials, the numerator and denominator of the zerofier polynomias
fn get_transition_zerofier_polynomials<
    T: Clone
        + Debug
        + Serialize
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Rem<Output = T>
        + Neg<Output = T>
        + IdentityValues
        + PartialEq
        + Eq
        + Hash
        + Display,
>(
    num_steps: usize,
    last_x_value_of_trace: &T,
) -> (Polynomial<T>, Polynomial<T>) {
    let mut transition_zerofier_numerator_coefficients =
        vec![last_x_value_of_trace.ring_zero(); num_steps + 2];
    transition_zerofier_numerator_coefficients[0] = -last_x_value_of_trace.ring_one();
    transition_zerofier_numerator_coefficients[num_steps + 1] = last_x_value_of_trace.ring_one();
    let transition_zerofier_numerator = Polynomial {
        coefficients: transition_zerofier_numerator_coefficients,
    };
    let transition_zerofier_denominator_coefficients = vec![
        -last_x_value_of_trace.clone(),
        last_x_value_of_trace.ring_one(),
    ];
    let transition_zerofier_denominator = Polynomial {
        coefficients: transition_zerofier_denominator_coefficients,
    };
    (
        transition_zerofier_numerator,
        transition_zerofier_denominator,
    )
}

fn get_boundary_zerofier_polynomial<
    T: Clone
        + Debug
        + Serialize
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Rem<Output = T>
        + Neg<Output = T>
        + IdentityValues
        + PartialEq
        + Eq
        + Hash
        + Display,
>(
    last_x_value_of_trace: &T,
) -> Polynomial<T> {
    Polynomial {
        coefficients: vec![
            last_x_value_of_trace.clone(),
            -last_x_value_of_trace.clone() - last_x_value_of_trace.ring_one(),
            last_x_value_of_trace.ring_one(),
        ],
    }
}

fn get_boundary_constraint_polynomial<'a>(
    mimc_input: &PrimeFieldElementBig<'a>,
    mimc_output: &PrimeFieldElementBig<'a>,
    last_x_value_of_trace: &PrimeFieldElementBig<'a>,
) -> Polynomial<PrimeFieldElementBig<'a>> {
    let (line_a, line_b): (PrimeFieldElementBig, PrimeFieldElementBig) =
        mimc_input.field.lagrange_interpolation_2(
            (mimc_input.ring_one(), mimc_input.clone()),
            (last_x_value_of_trace.to_owned(), mimc_output.to_owned()),
        );
    Polynomial {
        coefficients: vec![line_b, line_a],
    }
}

fn get_linear_combination_coefficients<'a>(
    field: &'a PrimeFieldBig,
    root_hash: &[u8; 32],
) -> Vec<PrimeFieldElementBig<'a>> {
    let k_seeds = utils::get_n_hash_rounds(root_hash, 5);
    k_seeds
        .iter()
        .map(|seed| PrimeFieldElementBig::from_bytes(field, seed))
        .collect::<Vec<PrimeFieldElementBig>>()
}

struct PolynomialEvaluations<T: Clone + Debug + Serialize + PartialEq> {
    extended_computational_trace: T,
    shifted_trace_codeword: T,
    transition_quotient_codeword: T,
    shifted_transition_quotient_codeword: T,
    boundary_quotient_codeword: T,
    shifted_boundary_quotient_codeword: T,
}

fn get_linear_combination<
    T: Clone + Debug + Serialize + PartialEq + Mul<Output = T> + Add<Output = T>,
>(
    coefficients: &[T],
    values: PolynomialEvaluations<T>,
) -> T {
    values.extended_computational_trace.clone()
        + coefficients[0].clone() * values.shifted_trace_codeword
        + coefficients[1].clone() * values.transition_quotient_codeword
        + coefficients[2].clone() * values.shifted_transition_quotient_codeword
        + coefficients[3].clone() * values.boundary_quotient_codeword
        + coefficients[4].clone() * values.shifted_boundary_quotient_codeword
}

fn get_linear_combinations<
    T: Clone + Debug + Serialize + PartialEq + Mul<Output = T> + Add<Output = T>,
>(
    coefficients: &[T],
    values: Vec<PolynomialEvaluations<T>>,
) -> Vec<T> {
    values
        .into_iter()
        .map(|value| get_linear_combination(coefficients, value))
        .collect::<Vec<T>>()
}

pub fn mimc_forward<'a>(
    input: &'a PrimeFieldElementBig,
    num_steps: usize,
    round_costants: &'a [PrimeFieldElementBig],
) -> Vec<PrimeFieldElementBig<'a>> {
    let mut computational_trace: Vec<PrimeFieldElementBig> = Vec::with_capacity(num_steps);
    let mut res: PrimeFieldElementBig = input.to_owned();
    computational_trace.push(input.to_owned());
    for i in 0..num_steps {
        res = res.clone().mod_pow(Into::<BigInt>::into(3))
            + round_costants[i % round_costants.len()].clone();
        computational_trace.push(res.clone());
    }

    computational_trace
}

pub fn mimc_forward_i128<'a>(
    input: &'a PrimeFieldElement,
    steps: usize,
    round_costants: &'a [PrimeFieldElement],
) -> Vec<PrimeFieldElement<'a>> {
    let mut computational_trace: Vec<PrimeFieldElement<'a>> = Vec::with_capacity(steps);
    let mut res: PrimeFieldElement<'a> = *input;
    computational_trace.push(*input);
    for i in 0..steps {
        res = res.mod_pow(3) + round_costants[i % round_costants.len()];
        computational_trace.push(res);
    }

    computational_trace
}

pub fn mimc_backward<'a>(
    input: &'a PrimeFieldElement,
    steps: usize,
    round_costants: &'a [PrimeFieldElement],
) -> PrimeFieldElement<'a> {
    // Verify that field.q is of the form 6k + 5
    if !input.field.prime_check(6, 5) {
        panic!("Invalid prime field selected");
    }

    let rc_length = round_costants.len() as i64;
    let start_index = steps as i64 % round_costants.len() as i64 - 1;
    let mut res: PrimeFieldElement<'a> = *input;
    for i in 0..steps as i64 {
        let index = (((start_index - i % rc_length) + rc_length) % rc_length) as usize;
        res = (res - round_costants[index]).mod_pow((2 * (input.field.q) - 1) / 3);
    }

    res
}

pub fn stark_of_mimc(
    security_checks: usize,
    num_steps: usize,
    expansion_factor: usize,
    omega: PrimeFieldElementBig,
    mimc_input: PrimeFieldElementBig,
    mimc_output: PrimeFieldElementBig,
    mimc_round_constants: &[PrimeFieldElementBig],
) -> Result<StarkProof<BigInt>, StarkProofError> {
    // Omega is the generator of the big domain
    // Omicron is the generator of the small domain
    let omicron: PrimeFieldElementBig = omega.mod_pow(Into::<BigInt>::into(expansion_factor));
    let extended_domain_length: usize = (num_steps + 1) * expansion_factor;
    let field: &PrimeFieldBig = omega.field;

    // compute computational trace
    let computational_trace: Vec<PrimeFieldElementBig> =
        mimc_forward(&mimc_input, num_steps, mimc_round_constants);

    // compute low-degree extension of computational trace
    let trace_interpolant_coefficients = intt(&computational_trace, &omicron);
    let trace_interpolant = Polynomial {
        coefficients: trace_interpolant_coefficients.clone(),
    };
    println!("trace_interpolant = {}", trace_interpolant); // TODO: REMOVE
    let mut padded_trace_interpolant_coefficients = trace_interpolant_coefficients.clone();
    padded_trace_interpolant_coefficients.append(&mut vec![
        omega.ring_zero();
        (expansion_factor - 1) * (num_steps + 1)
    ]);
    let extended_computational_trace = ntt(&padded_trace_interpolant_coefficients, &omega);
    println!(
        "extended_computational_trace = {:?}",
        extended_computational_trace
    ); // TODO: REMOVE

    // compute low-degree extension of the round constants polynomial
    let mut mimc_round_constants_padded = mimc_round_constants.to_vec();
    mimc_round_constants_padded.append(&mut vec![omega.ring_zero()]);
    let round_constants_interpolant = intt(&mimc_round_constants_padded, &omicron);
    // println!(
    //     "round_constants_interpolant = {:?}",
    //     round_constants_interpolant
    // ); // TODO: REMOVE
    let mut padded_round_constants_interpolant = round_constants_interpolant.clone();
    padded_round_constants_interpolant.append(&mut vec![
        omega.ring_zero();
        (expansion_factor - 1) * (num_steps + 1)
    ]);
    let extended_round_constants = ntt(&padded_round_constants_interpolant, &omega);
    // println!("extended_round_constants = {:?}", extended_round_constants); // TODO: REMOVE

    // evaluate and interpolate AIR
    let mut air_codeword = Vec::<PrimeFieldElementBig>::with_capacity(extended_domain_length);
    for i in 0..extended_domain_length {
        air_codeword.push(
            extended_computational_trace[i].mod_pow(Into::<BigInt>::into(3))
                + extended_round_constants[i].clone()
                - extended_computational_trace[(i + expansion_factor) % extended_domain_length]
                    .clone(),
        );
    }

    // It's important to interpolate across the *extended* domain, not the original smaller domain, because the degree of air(x) is greater than num_steps
    let air_polynomial_coefficients = intt(&air_codeword, &omega);
    let air_polynomial = Polynomial {
        coefficients: air_polynomial_coefficients,
    };

    let omega_domain: Vec<PrimeFieldElementBig> = omega.get_generator_domain();
    let omicron_domain: Vec<PrimeFieldElementBig> = omicron.get_generator_domain();
    let xlast: &PrimeFieldElementBig = omicron_domain.last().unwrap();

    // compute transition-zerofier polynomial
    let (transition_zerofier_numerator_polynomial, transition_zerofier_denominator_polynomial) =
        get_transition_zerofier_polynomials(num_steps, xlast);

    // compute the transition-quotient polynomial
    let (transition_quotient_polynomial, rem) = (air_polynomial.clone()
        * transition_zerofier_denominator_polynomial.clone())
    .divide(transition_zerofier_numerator_polynomial.clone());
    // Perform sanity check that remainder of division by transition-zerofier is zero
    if !(rem.is_zero()) {
        return Err(StarkProofError::NonZeroTransitionRemainder);
    }

    // Compute the codeword of the transition-quotient polynomial in the entire omega domain,
    let mut transition_quotient_coefficients = transition_quotient_polynomial.coefficients.clone();
    transition_quotient_coefficients.append(&mut vec![
        omega.ring_zero();
        extended_domain_length
            - transition_quotient_coefficients.len()
    ]);
    let transition_quotient_codeword = ntt(&transition_quotient_coefficients, &omega);

    // compute the boundary-zerofier
    let boundary_zerofier_polynomial = get_boundary_zerofier_polynomial(xlast);

    let boundary_constraint_polynomial =
        get_boundary_constraint_polynomial(&mimc_input, &mimc_output, xlast);

    // compute the boundary-quotient polynomial and codeword
    let (boundary_quotient_polynomial, bq_rem) =
        (trace_interpolant - boundary_constraint_polynomial).divide(boundary_zerofier_polynomial);
    if !(bq_rem.is_zero()) {
        return Err(StarkProofError::NonZeroBoundaryRemainder);
    }

    let mut boundary_constraint_coefficients_padded =
        boundary_quotient_polynomial.coefficients.clone();
    boundary_constraint_coefficients_padded.append(&mut vec![
        omega.ring_zero();
        expansion_factor * (num_steps + 1)
            - boundary_quotient_polynomial
                .coefficients
                .len()
    ]);
    let boundary_quotient_codeword = ntt(&boundary_constraint_coefficients_padded, &omega);

    // Commit to all evaluations by constructing a Merkle tree of the polynomial evaluations
    // TODO: We also need the offset values of the ect, since this is needed to verify that
    // `air(x) = TQ(x) * Z_t(x)`.
    let polynomial_evaluations: Vec<(BigInt, BigInt, BigInt)> = extended_computational_trace
        .iter()
        .zip(transition_quotient_codeword.iter())
        .zip(boundary_quotient_codeword.iter())
        .map(|((ect, tq), bq)| {
            (
                ect.to_owned().value,
                tq.to_owned().value,
                bq.to_owned().value,
            )
        })
        .collect::<Vec<(BigInt, BigInt, BigInt)>>();
    let tuple_merkle_tree: MerkleTree<(BigInt, BigInt, BigInt)> =
        MerkleTree::from_vec(&polynomial_evaluations);

    let lc_coefficients =
        get_linear_combination_coefficients(&field, &tuple_merkle_tree.get_root());

    // compute shifted trace codeword
    let mut shifted_trace_codeword: Vec<PrimeFieldElementBig> =
        extended_computational_trace.clone();
    let mut xi: PrimeFieldElementBig = omega.ring_one();
    for stc in shifted_trace_codeword.iter_mut() {
        *stc = stc.to_owned() * xi.clone();
        xi = xi * omega_domain[num_steps + 1].clone();
    }

    // compute shifted transition quotient codeword
    let mut shifted_transition_quotient_codeword: Vec<PrimeFieldElementBig> =
        transition_quotient_codeword.clone();
    for i in 0..extended_domain_length {
        shifted_transition_quotient_codeword[i] =
            shifted_transition_quotient_codeword[i].clone() * omega_domain[i].clone();
    }

    // compute shifted boundary quotient codeword
    let mut shifted_boundary_quotient_codeword: Vec<PrimeFieldElementBig> =
        boundary_quotient_codeword.clone();
    xi = omega.ring_one();
    for sbqc in shifted_boundary_quotient_codeword.iter_mut() {
        *sbqc = sbqc.to_owned() * xi.clone();
        xi = xi * omega_domain[num_steps + 3].clone();
    }

    // Compute linear combination of previous polynomial values
    let polynomial_evaluations = (0..extended_domain_length)
        .map(|i| PolynomialEvaluations {
            extended_computational_trace: extended_computational_trace[i].clone(),
            shifted_trace_codeword: shifted_trace_codeword[i].clone(),
            transition_quotient_codeword: transition_quotient_codeword[i].clone(),
            shifted_transition_quotient_codeword: shifted_transition_quotient_codeword[i].clone(),
            boundary_quotient_codeword: boundary_quotient_codeword[i].clone(),
            shifted_boundary_quotient_codeword: shifted_boundary_quotient_codeword[i].clone(),
        })
        .collect::<Vec<PolynomialEvaluations<PrimeFieldElementBig>>>();
    let linear_combination_codeword =
        get_linear_combinations(&lc_coefficients, polynomial_evaluations)
            .iter()
            .map(|x| x.value.clone())
            .collect::<Vec<BigInt>>();

    let linear_combination_mt = MerkleTree::from_vec(&linear_combination_codeword);

    // Compute the FRI low-degree proofs
    let extended_computational_trace_bigint = extended_computational_trace
        .iter()
        .map(|x| x.value.clone())
        .collect::<Vec<BigInt>>();
    let boundary_quotient_codeword_bigint = boundary_quotient_codeword
        .iter()
        .map(|x| x.value.clone())
        .collect::<Vec<BigInt>>();
    let transition_quotient_codeword_bigint = transition_quotient_codeword
        .iter()
        .map(|x| x.value.clone())
        .collect::<Vec<BigInt>>();
    let linear_combination_evaluations_bigint = linear_combination_codeword;

    let mut output = vec![];
    let max_degree_ect = num_steps as u32;
    let low_degree_proof_ect = low_degree_test::prover_bigint(
        &extended_computational_trace_bigint,
        field.q.clone(),
        max_degree_ect,
        security_checks,
        &mut output,
        omega.value.clone(),
    )
    .unwrap();
    let verify: Result<(), low_degree_test::ValidationError> =
        low_degree_test::verify_bigint(low_degree_proof_ect, field.q.clone());
    match verify {
        Ok(_) => println!(
            "Succesfully verified low degree ({}) of extended computational trace",
            max_degree_ect
        ),
        Err(_err) => return Err(StarkProofError::HighDegreeExtendedComputationalTrace),
    }
    let max_degree_bq = num_steps as u32;
    let low_degree_proof_bq = low_degree_test::prover_bigint(
        &boundary_quotient_codeword_bigint,
        field.q.clone(),
        max_degree_bq,
        security_checks,
        &mut output,
        omega.value.clone(),
    )
    .unwrap();
    let verify: Result<(), low_degree_test::ValidationError> =
        low_degree_test::verify_bigint(low_degree_proof_bq, field.q.clone());
    match verify {
        Ok(_) => println!(
            "Succesfully verified low degree ({}) of boundary quotient",
            max_degree_bq
        ),
        Err(_err) => return Err(StarkProofError::HighDegreeBoundaryQuotient),
        // panic!(
        //     "Failed to verify low degree ({}) of boundary quotient. Got: {:?}",
        //     max_degree_bq, err
        // ),
    }
    let max_degree_tq = ((num_steps + 1) * 2 - 1) as u32;
    let low_degree_proof_tq = low_degree_test::prover_bigint(
        &transition_quotient_codeword_bigint,
        field.q.clone(),
        max_degree_tq,
        security_checks,
        &mut output,
        omega.value.clone(),
    )
    .unwrap();
    let verify: Result<(), low_degree_test::ValidationError> =
        low_degree_test::verify_bigint(low_degree_proof_tq, field.q.clone());
    match verify {
        Ok(_) => println!(
            "Succesfully verified low degree ({}) of transition quotient",
            max_degree_tq
        ),
        Err(_err) => return Err(StarkProofError::HighDegreeTransitionQuotient),
    }
    let max_degree_shifted_tq = ((num_steps + 1) * 2 - 1) as u32;
    let shifted_transition_quotient_codeword_bi = shifted_transition_quotient_codeword
        .iter()
        .map(|x| x.value.clone())
        .collect::<Vec<BigInt>>();
    let low_degree_proof_shifted_tq = low_degree_test::prover_bigint(
        &shifted_transition_quotient_codeword_bi,
        field.q.clone(),
        max_degree_shifted_tq,
        security_checks,
        &mut output,
        omega.value.clone(),
    )
    .unwrap();
    let verify: Result<(), low_degree_test::ValidationError> =
        low_degree_test::verify_bigint(low_degree_proof_shifted_tq, field.q.clone());
    match verify {
        Ok(_) => println!(
            "Succesfully verified low degree ({}) of shifted transition quotient",
            max_degree_tq
        ),
        Err(_err) => return Err(StarkProofError::HighDegreeTransitionQuotient),
    }
    let max_degree_shifted_bq = ((num_steps + 1) * 2 - 1) as u32;
    let shifted_boundary_quotient_codeword_bi = shifted_boundary_quotient_codeword
        .iter()
        .map(|x| x.value.clone())
        .collect::<Vec<BigInt>>();
    let low_degree_proof_shifted_bq = low_degree_test::prover_bigint(
        &shifted_boundary_quotient_codeword_bi,
        field.q.clone(),
        max_degree_shifted_bq,
        security_checks,
        &mut output,
        omega.value.clone(),
    )
    .unwrap();
    let verify: Result<(), low_degree_test::ValidationError> =
        low_degree_test::verify_bigint(low_degree_proof_shifted_bq, field.q.clone());
    match verify {
        Ok(_) => println!(
            "Succesfully verified low degree ({}) of shifted boundary quotient",
            max_degree_tq
        ),
        Err(_err) => return Err(StarkProofError::HighDegreeTransitionQuotient),
    }
    let max_degree_shifted_ti = ((num_steps + 1) * 2 - 1) as u32;
    let shifted_trace_codeword_bi = shifted_trace_codeword
        .iter()
        .map(|x| x.value.clone())
        .collect::<Vec<BigInt>>();
    let low_degree_proof_shifted_ti = low_degree_test::prover_bigint(
        &shifted_trace_codeword_bi,
        field.q.clone(),
        max_degree_shifted_ti,
        security_checks,
        &mut output,
        omega.value.clone(),
    )
    .unwrap();
    let verify: Result<(), low_degree_test::ValidationError> =
        low_degree_test::verify_bigint(low_degree_proof_shifted_ti, field.q.clone());
    match verify {
        Ok(_) => println!(
            "Succesfully verified low degree ({}) of shifted trace interpolant",
            max_degree_tq
        ),
        Err(_err) => return Err(StarkProofError::HighDegreeTransitionQuotient),
    }
    // let max_degree_lc = num_steps as u32;
    let max_degree_lc = ((num_steps + 1) * 2 - 1) as u32;
    println!("max_degree of linear combination is: {}", max_degree_lc);
    output = vec![];
    // output = linear_combination_mt.get_root().to_vec();
    let low_degree_proof_lc_result = low_degree_test::prover_bigint(
        &linear_combination_evaluations_bigint,
        field.q.clone(),
        max_degree_lc,
        security_checks,
        &mut output,
        omega.value.clone(),
    );
    let low_degree_proof_lc: LowDegreeProof<BigInt>;
    match low_degree_proof_lc_result {
        Err(_err) => return Err(StarkProofError::HighDegreeLinearCombination),
        Ok(proof) => {
            println!(
                "Successfully verified low degree ({}) of linear combination!",
                max_degree_lc
            );
            low_degree_proof_lc = proof;
        }
    }
    let verify: Result<(), low_degree_test::ValidationError> =
        low_degree_test::verify_bigint(low_degree_proof_lc.clone(), field.q.clone());
    let linear_combination_fri: LowDegreeProof<BigInt>;
    match verify {
        Ok(_) => {
            linear_combination_fri = low_degree_proof_lc;
        }
        Err(_err) => {
            println!(
                "\n\n\n\nFailed to low degreeness of linear combination values.\n\n Coefficients: {:?}\n\nCodeword: {:?}\n\nDomain: {:?}",
                lc_coefficients, linear_combination_evaluations_bigint, omega_domain
            );
            return Err(StarkProofError::HighDegreeLinearCombination);
        }
    }

    // Produce authentication paths for the relevant codewords
    let abc_indices: Vec<(usize, usize, usize)> = linear_combination_fri.get_abc_indices(0);
    // let indices = ab_indices.iter().map(|(a, b)| )
    // todo: use https://users.rust-lang.org/t/flattening-a-vector-of-tuples/11409/2
    let mut ab_indices: Vec<usize> = vec![];
    for (a, b, _) in abc_indices.iter() {
        ab_indices.push(*a);
        ab_indices.push(*b);
    }
    // index_preimage.push(0);
    // let index_preimages =
    // let index_preimages =
    //     get_n_hash_rounds(&linear_combination_fri.index_picker_preimage)
    // get_n_hash_rounds(&linear_combination_mt.get_root(), security_checks as u32);
    // let indices: Vec<usize> = index_preimages
    //     .iter()
    //     .map(|x| get_index_from_bytes(x, extended_domain_length))
    //     .collect::<Vec<usize>>();
    let tuple_authentication_paths: Vec<CompressedAuthenticationPath<(BigInt, BigInt, BigInt)>> =
        tuple_merkle_tree.get_multi_proof(&ab_indices);
    let shifted_indices = ab_indices
        .iter()
        .map(|x| (x + expansion_factor) % extended_domain_length)
        .collect::<Vec<usize>>();
    let shifted_tuple_authentication_paths: Vec<
        CompressedAuthenticationPath<(BigInt, BigInt, BigInt)>,
    > = tuple_merkle_tree.get_multi_proof(&shifted_indices);

    Ok(StarkProof {
        tuple_merkle_root: tuple_merkle_tree.get_root(),
        linear_combination_merkle_root: linear_combination_mt.get_root(),
        shifted_tuple_authentication_paths,
        tuple_authentication_paths,
        linear_combination_fri,
    })
}

// pub fn stark_of_mimc_i128(
//     security_checks: usize,
//     steps: usize,
//     expansion_factor: usize,
//     field_modulus: i128,
//     g2: PrimeFieldElement,
//     mimc_input: PrimeFieldElement,
// ) {
//     let g1: PrimeFieldElement = g2.mod_pow(expansion_factor as i128);
//     let field = PrimeField::new(field_modulus);

//     let extended_domain_length: usize = steps * expansion_factor;
//     println!("g1 = {}", g1);

//     // Get the computational computational_trace of MIMC for some round_constant values
//     // TODO: FIX!! REPLACE modulus
//     // let round_constants_raw: Vec<i128> = vec![2i128; 128];
//     // let round_constants_raw: Vec<i128> = utils::generate_random_numbers(128, 5);
//     // TODO: FIX!! REPLACE modulus BY 65573
//     let round_constants_raw: Vec<i128> = utils::generate_random_numbers(64, 65573);
//     let round_constants: Vec<PrimeFieldElement> = round_constants_raw
//         .iter()
//         .map(|x| PrimeFieldElement::new(*x, &field)) // TODO: FIX!! REPLACE value BY x
//         //.map(|x| PrimeFieldElement::new(1, &field)) // TODO: FIX!! REPLACE value BY x
//         .collect::<Vec<PrimeFieldElement>>();
//     println!("Generating computational_trace");
//     let computational_trace: Vec<PrimeFieldElement> =
//         mimc_forward_i128(&mimc_input, steps - 1, &round_constants);
//     let output: &PrimeFieldElement = computational_trace.last().unwrap();
//     println!(
//         "Done generating trace with length {}",
//         computational_trace.len()
//     );
//     let mut computational_trace_polynomial_coeffs: Vec<PrimeFieldElement> =
//         fft::fast_polynomial_interpolate_prime_elements(&computational_trace, &g1);
//     println!(
//         "computational_trace_polynomial_coeffs has length {}",
//         computational_trace_polynomial_coeffs.len()
//     );
//     computational_trace_polynomial_coeffs.append(&mut vec![
//         PrimeFieldElement::new(0, &field);
//         (expansion_factor - 1) * steps
//     ]);
//     println!(
//         "expanded computational_trace_polynomial_coeffs has length {}",
//         computational_trace_polynomial_coeffs.len()
//     );
//     let trace_extension =
//         fft::fast_polynomial_evaluate_prime_elements(&computational_trace_polynomial_coeffs, &g2);
//     println!("Done evaluation over expanded domain!");
//     println!("trace_extension has length {}", trace_extension.len());
//     let g1_domain = g1.get_generator_domain();

//     // The round_constants_polynomial is here constructed as a `steps - 1` degree polynomial
//     // but it only depends on `round_constants.len()` values, so it should be representable
//     // in a simpler form.
//     let rc_length = round_constants.len();
//     let round_constants_repeated = (0..steps)
//         .map(|i| round_constants[i % rc_length])
//         .collect::<Vec<PrimeFieldElement>>();
//     let mut round_constants_polynomial =
//         fft::fast_polynomial_interpolate_prime_elements(&round_constants_repeated, &g1);
//     round_constants_polynomial.append(&mut vec![
//         PrimeFieldElement::new(0, &field);
//         (expansion_factor - 1) * steps
//     ]);
//     let round_constants_extension =
//         fft::fast_polynomial_evaluate_prime_elements(&round_constants_polynomial, &g2);

//     // Evaluate the Algebraic Intermediate Representation (composed polynomial), such that
//     // AIR(x) = A(te(x), te(g1*x), rc(x)) = te(g1*x) - te(x)**3 - rc(x)
//     let mut air: Vec<PrimeFieldElement> = Vec::with_capacity(extended_domain_length);
//     for i in 0..extended_domain_length {
//         let evaluation = trace_extension[((i + expansion_factor) % extended_domain_length)]
//             - trace_extension[i].mod_pow(3)
//             - round_constants_extension[i % round_constants_extension.len()];
//         air.push(evaluation);
//     }
//     println!("Computed air(x)");

//     // TODO: Alan wants to replace this with a monomial-basis calculation of Q,
//     // and then use that to calculate q_evaluations
//     // Calculate Zerofier(x) = prod_{i=i}^{N}(x - g1^i) = (x^N - 1) / (x - g2^N)
//     // Calculate the inverse of Z(x) so we can divide with it by multiplying with `1/Z(x)`
//     let g2_domain: Vec<PrimeFieldElement> = g2.get_generator_domain();
//     let one = PrimeFieldElement::new(1, &field);
//     let mut zerofier_numerator: Vec<PrimeFieldElement> =
//         vec![PrimeFieldElement::new(0, &field); extended_domain_length];
//     for i in 0..extended_domain_length {
//         // calculate x**N - 1 = g2**(i*steps) - 1
//         zerofier_numerator[i] = g2_domain[i * steps % extended_domain_length] - one;
//     }

//     let zerofier_numerator_inv: Vec<PrimeFieldElement> =
//         field.batch_inversion_elements(zerofier_numerator);
//     let last_step: &PrimeFieldElement = g1_domain.last().unwrap();
//     let zerofier_denominator: Vec<PrimeFieldElement> = g2_domain
//         .iter()
//         .map(|x| *x - *last_step)
//         .collect::<Vec<PrimeFieldElement>>();
//     let zerofier_inv = zerofier_numerator_inv
//         .iter()
//         .zip(zerofier_denominator.iter())
//         .map(|(a, b)| *a * *b)
//         .collect::<Vec<PrimeFieldElement>>();

//     // Calculate Q(x) = air(x) / Zerofier(x)
//     let mut q_evaluations: Vec<PrimeFieldElement> =
//         vec![PrimeFieldElement::new(0, &field); extended_domain_length];
//     for i in 0..extended_domain_length {
//         q_evaluations[i] = air[i] * zerofier_inv[i];
//     }
//     println!("Computed Q(x)");

//     // TODO: DEBUG: REMOVE!!!
//     for i in 0..2000 {
//         if q_evaluations[i] == air[i] * zerofier_inv[i] {
//         } else {
//             println!("Values did **not** match for i = {}", i);
//             return;
//         }
//     }

//     let line = field.lagrange_interpolation_2((one, mimc_input), (*last_step, *output));
//     let boundary_interpolant_evaluations = field.evaluate_straight_line(line, &g2_domain);

//     // TODO: Add polynomial support to the PrimeFieldElement struct, so we have
//     // support for regular polynomials and not just, as now, extension field polynomials
//     let pqr_mock = PolynomialQuotientRing::new(100, field.q);
//     let vanishing_polynomial_factor0 = PrimeFieldPolynomial {
//         coefficients: vec![-1, 1], // x - 1
//         pqr: &pqr_mock,
//     };
//     let vanishing_polynomial_factor1 = PrimeFieldPolynomial {
//         coefficients: vec![(-*last_step).value, 1], // x - g^steps
//         pqr: &pqr_mock,
//     };
//     // Boundary interpolant, is zero in the boundary-checking x values which are x = 1 and x = g1^steps
//     let vanishing_polynomial: PrimeFieldPolynomial =
//         vanishing_polynomial_factor0.mul(&vanishing_polynomial_factor1);

//     // TODO: Calculate BQ(x) analytically instead of calculating it through other polynomials

//     let mut vanishing_polynomial_coefficients: Vec<PrimeFieldElement> = vanishing_polynomial
//         .coefficients
//         .iter()
//         .map(|x| PrimeFieldElement::new(*x, &field))
//         .collect::<Vec<PrimeFieldElement>>();
//     vanishing_polynomial_coefficients.append(&mut vec![
//         PrimeFieldElement::new(0, &field);
//         extended_domain_length - 3
//     ]);
//     let vanishing_polynomial_evaluations: Vec<PrimeFieldElement> =
//         fft::fast_polynomial_evaluate_prime_elements(&vanishing_polynomial_coefficients, &g2);
//     let vanishing_polynomial_evaluations_inv: Vec<PrimeFieldElement> =
//         field.batch_inversion_elements(vanishing_polynomial_evaluations.clone());

//     // Evaluate BQ(x) = (te(x) - I(x)) / Z_boundary(x)
//     let mut bq_evaluations: Vec<PrimeFieldElement> =
//         vec![PrimeFieldElement::new(0, &field); extended_domain_length];
//     for i in 0..extended_domain_length {
//         bq_evaluations[i] = (trace_extension[i] - boundary_interpolant_evaluations[i])
//             * vanishing_polynomial_evaluations_inv[i];
//     }
//     println!("Computed BQ(x)");

//     // Wrapping the primefield elements into a triplet will allow us to use our fast Merkle Tree
//     // code that requires the number of nodes to be a power of two
//     let polynomial_evaluations: Vec<(PrimeFieldElement, PrimeFieldElement, PrimeFieldElement)> =
//         trace_extension
//             .iter()
//             .zip(q_evaluations.iter())
//             .zip(bq_evaluations.iter())
//             .map(|((te, q), bq)| (*te, *q, *bq))
//             .collect::<Vec<(PrimeFieldElement, PrimeFieldElement, PrimeFieldElement)>>();
//     let tuple_merkle_tree: MerkleTree<(
//         PrimeFieldElement,
//         PrimeFieldElement,
//         PrimeFieldElement,
//     )> = MerkleTree::from_vec(&polynomial_evaluations);
//     println!(
//         "Computed merkle tree. Root: {:?}",
//         tuple_merkle_tree.get_root()
//     );

//     // Verify that we can extract values from the Merkle Tree
//     // TODO: DEBUG: REMOVE!!!
//     for i in 700..790 {
//         let val: Vec<Node<(PrimeFieldElement, PrimeFieldElement, PrimeFieldElement)>> =
//             tuple_merkle_tree.get_proof(i);
//         let (p, d, b) = val[0].value.unwrap();
//         if p == trace_extension[i] {
//             // println!("Values matched for i = {}", i);
//         } else {
//             panic!("Values did **not** match for p i = {}", i);
//         }
//         if d == q_evaluations[i] {
//             // println!("Values matched for i = {}", i);
//         } else {
//             panic!("Values did **not** match for d i = {}", i);
//         }
//         if b == bq_evaluations[i] {
//             // println!("Values matched for i = {}", i);
//         } else {
//             panic!("Values did **not** match for b i = {}", i);
//         }
//         if !MerkleTree::verify_proof(tuple_merkle_tree.get_root(), i as u64, val) {
//             panic!("Failed to verify Merkle Tree proof for i = {}", i);
//         }
//     }

//     //for i in 0..range  {}
//     // z_x_2.evaluate(x: &'d PrimeFieldElement)

//     for i in 0..steps - 1 {
//         // println!(
//         //     "C(P({})) = {}",
//         //     g1_domain[i].value,
//         //     c_of_p_evaluations[i * expansion_factor ]
//         // );
//         if air[i * expansion_factor].value != 0 {
//             panic!(
//                 "C(P(x)) != 0 for x = {} => i = {}. Got C(P(x)) = {}",
//                 g2_domain[i * expansion_factor],
//                 i * expansion_factor,
//                 air[i * expansion_factor].value
//             );
//         }
//     }

//     // Find a pseudo-random linear combination of te, te*x^steps, BQ, BQ*x^steps, Q and prove
//     // low-degreenes of this
//     // Alan wants to take a random linear combination of two pairs of all polynomials, such that the second element of each pair is always degree 2^n-1
//     let mt_root_hash = tuple_merkle_tree.get_root();
//     let k_seeds = utils::get_n_hash_rounds(&mt_root_hash, 4);
//     let ks = k_seeds
//         .iter()
//         .map(|seed| PrimeFieldElement::from_bytes(&field, seed))
//         .collect::<Vec<PrimeFieldElement>>();

//     // Calculate `powers = x^steps`
//     let g2_pow_steps = g2_domain[steps];
//     let mut powers = vec![PrimeFieldElement::new(0, &field); extended_domain_length];
//     powers[0] = PrimeFieldElement::new(1, &field);
//     for i in 1..extended_domain_length {
//         // g2^i = g2^(i - 1) * g2^steps => x[i] = x[i - 1] * g2^steps
//         powers[i] = powers[i - 1] * g2_pow_steps;
//     }

//     let mut lc_codeword = vec![PrimeFieldElement::new(0, &field); extended_domain_length];
//     for i in 1..extended_domain_length {
//         lc_codeword[i] = q_evaluations[i]
//             + ks[0] * trace_extension[i]
//             + ks[1] * trace_extension[i] * powers[i]
//             + ks[2] * bq_evaluations[i]
//             + ks[3] * powers[i] * bq_evaluations[i];
//     }

//     //

//     // Alan: Don't need to send this tree
//     let l_mtree = MerkleTree::from_vec(&lc_codeword);
//     println!(
//         "Computed linear combination of low-degree polynomials. Got hash: {:?}",
//         l_mtree.get_root()
//     );

//     // Get pseudo-random indices from `l_mtree.get_root()`.
//     let index_preimages = get_n_hash_rounds(&l_mtree.get_root(), security_checks as u32);
//     let indices: Vec<usize> = index_preimages
//         .iter()
//         .map(|x| get_index_from_bytes(x, extended_domain_length))
//         .collect::<Vec<usize>>();
//     let te_q_bq_proofs = tuple_merkle_tree.get_multi_proof(&indices);
//     let l_proofs = l_mtree.get_multi_proof(&indices);
//     println!("te_q_bq_proofs = {:?}", te_q_bq_proofs);
//     println!("l_proofs = {:?}", l_proofs);
//     // TODO: REMOVE this when low_degree_test is changed to use PrimeFieldElements instead
//     // of i128
//     let trace_extension_i128 = trace_extension
//         .iter()
//         .map(|x| x.value)
//         .collect::<Vec<i128>>();
//     let bq_evaluations_i128 = bq_evaluations
//         .iter()
//         .map(|x| x.value)
//         .collect::<Vec<i128>>();
//     let q_evaluations_i128 = q_evaluations.iter().map(|x| x.value).collect::<Vec<i128>>();
//     let lc_codeword_i128 = lc_codeword.iter().map(|x| x.value).collect::<Vec<i128>>();

//     let mut output = vec![];
//     let low_degree_proof = low_degree_test::prover_i128(
//         &lc_codeword_i128,
//         field.q,
//         (steps * 2 - 1) as u32,
//         security_checks,
//         &mut output,
//         g2.value,
//     )
//     .unwrap();
//     let verify: Result<(), low_degree_test::ValidationError> =
//         low_degree_test::verify_i128(low_degree_proof, field.q);
//     if verify.is_err() {
//         println!("L failed low-degree test");
//     }

//     // TODO: DEBUG: REMOVE!
//     output = vec![];
//     println!("Length of lc_codeword_i128 = {}", lc_codeword_i128.len());
//     let mut low_degree_proof: low_degree_test::LowDegreeProof<i128> = low_degree_test::prover_i128(
//         &trace_extension_i128,
//         field.q,
//         (steps - 1) as u32,
//         security_checks,
//         &mut output,
//         g2.value,
//     )
//     .unwrap();
//     let mut verify: Result<(), low_degree_test::ValidationError> =
//         low_degree_test::verify_i128(low_degree_proof, field.q);
//     if verify.is_err() {
//         println!("P failed low-degree test");
//     }
//     // assert!(verify.is_ok());
//     output = vec![];
//     low_degree_proof = low_degree_test::prover_i128(
//         &bq_evaluations_i128,
//         field.q,
//         (steps - 1) as u32,
//         security_checks,
//         &mut output,
//         g2.value,
//     )
//     .unwrap();
//     verify = low_degree_test::verify_i128(low_degree_proof, field.q);
//     if verify.is_err() {
//         println!("B failed low-degree test");
//     }
//     // assert!(verify.is_ok());
//     output = vec![];
//     low_degree_proof = low_degree_test::prover_i128(
//         &q_evaluations_i128,
//         field.q,
//         (2 * steps - 1) as u32,
//         security_checks,
//         &mut output,
//         g2.value,
//     )
//     .unwrap();
//     verify = low_degree_test::verify_i128(low_degree_proof, field.q);
//     if verify.is_err() {
//         println!("D failed low-degree test");
//     }
// }

#[cfg(test)]
mod test_modular_arithmetic {
    use super::*;
    use crate::shared_math::prime_field_element_big::PrimeFieldBig;

    fn b(x: i128) -> BigInt {
        Into::<BigInt>::into(x)
    }

    #[test]
    fn mimc_big_test() {
        // let mut ret: Option<(PrimeFieldBig, BigInt)> = None;
        // PrimeFieldBig::get_field_with_primitive_root_of_unity(8, 100, &mut ret);
        // println!("Found: ret = {:?}", ret);
        let no_steps = 3;
        let expansion_factor = 4;
        let extended_domain_length = (no_steps + 1) * expansion_factor;
        let security_factor = 10;
        let field = PrimeFieldBig::new(b(5 * 2i128.pow(25) + 1));
        // let round_constants_raw: Vec<i128> = utils::generate_random_numbers(no_steps, 17);
        let round_constants_raw: Vec<i128> = vec![7, 256, 117];
        let round_constants: Vec<PrimeFieldElementBig> = round_constants_raw
            .iter()
            .map(|x| PrimeFieldElementBig::new(b(x.to_owned()), &field)) // TODO: FIX!! REPLACE value BY x
            .collect::<Vec<PrimeFieldElementBig>>();
        let (g2_option, _) = field.get_primitive_root_of_unity((no_steps + 1) * expansion_factor);
        let omega = g2_option.unwrap();
        let omicron = omega.mod_pow(b(expansion_factor));
        println!("Found omega = {}", omega);
        println!("Found omicron = {}", omicron);
        println!("omicron domain = {:?}", omicron.get_generator_domain());
        let omega_domain = omega.get_generator_domain();
        println!("omega_domain = {:?}", omega_domain);

        for i in 0..20 {
            println!("i = {}", i);
            let mimc_input = PrimeFieldElementBig::new(b(i), &field);
            let mimc_trace = mimc_forward(&mimc_input, no_steps as usize, &round_constants);
            let mimc_output = mimc_trace[no_steps as usize].clone();
            println!("\n\n\n\n\n\n\n\n\n\nmimc_input = {}", mimc_input);
            println!("mimc_output = {}", mimc_output);
            println!("x_last = {}", omicron.mod_pow(b(no_steps - 1)));
            let stark_res = stark_of_mimc(
                security_factor,
                no_steps as usize,
                expansion_factor as usize,
                omega.clone(),
                mimc_input.clone(),
                mimc_output.clone(),
                &round_constants,
            );

            let stark_proof: StarkProof<BigInt> = match stark_res {
                Ok(stark_proof) => stark_proof,
                Err(_err) => panic!("Failed to produce STARK proof"),
            };

            // Verify low-degreeness of linear combination
            assert!(low_degree_test::verify_bigint(
                stark_proof.linear_combination_fri.clone(),
                field.q.clone()
            )
            .is_ok());

            // Verify properties of polynomials

            // ## verify linear combination
            let lc_coefficients =
                get_linear_combination_coefficients(omega.field, &stark_proof.tuple_merkle_root);

            // get all indices (a,b) of first FRI round
            let abc_indices: Vec<(usize, usize, usize)> =
                stark_proof.linear_combination_fri.get_abc_indices(0);
            // let indices = ab_indices.iter().map(|(a, b)| )
            // todo: use https://users.rust-lang.org/t/flattening-a-vector-of-tuples/11409/2
            let mut ab_indices: Vec<usize> = vec![];
            for (a, b, _) in abc_indices.iter() {
                ab_indices.push(*a);
                ab_indices.push(*b);
            }
            let shifted_indices = ab_indices
                .iter()
                .map(|x| (x + expansion_factor as usize) % extended_domain_length as usize)
                .collect::<Vec<usize>>();

            // verify tuple authentication paths and shifted authentication paths
            let tuple_authentication_paths: Vec<
                CompressedAuthenticationPath<(BigInt, BigInt, BigInt)>,
            > = stark_proof.tuple_authentication_paths.clone();

            // Merkle::verify(root, indices, leafs, paths);
            assert!(MerkleTree::verify_multi_proof(
                stark_proof.tuple_merkle_root,
                &ab_indices,
                &tuple_authentication_paths,
            ));
            let shifted_tuple_authentication_paths = stark_proof.shifted_tuple_authentication_paths;
            assert!(MerkleTree::verify_multi_proof(
                stark_proof.tuple_merkle_root,
                &shifted_indices,
                &shifted_tuple_authentication_paths
            ));

            for j in 0..stark_proof.linear_combination_fri.s as usize * 2 {
                // verify linear relation on indicated elements
                let x = omega_domain[ab_indices[j]].clone();
                let x_power_ns_plus_1 = omega_domain
                    [ab_indices[j] * (no_steps + 1) as usize % extended_domain_length as usize]
                    .clone();
                let x_power_ns_plus_3 = omega_domain
                    [ab_indices[j] * (no_steps + 3) as usize % extended_domain_length as usize]
                    .clone();
                let extended_computational_trace: PrimeFieldElementBig = PrimeFieldElementBig::new(
                    stark_proof.tuple_authentication_paths[j]
                        .get_value()
                        .0
                        .clone(),
                    omega.field,
                );
                let shifted_trace_codeword: PrimeFieldElementBig = PrimeFieldElementBig::new(
                    stark_proof.tuple_authentication_paths[j]
                        .get_value()
                        .0
                        .clone(),
                    omega.field,
                ) * x_power_ns_plus_1.clone();
                let transition_quotient_codeword: PrimeFieldElementBig = PrimeFieldElementBig::new(
                    stark_proof.tuple_authentication_paths[j]
                        .get_value()
                        .1
                        .clone(),
                    omega.field,
                );
                let shifted_transition_quotient_codeword = PrimeFieldElementBig::new(
                    stark_proof.tuple_authentication_paths[j]
                        .get_value()
                        .1
                        .clone(),
                    omega.field,
                ) * x.clone();
                let boundary_quotient_codeword = PrimeFieldElementBig::new(
                    stark_proof.tuple_authentication_paths[j]
                        .get_value()
                        .2
                        .clone(),
                    omega.field,
                );
                let shifted_boundary_quotient_codeword = PrimeFieldElementBig::new(
                    stark_proof.tuple_authentication_paths[j]
                        .get_value()
                        .2
                        .clone(),
                    omega.field,
                ) * x_power_ns_plus_3.clone();
                let right_hand_side: PrimeFieldElementBig = get_linear_combination(
                    &lc_coefficients,
                    PolynomialEvaluations {
                        extended_computational_trace,
                        shifted_trace_codeword,
                        transition_quotient_codeword,
                        shifted_transition_quotient_codeword,
                        boundary_quotient_codeword,
                        shifted_boundary_quotient_codeword,
                    },
                );
                println!("right_hand_side = {}", right_hand_side);
                let left_hand_side: PrimeFieldElementBig = PrimeFieldElementBig::new(
                    stark_proof.linear_combination_fri.ab_proofs[0][j].get_value(),
                    omega.field,
                );
                println!("left_hand_side = {}", left_hand_side);
                assert_eq!(left_hand_side, right_hand_side);
            }
        }
    }

    // #[test]
    // fn mimc_forward_small() {
    //     let field = PrimeField::new(17);
    //     let input = PrimeFieldElement::new(6, &field);
    //     let round_constant = vec![
    //         PrimeFieldElement::new(6, &field),
    //         PrimeFieldElement::new(2, &field),
    //         PrimeFieldElement::new(1, &field),
    //         PrimeFieldElement::new(9, &field),
    //     ];
    //     let result: Vec<PrimeFieldElement> = mimc_forward_i128(&input, 7, &round_constant);

    //     // Result was verified on WolframAlpha: works for input = 6 mod 17
    //     assert_eq!(13, result.last().unwrap().value);

    //     for j in 0..10 {
    //         for i in 0..16 {
    //             let input2 = PrimeFieldElement::new(i, &field);
    //             let result2 = mimc_forward_i128(&input2, j, &round_constant);
    //             assert_eq!(
    //                 input2,
    //                 mimc_backward(&result2.last().unwrap(), j, &round_constant)
    //             );
    //         }
    //     }
    // }

    // #[test]
    // fn small_stark_of_mimc() {
    //     #![allow(clippy::just_underscores_and_digits)]
    //     // Create a field with a 8192th primitive root of unity and a (8*8192)th primitive root of unity
    //     // This field is mod 65537 and the roots are 6561, 3, respectively.

    //     // Use these to index into m and l.
    //     // Then generate a proof that lc_codeword is of low degree (steps * 2)
    //     let field = PrimeField::new(65537);
    //     stark_of_mimc_i128(
    //         80,
    //         8192,
    //         8,
    //         65537,
    //         PrimeFieldElement::new(3, &field),
    //         PrimeFieldElement::new(827, &field),
    //     );
    // }
}
