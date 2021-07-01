use crate::fft;
use crate::shared_math::low_degree_test;
use crate::shared_math::polynomial_quotient_ring::PolynomialQuotientRing;
use crate::shared_math::prime_field_element::{PrimeField, PrimeFieldElement};
use crate::shared_math::prime_field_polynomial::PrimeFieldPolynomial;
use crate::util_types::merkle_tree_vector::{MerkleTreeVector, Node};
use crate::utils;
use crate::utils::{get_index_from_bytes_exclude_multiples, get_n_hash_rounds};

pub fn mimc_forward<'a>(
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
    steps: usize,
    expansion_factor: usize,
    field_modulus: i128,
    g2: PrimeFieldElement,
    mimc_input: PrimeFieldElement,
) {
    let g1: PrimeFieldElement = g2.mod_pow(expansion_factor as i128);
    let field = PrimeField::new(field_modulus);

    let extended_domain_length: usize = steps * expansion_factor;
    println!("g1 = {}", g1);

    // Get the computational computational_trace of MIMC for some round_constant values
    // TODO: FIX!! REPLACE modulus
    // let round_constants_raw: Vec<i128> = vec![2i128; 128];
    // let round_constants_raw: Vec<i128> = utils::generate_random_numbers(128, 5);
    // TODO: FIX!! REPLACE modulus BY 65573
    let round_constants_raw: Vec<i128> = utils::generate_random_numbers(64, 65573);
    let round_constants: Vec<PrimeFieldElement> = round_constants_raw
        .iter()
        .map(|x| PrimeFieldElement::new(*x, &field)) // TODO: FIX!! REPLACE value BY x
        //.map(|x| PrimeFieldElement::new(1, &field)) // TODO: FIX!! REPLACE value BY x
        .collect::<Vec<PrimeFieldElement>>();
    println!("Generating computational_trace");
    let computational_trace: Vec<PrimeFieldElement> =
        mimc_forward(&mimc_input, steps - 1, &round_constants);
    let output: &PrimeFieldElement = computational_trace.last().unwrap();
    println!(
        "Done generating trace with length {}",
        computational_trace.len()
    );
    let mut computational_trace_polynomial_coeffs: Vec<PrimeFieldElement> =
        fft::fast_polynomial_interpolate_prime_elements(&computational_trace, &g1);
    println!(
        "computational_trace_polynomial_coeffs has length {}",
        computational_trace_polynomial_coeffs.len()
    );
    computational_trace_polynomial_coeffs.append(&mut vec![
        PrimeFieldElement::new(0, &field);
        (expansion_factor - 1) * steps
    ]);
    println!(
        "expanded computational_trace_polynomial_coeffs has length {}",
        computational_trace_polynomial_coeffs.len()
    );
    let trace_extension =
        fft::fast_polynomial_evaluate_prime_elements(&computational_trace_polynomial_coeffs, &g2);
    println!("Done evaluation over expanded domain!");
    println!("trace_extension has length {}", trace_extension.len());
    let g1_domain = g1.get_generator_domain();

    // The round_constants_polynomial is here constructed as a `steps - 1` degree polynomial
    // but it only depends on `round_constants.len()` values, so it should be representable
    // in a simpler form.
    let rc_length = round_constants.len();
    let round_constants_repeated = (0..steps)
        .map(|i| round_constants[i % rc_length])
        .collect::<Vec<PrimeFieldElement>>();
    let mut round_constants_polynomial =
        fft::fast_polynomial_interpolate_prime_elements(&round_constants_repeated, &g1);
    round_constants_polynomial.append(&mut vec![
        PrimeFieldElement::new(0, &field);
        (expansion_factor - 1) * steps
    ]);
    let round_constants_extension =
        fft::fast_polynomial_evaluate_prime_elements(&round_constants_polynomial, &g2);

    // Evaluate the Algebraic Intermediate Representation (composed polynomial), such that
    // AIR(x) = A(te(x), te(g1*x), rc(x)) = te(g1*x) - te(x)**3 - rc(x)
    let mut air: Vec<PrimeFieldElement> = Vec::with_capacity(extended_domain_length);
    for i in 0..extended_domain_length {
        let evaluation = trace_extension[((i + expansion_factor) % extended_domain_length)]
            - trace_extension[i].mod_pow(3)
            - round_constants_extension[i % round_constants_extension.len()];
        air.push(evaluation);
    }
    println!("Computed air(x)");

    // TODO: Alan wants to replace this with a monomial-basis calculation of Q,
    // and then use that to calculate q_evaluations
    // Calculate Zerofier(x) = prod_{i=i}^{N}(x - g1^i) = (x^N - 1) / (x - g2^N)
    // Calculate the inverse of Z(x) so we can divide with it by multiplying with `1/Z(x)`
    let g2_domain: Vec<PrimeFieldElement> = g2.get_generator_domain();
    let one = PrimeFieldElement::new(1, &field);
    let mut zerofier_numerator: Vec<PrimeFieldElement> =
        vec![PrimeFieldElement::new(0, &field); extended_domain_length];
    for i in 0..extended_domain_length {
        zerofier_numerator[i] = g2_domain[i * steps % extended_domain_length] - one;
    }

    let zerofier_numerator_inv: Vec<PrimeFieldElement> =
        field.batch_inversion_elements(zerofier_numerator);
    let last_step: &PrimeFieldElement = g1_domain.last().unwrap();
    let zerofier_denominator: Vec<PrimeFieldElement> = g2_domain
        .iter()
        .map(|x| *x - *last_step)
        .collect::<Vec<PrimeFieldElement>>();
    let zerofier_inv = zerofier_numerator_inv
        .iter()
        .zip(zerofier_denominator.iter())
        .map(|(a, b)| *a * *b)
        .collect::<Vec<PrimeFieldElement>>();

    // Calculate Q(x) = air(x) / Zerofier(x)
    let mut q_evaluations: Vec<PrimeFieldElement> =
        vec![PrimeFieldElement::new(0, &field); extended_domain_length];
    for i in 0..extended_domain_length {
        q_evaluations[i] = air[i] * zerofier_inv[i];
    }
    println!("Computed D(x)");

    // TODO: DEBUG: REMOVE!!!
    for i in 0..2000 {
        if q_evaluations[i] == air[i] * zerofier_inv[i] {
        } else {
            println!("Values did **not** match for i = {}", i);
            return;
        }
    }

    let line = field.lagrange_interpolation_2((one, mimc_input), (*last_step, *output));
    let boundary_interpolant_evaluations = field.evaluate_straight_line(line, &g2_domain);

    // TODO: Add polynomial support to the PrimeFieldElement struct, so we have
    // support for regular polynomials and not just, as now, extension field polynomials
    let pqr_mock = PolynomialQuotientRing::new(100, field.q);
    let vanishing_polynomial_factor0 = PrimeFieldPolynomial {
        coefficients: vec![-1, 1], // x - 1
        pqr: &pqr_mock,
    };
    let vanishing_polynomial_factor1 = PrimeFieldPolynomial {
        coefficients: vec![(-*last_step).value, 1], // x - g^steps
        pqr: &pqr_mock,
    };
    // Boundary interpolant, is zero in the boundary-checking x values which are x = 1 and x = g1^steps
    let vanishing_polynomial = vanishing_polynomial_factor0.mul(&vanishing_polynomial_factor1);

    // TODO: Calculate BQ(x) analytically instead of calculating it through other polynomials

    let mut vanishing_polynomial_coefficients: Vec<PrimeFieldElement> = vanishing_polynomial
        .coefficients
        .iter()
        .map(|x| PrimeFieldElement::new(*x, &field))
        .collect::<Vec<PrimeFieldElement>>();
    vanishing_polynomial_coefficients.append(&mut vec![
        PrimeFieldElement::new(0, &field);
        extended_domain_length - 3
    ]);
    let vanishing_polynomial_evaluations: Vec<PrimeFieldElement> =
        fft::fast_polynomial_evaluate_prime_elements(&vanishing_polynomial_coefficients, &g2);
    let vanishing_polynomial_evaluations_inv: Vec<PrimeFieldElement> =
        field.batch_inversion_elements(vanishing_polynomial_evaluations.clone());

    // Evaluate BQ(x) = (te(x) - I(x)) / Z_boundary(x)
    let mut bq_evaluations: Vec<PrimeFieldElement> =
        vec![PrimeFieldElement::new(0, &field); extended_domain_length];
    for i in 0..extended_domain_length {
        bq_evaluations[i] = (trace_extension[i] - boundary_interpolant_evaluations[i])
            * vanishing_polynomial_evaluations_inv[i];
    }
    println!("Computed BQ(x)");

    // Wrapping the primefield elements into a triplet will allow us to use our fast Merkle Tree
    // code that requires the number of nodes to be a power of two
    let polynomial_evaluations: Vec<(PrimeFieldElement, PrimeFieldElement, PrimeFieldElement)> =
        trace_extension
            .iter()
            .zip(q_evaluations.iter())
            .zip(bq_evaluations.iter())
            .map(|((p, d), b)| (*p, *d, *b))
            .collect::<Vec<(PrimeFieldElement, PrimeFieldElement, PrimeFieldElement)>>();
    let polynomials_merkle_tree: MerkleTreeVector<(
        PrimeFieldElement,
        PrimeFieldElement,
        PrimeFieldElement,
    )> = MerkleTreeVector::from_vec(&polynomial_evaluations);
    println!(
        "Computed merkle tree. Root: {:?}",
        polynomials_merkle_tree.get_root()
    );

    // Verify that we can extract values from the Merkle Tree
    // TODO: DEBUG: REMOVE!!!
    for i in 700..790 {
        let val: Vec<Node<(PrimeFieldElement, PrimeFieldElement, PrimeFieldElement)>> =
            polynomials_merkle_tree.get_proof(i);
        let (p, d, b) = val[0].value.unwrap();
        if p == trace_extension[i] {
            // println!("Values matched for i = {}", i);
        } else {
            panic!("Values did **not** match for p i = {}", i);
        }
        if d == q_evaluations[i] {
            // println!("Values matched for i = {}", i);
        } else {
            panic!("Values did **not** match for d i = {}", i);
        }
        if b == bq_evaluations[i] {
            // println!("Values matched for i = {}", i);
        } else {
            panic!("Values did **not** match for b i = {}", i);
        }
        if !MerkleTreeVector::verify_proof(polynomials_merkle_tree.get_root(), i as u64, val) {
            panic!("Failed to verify Merkle Tree proof for i = {}", i);
        }
    }

    //for i in 0..range  {}
    // z_x_2.evaluate(x: &'d PrimeFieldElement)

    for i in 0..steps - 1 {
        // println!(
        //     "C(P({})) = {}",
        //     g1_domain[i].value,
        //     c_of_p_evaluations[i * expansion_factor ]
        // );
        if air[i * expansion_factor].value != 0 {
            panic!(
                "C(P(x)) != 0 for x = {} => i = {}. Got C(P(x)) = {}",
                g2_domain[i * expansion_factor],
                i * expansion_factor,
                air[i * expansion_factor].value
            );
        }
    }

    // Find a pseudo-random linear combination of P, P*x^steps, B, B*x^steps, D and prove
    // low-degreenes of this
    let mt_root_hash = polynomials_merkle_tree.get_root();
    let k_seeds = utils::get_n_hash_rounds(&mt_root_hash, 4);
    let ks = k_seeds
        .iter()
        .map(|seed| PrimeFieldElement::from_bytes(&field, seed))
        .collect::<Vec<PrimeFieldElement>>();

    // Calculate x^steps
    let g2_pow_steps = g2_domain[steps];
    let mut powers = vec![PrimeFieldElement::new(0, &field); extended_domain_length];
    powers[0] = PrimeFieldElement::new(1, &field);
    for i in 1..extended_domain_length {
        // g2^i = g2^(i - 1) * g2^steps => x[i] = x[i - 1] * g2^steps
        powers[i] = powers[i - 1] * g2_pow_steps;
    }

    let mut l_evaluations = vec![PrimeFieldElement::new(0, &field); extended_domain_length];
    for i in 1..extended_domain_length {
        l_evaluations[i] = q_evaluations[i]
            + ks[0] * trace_extension[i]
            + ks[1] * trace_extension[i] * powers[i]
            + ks[2] * bq_evaluations[i]
            + ks[3] * powers[i] * bq_evaluations[i];
    }

    let l_mtree = MerkleTreeVector::from_vec(&l_evaluations);
    println!(
        "Computed linear combination of low-degree polynomials. Got hash: {:?}",
        l_mtree.get_root()
    );

    // Get pseudo-random indices from `l_mtree.get_root()`.
    let index_preimages = get_n_hash_rounds(&l_mtree.get_root(), security_checks as u32);
    let indices: Vec<usize> = index_preimages
        .iter()
        .map(|x| {
            get_index_from_bytes_exclude_multiples(x, extended_domain_length, expansion_factor)
        })
        .collect::<Vec<usize>>();
    let p_d_b_proofs = polynomials_merkle_tree.get_multi_proof(&indices);
    let l_proofs = l_mtree.get_multi_proof(&indices);
    println!("p_d_b_proofs = {:?}", p_d_b_proofs);
    println!("l_proofs = {:?}", l_proofs);
    // TODO: REMOVE this when low_degree_test is changed to use PrimeFieldElements instead
    // of i128
    let trace_extension_i128 = trace_extension
        .iter()
        .map(|x| x.value)
        .collect::<Vec<i128>>();
    let b_evaluations_i128 = bq_evaluations
        .iter()
        .map(|x| x.value)
        .collect::<Vec<i128>>();
    let q_evaluations_i128 = q_evaluations.iter().map(|x| x.value).collect::<Vec<i128>>();
    let l_evaluations_i128 = l_evaluations.iter().map(|x| x.value).collect::<Vec<i128>>();
    let mut output = vec![];
    println!(
        "Length of l_evaluations_i128 = {}",
        l_evaluations_i128.len()
    );
    let mut low_degree_proof: low_degree_test::LowDegreeProof<i128> = low_degree_test::prover(
        &trace_extension_i128,
        field.q,
        (steps - 1) as u32,
        security_checks,
        &mut output,
        g2.value,
    );
    let mut verify: Result<(), low_degree_test::ValidationError> =
        low_degree_test::verify(low_degree_proof, field.q);
    if verify.is_err() {
        println!("P failed low-degree test");
    }
    // assert!(verify.is_ok());
    output = vec![];
    low_degree_proof = low_degree_test::prover(
        &b_evaluations_i128,
        field.q,
        (steps - 1) as u32,
        security_checks,
        &mut output,
        g2.value,
    );
    verify = low_degree_test::verify(low_degree_proof, field.q);
    if verify.is_err() {
        println!("B failed low-degree test");
    }
    // assert!(verify.is_ok());
    output = vec![];
    low_degree_proof = low_degree_test::prover(
        &q_evaluations_i128,
        field.q,
        (2 * steps - 1) as u32,
        security_checks,
        &mut output,
        g2.value,
    );
    verify = low_degree_test::verify(low_degree_proof, field.q);
    if verify.is_err() {
        println!("D failed low-degree test");
    }
    // assert!(verify.is_ok());
    // l_evaluations_i128[100] = 100;
    output = vec![];
    let low_degree_proof = low_degree_test::prover(
        &l_evaluations_i128,
        field.q,
        (steps * 2 - 1) as u32,
        security_checks,
        &mut output,
        g2.value,
    );
    let verify: Result<(), low_degree_test::ValidationError> =
        low_degree_test::verify(low_degree_proof, field.q);
    if verify.is_err() {
        println!("L failed low-degree test");
    }
}

#[cfg(test)]
mod test_modular_arithmetic {
    use super::*;
    use crate::shared_math::prime_field_element::PrimeField;

    #[test]
    fn mimc_forward_small() {
        let field = PrimeField::new(17);
        let input = PrimeFieldElement::new(6, &field);
        let round_constant = vec![
            PrimeFieldElement::new(6, &field),
            PrimeFieldElement::new(2, &field),
            PrimeFieldElement::new(1, &field),
            PrimeFieldElement::new(9, &field),
        ];
        let result: Vec<PrimeFieldElement> = mimc_forward(&input, 7, &round_constant);

        // Result was verified on WolframAlpha: works for input = 6 mod 17
        assert_eq!(13, result.last().unwrap().value);

        for j in 0..10 {
            for i in 0..16 {
                let input2 = PrimeFieldElement::new(i, &field);
                let result2 = mimc_forward(&input2, j, &round_constant);
                assert_eq!(
                    input2,
                    mimc_backward(&result2.last().unwrap(), j, &round_constant)
                );
            }
        }
    }

    #[test]
    fn small_stark_of_mimc() {
        #![allow(clippy::just_underscores_and_digits)]
        // Create a field with a 8192th primitive root of unity and a (8*8192)th primitive root of unity
        // This field is mod 65537 and the roots are 6561, 3, respectively.

        // Use these to index into m and l.
        // Then generate a proof that l_evaluations is of low degree (steps * 2)
        let field = PrimeField::new(65537);
        stark_of_mimc(
            80,
            8192,
            8,
            65537,
            PrimeFieldElement::new(3, &field),
            PrimeFieldElement::new(827, &field),
        );
    }
}
