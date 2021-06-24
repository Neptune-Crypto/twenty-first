use crate::shared_math::prime_field_element::PrimeFieldElement;

pub fn mimc_forward<'a>(
    input: &'a PrimeFieldElement,
    steps: usize,
    round_costants: &'a [PrimeFieldElement],
) -> Vec<PrimeFieldElement<'a>> {
    let mut trace: Vec<PrimeFieldElement<'a>> = Vec::with_capacity(steps);
    let mut res: PrimeFieldElement<'a> = *input;
    trace.push(*input);
    for i in 0..steps {
        res = res.mod_pow(3) + round_costants[i % round_costants.len()];
        trace.push(res);
    }

    trace
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

#[cfg(test)]
mod test_modular_arithmetic {
    use super::*;
    use crate::fft;
    use crate::shared_math::low_degree_test;
    use crate::shared_math::polynomial_quotient_ring::PolynomialQuotientRing;
    use crate::shared_math::prime_field_element::PrimeField;
    use crate::shared_math::prime_field_polynomial::PrimeFieldPolynomial;
    use crate::util_types::merkle_tree_vector::{MerkleTreeVector, Node};
    use crate::utils;
    use crate::utils::{get_index_from_bytes_exclude_multiples, get_n_hash_rounds};
    use std::result;

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
        let security_checks = 80usize;
        let steps = 8192usize;
        let expansion_factor = 8usize;
        let range: usize = steps * expansion_factor;
        let mut ret: Option<(PrimeField, i128)> = None;
        PrimeField::get_field_with_primitive_root_of_unity(range as i128, 2i128.pow(16), &mut ret);
        let (field, root) = ret.unwrap();
        assert_eq!(65537, field.q);
        assert_eq!(3, root);
        let g1_ret = field.get_primitive_root_of_unity(steps as i128);
        let g1: PrimeFieldElement = g1_ret.0.unwrap();
        let g2: PrimeFieldElement = PrimeFieldElement::new(root, &field);
        assert_eq!(6561, g1.value);
        assert_eq!(g2.mod_pow(expansion_factor as i128), g1);
        println!("g1 = {}", g1);

        // Get the computational trace of MIMC for some round_constant values
        let input = PrimeFieldElement::new(827, &field);
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
        println!("Generating trace");
        let trace: Vec<PrimeFieldElement> = mimc_forward(&input, steps - 1, &round_constants);
        let output: &PrimeFieldElement = trace.last().unwrap();
        println!("Done generating trace with length {}", trace.len());
        let mut trace_polynomial_coeffs: Vec<PrimeFieldElement> =
            fft::fast_polynomial_interpolate_prime_elements(&trace, &g1);
        println!(
            "trace_polynomial_coeffs has length {}",
            trace_polynomial_coeffs.len()
        );
        trace_polynomial_coeffs.append(&mut vec![
            PrimeFieldElement::new(0, &field);
            (expansion_factor - 1) * steps
        ]);
        println!(
            "expanded trace_polynomial_coeffs has length {}",
            trace_polynomial_coeffs.len()
        );
        let p_evaluations =
            fft::fast_polynomial_evaluate_prime_elements(&trace_polynomial_coeffs, &g2);
        println!("Done evaluation over expanded domain!");
        println!("p_evaluations has length {}", p_evaluations.len());
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
        let round_constants_extended =
            fft::fast_polynomial_evaluate_prime_elements(&round_constants_polynomial, &g2);

        // Evaluate the composed polynomial such that
        // C(P(x), P(g1*x), K(x)) = P(g1*x) - P(x)**3 - K(x)
        let mut c_of_p_evaluations: Vec<PrimeFieldElement> = Vec::with_capacity(range);
        for i in 0..range {
            let evaluation = p_evaluations[((i + expansion_factor) % range)]
                - p_evaluations[i].mod_pow(3)
                - round_constants_extended[i % round_constants_extended.len()];
            c_of_p_evaluations.push(evaluation);
        }
        println!("Computed C(P(x))");

        // Calculate Z(x) = prod_{i=i}^{N}(x - g1^i) = (x^N - 1) / (x - g2^N)
        // Calculate the inverse of Z(x) so we can divide with it by multiplying with `1/Z(x)`
        let g2_domain: Vec<PrimeFieldElement> = g2.get_generator_domain();
        let one = PrimeFieldElement::new(1, &field);
        let mut z_x_numerator: Vec<PrimeFieldElement> =
            vec![PrimeFieldElement::new(0, &field); range];
        for i in 0..range {
            z_x_numerator[i] = g2_domain[i * steps % range] - one;
        }

        let z_x_numerator_inv: Vec<PrimeFieldElement> =
            field.batch_inversion_elements(z_x_numerator);
        let last_step: &PrimeFieldElement = g1_domain.last().unwrap();
        let z_x_denominator: Vec<PrimeFieldElement> = g2_domain
            .iter()
            .map(|x| *x - *last_step)
            .collect::<Vec<PrimeFieldElement>>();
        let z_inv = z_x_numerator_inv
            .iter()
            .zip(z_x_denominator.iter())
            .map(|(a, b)| *a * *b)
            .collect::<Vec<PrimeFieldElement>>();

        // Calculate D(x) = C(P(x)) / Z(x)
        let mut d_evaluations: Vec<PrimeFieldElement> =
            vec![PrimeFieldElement::new(0, &field); range];
        for i in 0..range {
            d_evaluations[i] = c_of_p_evaluations[i] * z_inv[i];
        }
        println!("Computed D(x)");

        // TODO: DEBUG: REMOVE!!!
        for i in 0..2000 {
            // println!("c(p(g2^{})) = {}", i, c_of_p_evaluations[i].value);
            // println!("d(g2^{}) = {}", i, d_evaluations[i].value);
            // println!("1/z(g2^{}) = {}", i, z_inv[i].value);
            if d_evaluations[i] == c_of_p_evaluations[i] * z_inv[i] {
                // println!("Values matched for i = {}", i);
            } else {
                println!("Values did **not** match for i = {}", i);
                return;
            }
        }

        let line = field.lagrange_interpolation_2((one, input), (*last_step, *output));
        let i_evaluations = field.evaluate_straight_line(line, &g2_domain);

        // TODO: Add polynomial support to the PrimeFieldElement struct, so we have
        // support for regular polynomials and not just, as now, extension field polynomials
        let pqr_mock = PolynomialQuotientRing::new(100, field.q);
        let z2_xfactor0 = PrimeFieldPolynomial {
            coefficients: vec![-1, 1],
            pqr: &pqr_mock,
        };
        let z2_xfactor1 = PrimeFieldPolynomial {
            coefficients: vec![(-*last_step).value, 1],
            pqr: &pqr_mock,
        };
        let z2_x = z2_xfactor0.mul(&z2_xfactor1);

        let mut z2_x_coefficients: Vec<PrimeFieldElement> = z2_x
            .coefficients
            .iter()
            .map(|x| PrimeFieldElement::new(*x, &field))
            .collect::<Vec<PrimeFieldElement>>();
        z2_x_coefficients.append(&mut vec![PrimeFieldElement::new(0, &field); range - 3]);
        let z2_x_evaluations: Vec<PrimeFieldElement> =
            fft::fast_polynomial_evaluate_prime_elements(&z2_x_coefficients, &g2);
        let z2_inv_x_evaluations: Vec<PrimeFieldElement> =
            field.batch_inversion_elements(z2_x_evaluations.clone());

        let mut b_evaluations: Vec<PrimeFieldElement> =
            vec![PrimeFieldElement::new(0, &field); range];
        for i in 0..range {
            b_evaluations[i] = (p_evaluations[i] - i_evaluations[i]) * z2_inv_x_evaluations[i];
        }
        println!("Computed B(x)");

        // Wrapping the primefield elements into a triplet will allow us to use our fast Merkle Tree
        // code that requires the number of nodes to be a power of two
        let polynomial_evaluations: Vec<(PrimeFieldElement, PrimeFieldElement, PrimeFieldElement)> =
            p_evaluations
                .iter()
                .zip(d_evaluations.iter())
                .zip(b_evaluations.iter())
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
            if p == p_evaluations[i] {
                // println!("Values matched for i = {}", i);
            } else {
                panic!("Values did **not** match for p i = {}", i);
            }
            if d == d_evaluations[i] {
                // println!("Values matched for i = {}", i);
            } else {
                panic!("Values did **not** match for d i = {}", i);
            }
            if b == b_evaluations[i] {
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
            if c_of_p_evaluations[i * expansion_factor].value != 0 {
                panic!(
                    "C(P(x)) != 0 for x = {} => i = {}. Got C(P(x)) = {}",
                    g2_domain[i * expansion_factor],
                    i * expansion_factor,
                    c_of_p_evaluations[i * expansion_factor].value
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
        let mut powers = vec![PrimeFieldElement::new(0, &field); range];
        powers[0] = PrimeFieldElement::new(1, &field);
        for i in 1..range {
            // g2^i = g2^(i - 1) * g2^steps => x[i] = x[i - 1] * g2^steps
            powers[i] = powers[i - 1] * g2_pow_steps;
        }

        let mut l_evaluations = vec![PrimeFieldElement::new(0, &field); range];
        for i in 1..range {
            l_evaluations[i] = d_evaluations[i]
                + ks[0] * p_evaluations[i]
                + ks[1] * p_evaluations[i] * powers[i]
                + ks[2] * b_evaluations[i]
                + ks[3] * powers[i] * b_evaluations[i];
        }

        let l_mtree = MerkleTreeVector::from_vec(&l_evaluations);
        println!("Computed linear combination of low-degree polynomials");

        // Get pseudo-random indices from `l_mtree.get_root()`.
        let index_preimages = get_n_hash_rounds(&l_mtree.get_root(), security_checks as u32);
        let indices: Vec<usize> = index_preimages
            .iter()
            .map(|x| get_index_from_bytes_exclude_multiples(x, range, expansion_factor))
            .collect::<Vec<usize>>();
        let p_d_b_proofs = polynomials_merkle_tree.get_multi_proof(&indices);
        let l_proofs = l_mtree.get_multi_proof(&indices);
        // TODO: REMOVE this when low_degree_test is changed to use PrimeFieldElements instead
        // of i128
        let mut p_evaluations_i128 = p_evaluations.iter().map(|x| x.value).collect::<Vec<i128>>();
        let mut b_evaluations_i128 = b_evaluations.iter().map(|x| x.value).collect::<Vec<i128>>();
        let mut d_evaluations_i128 = d_evaluations.iter().map(|x| x.value).collect::<Vec<i128>>();
        let mut l_evaluations_i128 = l_evaluations.iter().map(|x| x.value).collect::<Vec<i128>>();
        let mut output = vec![];
        println!(
            "Length of l_evaluations_i128 = {}",
            l_evaluations_i128.len()
        );
        let mut low_degree_proof: low_degree_test::LowDegreeProof<i128> = low_degree_test::prover(
            &p_evaluations_i128,
            field.q,
            (steps - 1) as u32,
            security_checks,
            &mut output,
            g2.value,
        );
        let mut verify: Result<(), low_degree_test::ValidationError> =
            low_degree_test::verify(low_degree_proof, field.q);
        if !verify.is_ok() {
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
        if !verify.is_ok() {
            println!("B failed low-degree test");
        }
        // assert!(verify.is_ok());
        output = vec![];
        low_degree_proof = low_degree_test::prover(
            &d_evaluations_i128,
            field.q,
            (2 * steps - 1) as u32,
            security_checks,
            &mut output,
            g2.value,
        );
        verify = low_degree_test::verify(low_degree_proof, field.q);
        if !verify.is_ok() {
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
        if !verify.is_ok() {
            println!("L failed low-degree test");
        }

        // Use these to index into m and l.
        // Then generate a proof that l_evaluations is of low degree (steps * 2)
    }
}
