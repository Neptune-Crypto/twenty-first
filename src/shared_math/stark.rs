use crate::shared_math::prime_field_element::PrimeFieldElement;

pub fn mimc_forward<'a>(
    input: &'a PrimeFieldElement,
    steps: u64,
    round_costants: &'a [PrimeFieldElement],
) -> Vec<PrimeFieldElement<'a>> {
    let mut trace: Vec<PrimeFieldElement<'a>> = Vec::with_capacity(steps as usize);
    let mut res: PrimeFieldElement<'a> = *input;
    trace.push(*input);
    for i in 0..steps as usize {
        res = res.mod_pow(3) + round_costants[i % round_costants.len()];
        trace.push(res);
    }

    trace
}

pub fn mimc_backward<'a>(
    input: &'a PrimeFieldElement,
    steps: u64,
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
    use crate::shared_math::polynomial_quotient_ring::PolynomialQuotientRing;
    use crate::shared_math::prime_field_element::PrimeField;
    use crate::shared_math::prime_field_polynomial::PrimeFieldPolynomial;
    use crate::util_types::merkle_tree_vector::{MerkleTreeVector, Node};
    use crate::utils;

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
        let steps = 8192i128;
        let expansion_factor = 8i128;
        let range: i128 = steps * expansion_factor;
        let mut ret: Option<(PrimeField, i128)> = None;
        PrimeField::get_field_with_primitive_root_of_unity(range, 2i128.pow(16), &mut ret);
        let (field, root) = ret.unwrap();
        assert_eq!(65537, field.q);
        assert_eq!(3, root);
        let g1_ret = field.get_primitive_root_of_unity(steps);
        let g1: PrimeFieldElement = g1_ret.0.unwrap();
        let g2: PrimeFieldElement = PrimeFieldElement::new(root, &field);
        assert_eq!(6561, g1.value);
        assert_eq!(g2.mod_pow(expansion_factor), g1);
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
        let trace: Vec<PrimeFieldElement> =
            mimc_forward(&input, steps as u64 - 1, &round_constants);
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
            ((expansion_factor - 1) * steps) as usize
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
        // for i in 0..20 {
        //     println!(
        //         "p({}) = {}",
        //         g1_domain[i].value,
        //         p_evaluations[i * expansion_factor as usize]
        //     );
        // }
        let skips2 = (steps as usize / round_constants.len()) as i128;

        println!("steps = {}", steps);
        println!("round_constants.len() = {}", round_constants.len());
        println!("skips2 = {}", skips2);
        println!("g1.mod_pow(skips2) = {}", g1.mod_pow(skips2));
        println!(
            "g1.mod_pow(skips2) ^ round_constants.len() = {}",
            g1.mod_pow(skips2).mod_pow(round_constants.len() as i128)
        );
        println!(
            "g1.mod_pow(skips2) ^ (round_constants.len() / 2) = {}",
            g1.mod_pow(skips2)
                .mod_pow(round_constants.len() as i128 / 2)
        );
        // Find a polynomial for the round constants in a domain over which they span
        // TODO: THE ERROR IS HERE SOMEWHERE!!
        let round_constants_polynomial =
            fft::fast_polynomial_interpolate_prime_elements(&round_constants, &g1.mod_pow(skips2));
        println!("round_constants.len() = {}", round_constants.len());
        println!(
            "round_constants_polynomial.len() = {}",
            round_constants_polynomial.len()
        );
        // let mut round_constants_polynomial_extended = round_constants_polynomial.clone();
        // round_constants_polynomial_extended.append(&mut vec![
        //     PrimeFieldElement::new(0, &field);
        //     range as usize - round_constants.len()
        // ]);
        // println!(
        //     "round_constants_polynomial_extended.len() = {}",
        //     round_constants_polynomial_extended.len()
        // );
        let mut round_constants_polynomial_extended: Vec<PrimeFieldElement> =
            Vec::with_capacity(range as usize);
        for i in 0..range {
            // if i < skips2 {
            //     round_constants_polynomial_extended
            //         .push(round_constants_polynomial[(i / skips2) as usize]);
            // } else {
            //     round_constants_polynomial_extended.push(PrimeFieldElement::new(0, &field));
            // }
            if i % skips2 == 0 {
                round_constants_polynomial_extended
                    .push(round_constants_polynomial[(i / skips2) as usize]);
            } else {
                round_constants_polynomial_extended.push(PrimeFieldElement::new(0, &field));
            }
        }

        println!(
            "Calling NTT with ω = {}, length = {}",
            &g2.mod_pow(skips2).value,
            round_constants_polynomial_extended.len()
        );
        let round_constants_extension =
            fft::fast_polynomial_evaluate_prime_elements(&round_constants_polynomial_extended, &g2);
        println!(
            "g2.mod_pow(skips2) ^ round_constants_extension.len() = {}",
            g2.mod_pow(skips2)
                .mod_pow(round_constants_extension.len() as i128)
        );
        println!(
            "g2.mod_pow(skips2) ^ (round_constants_extension.len() / 2) = {}",
            g2.mod_pow(skips2)
                .mod_pow(round_constants_extension.len() as i128 / 2)
        );
        println!("g2.mod_pow(skips2) = {}", g2.mod_pow(skips2));
        // for i in 0..10 {
        //     println!("round_constants[{}] = {}", i, round_constants[i].value);
        // }
        // for i in 0..10 {
        //     println!(
        //         "round_constants_polynomial[{}] = {}",
        //         i, round_constants_polynomial[i].value
        //     );
        // }
        println!("Low-degree extended round constants");
        // for i in 0..10 {
        //     println!(
        //         "round_constants_extension[{}] = {}",
        //         i, round_constants_extension[i].value
        //     );
        //     println!(
        //         "round_constants_extension[{}] = {}",
        //         i * expansion_factor as usize,
        //         round_constants_extension[i * expansion_factor as usize].value
        //     );
        // }

        // Evaluate the composed polynomial such that
        // C(P(x), P(g1*x), K(x)) = P(g1*x) - P(x)**3 - K(x)
        let mut c_of_p_evaluations: Vec<PrimeFieldElement> = Vec::with_capacity(range as usize);
        for i in 0..range {
            let evaluation = p_evaluations[((i + expansion_factor) % range) as usize]
                - p_evaluations[i as usize].mod_pow(3)
                - round_constants_extension[i as usize % round_constants_extension.len()];
            c_of_p_evaluations.push(evaluation);
        }
        println!("Computed C(P(x))");
        for i in 1000..1200 {
            println!("C(P(g2^{})) = {}", i, c_of_p_evaluations[i].value);
        }

        // Calculate Z(x) = prod_{i=i}^{N}(x - g1^i) = (x^N - 1) / (x - g2^N)
        // Calculate the inverse of Z(x) so we can divide with it by multiplying with `1/Z(x)`
        let g2_domain: Vec<PrimeFieldElement> = g2.get_generator_domain();
        let one = PrimeFieldElement::new(1, &field);
        let mut z_x_numerator: Vec<PrimeFieldElement> =
            vec![PrimeFieldElement::new(0, &field); range as usize];
        for i in 0..range as usize {
            z_x_numerator[i] = g2_domain[i * steps as usize % range as usize] - one;
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
            vec![PrimeFieldElement::new(0, &field); range as usize];
        for i in 0..range as usize {
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
        // let z_x_2_evaluations: Vec<PrimeFieldElement> =
        //     vec![PrimeFieldElement::new(0, &field); range as usize];

        let mut z2_x_coefficients: Vec<PrimeFieldElement> = z2_x
            .coefficients
            .iter()
            .map(|x| PrimeFieldElement::new(*x, &field))
            .collect::<Vec<PrimeFieldElement>>();
        z2_x_coefficients.append(&mut vec![
            PrimeFieldElement::new(0, &field);
            range as usize - 3
        ]);
        let z2_x_evaluations: Vec<PrimeFieldElement> =
            fft::fast_polynomial_evaluate_prime_elements(&z2_x_coefficients, &g2);
        let z2_inv_x_evaluations: Vec<PrimeFieldElement> =
            field.batch_inversion_elements(z2_x_evaluations.clone());

        // for i in 0..10 {
        //     println!("z2_x_evaluations(g2^{}) = {}", i, z2_x_evaluations[i].value);
        //     println!(
        //         "z2_inv_x_evaluations(g2^{}) = {}",
        //         i, z2_inv_x_evaluations[i].value
        //     );
        // }
        // (steps-1)*extension_factor
        // println!(
        //     "z2_x_evaluations(g2^{}) = {}",
        //     (steps - 1) * expansion_factor,
        //     z2_x_evaluations[((steps - 1) * expansion_factor) as usize]
        // );
        // println!(
        //     "z2_inv_x_evaluations(g2^{}) = {}",
        //     (steps - 1) * expansion_factor,
        //     z2_inv_x_evaluations[((steps - 1) * expansion_factor) as usize]
        // );

        let mut b_evaluations: Vec<PrimeFieldElement> =
            vec![PrimeFieldElement::new(0, &field); range as usize];
        for i in 0..range as usize {
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
        let mt: MerkleTreeVector<(PrimeFieldElement, PrimeFieldElement, PrimeFieldElement)> =
            MerkleTreeVector::from_vec(&polynomial_evaluations);
        println!("Computed merkle tree. Root: {:?}", mt.get_root());

        // Verify that we can extract values from the Merkle Tree
        // TODO: DEBUG: REMOVE!!!
        for i in 700..790 {
            let val: Vec<Node<(PrimeFieldElement, PrimeFieldElement, PrimeFieldElement)>> =
                mt.get_proof(i);
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
            if !MerkleTreeVector::verify_proof(mt.get_root(), i as u64, val) {
                panic!("Failed to verify Merkle Tree proof for i = {}", i);
            }
        }

        //for i in 0..range as usize {}
        // z_x_2.evaluate(x: &'d PrimeFieldElement)

        for i in 0..8000 as usize {
            // println!(
            //     "C(P({})) = {}",
            //     g1_domain[i].value,
            //     c_of_p_evaluations[i * expansion_factor as usize]
            // );
            if c_of_p_evaluations[i * expansion_factor as usize].value != 0 {
                panic!(
                    "C(P(x)) != 0 for x = {} => i = {}. Got C(P(x)) = {}",
                    g2_domain[i * expansion_factor as usize],
                    i * expansion_factor as usize,
                    c_of_p_evaluations[i * expansion_factor as usize].value
                );
            }
        }
    }
}
