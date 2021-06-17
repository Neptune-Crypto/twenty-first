use crate::shared_math::other::log_2;
use crate::shared_math::polynomial_quotient_ring::PolynomialQuotientRing;
use crate::shared_math::prime_field_element::{PrimeField, PrimeFieldElement};
use crate::shared_math::prime_field_polynomial::PrimeFieldPolynomial;
use crate::util_types::merkle_tree_vector::{MerkleTreeVector, Node};
use crate::utils::{get_index_from_bytes, get_n_hash_rounds};
use std::convert::TryInto;

pub fn verify(
    modulus: i128,
    s: usize,
    output: &[u8],
    codeword_size: usize,
    mut primitive_root_of_unity: i128,
) -> bool {
    // let rounds_count = output.append(&mut bincode::serialize(&(round_count as u16)).unwrap());
    let rounds_count_u16: u16 = bincode::deserialize(&output[0..2]).unwrap();
    let rounds_count: usize = rounds_count_u16 as usize;
    let field = PrimeField::new(modulus);
    let roots: Vec<[u8; 32]> = (0..rounds_count)
        .map(|i| output[2 + i * 32..(i + 1) * 32 + 2].try_into().unwrap())
        .collect();
    let challenge_hash_preimages: Vec<Vec<u8>> = (0..rounds_count)
        .map(|i| output[0..((i + 1) * 32 + 2)].to_vec())
        .collect();
    let challenge_hashes: Vec<[u8; 32]> = challenge_hash_preimages
        .iter()
        .map(|bs| *blake3::hash(bs.as_slice()).as_bytes())
        .collect();
    let challenges: Vec<i128> = challenge_hashes
        .iter()
        .map(|x| PrimeFieldElement::from_bytes_raw(&modulus, &x[0..16]))
        .collect();

    let partial_output = output[0..((rounds_count + 1) * 32 + 2)].to_vec();
    let mut number_of_leaves = codeword_size;
    let mut output_index: usize = (rounds_count + 1) * 32 + 2;
    for i in 0..rounds_count - 1 {
        let mut hash_preimage: Vec<u8> = partial_output.clone();
        hash_preimage.push(i as u8);
        let hashes = get_n_hash_rounds(hash_preimage.as_slice(), s);
        let mut y_indices: Vec<usize> = vec![];
        let mut s_indices: Vec<usize> = vec![];
        for hash in hashes.iter() {
            let y_index = get_index_from_bytes(&hash[0..16], number_of_leaves / 2);
            y_indices.push(y_index);
            let s0_index = y_index;
            s_indices.push(s0_index);
            let s1_index = y_index + number_of_leaves / 2;
            s_indices.push(s1_index);
        }
        number_of_leaves /= 2;

        let mut proof_size: u16 =
            bincode::deserialize(&output[output_index..output_index + 2]).unwrap();
        output_index += 2;
        let mut cursor = &output[output_index..output_index + proof_size as usize];
        let y_proofs: Vec<Vec<Option<Node<i128>>>> = bincode::deserialize_from(cursor).unwrap();
        output_index += proof_size as usize;
        proof_size = bincode::deserialize(&output[output_index..output_index + 2]).unwrap();
        output_index += 2;
        cursor = &output[output_index..output_index + proof_size as usize];
        let s_proofs: Vec<Vec<Option<Node<i128>>>> = bincode::deserialize_from(cursor).unwrap();
        output_index += proof_size as usize;

        let valid_ys = MerkleTreeVector::verify_multi_proof(roots[i + 1], &y_indices, &y_proofs);
        let valid_ss = MerkleTreeVector::verify_multi_proof(roots[i], &s_indices, &s_proofs);
        if !valid_ys || !valid_ss {
            println!(
                "Found invalidity of indices on iteration {}: y = {}, s = {}",
                i, valid_ys, valid_ss
            );
            print!("Invalid proofs:");
            if !valid_ss {
                println!("{:?}", y_proofs);
            }
            if !valid_ys {
                println!("{:?}", s_proofs);
            }
            return false;
        }

        let root = PrimeFieldElement::new(primitive_root_of_unity, &field);
        println!("primitive_root_of_unity = {}", primitive_root_of_unity);
        for i in 0..s {
            let s0_index = s_indices[2 * i] as i128;
            let s0 = root.mod_pow_raw(s0_index);
            let alpha0 = s_proofs[2 * i][0].as_ref().unwrap().value.unwrap();
            let s1_index = s_indices[2 * i + 1] as i128;
            let s1 = root.mod_pow_raw(s1_index);
            let alpha1 = s_proofs[2 * i + 1][0].as_ref().unwrap().value.unwrap();
            let y_index = y_indices[i] as i128;
            let y = root.mod_pow_raw(y_index * 2);
            let beta_temp = y_proofs[i][0].as_ref().unwrap().value.unwrap();
            let beta = beta_temp;
            // let beta = beta_temp * beta_temp % modulus;
            println!(
                "{{({},{}),({},{}),({},{})}}",
                s0, alpha0, s1, alpha1, y, beta
            ); // TODO: REMOVE
            if !PrimeFieldPolynomial::are_colinear_raw(
                &[(s0, alpha0), (s1, alpha1), (y, beta)],
                modulus,
            ) {
                println!("Failed to verify colinearity!");
                return false;
            } else {
                println!(
                    "({}, {}), ({}, {}), ({}, {}) are colinear",
                    s0, alpha0, s1, alpha1, y, beta
                );
            }
        }

        primitive_root_of_unity = primitive_root_of_unity * primitive_root_of_unity % modulus;
    }
    true
}

pub fn fri_prover_iteration(
    codeword: &[i128],
    challenge: &i128,
    modulus: &i128,
    inv_two: &i128,
) -> Vec<i128> {
    let mut new_codeword: Vec<i128> = vec![0i128; codeword.len() / 2];

    for i in 0..new_codeword.len() {
        // If codeword is the evaluation of a polynomial of degree N,
        // this is an evaluation of a polynomial of degree N/2
        new_codeword[i] = ((challenge + 1) * codeword[i]
            + (challenge - 1) * codeword[i + codeword.len() / 2])
            * *inv_two
            % *modulus;
    }
    println!("codeword: {:?}", codeword);
    println!("new_codeword: {:?}", new_codeword);
    new_codeword
}

// TODO: We want this implemented for prime field elements, and preferably for
// any finite field/extension field.
// Prove that codeword elements come from the evaluation of a polynomial of
// `degree < codeword.len() / rho`
pub fn prover(
    codeword: &[i128],
    modulus: i128,
    rho: usize,
    s: usize,
    output: &mut Vec<u8>,
    mut primitive_root_of_unity: i128, // TODO: REMOVE -- only used for debugging
) {
    let round_count = log_2((codeword.len() / rho) as u64);
    output.append(&mut bincode::serialize(&(round_count as u16)).unwrap());
    let mut mt = MerkleTreeVector::from_vec(codeword);
    output.append(&mut mt.get_root().to_vec());
    let mut mts: Vec<MerkleTreeVector<i128>> = vec![mt];
    let mut mut_codeword: Vec<i128> = codeword.to_vec().clone();

    // commit phase
    let (_, _, inv2_temp) = PrimeFieldElement::eea(modulus, 2);
    let inv2 = (inv2_temp + modulus) % modulus;
    let mut num_rounds = 0;
    while mut_codeword.len() > rho {
        // get challenge
        let hash = *blake3::hash(output.as_slice()).as_bytes();
        let challenge: i128 = PrimeFieldElement::from_bytes_raw(&modulus, &hash[0..16]);

        // run fri iteration
        mut_codeword = fri_prover_iteration(&mut_codeword.clone(), &challenge, &modulus, &inv2);

        // wrap into merkle tree
        mt = MerkleTreeVector::from_vec(&mut_codeword);

        // append root to proof
        output.append(&mut mt.get_root().to_vec());

        // collect into memory
        mts.push(mt.clone());

        num_rounds += 1;
    }

    // query phase
    // for all subsequent pairs of merkle trees:
    // - do s times:
    // -- sample random point y in L2
    // -- compute square roots s1 s2
    // -- query P1 in y -> beta
    // -- query P2 in s1 -> alpha1
    // -- query P2 in s2 -> alpha2
    // -- check collinearity (s0, alpha0), (s1, alpha1), (y, beta) <-- we don't care about thi right nw>
    let partial_output = output.clone();
    for i in 0usize..num_rounds - 1 {
        let number_of_leaves = mts[i].get_number_of_leafs();
        let mut y_indices: Vec<usize> = vec![];
        let mut s_indices: Vec<usize> = vec![];

        // it's unrealistic that the number of rounds exceed 256 but this should wrap around if it does
        let mut hash_preimage: Vec<u8> = partial_output.clone();
        hash_preimage.push(i as u8);

        let hashes = get_n_hash_rounds(hash_preimage.as_slice(), s);
        for hash in hashes.iter() {
            let y_index = get_index_from_bytes(&hash[0..16], number_of_leaves / 2);
            y_indices.push(y_index);
            let s0_index = y_index;
            s_indices.push(s0_index);
            let s1_index = y_index + number_of_leaves / 2;
            s_indices.push(s1_index);
        }

        let authentication_paths_y: Vec<Vec<Option<Node<i128>>>> =
            mts[i + 1].get_multi_proof(&y_indices);
        let authentication_paths_s: Vec<Vec<Option<Node<i128>>>> =
            mts[i].get_multi_proof(&s_indices);

        // Debug
        let field = PrimeField::new(modulus);
        let root = PrimeFieldElement::new(primitive_root_of_unity, &field);
        println!("number_of_leaves = {}", number_of_leaves);
        println!("y index = {}", y_indices[0]);
        println!("y x value = {}", root.mod_pow_raw(y_indices[0] as i128 * 2));
        println!(
            "y value revealed = {}",
            authentication_paths_y[0][0]
                .as_ref()
                .unwrap()
                .value
                .unwrap()
        );
        println!("s0 index = {}", s_indices[0]);
        println!("s0 x value = {}", root.mod_pow_raw(s_indices[0] as i128));
        println!(
            "s0 f(x) value revealed = {}",
            authentication_paths_s[0][0]
                .as_ref()
                .unwrap()
                .value
                .unwrap()
        );
        println!("s1 index = {}", s_indices[1]);
        println!("s1 x value = {}", root.mod_pow_raw(s_indices[1] as i128));
        println!(
            "s1 f(x) value revealed = {}",
            authentication_paths_s[1][0]
                .as_ref()
                .unwrap()
                .value
                .unwrap()
        );

        // serialize proofs and store in output
        let mut y_paths_encoded = bincode::serialize(&authentication_paths_y.clone()).unwrap();
        output.append(&mut bincode::serialize(&(y_paths_encoded.len() as u16)).unwrap());
        output.append(&mut y_paths_encoded);

        let mut s_paths_encoded = bincode::serialize(&authentication_paths_s.clone()).unwrap();
        output.append(&mut bincode::serialize(&(s_paths_encoded.len() as u16)).unwrap());
        output.append(&mut s_paths_encoded);

        primitive_root_of_unity = primitive_root_of_unity * primitive_root_of_unity % modulus;
        // TODO: REMOVE -- only used for debugging
    }
}

#[cfg(test)]
mod test_utils {
    use super::*;
    use crate::shared_math::prime_field_element::PrimeField;

    #[test]
    fn generate_proof_small() {
        let mut ret: Option<(PrimeField, i128)> = None;
        PrimeField::get_field_with_primitive_root_of_unity(4, 100, &mut ret);
        assert_eq!(101i128, ret.clone().unwrap().0.q);
        let (field, primitive_root_of_unity) = ret.clone().unwrap();
        let power_series = field.get_power_series(primitive_root_of_unity);
        assert_eq!(4, power_series.len());
        assert_eq!(vec![1i128, 10, 100, 91], power_series);
        let mut output = vec![];

        // corresponds to the polynomial P(x) = x
        // degree < codeword.len() / rho
        let y_values = power_series;
        let rho = 1;
        let s = 5;
        // prover(&y_values, field.q, rho, s, &mut output);
        prover(
            &y_values,
            field.q,
            rho,
            s,
            &mut output,
            primitive_root_of_unity,
        );
        println!("\n\n\n\n\n\n\n\n\n\n\n***************** PROOF DONE *****************");
        println!("***************** START VERIFY ***************** \n\n");
        assert!(verify(
            field.q,
            s,
            &output,
            y_values.len(),
            primitive_root_of_unity
        ));
    }

    #[test]
    fn generate_proof_parabola() {
        let mut ret: Option<(PrimeField, i128)> = None;
        PrimeField::get_field_with_primitive_root_of_unity(16, 100, &mut ret);
        let (field, primitive_root_of_unity) = ret.clone().unwrap();
        println!(
            "Field: q = {}, root of unity = {}",
            field.q, primitive_root_of_unity
        );
        let power_series = field.get_power_series(primitive_root_of_unity);
        // coefficients: vec![6, 2, 5] => P(x) = 5x^2 + 2x + 6
        let y_values = power_series
            .iter()
            .map(|&x| ((5 + x * (2 + 5 * x)) % field.q + field.q) % field.q)
            .collect::<Vec<i128>>();

        println!("power_series = {:?}", power_series);
        println!("y_values = {:?}", y_values);
        let rho = 4;
        let s = 1;
        let mut output = vec![];
        prover(
            &y_values,
            field.q,
            rho,
            s,
            &mut output,
            primitive_root_of_unity,
        );
        assert!(verify(
            field.q,
            s,
            &output,
            y_values.len(),
            primitive_root_of_unity
        ));
    }

    #[test]
    fn generate_proof_16_alt() {
        let mut ret: Option<(PrimeField, i128)> = None;
        // should return (field = mod 193; root = 64) for (n = 16, min_value = 113)
        PrimeField::get_field_with_primitive_root_of_unity(16, 113, &mut ret);
        assert_eq!(193i128, ret.clone().unwrap().0.q);
        let (field, primitive_root_of_unity) = ret.clone().unwrap();
        println!("primitive_root_of_unity = {}", primitive_root_of_unity);
        let power_series = field.get_power_series(primitive_root_of_unity);
        assert_eq!(16, power_series.len());
        assert_eq!(
            vec![1, 64, 43, 50, 112, 27, 184, 3, 192, 129, 150, 143, 81, 166, 9, 190],
            power_series
        );
        let mut output = vec![];

        // corresponds to the polynomial P(x) = x
        // degree < codeword.len() / rho
        let rho = 2;
        let s = 2;
        let y_values = power_series;
        prover(
            &y_values,
            field.q,
            rho,
            s,
            &mut output,
            primitive_root_of_unity,
        );
        assert!(verify(
            field.q,
            s,
            &output,
            y_values.len(),
            primitive_root_of_unity
        ));
    }
}
