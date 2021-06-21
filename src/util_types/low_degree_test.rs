use crate::shared_math::other::log_2_ceil;
use crate::shared_math::prime_field_element::{PrimeField, PrimeFieldElement};
use crate::shared_math::prime_field_polynomial::PrimeFieldPolynomial;
use crate::util_types::merkle_tree_vector::{MerkleTreeVector, Node};
use crate::utils::{get_index_from_bytes, get_n_hash_rounds};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::convert::TryInto;
use std::error::Error;
use std::fmt;
use std::result::Result;

#[derive(PartialEq, Eq, Debug)]
pub enum ValidationError {
    BadMerkleProof,
    BadSizedProof,
    NotColinear,
    LastIterationNotConstant,
}

#[derive(PartialEq, Debug, Serialize, Clone)]
pub struct LowDegreeProof<T>
where
    // DeserializeOwned is used here to indicate that no data is borrowed
    // in the deserialization process.
    T: DeserializeOwned + Serialize,
{
    ab_proofs: Vec<Vec<Vec<Option<Node<i128>>>>>,
    challenge_hash_preimages: Vec<Vec<u8>>,
    codeword_size: u32,
    c_proofs: Vec<Vec<Vec<Option<Node<i128>>>>>,
    index_picker_preimage: Vec<u8>,
    max_degree: u32,
    merkle_roots: Vec<[u8; 32]>,
    primitive_root_of_unity: T,
    rounds_count: u8,
    s: u32,
}

#[derive(Debug)]
struct MyError(String);
impl Error for MyError {}

impl fmt::Display for MyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Deserialization error for LowDegreeProof: {}", self.0)
    }
}

impl<U: DeserializeOwned + Serialize> LowDegreeProof<U> {
    pub fn from_serialization<T: DeserializeOwned + Serialize>(
        serialization: Vec<u8>,
        start_index: usize,
    ) -> Result<LowDegreeProof<U>, Box<dyn Error>> {
        let mut index = start_index;
        // let slice = serialization.clone().as_slice();
        let codeword_size: u32 = bincode::deserialize(&serialization[index..index + 4])?;
        index += 4;
        let max_degree: u32 = bincode::deserialize(&serialization[index..index + 4])?;
        index += 4;
        let s: u32 = bincode::deserialize(&serialization[index..index + 4])?;
        index += 4;
        let size_of_u = std::mem::size_of::<U>();
        let primitive_root_of_unity: U =
            bincode::deserialize(&serialization[index..index + size_of_u])?;
        index += size_of_u;
        let rounds_count = log_2_ceil(max_degree as u64 + 1) as u8;
        let rounds_count_usize = rounds_count as usize;
        let challenge_hash_preimages: Vec<Vec<u8>> = (0..rounds_count_usize)
            .map(|i| serialization[0..((i + 1) * 32 + index)].to_vec())
            .collect();
        let index_picker_preimage =
            serialization[0..((rounds_count_usize + 1) * 32 + index)].to_vec();
        let mut merkle_roots: Vec<[u8; 32]> = Vec::with_capacity(rounds_count_usize + 1);
        for _ in 0usize..(rounds_count_usize + 1) {
            let root: [u8; 32] = serialization[index..index + 32].try_into()?;
            index += 32;
            merkle_roots.push(root);
        }

        let mut c_proofs: Vec<Vec<Vec<Option<Node<i128>>>>> =
            Vec::with_capacity(rounds_count_usize);
        let mut ab_proofs: Vec<Vec<Vec<Option<Node<i128>>>>> =
            Vec::with_capacity(rounds_count_usize);
        for _ in 0..rounds_count {
            let mut proof_size: u16 = bincode::deserialize(&serialization[index..index + 2])?;
            index += 2;
            let c_proof: Vec<Vec<Option<Node<i128>>>> =
                bincode::deserialize_from(&serialization[index..index + proof_size as usize])?;
            index += proof_size as usize;
            c_proofs.push(c_proof);
            proof_size = bincode::deserialize(&serialization[index..index + 2])?;
            index += 2;
            let ab_proof: Vec<Vec<Option<Node<i128>>>> =
                bincode::deserialize_from(&serialization[index..index + proof_size as usize])?;
            index += proof_size as usize;
            ab_proofs.push(ab_proof);
        }
        Ok(LowDegreeProof::<U> {
            ab_proofs,
            challenge_hash_preimages,
            codeword_size,
            c_proofs,
            index_picker_preimage,
            max_degree,
            merkle_roots,
            primitive_root_of_unity,
            rounds_count,
            s,
        })
    }
}

pub fn verify(proof: LowDegreeProof<i128>, modulus: i128) -> Result<(), ValidationError> {
    let rounds_count: usize = log_2_ceil(proof.max_degree as u64 + 1) as usize;
    if rounds_count != proof.ab_proofs.len()
        || rounds_count != proof.c_proofs.len()
        || rounds_count != proof.challenge_hash_preimages.len()
        || rounds_count + 1 != proof.merkle_roots.len()
    {
        return Err(ValidationError::BadSizedProof);
    }

    let field = PrimeField::new(modulus);
    let challenge_hashes: Vec<[u8; 32]> = proof
        .challenge_hash_preimages
        .iter()
        .map(|bs| *blake3::hash(bs.as_slice()).as_bytes())
        .collect();
    let challenges: Vec<i128> = challenge_hashes
        .iter()
        .map(|x| PrimeFieldElement::from_bytes_raw(&modulus, &x[0..16]))
        .collect();

    let mut number_of_leaves = proof.codeword_size as usize;
    let mut c_values: Vec<i128> = vec![];
    let mut primitive_root_of_unity = proof.primitive_root_of_unity;
    for (i, challenge) in challenges.iter().enumerate() {
        let mut hash_preimage: Vec<u8> = proof.index_picker_preimage.clone();
        hash_preimage.push(i as u8);
        let hashes = get_n_hash_rounds(hash_preimage.as_slice(), proof.s);
        let mut c_indices: Vec<usize> = vec![];
        let mut ab_indices: Vec<usize> = vec![];
        for hash in hashes.iter() {
            let c_index = get_index_from_bytes(&hash[0..16], number_of_leaves / 2);
            c_indices.push(c_index);
            let a_index = c_index;
            ab_indices.push(a_index);
            let b_index = c_index + number_of_leaves / 2;
            ab_indices.push(b_index);
        }
        number_of_leaves /= 2;
        c_values = proof.c_proofs[i]
            .iter()
            .map(|x| x[0].as_ref().unwrap().value.unwrap())
            .collect::<Vec<i128>>();

        let valid_cs = MerkleTreeVector::verify_multi_proof(
            proof.merkle_roots[i + 1],
            &c_indices,
            &proof.c_proofs[i],
        );
        let valid_abs = MerkleTreeVector::verify_multi_proof(
            proof.merkle_roots[i],
            &ab_indices,
            &proof.ab_proofs[i],
        );
        if !valid_cs || !valid_abs {
            println!(
                "Found invalidity of indices on iteration {}: y = {}, s = {}",
                i, valid_cs, valid_abs
            );
            print!("Invalid proofs:");
            if !valid_abs {
                println!("{:?}", &proof.c_proofs[i]);
            }
            if !valid_cs {
                println!("{:?}", &proof.ab_proofs[i]);
            }
            return Err(ValidationError::BadMerkleProof);
        }

        let root = PrimeFieldElement::new(primitive_root_of_unity, &field);
        for j in 0..proof.s as usize {
            let a_index = ab_indices[2 * j] as i128;
            let a_x = root.mod_pow_raw(a_index);
            let a_y = proof.ab_proofs[i][2 * j][0]
                .as_ref()
                .unwrap()
                .value
                .unwrap();
            let b_index = ab_indices[2 * j + 1] as i128;
            let b_x = root.mod_pow_raw(b_index);
            let b_y = proof.ab_proofs[i][2 * j + 1][0]
                .as_ref()
                .unwrap()
                .value
                .unwrap();
            let c_y = proof.c_proofs[i][j][0].as_ref().unwrap().value.unwrap();
            if !PrimeFieldPolynomial::are_colinear_raw(
                &[(a_x, a_y), (b_x, b_y), (*challenge, c_y)],
                modulus,
            ) {
                println!(
                    "{{({},{}),({},{}),({},{})}} are not colinear",
                    a_x, a_y, b_x, b_y, challenge, c_y
                );
                println!("Failed to verify colinearity!");
                return Err(ValidationError::NotColinear);
            } else {
                println!(
                    "({}, {}), ({}, {}), ({}, {}) are colinear",
                    a_x, a_y, b_x, b_y, challenge, c_y
                );
            }
        }

        primitive_root_of_unity = primitive_root_of_unity * primitive_root_of_unity % modulus;
    }

    // Base case: Verify that the last merkle tree is a constant function
    // Verify only the c indicies
    if c_values.is_empty() || !c_values.iter().all(|&x| c_values[0] == x) {
        println!("Last y values were not constant. Got: {:?}", c_values);
        return Err(ValidationError::LastIterationNotConstant);
    }

    Ok(())
}

fn fri_prover_iteration(
    codeword: &[i128],
    challenge: &i128,
    modulus: &i128,
    inv_two: &i128,
    primitive_root_of_unity: &i128,
) -> Vec<i128> {
    let mut new_codeword: Vec<i128> = vec![0i128; codeword.len() / 2];

    let mut x = 1i128;
    for i in 0..new_codeword.len() {
        let (_, x_inv, _) = PrimeFieldElement::eea(x, *modulus);
        // If codeword is the evaluation of a polynomial of degree N,
        // this is an evaluation of a polynomial of degree N/2
        new_codeword[i] = (((1 + challenge * x_inv) * codeword[i]
            + (1 - challenge * x_inv) * codeword[i + codeword.len() / 2])
            * *inv_two
            % *modulus
            + *modulus)
            % *modulus;
        x = x * *primitive_root_of_unity % modulus;
    }
    new_codeword
}

// TODO: We want this implemented for prime field elements, and preferably for
// any finite field/extension field.
// Prove that codeword elements come from the evaluation of a polynomial of
// `degree < codeword.len() / expansion_factor`
pub fn prover(
    codeword: &[i128],
    modulus: i128,
    max_degree: u32,
    s: usize,
    output: &mut Vec<u8>,
    primitive_root_of_unity: i128,
) -> LowDegreeProof<i128> {
    let rounds_count = log_2_ceil(max_degree as u64 + 1) as usize;
    output.append(&mut bincode::serialize(&(codeword.len() as u32)).unwrap());
    output.append(&mut bincode::serialize(&(max_degree as u32)).unwrap());
    output.append(&mut bincode::serialize(&(s as u32)).unwrap());
    output.append(&mut bincode::serialize(&(primitive_root_of_unity)).unwrap());
    let mut mt = MerkleTreeVector::from_vec(codeword);
    let mut mts: Vec<MerkleTreeVector<i128>> = vec![mt];

    // Arrays for return values
    let mut c_proofs: Vec<Vec<Vec<Option<Node<i128>>>>> = vec![];
    let mut ab_proofs: Vec<Vec<Vec<Option<Node<i128>>>>> = vec![];

    output.append(&mut mts[0].get_root().to_vec());
    let mut mut_codeword: Vec<i128> = codeword.to_vec();

    // commit phase
    let (_, _, inv2_temp) = PrimeFieldElement::eea(modulus, 2);
    let inv2 = (inv2_temp + modulus) % modulus;
    let mut primitive_root_of_unity_temp = primitive_root_of_unity;
    let mut challenge_hash_preimages: Vec<Vec<u8>> = vec![];
    for _ in 0..rounds_count {
        // get challenge
        challenge_hash_preimages.push(output.clone());
        let hash = *blake3::hash(output.as_slice()).as_bytes();
        let challenge: i128 = PrimeFieldElement::from_bytes_raw(&modulus, &hash[0..16]);

        // run fri iteration reducing the degree of the polynomial by one half.
        // This is achieved by realizing that
        // P(x) + P(-x) = 2*P_e(x^2) and P(x) - P(-x) = 2*P_o(x^2) where P_e, P_o both
        // have half the degree of P.
        mut_codeword = fri_prover_iteration(
            &mut_codeword.clone(),
            &challenge,
            &modulus,
            &inv2,
            &primitive_root_of_unity_temp,
        );

        // Construct Merkle Tree from the new codeword of degree `max_degree / 2`
        mt = MerkleTreeVector::from_vec(&mut_codeword);

        // append root to proof
        output.append(&mut mt.get_root().to_vec());

        // collect into memory
        mts.push(mt);

        // num_rounds += 1;
        primitive_root_of_unity_temp =
            primitive_root_of_unity_temp * primitive_root_of_unity_temp % modulus;
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
    // let index_picker_preimage =
    // serialization[0..((rounds_count_usize + 1) * 32 + index)].to_vec();
    let index_picker_preimage = output.clone();
    primitive_root_of_unity_temp = primitive_root_of_unity;
    for i in 0usize..rounds_count {
        let number_of_leaves = mts[i].get_number_of_leafs();
        let mut c_indices: Vec<usize> = vec![];
        let mut ab_indices: Vec<usize> = vec![];

        // it's unrealistic that the number of rounds exceed 256 but this should wrap around if it does
        let mut hash_preimage: Vec<u8> = index_picker_preimage.clone();
        hash_preimage.push(i as u8);

        let hashes = get_n_hash_rounds(hash_preimage.as_slice(), s as u32);
        for hash in hashes.iter() {
            let c_index = get_index_from_bytes(&hash[0..16], number_of_leaves / 2);
            c_indices.push(c_index);
            let s0_index = c_index;
            ab_indices.push(s0_index);
            let s1_index = c_index + number_of_leaves / 2;
            ab_indices.push(s1_index);
        }

        let authentication_paths_c: Vec<Vec<Option<Node<i128>>>> =
            mts[i + 1].get_multi_proof(&c_indices);
        let authentication_paths_ab: Vec<Vec<Option<Node<i128>>>> =
            mts[i].get_multi_proof(&ab_indices);

        // serialize proofs and store in output
        let mut c_paths_encoded = bincode::serialize(&authentication_paths_c.clone()).unwrap();
        output.append(&mut bincode::serialize(&(c_paths_encoded.len() as u16)).unwrap());
        output.append(&mut c_paths_encoded);

        let mut ab_paths_encoded = bincode::serialize(&authentication_paths_ab.clone()).unwrap();
        output.append(&mut bincode::serialize(&(ab_paths_encoded.len() as u16)).unwrap());
        output.append(&mut ab_paths_encoded);

        primitive_root_of_unity_temp =
            primitive_root_of_unity_temp * primitive_root_of_unity_temp % modulus;

        // Accumulate values to be returned
        c_proofs.push(authentication_paths_c);
        ab_proofs.push(authentication_paths_ab);
    }

    LowDegreeProof::<i128> {
        rounds_count: rounds_count as u8,
        challenge_hash_preimages,
        c_proofs,
        ab_proofs,
        index_picker_preimage,
        s: s as u32,
        merkle_roots: mts.iter().map(|x| x.get_root()).collect::<Vec<[u8; 32]>>(),
        codeword_size: codeword.len() as u32,
        primitive_root_of_unity,
        max_degree,
    }
}

#[cfg(test)]
mod test_low_degree_proof {
    use super::*;
    use crate::fft::fast_polynomial_evaluate;
    use crate::shared_math::prime_field_element::PrimeField;
    use crate::utils::generate_random_numbers;

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
        let y_values = power_series;
        let max_degree = 1;
        let s = 5; // The security factor
        let mut proof: LowDegreeProof<i128> = prover(
            &y_values,
            field.q,
            max_degree,
            s,
            &mut output,
            primitive_root_of_unity,
        );
        assert_eq!(1, proof.max_degree);
        assert_eq!(4, proof.codeword_size);
        assert_eq!(10, proof.primitive_root_of_unity);
        assert_eq!(1, proof.rounds_count);
        assert_eq!(5, proof.s);
        assert_eq!(1, proof.ab_proofs.len());
        assert_eq!(1, proof.c_proofs.len());
        assert_eq!(2, proof.merkle_roots.len());

        let mut deserialized_proof: LowDegreeProof<i128> =
            LowDegreeProof::from_serialization::<i128>(output.clone(), 0).unwrap();
        assert_eq!(1, deserialized_proof.max_degree);
        assert_eq!(4, deserialized_proof.codeword_size);
        assert_eq!(10, deserialized_proof.primitive_root_of_unity);
        assert_eq!(1, deserialized_proof.rounds_count);
        assert_eq!(5, deserialized_proof.s);
        assert_eq!(1, deserialized_proof.ab_proofs.len());
        assert_eq!(1, deserialized_proof.c_proofs.len());
        assert_eq!(2, deserialized_proof.merkle_roots.len());
        assert_eq!(proof.ab_proofs, deserialized_proof.ab_proofs);
        assert_eq!(proof.c_proofs, deserialized_proof.c_proofs);
        assert_eq!(
            proof.index_picker_preimage,
            deserialized_proof.index_picker_preimage
        );
        assert_eq!(Ok(()), verify(proof, field.q));

        // Change one of the values in a leaf in the committed Merkle tree, and verify that the Merkle proof fails
        output = vec![];
        proof = prover(
            &y_values,
            field.q,
            max_degree,
            s,
            &mut output,
            primitive_root_of_unity,
        );
        let mut new_value = proof.ab_proofs[0][1][0].clone().unwrap();
        new_value.value = Some(237);
        proof.ab_proofs[0][1][0] = Some(new_value);
        assert_eq!(Err(ValidationError::BadMerkleProof), verify(proof, field.q));

        // Verify that the proof still works if the output vector is non-empty
        output = vec![145, 96];
        proof = prover(
            &y_values,
            field.q,
            max_degree,
            s,
            &mut output,
            primitive_root_of_unity,
        );
        deserialized_proof = LowDegreeProof::from_serialization::<i128>(output, 2).unwrap();
        assert_eq!(deserialized_proof, proof);
        assert_eq!(Ok(()), verify(deserialized_proof, field.q));
        assert_eq!(Ok(()), verify(proof, field.q));
    }

    #[test]
    fn generate_proof_parabola() {
        let mut ret: Option<(PrimeField, i128)> = None;
        PrimeField::get_field_with_primitive_root_of_unity(16, 100, &mut ret);
        let (field, primitive_root_of_unity) = ret.clone().unwrap();
        let domain = field.get_power_series(primitive_root_of_unity);
        // coefficients: vec![6, 2, 5] => P(x) = 5x^2 + 2x + 6
        let mut y_values = domain
            .iter()
            .map(|&x| ((6 + x * (2 + 5 * x)) % field.q + field.q) % field.q)
            .collect::<Vec<i128>>();

        let max_degree = 2;
        let s = 6;
        let mut output = vec![124, 62, 98, 10, 207];
        let mut proof: LowDegreeProof<i128> = prover(
            &y_values,
            field.q,
            max_degree,
            s,
            &mut output,
            primitive_root_of_unity,
        );
        assert_eq!(
            proof,
            LowDegreeProof::from_serialization::<i128>(output.clone(), 5).unwrap()
        );
        assert_eq!(Ok(()), verify(proof.clone(), field.q));

        // Change a single y value such that it no longer corresponds to a polynomil
        // a verify that the test fails
        output = vec![];
        let original_y_values = y_values.clone();
        y_values[3] = 100;
        y_values[4] = 100;
        proof = prover(
            &y_values,
            field.q,
            max_degree,
            s,
            &mut output,
            primitive_root_of_unity,
        );
        assert_eq!(
            Err(ValidationError::LastIterationNotConstant),
            verify(proof.clone(), field.q)
        );

        // make a proof with a too low max_degree parameter and verify that it fails verification
        // with the expected output
        let wrong_max_degree = 1;
        output = vec![];
        let proof = prover(
            &original_y_values,
            field.q,
            wrong_max_degree,
            s,
            &mut output,
            primitive_root_of_unity,
        );
        assert_eq!(
            Err(ValidationError::LastIterationNotConstant),
            verify(proof.clone(), field.q)
        );
    }

    #[test]
    fn generate_proof_16_alt() {
        let mut ret: Option<(PrimeField, i128)> = None;
        // should return (field = mod 193; root = 64) for (n = 16, min_value = 113)
        PrimeField::get_field_with_primitive_root_of_unity(16, 113, &mut ret);
        assert_eq!(193i128, ret.clone().unwrap().0.q);
        let (field, primitive_root_of_unity) = ret.clone().unwrap();
        let domain = field.get_power_series(primitive_root_of_unity);
        assert_eq!(16, domain.len());
        assert_eq!(
            vec![1, 64, 43, 50, 112, 27, 184, 3, 192, 129, 150, 143, 81, 166, 9, 190],
            domain
        );
        let mut output = vec![];

        // corresponds to the polynomial P(x) = x
        // degree < codeword.len() / expansion_factor
        let max_degree = 1;
        let s = 2;
        let y_values = domain;
        let proof = prover(
            &y_values,
            field.q,
            max_degree,
            s,
            &mut output,
            primitive_root_of_unity,
        );
        assert_eq!(
            proof,
            LowDegreeProof::from_serialization::<i128>(output.clone(), 0).unwrap()
        );
        assert_eq!(Ok(()), verify(proof, field.q));
    }

    #[test]
    fn generate_proof_1024() {
        let mut ret: Option<(PrimeField, i128)> = None;
        let size = 2usize.pow(14);
        let max_degree = 1024;
        PrimeField::get_field_with_primitive_root_of_unity(size as i128, size as i128, &mut ret);
        let (field_temp, primitive_root_of_unity) = ret.clone().unwrap();
        let field: PrimeField = field_temp.clone();
        assert_eq!(65537i128, field.q);
        assert_eq!(81i128, primitive_root_of_unity);
        let mut coefficients = generate_random_numbers(max_degree, field.q);
        coefficients.extend_from_slice(&vec![0; size - max_degree]);
        let mut y_values =
            fast_polynomial_evaluate(coefficients.as_slice(), field.q, primitive_root_of_unity);

        let mut output = vec![1, 2];

        let s = 40;
        let mut proof = prover(
            &y_values,
            field.q,
            max_degree as u32,
            s,
            &mut output,
            primitive_root_of_unity,
        );
        assert_eq!(
            proof,
            LowDegreeProof::from_serialization::<i128>(output.clone(), 2).unwrap()
        );
        assert_eq!(Ok(()), verify(proof, field.q));

        // Change a single y value such that it no longer corresponds to a polynomial
        // and verify that the test fails
        output = vec![];
        y_values[3] = 100;
        proof = prover(
            &y_values,
            field.q,
            max_degree as u32,
            s,
            &mut output,
            primitive_root_of_unity,
        );
        assert_eq!(
            proof,
            LowDegreeProof::from_serialization::<i128>(output.clone(), 0).unwrap()
        );
        assert_eq!(
            Err(ValidationError::LastIterationNotConstant),
            verify(proof, field.q)
        );
    }
}
