use crate::shared_math::other::log_2_ceil;
use crate::shared_math::prime_field_element::{PrimeField, PrimeFieldElement};
use crate::shared_math::prime_field_polynomial::PrimeFieldPolynomial;
use crate::util_types::merkle_tree_vector::{MerkleTreeVector, Node};
use crate::utils::{get_index_from_bytes, get_n_hash_rounds};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::convert::TryFrom;
use std::convert::TryInto;
use std::error::Error;
use std::fmt;
use std::result::Result;

#[derive(PartialEq, Eq, Debug)]
pub enum ValidationError {
    BadMerkleProof,
    NotColinear,
    LastIterationNotConstant,
}

#[derive(PartialEq, Debug, Serialize)]
pub struct LowDegreeProof<T>
where
    // DeserializeOwned is used here to indicate that no data is borrowed
    // in the deserialization process.
    T: DeserializeOwned + Serialize,
{
    ab_proofs: Vec<Vec<Vec<Option<Node<i128>>>>>,
    codeword_size: u32,
    c_proofs: Vec<Vec<Vec<Option<Node<i128>>>>>,
    max_degree: u32,
    merkle_roots: Vec<[u8; 32]>,
    // phantomData is included in this struct since the lifetime parameter
    // would otherwise be unused, and the deserialization trait on T requires
    // a lifetime parameter.
    primitive_root_of_unity: T,
    rounds_count: u8,
    s: u32,
    serialization: Vec<u8>,
}

#[derive(Debug)]
struct MyError(String);
impl Error for MyError {}

impl fmt::Display for MyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Deserialization error for LowDegreeProof: {}", self.0)
    }
}

impl<T> TryFrom<Vec<u8>> for LowDegreeProof<T>
where
    T: DeserializeOwned + Serialize,
{
    type Error = Box<dyn Error>;

    fn try_from(serialization: Vec<u8>) -> Result<Self, Box<dyn Error>> {
        let mut index = 0;
        // let slice = serialization.clone().as_slice();
        let codeword_size: u32 = bincode::deserialize(&serialization[index..index + 4])?;
        index += 4;
        let max_degree: u32 = bincode::deserialize(&serialization[index..index + 4])?;
        index += 4;
        let s: u32 = bincode::deserialize(&serialization[index..index + 4])?;
        index += 4;
        let size_of_t = std::mem::size_of::<T>();
        let primitive_root_of_unity: T =
            bincode::deserialize(&serialization[index..index + size_of_t])?;
        index += size_of_t;
        let rounds_count = log_2_ceil(max_degree as u64 + 1) as u8;
        let rounds_count_usize = rounds_count as usize;
        let mut merkle_roots: Vec<[u8; 32]> = Vec::with_capacity(rounds_count_usize + 1);
        for _ in 0usize..(rounds_count_usize + 1) {
            let root: [u8; 32] = serialization[index..index + 32].try_into()?;
            index += 32;
            merkle_roots.push(root);
        }

        // let mut proof_size: u16 =
        //     bincode::deserialize(&serialization[output_index..output_index + 2]).unwrap();
        // Get the proofs
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

        Ok(Self {
            ab_proofs,
            codeword_size,
            c_proofs,
            max_degree,
            merkle_roots,
            primitive_root_of_unity,
            rounds_count,
            s,
            serialization: serialization.clone(),
        })
    }
}

pub fn verify(modulus: i128, output: &[u8]) -> Result<(), ValidationError> {
    // let rounds_count_u8: u8 = bincode::deserialize(&output[0..1]).unwrap();
    let codeword_size: u32 = bincode::deserialize(&output[0..4]).unwrap();
    let max_degree: u32 = bincode::deserialize(&output[4..8]).unwrap();
    let s: u32 = bincode::deserialize(&output[8..12]).unwrap();
    let mut primitive_root_of_unity: i128 = bincode::deserialize(&output[12..28]).unwrap();
    let rounds_count: usize = log_2_ceil(max_degree as u64 + 1) as usize;
    // let rounds_count: usize = rounds_count_u8 as usize;
    println!("codeword_size = {}", codeword_size);
    println!("max_degree = {}", max_degree);
    println!("s = {}", s);
    println!("primitive_root_of_unity = {}", primitive_root_of_unity);
    println!("rounds_count = {}", rounds_count);
    let field = PrimeField::new(modulus);
    let roots: Vec<[u8; 32]> = (0..=rounds_count)
        .map(|i| output[28 + i * 32..(i + 1) * 32 + 28].try_into().unwrap())
        .collect();
    println!("Last root: {:?}", roots.last().unwrap()); // TODO: REMOVE
    let challenge_hash_preimages: Vec<Vec<u8>> = (0..rounds_count)
        .map(|i| output[0..((i + 1) * 32 + 28)].to_vec())
        .collect();
    let challenge_hashes: Vec<[u8; 32]> = challenge_hash_preimages
        .iter()
        .map(|bs| *blake3::hash(bs.as_slice()).as_bytes())
        .collect();
    let challenges: Vec<i128> = challenge_hashes
        .iter()
        .map(|x| PrimeFieldElement::from_bytes_raw(&modulus, &x[0..16]))
        .collect();
    println!("challenges = {:?}", challenges);

    let partial_output = output[0..((rounds_count + 1) * 32 + 28)].to_vec();
    let mut number_of_leaves = codeword_size as usize;
    let mut output_index: usize = (rounds_count + 1) * 32 + 28;
    let mut c_values: Vec<i128> = vec![];
    for i in 0..rounds_count {
        println!("i = {}", i);
        let mut hash_preimage: Vec<u8> = partial_output.clone();
        hash_preimage.push(i as u8);
        let hashes = get_n_hash_rounds(hash_preimage.as_slice(), s);
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
        // println!("c_indices = {:?}", c_indices); // TODO: REMOVE

        let mut proof_size: u16 =
            bincode::deserialize(&output[output_index..output_index + 2]).unwrap();
        output_index += 2;
        let mut cursor = &output[output_index..output_index + proof_size as usize];
        println!("c_proofs is located at address: {}", output_index); // TODO: REMOVE
        println!("cursor is: {:?}", cursor);
        let c_proofs: Vec<Vec<Option<Node<i128>>>> = bincode::deserialize_from(cursor).unwrap();
        c_values = c_proofs
            .iter()
            .map(|x| x[0].as_ref().unwrap().value.unwrap())
            .collect::<Vec<i128>>();
        println!("c_values = {:?}", c_values); // TODO: REMOVE
        output_index += proof_size as usize;
        proof_size = bincode::deserialize(&output[output_index..output_index + 2]).unwrap();
        output_index += 2;
        cursor = &output[output_index..output_index + proof_size as usize];
        let ab_proofs: Vec<Vec<Option<Node<i128>>>> = bincode::deserialize_from(cursor).unwrap();
        output_index += proof_size as usize;

        let valid_cs = MerkleTreeVector::verify_multi_proof(roots[i + 1], &c_indices, &c_proofs);
        let valid_abs = MerkleTreeVector::verify_multi_proof(roots[i], &ab_indices, &ab_proofs);
        if !valid_cs || !valid_abs {
            println!(
                "Found invalidity of indices on iteration {}: y = {}, s = {}",
                i, valid_cs, valid_abs
            );
            print!("Invalid proofs:");
            if !valid_abs {
                println!("{:?}", c_proofs);
            }
            if !valid_cs {
                println!("{:?}", ab_proofs);
            }
            return Err(ValidationError::BadMerkleProof);
        }

        let root = PrimeFieldElement::new(primitive_root_of_unity, &field);
        println!("primitive_root_of_unity = {}", primitive_root_of_unity);
        let challenge = challenges[i];
        for j in 0..s as usize {
            let a_index = ab_indices[2 * j] as i128;
            let a_x = root.mod_pow_raw(a_index);
            let a_y = ab_proofs[2 * j][0].as_ref().unwrap().value.unwrap();
            let b_index = ab_indices[2 * j + 1] as i128;
            let b_x = root.mod_pow_raw(b_index);
            let b_y = ab_proofs[2 * j + 1][0].as_ref().unwrap().value.unwrap();
            // let c_index = c_indices[j] as i128;
            // let c_x = root.mod_pow_raw(c_index * 2);
            let c_y = c_proofs[j][0].as_ref().unwrap().value.unwrap();
            if !PrimeFieldPolynomial::are_colinear_raw(
                &[(a_x, a_y), (b_x, b_y), (challenge, c_y)],
                modulus,
            ) {
                println!(
                    "{{({},{}),({},{}),({},{})}} is not colinear",
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
    // let last_y_value =
    if !c_values.iter().all(|&x| c_values[0] == x) {
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

    println!("challenge = {}", challenge);
    let mut x = 1i128;
    for i in 0..new_codeword.len() {
        // let (_, two_x_inv, _) = PrimeFieldElement::eea(2 * x, *modulus);
        let (_, x_inv, _) = PrimeFieldElement::eea(x, *modulus);
        // If codeword is the evaluation of a polynomial of degree N,
        // this is an evaluation of a polynomial of degree N/2
        new_codeword[i] = (((1 + challenge * x_inv) * codeword[i]
            + (1 - challenge * x_inv) * codeword[i + codeword.len() / 2])
            * *inv_two
            % *modulus
            + *modulus)
            % *modulus;
        // println!("codeword[i] = {}", codeword[i]); //
        // println!(
        //     "codeword[i + codeword.len() / 2] = {}",
        //     codeword[i + codeword.len() / 2]
        // );
        // let p_even = ((codeword[i] + codeword[i + codeword.len() / 2]) * *inv_two % *modulus
        //     + *modulus)
        //     % *modulus;
        // let p_odd = ((codeword[i] - codeword[i + codeword.len() / 2]) * two_x_inv % *modulus
        //     + *modulus)
        //     % *modulus;
        // println!("p_even = {}", p_even);
        // println!("p_odd = {}", p_odd);
        x = x * *primitive_root_of_unity % modulus;
    }
    println!("modulus = {}", modulus); // TODO: REMOVE
    println!("inv_two = {}", inv_two); // TODO: REMOVE
    println!("codeword: {:?}", codeword); // TODO: REMOVE
    println!("new_codeword: {:?}", new_codeword); // TODO: REMOVE
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
    let expansion_factor = (codeword.len() as f64 / max_degree as f64).ceil() as usize;
    println!("max_degree = {}", max_degree);
    println!("expansion_factor = {}", expansion_factor);
    println!("codeword.len() = {}", codeword.len());
    let rounds_count = log_2_ceil(max_degree as u64 + 1) as usize;
    println!("rounds_count = {}", rounds_count);
    // output.append(&mut bincode::serialize(&(rounds_count as u8)).unwrap());
    output.append(&mut bincode::serialize(&(codeword.len() as u32)).unwrap());
    output.append(&mut bincode::serialize(&(max_degree as u32)).unwrap());
    output.append(&mut bincode::serialize(&(s as u32)).unwrap());
    output.append(&mut bincode::serialize(&(primitive_root_of_unity)).unwrap());
    let mut mt = MerkleTreeVector::from_vec(codeword);
    println!("mts[0].get_root() = {:?}", mt.get_root());
    let mut mts: Vec<MerkleTreeVector<i128>> = vec![mt];

    // Arrays for return values
    let mut c_proofs: Vec<Vec<Vec<Option<Node<i128>>>>> = vec![];
    let mut ab_proofs: Vec<Vec<Vec<Option<Node<i128>>>>> = vec![];

    output.append(&mut mts[0].get_root().to_vec());
    let mut mut_codeword: Vec<i128> = codeword.to_vec().clone();

    // commit phase
    let (_, _, inv2_temp) = PrimeFieldElement::eea(modulus, 2);
    let inv2 = (inv2_temp + modulus) % modulus;
    // let mut num_rounds = 0;
    let mut primitive_root_of_unity_temp = primitive_root_of_unity;
    for _ in 0..rounds_count {
        // while mut_codeword.len() > expansion_factor {
        // get challenge
        let hash = *blake3::hash(output.as_slice()).as_bytes();
        let challenge: i128 = PrimeFieldElement::from_bytes_raw(&modulus, &hash[0..16]);

        // run fri iteration
        mut_codeword = fri_prover_iteration(
            &mut_codeword.clone(),
            &challenge,
            &modulus,
            &inv2,
            &primitive_root_of_unity_temp,
        );

        // wrap into merkle tree
        mt = MerkleTreeVector::from_vec(&mut_codeword);
        println!("mt.get_root() = {:?}", mt.get_root());

        // append root to proof
        output.append(&mut mt.get_root().to_vec());

        // collect into memory
        mts.push(mt.clone());

        // num_rounds += 1;
        primitive_root_of_unity_temp =
            primitive_root_of_unity_temp * primitive_root_of_unity_temp % modulus;
    }
    println!("last Merkle Tree: {:?}", mts[rounds_count]); // TODO: REMOVE

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
    primitive_root_of_unity_temp = primitive_root_of_unity;
    for i in 0usize..rounds_count {
        println!("i = {}", i);
        let number_of_leaves = mts[i].get_number_of_leafs();
        let mut c_indices: Vec<usize> = vec![];
        let mut ab_indices: Vec<usize> = vec![];

        // it's unrealistic that the number of rounds exceed 256 but this should wrap around if it does
        let mut hash_preimage: Vec<u8> = partial_output.clone();
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

        // Debug, TODO: REMOVE
        if i >= rounds_count - 1 {
            println!(
                "i = {}, last Merkle Tree root: {:?}",
                i,
                mts[i + 1].get_root()
            ); // TODO: REMOVE
        }
        println!("c_indices = {:?}", c_indices);
        let field = PrimeField::new(modulus);
        let root = PrimeFieldElement::new(primitive_root_of_unity, &field);
        println!("number_of_leaves = {}", number_of_leaves);
        println!("c index = {}", c_indices[0]);
        println!("c x value = {}", root.mod_pow_raw(c_indices[0] as i128 * 2));
        println!(
            "c y value revealed = {}",
            authentication_paths_c[0][0]
                .as_ref()
                .unwrap()
                .value
                .unwrap()
        );
        println!("a index = {}", ab_indices[0]);
        println!("a x value = {}", root.mod_pow_raw(ab_indices[0] as i128));
        println!(
            "a y value revealed = {}",
            authentication_paths_ab[0][0]
                .as_ref()
                .unwrap()
                .value
                .unwrap()
        );
        println!("b index = {}", ab_indices[1]);
        println!("b x value = {}", root.mod_pow_raw(ab_indices[1] as i128));
        println!(
            "b y value revealed = {}",
            authentication_paths_ab[1][0]
                .as_ref()
                .unwrap()
                .value
                .unwrap()
        );

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
        c_proofs,
        ab_proofs,
        s: s as u32,
        merkle_roots: mts.iter().map(|x| x.get_root()).collect::<Vec<[u8; 32]>>(),
        codeword_size: codeword.len() as u32,
        primitive_root_of_unity,
        max_degree,
        serialization: output.clone(),
    }
}

// pub struct LowDegreeProof {
//     rounds: u8,
//     max_degree: u32,
//     s: u32,
//     merkle_roots: Vec<[u8; 32]>,
//     codeword_size: u32,
//     primitive_root_of_unity: i128,
//     c_proofs: Vec<Vec<Vec<Option<Node<i128>>>>>,
//     ab_proofs: Vec<Vec<Vec<Option<Node<i128>>>>>,
//     serialization: Vec<u8>,
// }

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
        println!("prime = {}, root = {}", field.q, primitive_root_of_unity);
        let power_series = field.get_power_series(primitive_root_of_unity);
        assert_eq!(4, power_series.len());
        assert_eq!(vec![1i128, 10, 100, 91], power_series);
        let mut output = vec![];

        // corresponds to the polynomial P(x) = x
        let y_values = power_series;
        let max_degree = 1;
        let s = 5; // The security factor
        let proof: LowDegreeProof<i128> = prover(
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

        let deserialized_proof: LowDegreeProof<i128> = proof.serialization.try_into().unwrap();
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

        println!("\n\n\n\n\n\n\n\n\n\n\n***************** PROOF DONE *****************");
        println!("***************** START VERIFY ***************** \n\n");
        assert_eq!(Ok(()), verify(field.q, &output));
        println!("***************** VERIFY SUCCEEDED! ***************** \n\n");

        // Change one of the values in a leaf in the committed Merkle tree, and verify that the Merkle proof fails
        output = vec![];
        prover(
            &y_values,
            field.q,
            max_degree,
            s,
            &mut output,
            primitive_root_of_unity,
        );
        // println!("output[1033] = {}", output[1033]);
        // output[1033] = (output[1033] + 1) as u8; // Should change a committed value from 42 to 43
        assert_eq!(
            Err(ValidationError::BadMerkleProof),
            verify(field.q, &output)
        );
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
        let domain = field.get_power_series(primitive_root_of_unity);
        // coefficients: vec![6, 2, 5] => P(x) = 5x^2 + 2x + 6
        let mut y_values = domain
            .iter()
            .map(|&x| ((6 + x * (2 + 5 * x)) % field.q + field.q) % field.q)
            .collect::<Vec<i128>>();

        println!("domain = {:?}", domain);
        println!("y_values = {:?}", y_values);
        let max_degree = 2;
        let s = 6;
        let mut output = vec![];
        let ret = prover(
            &y_values,
            field.q,
            max_degree,
            s,
            &mut output,
            primitive_root_of_unity,
        );
        assert_eq!(ret, ret.serialization.clone().try_into().unwrap());
        println!("\n\n\n\n\n\n\n\n\n\n\n***************** PROOF DONE *****************");
        println!("***************** START VERIFY ***************** \n\n");
        assert_eq!(Ok(()), verify(field.q, &output));

        // Change a single y value such that it no longer corresponds to a polynomil
        // a verify that the test fails
        output = vec![];
        let original_y_values = y_values.clone();
        y_values[3] = 100;
        y_values[4] = 100;
        prover(
            &y_values,
            field.q,
            max_degree,
            s,
            &mut output,
            primitive_root_of_unity,
        );
        assert_eq!(
            Err(ValidationError::LastIterationNotConstant),
            verify(field.q, &output)
        );

        // make a proof with a too low max_degree parameter and verify that it fails verification
        // with the expected output
        let wrong_max_degree = 1;
        output = vec![];
        prover(
            &original_y_values,
            field.q,
            wrong_max_degree,
            s,
            &mut output,
            primitive_root_of_unity,
        );
        assert_eq!(
            Err(ValidationError::LastIterationNotConstant),
            verify(field.q, &output)
        );
    }

    #[test]
    fn generate_proof_16_alt() {
        let mut ret: Option<(PrimeField, i128)> = None;
        // should return (field = mod 193; root = 64) for (n = 16, min_value = 113)
        PrimeField::get_field_with_primitive_root_of_unity(16, 113, &mut ret);
        assert_eq!(193i128, ret.clone().unwrap().0.q);
        let (field, primitive_root_of_unity) = ret.clone().unwrap();
        println!("primitive_root_of_unity = {}", primitive_root_of_unity);
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
        assert_eq!(proof, proof.serialization.clone().try_into().unwrap());
        assert_eq!(Ok(()), verify(field.q, &output));
    }

    #[test]
    fn generate_proof_1024() {
        let mut ret: Option<(PrimeField, i128)> = None;
        let size = 2usize.pow(14);
        let max_degree = 1024;
        PrimeField::get_field_with_primitive_root_of_unity(size as i128, size as i128, &mut ret);
        let (field_temp, primitive_root_of_unity) = ret.clone().unwrap();
        let field: PrimeField = field_temp.clone();
        let (generator_2_option, _) = field.get_primitive_root_of_unity(max_degree as i128);
        let generator_2 = generator_2_option.unwrap();
        println!(
            "primitive_root_of_unity = {}, prime = {}",
            primitive_root_of_unity, field.q
        );
        println!(
            "generator_2 = {}, {}th root of unity",
            generator_2, max_degree
        );
        assert_eq!(65537i128, field.q);
        assert_eq!(81i128, primitive_root_of_unity);
        let mut coefficients = generate_random_numbers(max_degree, field.q);
        coefficients.extend_from_slice(&vec![0; size - max_degree]);
        let mut y_values =
            fast_polynomial_evaluate(coefficients.as_slice(), field.q, primitive_root_of_unity);

        let mut output = vec![];

        let s = 40;
        let mut proof = prover(
            &y_values,
            field.q,
            max_degree as u32,
            s,
            &mut output,
            primitive_root_of_unity,
        );
        assert_eq!(proof, proof.serialization.clone().try_into().unwrap());
        assert_eq!(Ok(()), verify(field.q, &output));

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
        assert_eq!(proof, proof.serialization.clone().try_into().unwrap());
        assert_eq!(
            Err(ValidationError::LastIterationNotConstant),
            verify(field.q, &output)
        );
    }
}
