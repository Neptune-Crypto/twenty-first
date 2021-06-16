use crate::shared_math::prime_field_element::PrimeFieldElement;
use crate::util_types::merkle_tree_vector::{MerkleTreeVector, Node};
use crate::utils::{get_index_from_bytes, get_n_hash_rounds};
use std::convert::TryInto;

pub fn verify(modulus: i128, max_degree: i128, s: usize, rounds_count: usize, output: &[u8]) {
    // get the 1st merkle tree root
    // let roots: Vec<[u8; 32]> = vec![];
    // let challenges: Vec<i128> = vec![];
    // for i in 0..rounds_count {
    //     roots.push(output[i * 32..(i + 1) * 32].try_into().unwrap());
    // }
    let roots: Vec<[u8; 32]> = (0..rounds_count)
        .map(|i| output[i * 32..(i + 1) * 32].try_into().unwrap())
        .collect();
    // let challenges: Vec<i128> = roots
    //     .iter()
    //     .map(|x| PrimeFieldElement::from_bytes_raw(&modulus, &x[0..16]))
    //     .collect();
    let challenge_hash_preimages: Vec<Vec<u8>> = (0..rounds_count)
        .map(|i| output[0..((i + 1) * 32)].to_vec())
        .collect();
    let challenge_hashes: Vec<[u8; 32]> = challenge_hash_preimages
        .iter()
        .map(|bs| *blake3::hash(bs.as_slice()).as_bytes())
        .collect();
    let challenges: Vec<i128> = challenge_hashes
        .iter()
        .map(|x| PrimeFieldElement::from_bytes_raw(&modulus, &x[0..16]))
        .collect();
    println!("challenges = {:?}", challenges); // TODO: REMOVE

    // Calculate the indicies values from the `output` u8 vector
    // for i in 0usize..rounds_count - 1 {
    //     let n = mts[i].get_number_of_leafs();
    //     let mut y_indices: Vec<usize> = vec![];
    //     let mut s_indices: Vec<usize> = vec![];
    // }
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
    new_codeword
}

// TODO: We want this implemented for prime field elements, and preferably for
// any finite field/extension field.
// Prove that codeword elements come from the evaluation of a polynomial of
// `degree < codeword.len() / rho`
pub fn prover(codeword: &[i128], modulus: i128, rho: usize, s: usize, output: &mut Vec<u8>) {
    let mut mt = MerkleTreeVector::from_vec(codeword);
    output.append(&mut mt.get_root().to_vec());
    let mut mts: Vec<MerkleTreeVector<i128>> = vec![mt];
    let mut mut_codeword: Vec<i128> = codeword.to_vec().clone();

    // commit phase
    let (_, inv2, _) = PrimeFieldElement::eea(modulus, 2);
    let mut num_rounds = 0;
    while mut_codeword.len() >= rho {
        // get challenge
        let hash = *blake3::hash(output.as_slice()).as_bytes();
        let challenge: i128 = PrimeFieldElement::from_bytes_raw(&modulus, &hash[0..16]);
        println!("challenge = {}", challenge); // TODO: REMOVE

        // run fri iteration
        mut_codeword = fri_prover_iteration(&mut_codeword.clone(), &challenge, &modulus, &inv2);

        // wrap into merkle tree
        mt = MerkleTreeVector::from_vec(&mut_codeword);

        // append root to proof
        output.append(&mut mt.get_root().to_vec());
        println!("output = {:?}", output); // TODO: REMOVE

        // collect into cache
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
    for i in 0usize..num_rounds - 1 {
        let n = mts[i].get_number_of_leafs();
        let mut y_indices: Vec<usize> = vec![];
        let mut s_indices: Vec<usize> = vec![];

        let hashes = get_n_hash_rounds(output.as_slice(), s); // TODO: Are we reusing randomness here??
        for hash in hashes.iter() {
            let y_index = get_index_from_bytes(&hash[0..16], n / 2);
            y_indices.push(y_index);
            let s0_index = y_index;
            s_indices.push(s0_index);
            let s1_index = y_index + n / 2;
            s_indices.push(s1_index);
        }

        let authentication_paths_y: Vec<Vec<Option<Node<i128>>>> =
            mts[i + i].get_multi_proof(y_indices);
        let authentication_paths_s: Vec<Vec<Option<Node<i128>>>> =
            mts[i].get_multi_proof(s_indices);
        output.append(&mut bincode::serialize(&authentication_paths_y.clone()).unwrap());
        output.append(&mut bincode::serialize(&authentication_paths_s.clone()).unwrap());
    }
}

#[cfg(test)]
mod test_utils {
    use super::*;
    use crate::shared_math::prime_field_element::PrimeField;

    #[test]
    fn generate_proof() {
        let mut ret: Option<(PrimeField, i128)> = None;
        // should return (field = mod 113; root = 40) for (n = 16, min_value = 100)
        PrimeField::get_field_with_primitive_root_of_unity(16, 100, &mut ret);
        assert_eq!(113i128, ret.clone().unwrap().0.q);
        let (field, root) = ret.clone().unwrap();
        let power_series = field.get_power_series(root);
        assert_eq!(16, power_series.len());
        assert_eq!(
            vec![1i128, 40, 18, 42, 98, 78, 69, 48, 112, 73, 95, 71, 15, 35, 44, 65],
            power_series
        );
        let mut output = vec![];

        // corresponds to the polynomial P(x) = x
        let y_values = power_series;
        prover(&y_values, 113i128, 4, 4, &mut output);
        // println!("{:?}", output); // TODO: REMOVE
        verify(113i128, 4, 4, 3, &output);
    }
}
