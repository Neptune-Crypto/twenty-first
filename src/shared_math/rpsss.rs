use super::prime_field_element_big::{PrimeFieldBig, PrimeFieldElementBig};
use super::stark::Stark;
use crate::shared_math::rescue_prime_stark::RescuePrime;
use crate::util_types::proof_stream::ProofStream;
use rand::RngCore;
use std::error::Error;

#[derive(Clone, Debug)]
pub struct SecretKey<'a> {
    pub value: PrimeFieldElementBig<'a>,
}

#[derive(Clone, Debug)]
pub struct PublicKey<'a> {
    pub value: PrimeFieldElementBig<'a>,
}

#[derive(Clone, Debug)]
pub struct Signature {
    pub proof: Vec<u8>,
}

pub struct RPSSS<'a> {
    pub field: PrimeFieldBig,
    pub rp: RescuePrime<'a>,
    pub stark: Stark<'a>,
}

impl<'a> RPSSS<'a> {
    pub fn keygen(&'a self) -> (SecretKey<'a>, PublicKey<'a>) {
        let mut prng = rand::thread_rng();
        let mut bytes = vec![0u8; 17];
        prng.fill_bytes(&mut bytes);
        let sk: SecretKey = SecretKey {
            value: self.field.from_bytes(&bytes),
        };
        let pk = PublicKey {
            value: self.rp.hash(&sk.value),
        };
        (sk, pk)
    }

    pub fn sign(&self, sk: SecretKey, document: Vec<u8>) -> Result<Signature, Box<dyn Error>> {
        let (output, trace) = self.rp.eval_and_trace(&sk.value);
        // let mut proof_stream = ProofStream::signature_proof_stream(document);
        let mut proof_stream = ProofStream::new_with_prefix(&document);
        self.stark.prove(
            trace,
            self.rp.get_air_constraints(&self.stark.omicron),
            self.rp.get_boundary_constraints(&output),
            &mut proof_stream,
        )?;

        Ok(Signature {
            proof: proof_stream.serialize(),
        })
    }
}
