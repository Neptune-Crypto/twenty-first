use serde::{de::DeserializeOwned, Serialize};
use std::{error::Error, fmt, result::Result};

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ProofStream {
    read_index: usize,
    transcript: Vec<u8>,
}

impl From<Vec<u8>> for ProofStream {
    fn from(item: Vec<u8>) -> Self {
        ProofStream {
            read_index: 0,
            transcript: item,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ProofStreamError {
    TranscriptLengthExceeded,
}

impl Error for ProofStreamError {}

impl fmt::Display for ProofStreamError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl ProofStream {
    pub fn new_with_prefix(prefix: &[u8]) -> Self {
        Self {
            read_index: 0,
            transcript: prefix.to_vec(),
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        self.transcript.clone()
    }

    pub fn len(&self) -> usize {
        self.transcript.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transcript.is_empty()
    }

    pub fn get_read_index(&self) -> usize {
        self.read_index
    }

    pub fn set_index(&mut self, new_index: usize) {
        assert!(
            new_index <= self.transcript.len(),
            "new_index cannot exceed transcript length"
        );
        self.read_index = new_index;
    }

    pub fn enqueue<T>(&mut self, item: &T) -> Result<(), Box<dyn Error>>
    where
        T: Serialize,
    {
        let mut serialization_result = bincode::serialize(item)?;
        self.transcript.append(&mut serialization_result);

        Ok(())
    }

    pub fn enqueue_length_prepended<T>(&mut self, item: &T) -> Result<(), Box<dyn Error>>
    where
        T: Serialize,
    {
        let mut serialization_result: Vec<u8> = bincode::serialize(item)?;
        let serialization_result_length: u32 = serialization_result.len() as u32;
        self.transcript
            .append(&mut bincode::serialize(&serialization_result_length).unwrap());
        self.transcript.append(&mut serialization_result);

        Ok(())
    }

    pub fn dequeue<T>(&mut self, byte_length: usize) -> Result<T, Box<dyn Error>>
    where
        T: DeserializeOwned,
    {
        if byte_length + self.read_index > self.transcript.len() {
            return Err(Box::new(ProofStreamError::TranscriptLengthExceeded));
        }

        let item: T =
            bincode::deserialize(&self.transcript[self.read_index..self.read_index + byte_length])?;
        self.read_index += byte_length;

        Ok(item)
    }

    pub fn dequeue_length_prepended<T>(&mut self) -> Result<T, Box<dyn Error>>
    where
        T: DeserializeOwned,
    {
        let item_length: u32 =
            bincode::deserialize(&self.transcript[self.read_index..self.read_index + 4])?;
        self.read_index += 4;
        if item_length as usize + self.read_index > self.transcript.len() {
            return Err(Box::new(ProofStreamError::TranscriptLengthExceeded));
        }

        let item: T = bincode::deserialize(
            &self.transcript[self.read_index..self.read_index + item_length as usize],
        )?;
        self.read_index += item_length as usize;

        Ok(item)
    }

    pub fn prover_fiat_shamir(&self) -> Vec<u8> {
        blake3::hash(&self.transcript).as_bytes().to_vec()
    }

    pub fn verifier_fiat_shamir(&self) -> Vec<u8> {
        blake3::hash(&self.transcript[0..self.read_index])
            .as_bytes()
            .to_vec()
    }
}

#[cfg(test)]
pub mod test_proof_stream {

    use super::*;

    #[test]
    fn test_default_empty_initiation() {
        let proof_stream = ProofStream::default();
        assert!(proof_stream.is_empty());
        assert_eq!(0, proof_stream.get_read_index());
    }
}
