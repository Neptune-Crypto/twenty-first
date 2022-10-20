use serde::{de::DeserializeOwned, Serialize};
use std::{error::Error, fmt, result::Result};

use crate::shared_math::rescue_prime_digest::Digest;

use super::blake3_wrapper::from_blake3_digest;

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

    /// A package on a `ProofStream` consist of a `u32` containing the `item_length` of the payload (`item`)
    /// followed by the payload.  This is similar to _pascal style strings_.
    /// Corresponds to `pull` in [AoaS](https://aszepieniec.github.io/stark-anatomy/basic-tools#the-fiat-shamir-transform).

    /// # Arguments
    ///
    /// * `item` - The payload we want to dequeue and deserialize.
    /// * `item_length` - The length of the payload in bytes.
    /// * `sizeof_item_length` - The size of the prepended field.
    pub fn dequeue_length_prepended<T>(&mut self) -> Result<T, Box<dyn Error>>
    where
        T: DeserializeOwned,
    {
        let sizeof_item_length = std::mem::size_of::<u32>();
        assert_eq!(sizeof_item_length, 4, "32 bits should equal 4 bytes.");

        let item_length_start = self.read_index;
        let item_length_end = self.read_index + sizeof_item_length;
        let item_length: u32 =
            bincode::deserialize(&self.transcript[item_length_start..item_length_end])?;

        let item_start = self.read_index + sizeof_item_length;
        let item_end = item_start + item_length as usize; // Is this cast necessary?

        if self.len() < item_end {
            return Err(Box::new(ProofStreamError::TranscriptLengthExceeded));
        }

        let item: T = bincode::deserialize(&self.transcript[item_start..item_end])?;

        self.read_index = item_end;

        Ok(item)
    }

    pub fn prover_fiat_shamir(&self) -> Digest {
        from_blake3_digest(&blake3::hash(&self.transcript))
    }

    pub fn verifier_fiat_shamir(&self) -> Digest {
        from_blake3_digest(&blake3::hash(&self.transcript[0..self.read_index]))
    }
}

#[cfg(test)]
pub mod test_proof_stream {
    use super::*;
    use crate::shared_math::b_field_element::BFieldElement;

    #[test]
    fn ps_test_default_empty_initiation() {
        let proof_stream = ProofStream::default();
        assert!(proof_stream.is_empty());
        assert_eq!(0, proof_stream.get_read_index());
    }

    #[test]
    fn ps_empty_ts() {
        let ps = ProofStream::default();
        assert_eq!(ps.len(), 0, "The empty ProofStream must have length zero.");
        let ts = ps.serialize();
        assert_eq!(
            ts.len(),
            0,
            "The serialization of the empty ProofStream must have length zero."
        );
    }

    #[test]
    fn ps_enqueue_then_dequeue() {
        let mut ps = ProofStream::default();

        let bfe_before = BFieldElement::new(213);
        assert!(ps.enqueue_length_prepended(&bfe_before).is_ok());
        let bfe_after = ps.dequeue_length_prepended().unwrap();

        assert_eq!(
            bfe_before, bfe_after,
            "`enqueue` element followed by `dequeue` should return the same element."
        );
    }

    #[test]
    fn ps_thrice_enqueue_then_dequeue() {
        ps_enqueue_then_dequeue();
        ps_enqueue_then_dequeue();
        ps_enqueue_then_dequeue();
    }

    #[test]
    fn ps_enq_deq_enq_deq() {
        let bfe1_before = BFieldElement::new(213);
        let bfe2_before = BFieldElement::new(783);

        let mut ps = ProofStream::default();
        assert!(ps.enqueue_length_prepended(&bfe1_before).is_ok());
        assert!(ps.enqueue_length_prepended(&bfe2_before).is_ok());

        let bfe1_after = ps.dequeue_length_prepended().unwrap();
        let bfe2_after = ps.dequeue_length_prepended().unwrap();

        assert_eq!(
            bfe1_before, bfe1_after,
            "Element 1 has changed on the stream!"
        );

        assert_eq!(
            bfe2_before, bfe2_after,
            "Element 2 has changed on the stream!"
        );
    }

    #[test]
    fn ps_is_fifo_no_lifo() {
        let bfe1_before = BFieldElement::new(213);
        let bfe2_before = BFieldElement::new(783);

        let mut ps = ProofStream::default();
        assert!(ps.enqueue_length_prepended(&bfe1_before).is_ok());
        assert!(ps.enqueue_length_prepended(&bfe2_before).is_ok());

        // Intentionally wrong order
        let bfe2_after_phoney = ps.dequeue_length_prepended().unwrap();
        let bfe1_after_phoney = ps.dequeue_length_prepended().unwrap();

        assert_ne!(
            bfe1_before, bfe1_after_phoney,
            "ProofStream erroneously has LIFO behavior when it should have FIFO behavior."
        );

        assert_ne!(
            bfe2_before, bfe2_after_phoney,
            "ProofStream erroneously has LIFO behavior when it should have FIFO behavior."
        );
    }
}
