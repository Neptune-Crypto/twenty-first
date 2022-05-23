use crate::shared_math::b_field_element::BFieldElement;
use byteorder::{BigEndian, ReadBytesExt};
use std::io::{Cursor, Error, Write};
use std::io::{Stdin, Stdout};

pub trait InputStream {
    fn read_elem(&mut self) -> Result<BFieldElement, Error>;
}

pub trait OutputStream {
    fn write_elem(&mut self, elem: BFieldElement) -> Result<usize, Error>;
}

impl InputStream for Stdin {
    fn read_elem(&mut self) -> Result<BFieldElement, Error> {
        let elem = self.read_u64::<BigEndian>()?;
        Ok(BFieldElement::new(elem))
    }
}

impl OutputStream for Stdout {
    fn write_elem(&mut self, elem: BFieldElement) -> Result<usize, Error> {
        let bytes = elem.value().to_be_bytes();
        self.write(&bytes)
    }
}

pub struct VecStream {
    cursor: Cursor<Vec<u8>>,
}

impl VecStream {
    pub fn new(bytes: &[u8]) -> Self {
        VecStream {
            cursor: Cursor::new(bytes.to_vec()),
        }
    }

    pub fn to_vec(&self) -> Vec<u8> {
        // FIXME: Address cloning as unnecessary.
        self.cursor.clone().into_inner()
    }
}

impl InputStream for VecStream {
    fn read_elem(&mut self) -> Result<BFieldElement, Error> {
        let elem = self.cursor.read_u64::<BigEndian>()?;
        Ok(BFieldElement::new(elem))
    }
}

impl OutputStream for VecStream {
    fn write_elem(&mut self, elem: BFieldElement) -> Result<usize, Error> {
        let bytes = elem.value().to_be_bytes();
        self.cursor.write(&bytes)
    }
}
