use byteorder::{BigEndian, ReadBytesExt};
use std::io::{Cursor, Error, Write};
use std::io::{Stdin, Stdout};

pub trait InputStream {
    fn read_u32_be(&mut self) -> Result<u32, Error>;
}

pub trait OutputStream {
    fn write_u32_be(&mut self, codepoint: u32) -> Result<usize, Error>;
}

impl InputStream for Stdin {
    fn read_u32_be(&mut self) -> Result<u32, Error> {
        self.read_u32::<BigEndian>()
    }
}

impl OutputStream for Stdout {
    fn write_u32_be(&mut self, codepoint: u32) -> Result<usize, Error> {
        let bytes = codepoint.to_be_bytes();
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
    fn read_u32_be(&mut self) -> Result<u32, Error> {
        self.cursor.read_u32::<BigEndian>()
    }
}

impl OutputStream for VecStream {
    fn write_u32_be(&mut self, codepoint: u32) -> Result<usize, Error> {
        let bytes = codepoint.to_be_bytes();
        self.cursor.write(&bytes)
    }
}
