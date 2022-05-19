use byteorder::{BigEndian, ReadBytesExt};
use std::io::{Error, Write};
use std::io::{Stdin, Stdout};

pub trait InputStream {
    fn read_u32_be(&mut self) -> Result<u32, Error>;
}

impl InputStream for Stdin {
    fn read_u32_be(&mut self) -> Result<u32, Error> {
        self.read_u32::<BigEndian>()
    }
}

pub trait OutputStream {
    fn write_u32_be(&mut self, codepoint: u32) -> Result<usize, Error>;
}

impl OutputStream for Stdout {
    fn write_u32_be(&mut self, codepoint: u32) -> Result<usize, Error> {
        let bytes = codepoint.to_be_bytes();
        self.write(&bytes)
    }
}
