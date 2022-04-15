use std::fmt::Display;
use Ord16::*;
use Ord4::*;
use Ord6::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Ord4 {
    N0,
    N1,
    N2,
    N3,
}

impl From<Ord4> for usize {
    fn from(n: Ord4) -> Self {
        match n {
            N0 => 0,
            N1 => 1,
            N2 => 2,
            N3 => 3,
        }
    }
}

impl Display for Ord4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n: usize = (*self).into();
        write!(f, "{}", n)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Ord6 {
    IB0,
    IB1,
    IB2,
    IB3,
    IB4,
    IB5,
}

impl From<Ord6> for usize {
    fn from(n: Ord6) -> Self {
        match n {
            IB0 => 0,
            IB1 => 1,
            IB2 => 2,
            IB3 => 3,
            IB4 => 4,
            IB5 => 5,
        }
    }
}

impl Display for Ord6 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n: usize = (*self).into();
        write!(f, "{}", n)
    }
}

/// `Ord16` represents numbers that are exactly 0--15.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Ord16 {
    A0,
    A1,
    A2,
    A3,
    A4,
    A5,
    A6,
    A7,
    A8,
    A9,
    A10,
    A11,
    A12,
    A13,
    A14,
    A15,
}

impl From<Ord16> for usize {
    fn from(n: Ord16) -> Self {
        match n {
            A0 => 0,
            A1 => 1,
            A2 => 2,
            A3 => 3,
            A4 => 4,
            A5 => 5,
            A6 => 6,
            A7 => 7,
            A8 => 8,
            A9 => 9,
            A10 => 10,
            A11 => 11,
            A12 => 12,
            A13 => 13,
            A14 => 14,
            A15 => 15,
        }
    }
}

impl Display for Ord16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n: usize = (*self).into();
        write!(f, "{}", n)
    }
}
