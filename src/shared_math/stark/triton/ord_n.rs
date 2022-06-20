use std::fmt::Display;
use Ord16::*;
use Ord5::*;
use Ord6::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ord5 {
    HV0,
    HV1,
    HV2,
    HV3,
    HV4,
}

impl Display for Ord5 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n: usize = (*self).into();
        write!(f, "{}", n)
    }
}

impl From<Ord5> for u32 {
    fn from(n: Ord5) -> Self {
        match n {
            HV0 => 0,
            HV1 => 1,
            HV2 => 2,
            HV3 => 3,
            HV4 => 4,
        }
    }
}

impl From<&Ord5> for u32 {
    fn from(n: &Ord5) -> Self {
        (*n).into()
    }
}

impl From<Ord5> for usize {
    fn from(n: Ord5) -> Self {
        let n: u32 = n.into();
        n as usize
    }
}

impl From<&Ord5> for usize {
    fn from(n: &Ord5) -> Self {
        (*n).into()
    }
}

impl TryFrom<usize> for Ord5 {
    type Error = String;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(HV0),
            1 => Ok(HV1),
            2 => Ok(HV2),
            3 => Ok(HV3),
            4 => Ok(HV4),
            _ => Err(format!("{} is out of range for Ord5", value)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ord6 {
    IB0,
    IB1,
    IB2,
    IB3,
    IB4,
    IB5,
}

impl Display for Ord6 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n: usize = (*self).into();
        write!(f, "{}", n)
    }
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

impl TryFrom<usize> for Ord6 {
    type Error = String;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(IB0),
            1 => Ok(IB1),
            2 => Ok(IB2),
            3 => Ok(IB3),
            4 => Ok(IB4),
            5 => Ok(IB5),
            _ => Err(format!("{} is out of range for Ord6", value)),
        }
    }
}

/// `Ord16` represents numbers that are exactly 0--15.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl Display for Ord16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n: usize = (*self).into();
        write!(f, "{}", n)
    }
}

impl From<Ord16> for u32 {
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

impl From<&Ord16> for u32 {
    fn from(n: &Ord16) -> Self {
        (*n).into()
    }
}

impl From<Ord16> for usize {
    fn from(n: Ord16) -> Self {
        let n: u32 = n.into();
        n as usize
    }
}

impl From<&Ord16> for usize {
    fn from(n: &Ord16) -> Self {
        (*n).into()
    }
}

impl TryFrom<usize> for Ord16 {
    type Error = &'static str;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(A0),
            1 => Ok(A1),
            2 => Ok(A2),
            3 => Ok(A3),
            4 => Ok(A4),
            5 => Ok(A5),
            6 => Ok(A6),
            7 => Ok(A7),
            8 => Ok(A8),
            9 => Ok(A9),
            10 => Ok(A10),
            11 => Ok(A11),
            12 => Ok(A12),
            13 => Ok(A13),
            14 => Ok(A14),
            15 => Ok(A15),
            _ => Err("usize out of range for Ord16"),
        }
    }
}
