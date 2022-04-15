use crate::shared_math::b_field_element::BFieldElement;
use std::fmt::Display;
use Instruction::*;
use Ord16::*;
use Ord4::*;

type Word = BFieldElement;

/// A Triton VM instruction
///
/// The ISA is defined at:
///
/// https://neptune.builders/core-team/triton-vm/src/branch/master/specification/isa.md
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Instruction {
    // OpStack manipulation
    Pop,
    Push(Word),
    Pad,
    Dup(Ord4),
    Swap,
    Pull2,
    Pull3,
    // Control flow
    Nop,
    Skiz,
    Call(Word),
    Return,
    Recurse,
    Assert,
    Halt,
    // Memory access
    Load,
    LoadInc,
    LoadDec,
    Save,
    SaveInc,
    SaveDec,
    SetRamp,
    GetRamp,
    // Auxiliary register instructions
    Xlix,
    Ntt,
    Intt,
    ClearAll,
    Squeeze(Ord16),
    Absorb(Ord16),
    Clear(Ord16),
    Rotate(Ord16),
    // Arithmetic on stack instructions
    Add,
    Neg,
    Mul,
    Inv,
    Lnot,
    Split,
    Eq,
    Lt,
    And,
    Or,
    Xor,
    Reverse,
    Div,
}

pub fn push(value: u32) -> Instruction {
    Push(value.into())
}

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

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Pop => write!(f, "pop"),
            Push(elem) => write!(f, "push {}", elem),
            Pad => write!(f, "pad"),
            Dup(n) => write!(f, "dup {}", n),
            Swap => write!(f, "swap"),
            Pull2 => write!(f, "pull2"),
            Pull3 => write!(f, "pull3"),
            Nop => write!(f, "nop"),
            Skiz => write!(f, "skiz"),
            Call(addr) => write!(f, "call {}", addr),
            Return => write!(f, "return"),
            Recurse => write!(f, "recurse"),
            Assert => write!(f, "assert"),
            Halt => write!(f, "halt"),
            Load => write!(f, "load"),
            LoadInc => write!(f, "loadinc"),
            LoadDec => write!(f, "loaddec"),
            Save => write!(f, "save"),
            SaveInc => write!(f, "saveinc"),
            SaveDec => write!(f, "savedec"),
            SetRamp => write!(f, "setramp"),
            GetRamp => write!(f, "getramp"),
            Xlix => write!(f, "xlix"),
            Ntt => write!(f, "ntt"),
            Intt => write!(f, "intt"),
            ClearAll => write!(f, "clearall"),
            Squeeze(arg) => write!(f, "squeeze {}", arg),
            Absorb(arg) => write!(f, "absorb {}", arg),
            Clear(arg) => write!(f, "clear {}", arg),
            Rotate(arg) => write!(f, "rotate {}", arg),
            Add => write!(f, "add"),
            Neg => write!(f, "neg"),
            Mul => write!(f, "mul"),
            Inv => write!(f, "inv"),
            Lnot => write!(f, "lnot"),
            Split => write!(f, "split"),
            Eq => write!(f, "eq"),
            Lt => write!(f, "lt"),
            And => write!(f, "and"),
            Or => write!(f, "or"),
            Xor => write!(f, "xor"),
            Reverse => write!(f, "reverse"),
            Div => write!(f, "div"),
        }
    }
}

impl Instruction {
    /// Assign a unique positive integer to each `Instruction`.
    pub fn number(&self) -> u32 {
        match self {
            Pop => 0,
            Push(_) => 1,
            Pad => 2,
            Dup(N0) => 3,
            Dup(N1) => 4,
            Dup(N2) => 5,
            Dup(N3) => 6,
            Swap => 7,
            Pull2 => 8,
            Pull3 => 9,
            Nop => 10,
            Skiz => 11,
            Call(_) => 12,
            Return => 13,
            Recurse => 14,
            Assert => 15,
            Halt => 16,
            Load => 17,
            LoadInc => 18,
            LoadDec => 19,
            Save => 20,
            SaveInc => 21,
            SaveDec => 22,
            SetRamp => 23,
            GetRamp => 24,
            Xlix => 25,
            Ntt => 26,
            Intt => 27,
            ClearAll => 28,
            Squeeze(_) => 29,
            Absorb(_) => 30,
            Clear(_) => 31,
            Rotate(_) => 32,
            Add => 33,
            Neg => 34,
            Mul => 35,
            Inv => 36,
            Lnot => 37,
            Split => 38,
            Eq => 39,
            Lt => 40,
            And => 41,
            Or => 42,
            Xor => 43,
            Reverse => 44,
            Div => 45,
        }
    }
}
