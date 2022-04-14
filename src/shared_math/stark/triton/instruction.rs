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
///
/// TODO: Order these according to their "value" (Nop == 1) once value is known.
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

pub const DUP0: Instruction = Dup(Ord4::D0);
pub const DUP1: Instruction = Dup(Ord4::D1);
pub const DUP2: Instruction = Dup(Ord4::D2);
pub const DUP3: Instruction = Dup(Ord4::D3);

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Ord4 {
    D0,
    D1,
    D2,
    D3,
}

impl From<Ord4> for usize {
    fn from(n: Ord4) -> Self {
        match n {
            D0 => 0,
            D1 => 1,
            D2 => 2,
            D3 => 3,
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
