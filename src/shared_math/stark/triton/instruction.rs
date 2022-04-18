use super::ord_n::Ord16;
use super::ord_n::{Ord4, Ord4::*};
use crate::shared_math::b_field_element::BFieldElement;
use std::fmt::Display;
use Instruction::*;

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
    // Read/write
    Print,
    Scan,
}

pub fn push(value: u32) -> Instruction {
    Push(value.into())
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
            Print => write!(f, "print"),
            Scan => write!(f, "scan"),
        }
    }
}

impl Instruction {
    /// Assign a unique positive integer to each `Instruction`.
    pub fn value(&self) -> u32 {
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
            // FIXME: Gap left by Ntt/Intt.
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
            Print => 46,
            Scan => 47,
        }
    }
}
