use crate::shared_math::b_field_element::BFieldElement;
use std::fmt::Display;
use Instruction::*;
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
    Squeeze(Word),
    Absorb(Word),
    Clear(Word),
    Rotate(Word),
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

pub const DUP0: Instruction = Dup(Ord4::N0);
pub const DUP1: Instruction = Dup(Ord4::N1);
pub const DUP2: Instruction = Dup(Ord4::N2);
pub const DUP3: Instruction = Dup(Ord4::N3);

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
        write!(
            f,
            "{}",
            match self {
                N0 => "0",
                N1 => "1",
                N2 => "2",
                N3 => "3",
            }
        )
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
