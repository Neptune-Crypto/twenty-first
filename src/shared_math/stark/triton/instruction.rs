use super::ord_n::{Ord16, Ord4, Ord4::*};
use crate::shared_math::b_field_element::BFieldElement;
use std::error::Error;
use std::fmt::Display;
use std::str::SplitWhitespace;
use Instruction::*;
use TokenError::*;

type Word = BFieldElement;

/// A Triton VM instruction
///
/// The ISA is defined at:
///
/// https://neptune.builders/core-team/triton-vm/src/branch/master/specification/isa.md
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
            Dup(n) => write!(f, "dup{}", n),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program(pub Vec<Instruction>);

impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.iter().fold(Ok(()), |result, instruction| {
            result.and_then(|()| writeln!(f, "{}", instruction))
        })
    }
}

#[derive(Debug)]
pub enum TokenError<'a> {
    UnknownInstruction(&'a str),
    UnexpectedEndOfStream,
    // InvalidConstant(&'a str),
    // NotImplemented(&'a str),
}

impl<'a> Display for TokenError<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnknownInstruction(s) => write!(f, "UnknownInstructino({})", s),
            UnexpectedEndOfStream => write!(f, "UnexpectedEndOfStream"),
        }
    }
}

impl<'a> Error for TokenError<'a> {}

// pub fn run<'pgm>(program: &'pgm [Instruction]) -> Result<Vec<VMState<'pgm>>, Box<dyn Error>> {
pub fn parse(code: &str) -> Result<Program, Box<dyn Error>> {
    let mut tokens = code.split_whitespace();
    let mut program = vec![];

    while let Some(token) = tokens.next() {
        let instruction = parse_token(token, &mut tokens)?;
        program.push(instruction);
    }

    Ok(Program(program))
}

fn parse_token(token: &str, tokens: &mut SplitWhitespace) -> Result<Instruction, Box<dyn Error>> {
    let instruction = match token {
        "pop" => Pop,
        "push" => Push(parse_elem(tokens)?),
        "pad" => Pad,
        "dup0" => Dup(N0),
        "dup1" => Dup(N1),
        "dup2" => Dup(N2),
        "dup3" => Dup(N3),
        "swap" => Swap,
        "pull2" => Pull2,
        "pull3" => Pull3,
        "nop" => Nop,
        "skiz" => Skiz,
        "call" => Call(parse_elem(tokens)?),
        "return" => Return,
        "recurse" => Recurse,
        "assert" => Assert,
        "halt" => Halt,
        "load" => Load,
        "loadinc" => LoadInc,
        "loaddec" => LoadDec,
        "save" => Save,
        "saveinc" => SaveInc,
        "savedec" => SaveDec,
        "setramp" => SetRamp,
        "getramp" => GetRamp,
        "xlix" => Xlix,
        "clearall" => ClearAll,
        "squeeze" => Squeeze(parse_arg(tokens)?),
        "absorb" => Absorb(parse_arg(tokens)?),
        "clear" => Clear(parse_arg(tokens)?),
        "rotate" => Rotate(parse_arg(tokens)?),
        "add" => Add,
        "neg" => Neg,
        "mul" => Mul,
        "inv" => Inv,
        "lnot" => Lnot,
        "split" => Split,
        "eq" => Eq,
        "lt" => Lt,
        "and" => And,
        "or" => Or,
        "xor" => Xor,
        "reverse" => Reverse,
        "div" => Div,
        "print" => Print,
        "scan" => Scan,
        _ => Err(format!("Unknown token '{}'", token))?,
    };

    Ok(instruction)
}

fn parse_arg(tokens: &mut SplitWhitespace) -> Result<Ord16, Box<dyn Error>> {
    let constant_s = tokens.next().ok_or_else(|| UnexpectedEndOfStream)?;
    let constant_n = constant_s.parse::<usize>()?;
    let constant_arg = constant_n.try_into()?;

    Ok(constant_arg)
}

fn parse_elem(tokens: &mut SplitWhitespace) -> Result<BFieldElement, Box<dyn Error>> {
    let constant_s = tokens.next().ok_or_else(|| UnexpectedEndOfStream)?;
    let constant_n = constant_s.parse::<u64>()?;
    let constant_elem = BFieldElement::new(constant_n);

    Ok(constant_elem)
}
pub mod sample_programs {
    use super::{Instruction::*, Program};

    pub const PUSH_POP_S: &str = "
        push 1
        push 2
        pop
        pop
    ";

    pub fn push_pop_p() -> Program {
        Program(vec![Push(1.into()), Push(1.into()), Pop, Pop])
    }
}

#[cfg(test)]
mod instruction_tests {
    use super::parse;
    use super::sample_programs;

    #[test]
    fn parse_display_push_pop_test() {
        let foo = sample_programs::push_pop_p();
        let foo_s = format!("{}", foo);
        let foo_again = parse(&foo_s).unwrap();

        println!("{}", foo);
        println!("{}", foo_again);

        assert_eq!(foo, foo_again);

        let _bar = sample_programs::PUSH_POP_S;
    }
}
