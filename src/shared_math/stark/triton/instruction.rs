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
    Swap(Ord4),
    // Control flow
    Skiz,
    Call(Word),
    Return,
    Recurse,
    Assert,
    Halt,
    // Memory access
    Load,
    Save,
    // Auxiliary register instructions
    Xlix,
    ClearAll,
    Squeeze(Ord16),
    Absorb(Ord16),
    MerkleLeft,
    MerkleRight,
    CmpDigest,
    // Arithmetic on stack instructions
    Add,
    Mul,
    Inv,
    Split,
    Eq,
    Lt,
    And,
    Xor,
    Reverse,
    Div,
    XxAdd,
    XxMul,
    XInv,
    XsMul,
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
            // OpStack manipulation
            Pop => write!(f, "pop"),
            Push(elem) => write!(f, "push {}", elem),
            Pad => write!(f, "pad"),
            Dup(arg) => write!(f, "dup{}", {
                let n: usize = arg.into();
                n + 1
            }),
            Swap(arg) => write!(f, "swap{}", {
                let n: usize = arg.into();
                n + 1
            }),
            // Control flow
            Skiz => write!(f, "skiz"),
            Call(addr) => write!(f, "call {}", addr),
            Return => write!(f, "return"),
            Recurse => write!(f, "recurse"),
            Assert => write!(f, "assert"),
            Halt => write!(f, "halt"),
            // Memory access
            Load => write!(f, "load"),
            Save => write!(f, "save"),
            // Auxiliary register instructions
            Xlix => write!(f, "xlix"),
            ClearAll => write!(f, "clearall"),
            Squeeze(arg) => write!(f, "squeeze {}", arg),
            Absorb(arg) => write!(f, "absorb {}", arg),
            MerkleLeft => write!(f, "merkle_left"),
            MerkleRight => write!(f, "merkle_right"),
            CmpDigest => write!(f, "cmp_digest"),
            // Arithmetic on stack instructions
            Add => write!(f, "add"),
            Mul => write!(f, "mul"),
            Inv => write!(f, "inv"),
            Split => write!(f, "split"),
            Eq => write!(f, "eq"),
            Lt => write!(f, "lt"),
            And => write!(f, "and"),
            Xor => write!(f, "xor"),
            Reverse => write!(f, "reverse"),
            Div => write!(f, "div"),
            XxAdd => write!(f, "xxadd"),
            XxMul => write!(f, "xxmul"),
            XInv => write!(f, "xinv"),
            XsMul => write!(f, "xsmul"),
            // Read/write
            Print => write!(f, "print"),
            Scan => write!(f, "scan"),
        }
    }
}

impl Instruction {
    /// Assign a unique positive integer to each `Instruction`.
    pub fn value(&self) -> u32 {
        match self {
            // OpStack manipulation

            // FIXME: Halt is actually 0 for polynomials to work.
            Pop => 0,
            Push(_) => 1,
            Pad => 2,
            Dup(N0) => 3,
            Dup(N1) => 4,
            Dup(N2) => 5,
            Dup(N3) => 6,
            Swap(N0) => 7,
            Swap(N1) => 8,
            Swap(N2) => 9,
            Swap(N3) => 10,
            // Control flow
            Skiz => 21,
            Call(_) => 22,
            Return => 23,
            Recurse => 24,
            Assert => 25,
            Halt => 26,
            // Memory access
            Load => 30,
            Save => 31,
            // Auxiliary register instructions
            Xlix => 40,
            ClearAll => 41,
            Squeeze(_) => 42,
            Absorb(_) => 43,
            MerkleLeft => 44,
            MerkleRight => 45,
            CmpDigest => 46,
            // Arithmetic on stack instructions
            Add => 50,
            Mul => 51,
            Inv => 52,
            Split => 53,
            Eq => 54,
            Lt => 55,
            And => 56,
            Xor => 57,
            Reverse => 58,
            Div => 59,
            XxAdd => 60,
            XxMul => 61,
            XInv => 62,
            XsMul => 63,
            // Read/write
            Print => 70,
            Scan => 71,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    pub instructions: Vec<Instruction>,
}

impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.instructions
            .iter()
            .fold(Ok(()), |result, instruction| {
                result.and_then(|()| writeln!(f, "{}", instruction))
            })
    }
}

#[derive(Debug)]
pub enum TokenError {
    UnexpectedEndOfStream,
    UnknownInstruction(String),
}

impl Display for TokenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnknownInstruction(s) => write!(f, "UnknownInstruction({})", s),
            UnexpectedEndOfStream => write!(f, "UnexpectedEndOfStream"),
        }
    }
}

impl Error for TokenError {}

pub fn parse(code: &str) -> Result<Program, Box<dyn Error>> {
    let mut tokens = code.split_whitespace();
    let mut instructions = vec![];

    while let Some(token) = tokens.next() {
        instructions.push(parse_token(token, &mut tokens)?);
    }

    Ok(Program { instructions })
}

fn parse_token(token: &str, tokens: &mut SplitWhitespace) -> Result<Instruction, Box<dyn Error>> {
    let instruction = match token {
        // OpStack manipulation
        "pop" => Pop,
        "push" => Push(parse_elem(tokens)?),
        "pad" => Pad,
        "dup1" => Dup(N0),
        "dup2" => Dup(N1),
        "dup3" => Dup(N2),
        "dup4" => Dup(N3),
        "swap1" => Swap(N0),
        "swap2" => Swap(N1),
        "swap3" => Swap(N2),
        "swap4" => Swap(N3),
        // Control flow
        "skiz" => Skiz,
        "call" => Call(parse_elem(tokens)?),
        "return" => Return,
        "recurse" => Recurse,
        "assert" => Assert,
        "halt" => Halt,
        // Memory access
        "load" => Load,
        "save" => Save,
        // Auxiliary register instructions
        "xlix" => Xlix,
        "clearall" => ClearAll,
        "squeeze" => Squeeze(parse_arg(tokens)?),
        "absorb" => Absorb(parse_arg(tokens)?),
        "merkle_left" => MerkleLeft,
        "merkle_right" => MerkleRight,
        "cmp_digest" => CmpDigest,
        // Arithmetic on stack instructions
        "add" => Add,
        "mul" => Mul,
        "inv" => Inv,
        "split" => Split,
        "eq" => Eq,
        "lt" => Lt,
        "and" => And,
        "xor" => Xor,
        "reverse" => Reverse,
        "div" => Div,
        "xxadd" => XxAdd,
        "xxmul" => XsMul,
        "xinv" => XInv,
        "XsMul" => XsMul,
        // Read/write
        "print" => Print,
        "scan" => Scan,
        _ => return Err(Box::new(UnknownInstruction(token.to_string()))),
    };

    Ok(instruction)
}

fn parse_arg(tokens: &mut SplitWhitespace) -> Result<Ord16, Box<dyn Error>> {
    let constant_s = tokens.next().ok_or(UnexpectedEndOfStream)?;
    let constant_n = constant_s.parse::<usize>()?;
    let constant_arg = constant_n.try_into()?;

    Ok(constant_arg)
}

fn parse_elem(tokens: &mut SplitWhitespace) -> Result<BFieldElement, Box<dyn Error>> {
    let constant_s = tokens.next().ok_or(UnexpectedEndOfStream)?;
    let constant_n = constant_s.parse::<u64>()?;
    let constant_elem = BFieldElement::new(constant_n);

    Ok(constant_elem)
}
pub mod sample_programs {
    use super::{Instruction::*, Program};

    pub const PUSH_POP_S: &str = "
        push 1
        push 2
        add
        pop
    ";

    pub fn push_pop_p() -> Program {
        let instructions = vec![Push(1.into()), Push(2.into()), Add, Pop];
        Program { instructions }
    }
}

#[cfg(test)]
mod instruction_tests {
    use super::parse;
    use super::sample_programs;

    #[test]
    fn parse_display_push_pop_test() {
        let pgm_expected = sample_programs::push_pop_p();
        let pgm_pretty = format!("{}", pgm_expected);
        let pgm_actual = parse(&pgm_pretty).unwrap();

        println!("Expected:\n{}", pgm_expected);
        println!("Actual:\n{}", pgm_actual);

        assert_eq!(pgm_expected, pgm_actual);

        let pgm_text = sample_programs::PUSH_POP_S;
        let pgm_actual_2 = parse(pgm_text).unwrap();

        assert_eq!(pgm_expected, pgm_actual_2);
    }
}
