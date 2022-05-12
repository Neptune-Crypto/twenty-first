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
    Push,
    PushArg(Word),
    Pad,
    Dup,
    DupArg(Ord4),
    Swap,
    SwapArg(Ord4),
    // Control flow
    Skiz,
    Call,
    CallArg(Word),
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
    Squeeze,
    SqueezeArg(Ord16),
    Absorb,
    AbsorbArg(Ord16),
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

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // OpStack manipulation
            Pop => write!(f, "pop"),
            Push => write!(f, "push"),
            Pad => write!(f, "pad"),
            Dup => write!(f, "dup"),
            Swap => write!(f, "swap"),

            // Control flow
            Skiz => write!(f, "skiz"),
            Call => write!(f, "call"),
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
            Squeeze => write!(f, "squeeze"),
            Absorb => write!(f, "absorb"),
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

            PushArg(arg) => {
                let n: u64 = arg.into();
                write!(f, "{}", n)
            }

            DupArg(arg) => {
                let n: usize = arg.into();
                write!(f, "{}", n)
            }

            SwapArg(arg) => {
                let n: usize = arg.into();
                write!(f, "{}", n)
            }

            CallArg(arg) => {
                let n: u64 = arg.into();
                write!(f, "{}", n)
            }

            SqueezeArg(arg) => {
                let n: usize = arg.into();
                write!(f, "{}", n)
            }

            AbsorbArg(arg) => {
                let n: usize = arg.into();
                write!(f, "{}", n)
            }
        }
    }
}

impl Instruction {
    /// Assign a unique positive integer to each `Instruction`.
    pub fn opcode(&self) -> Option<u32> {
        let value = match self {
            // OpStack manipulation
            Pop => 1,
            Push => 2,
            Pad => 3,
            Dup => 4,
            Swap => 5,

            // Control flow
            Skiz => 10,
            Call => 11,
            Return => 12,
            Recurse => 13,
            Assert => 14,
            Halt => 0,

            // Memory access
            Load => 20,
            Save => 21,

            // Auxiliary register instructions
            Xlix => 30,
            ClearAll => 31,
            Squeeze => 32,
            Absorb => 33,
            MerkleLeft => 34,
            MerkleRight => 35,
            CmpDigest => 36,

            // Arithmetic on stack instructions
            Add => 40,
            Mul => 41,
            Inv => 42,
            Split => 43,
            Eq => 44,
            Lt => 45,
            And => 46,
            Xor => 47,
            Reverse => 48,
            Div => 49,

            XxAdd => 50,
            XxMul => 51,
            XInv => 52,
            XbMul => 53,

            // Read/write
            Print => 70,
            Scan => 71,

            PushArg(_) => return None,
            DupArg(_) => return None,
            SwapArg(_) => return None,
            CallArg(_) => return None,
            SqueezeArg(_) => return None,
            AbsorbArg(_) => return None,
        };

        Some(value)
    }

    pub fn size(&self) -> usize {
        match self {
            Push => 2,
            Dup => 2,
            Swap => 2,
            Call => 2,
            Squeeze => 2,
            Absorb => 2,

            Pop => 1,
            Pad => 1,
            Skiz => 1,
            Return => 1,
            Recurse => 1,
            Assert => 1,
            Halt => 1,
            Load => 1,
            Save => 1,
            Xlix => 1,
            ClearAll => 1,
            MerkleLeft => 1,
            MerkleRight => 1,
            CmpDigest => 1,
            Add => 1,
            Mul => 1,
            Inv => 1,
            Split => 1,
            Eq => 1,
            Lt => 1,
            And => 1,
            Xor => 1,
            Reverse => 1,
            Div => 1,
            XxAdd => 1,
            XxMul => 1,
            XInv => 1,
            XbMul => 1,
            Print => 1,
            Scan => 1,

            PushArg(_) => 0,
            DupArg(_) => 0,
            SwapArg(_) => 0,
            CallArg(_) => 0,
            SqueezeArg(_) => 0,
            AbsorbArg(_) => 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    pub instructions: Vec<Instruction>,
}

impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // FIXME: Print arguments to multi-word instructions nicely after.
        let mut iterator = self.instructions.iter();
        loop {
            let item = iterator.next();
            if item.is_none() {
                return Ok(());
            }

            let item = item.unwrap();

            match item {
                Push => writeln!(f, "{} {}", item, iterator.next().unwrap())?,
                Call => writeln!(f, "{} {}", item, iterator.next().unwrap())?,

                Dup => writeln!(f, "{}{}", item, iterator.next().unwrap())?,
                Swap => writeln!(f, "{}{}", item, iterator.next().unwrap())?,
                Squeeze => writeln!(f, "{}{}", item, iterator.next().unwrap())?,
                Absorb => writeln!(f, "{}{}", item, iterator.next().unwrap())?,

                instr => writeln!(f, "{}", instr)?,
            }
        }
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
        let mut instruction = parse_token(token, &mut tokens)?;
        instructions.append(&mut instruction);
    }

    Ok(Program { instructions })
}

fn parse_token(
    token: &str,
    tokens: &mut SplitWhitespace,
) -> Result<Vec<Instruction>, Box<dyn Error>> {
    let instruction = match token {
        // OpStack manipulation
        "pop" => vec![Pop],
        "push" => vec![Push, PushArg(parse_elem(tokens)?)],
        "pad" => vec![Pad],
        "dup1" => vec![Dup, DupArg(N0)],
        "dup2" => vec![Dup, DupArg(N1)],
        "dup3" => vec![Dup, DupArg(N2)],
        "dup4" => vec![Dup, DupArg(N3)],
        "swap1" => vec![Swap, SwapArg(N0)],
        "swap2" => vec![Swap, SwapArg(N1)],
        "swap3" => vec![Swap, SwapArg(N2)],
        "swap4" => vec![Swap, SwapArg(N3)],
        // Control flow
        "skiz" => vec![Skiz],
        "call" => vec![Call, CallArg(parse_elem(tokens)?)],
        "return" => vec![Return],
        "recurse" => vec![Recurse],
        "assert" => vec![Assert],
        "halt" => vec![Halt],
        // Memory access
        "load" => vec![Load],
        "save" => vec![Save],

        // Auxiliary register instructions
        "xlix" => vec![Xlix],
        "clearall" => vec![ClearAll],
        "squeeze" => vec![Squeeze, SqueezeArg(parse_arg(tokens)?)],
        "absorb" => vec![Absorb, AbsorbArg(parse_arg(tokens)?)],
        "merkle_left" => vec![MerkleLeft],
        "merkle_right" => vec![MerkleRight],
        "cmp_digest" => vec![CmpDigest],

        // Arithmetic on stack instructions
        "add" => vec![Add],
        "mul" => vec![Mul],
        "inv" => vec![Inv],
        "split" => vec![Split],
        "eq" => vec![Eq],
        "lt" => vec![Lt],
        "and" => vec![And],
        "xor" => vec![Xor],
        "reverse" => vec![Reverse],
        "div" => vec![Div],
        "xxadd" => vec![XxAdd],
        "xxmul" => vec![XsMul],
        "xinv" => vec![XInv],
        "XsMul" => vec![XsMul],
        // Read/write
        "print" => vec![Print],
        "scan" => vec![Scan],
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

    pub const PUSH_PUSH_ADD_POP_S: &str = "
        push 1
        push 2
        add
        pop
    ";

    pub fn push_push_add_pop_p() -> Program {
        let instructions = vec![Push, PushArg(1.into()), Push, PushArg(2.into()), Add, Pop];
        Program { instructions }
    }
}

#[cfg(test)]
mod instruction_tests {
    use super::parse;
    use super::sample_programs;

    #[test]
    fn parse_display_push_pop_test() {
        let pgm_expected: crate::shared_math::stark::triton::instruction::Program =
            sample_programs::push_push_add_pop_p();
        let pgm_pretty = format!("{}", pgm_expected);
        let pgm_actual = parse(&pgm_pretty).unwrap();

        println!("Expected:\n{}", pgm_expected);
        println!("Actual:\n{}", pgm_actual);

        assert_eq!(pgm_expected, pgm_actual);

        let pgm_text = sample_programs::PUSH_PUSH_ADD_POP_S;
        let pgm_actual_2 = parse(pgm_text).unwrap();

        assert_eq!(pgm_expected, pgm_actual_2);
    }
}
