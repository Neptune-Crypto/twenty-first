use super::ord_n::{Ord16, Ord16::*, Ord4, Ord4::*, Ord6};
use crate::shared_math::b_field_element::BFieldElement;
use num_traits::Zero;
use std::error::Error;
use std::fmt::Display;
use std::str::SplitWhitespace;
use Instruction::*;
use TokenError::*;

type BWord = BFieldElement;

/// A Triton VM instruction
///
/// The ISA is defined at:
///
/// https://neptune.builders/core-team/triton-vm/src/branch/master/specification/isa.md
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Instruction {
    // OpStack manipulation
    Pop,
    Push(BWord),
    Pad,
    Dup(Ord4),
    Swap(Ord4),

    // Control flow
    Skiz,
    Call(BWord),
    Return,
    Recurse,
    Assert,
    Halt,

    // Memory access
    ReadMem,
    WriteMem,

    // Auxiliary register instructions
    Xlix,
    ClearAll,
    Squeeze(Ord16),
    Absorb(Ord16),
    AssertDigest,

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
    XbMul,

    // Read/write
    ReadIo,
    WriteIo,
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // OpStack manipulation
            Pop => write!(f, "pop"),
            Push(arg) => write!(f, "push {}", {
                let n: u64 = arg.into();
                n
            }),
            Pad => write!(f, "pad"),
            Dup(arg) => write!(f, "dup{}", {
                let n: usize = arg.into();
                n
            }),
            Swap(arg) => write!(f, "swap{}", {
                let n: usize = arg.into();
                n + 1
            }),
            // Control flow
            Skiz => write!(f, "skiz"),
            Call(arg) => write!(f, "call {}", {
                let addr: u64 = arg.into();
                addr
            }),
            Return => write!(f, "return"),
            Recurse => write!(f, "recurse"),
            Assert => write!(f, "assert"),
            Halt => write!(f, "halt"),

            // Memory access
            ReadMem => write!(f, "read_mem"),
            WriteMem => write!(f, "write_mem"),

            // Auxiliary register instructions
            Xlix => write!(f, "xlix"),
            ClearAll => write!(f, "clearall"),
            Squeeze(arg) => write!(f, "squeeze{}", {
                let n: usize = arg.into();
                n
            }),
            Absorb(arg) => write!(f, "absorb{}", {
                let n: usize = arg.into();
                n
            }),
            AssertDigest => write!(f, "assert_digest"),

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
            XbMul => write!(f, "xbmul"),

            // Read/write
            ReadIo => write!(f, "read_io"),
            WriteIo => write!(f, "write_io"),
        }
    }
}

impl Instruction {
    /// Assign a unique positive integer to each `Instruction`.
    pub fn opcode(&self) -> u32 {
        match self {
            // OpStack manipulation
            Pop => 1,
            Push(_) => 2,
            Pad => 3,
            Dup(_) => 4,
            Swap(_) => 5,

            // Control flow
            Skiz => 10,
            Call(_) => 11,
            Return => 12,
            Recurse => 13,
            Assert => 14,
            Halt => 0,

            // Memory access
            ReadMem => 20,
            WriteMem => 21,

            // Auxiliary register instructions
            Xlix => 30,
            ClearAll => 31,
            Squeeze(_) => 32,
            Absorb(_) => 33,
            AssertDigest => 36,

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
            ReadIo => 70,
            WriteIo => 71,
        }
    }

    /// Returns whether a given instruction modifies the op-stack.
    ///
    /// A modification involves any amount of pushing and/or popping.
    pub fn is_op_stack_instruction(&self) -> bool {
        match self {
            Pad => false,
            Call(_) => false,
            Return => false,
            Recurse => false,
            Halt => false,
            Xlix => false,
            ClearAll => false,
            AssertDigest => false,

            Pop => true,
            Push(_) => true,
            Dup(_) => true,
            Swap(_) => true,
            Skiz => true,
            Assert => true,
            ReadMem => true,
            WriteMem => true,
            Squeeze(_) => true,
            Absorb(_) => true,
            Add => true,
            Mul => true,
            Inv => true,
            Split => true,
            Eq => true,
            Lt => true,
            And => true,
            Xor => true,
            Reverse => true,
            Div => true,
            XxAdd => true,
            XxMul => true,
            XInv => true,
            XbMul => true,
            ReadIo => true,
            WriteIo => true,
        }
    }

    pub fn opcode_b(&self) -> BFieldElement {
        self.opcode().into()
    }

    pub fn size(&self) -> usize {
        match self {
            // Double-word instructions (instructions that take arguments)
            Push(_) => 2,
            Dup(_) => 2,
            Swap(_) => 2,
            Call(_) => 2,
            Squeeze(_) => 2,
            Absorb(_) => 2,

            // Single-word instructions
            Pop => 1,
            Pad => 1,
            Skiz => 1,
            Return => 1,
            Recurse => 1,
            Assert => 1,
            Halt => 1,
            ReadMem => 1,
            WriteMem => 1,
            Xlix => 1,
            ClearAll => 1,
            AssertDigest => 1,
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
            WriteIo => 1,
            ReadIo => 1,
        }
    }

    /// Get the i'th instruction bit
    pub fn ib(&self, arg: Ord6) -> BFieldElement {
        let opcode = self.opcode();
        let bit_number: usize = arg.into();
        let bit_mask: u32 = 1 << bit_number;

        (opcode & bit_mask).into()
    }

    pub fn hv(&self, bit: Ord4) -> BFieldElement {
        match self {
            Dup(arg) => hv_calc(arg.into(), bit.into()),
            Swap(arg) => hv_calc(arg.into(), bit.into()),
            Squeeze(arg) => hv_calc(arg.into(), bit.into()),
            Absorb(arg) => hv_calc(arg.into(), bit.into()),
            _ => BFieldElement::ring_zero(),
        }
    }

    pub fn arg(&self) -> Option<BFieldElement> {
        match self {
            // Double-word instructions (instructions that take arguments)
            Push(arg) => Some(*arg),
            Dup(arg) => Some(ord4_to_bfe(arg)),
            Swap(arg) => Some(ord4_to_bfe(arg)),
            Call(arg) => Some(*arg),
            Squeeze(arg) => Some(ord16_to_bfe(arg)),
            Absorb(arg) => Some(ord16_to_bfe(arg)),
            _ => None,
        }
    }
}

fn hv_calc(opcode_arg: u32, bit_number: usize) -> BFieldElement {
    let bit_mask: u32 = 1 << bit_number;
    if (opcode_arg & bit_mask).is_zero() {
        BWord::ring_zero()
    } else {
        BWord::ring_one()
    }
}

fn ord4_to_bfe(n: &Ord4) -> BFieldElement {
    let n: u32 = n.into();
    n.into()
}

fn ord16_to_bfe(n: &Ord16) -> BFieldElement {
    let n: u32 = n.into();
    n.into()
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

pub fn parse(code: &str) -> Result<Vec<Instruction>, Box<dyn Error>> {
    let mut tokens = code.split_whitespace();
    let mut instructions = vec![];

    while let Some(token) = tokens.next() {
        let mut instruction = parse_token(token, &mut tokens)?;
        instructions.append(&mut instruction);
    }

    Ok(instructions)
}

fn parse_token(
    token: &str,
    tokens: &mut SplitWhitespace,
) -> Result<Vec<Instruction>, Box<dyn Error>> {
    let instruction = match token {
        // OpStack manipulation
        "pop" => vec![Pop],
        "push" => vec![Push(parse_elem(tokens)?)],
        "pad" => vec![Pad],
        "dup0" => vec![Dup(N0)],
        "dup1" => vec![Dup(N1)],
        "dup2" => vec![Dup(N2)],
        "dup3" => vec![Dup(N3)],
        "swap1" => vec![Swap(N0)],
        "swap2" => vec![Swap(N1)],
        "swap3" => vec![Swap(N2)],
        "swap4" => vec![Swap(N3)],

        // Control flow
        "skiz" => vec![Skiz],
        "call" => vec![Call(parse_elem(tokens)?)],
        "return" => vec![Return],
        "recurse" => vec![Recurse],
        "assert" => vec![Assert],
        "halt" => vec![Halt],

        // Memory access
        "read_mem" => vec![ReadMem],
        "write_mem" => vec![WriteMem],

        // Auxiliary register instructions
        "xlix" => vec![Xlix],
        "clearall" => vec![ClearAll],
        "squeeze0" => vec![Squeeze(A0)],
        "squeeze1" => vec![Squeeze(A1)],
        "squeeze2" => vec![Squeeze(A2)],
        "squeeze3" => vec![Squeeze(A3)],
        "squeeze4" => vec![Squeeze(A4)],
        "squeeze5" => vec![Squeeze(A5)],
        "squeeze6" => vec![Squeeze(A6)],
        "squeeze7" => vec![Squeeze(A7)],
        "squeeze8" => vec![Squeeze(A8)],
        "squeeze9" => vec![Squeeze(A9)],
        "squeeze10" => vec![Squeeze(A10)],
        "squeeze11" => vec![Squeeze(A11)],
        "squeeze12" => vec![Squeeze(A12)],
        "squeeze13" => vec![Squeeze(A13)],
        "squeeze14" => vec![Squeeze(A14)],
        "squeeze15" => vec![Squeeze(A15)],
        "absorb0" => vec![Absorb(A0)],
        "absorb1" => vec![Absorb(A1)],
        "absorb2" => vec![Absorb(A2)],
        "absorb3" => vec![Absorb(A3)],
        "absorb4" => vec![Absorb(A4)],
        "absorb5" => vec![Absorb(A5)],
        "absorb6" => vec![Absorb(A6)],
        "absorb7" => vec![Absorb(A7)],
        "absorb8" => vec![Absorb(A8)],
        "absorb9" => vec![Absorb(A9)],
        "absorb10" => vec![Absorb(A10)],
        "absorb11" => vec![Absorb(A11)],
        "absorb12" => vec![Absorb(A12)],
        "absorb13" => vec![Absorb(A13)],
        "absorb14" => vec![Absorb(A14)],
        "absorb15" => vec![Absorb(A15)],
        "assert_digest" => vec![AssertDigest],

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
        "xxmul" => vec![XxMul],
        "xinv" => vec![XInv],
        "xbmul" => vec![XbMul],

        // Read/write
        "read_io" => vec![ReadIo],
        "write_io" => vec![WriteIo],

        _ => return Err(Box::new(UnknownInstruction(token.to_string()))),
    };

    Ok(instruction)
}

fn _parse_arg(tokens: &mut SplitWhitespace) -> Result<Ord16, Box<dyn Error>> {
    let constant_s = tokens.next().ok_or(UnexpectedEndOfStream)?;
    let constant_n = constant_s.parse::<usize>()?;
    let constant_arg = constant_n.try_into()?;

    Ok(constant_arg)
}

fn parse_elem(tokens: &mut SplitWhitespace) -> Result<BFieldElement, Box<dyn Error>> {
    let constant_s = tokens.next().ok_or(UnexpectedEndOfStream)?;

    let mut constant_n128: i128 = constant_s.parse::<i128>()?;
    if constant_n128 < 0 {
        constant_n128 += BFieldElement::QUOTIENT as i128;
    }
    let constant_n64: u64 = constant_n128.try_into()?;
    let constant_elem = BFieldElement::new(constant_n64);

    Ok(constant_elem)
}
pub mod sample_programs {
    use super::super::vm::Program;
    use super::Instruction::{self, *};
    use super::{Ord16::*, Ord4::*};

    pub const PUSH_PUSH_ADD_POP_S: &str = "
        push 1
        push 2
        add
        pop
    ";

    pub fn push_push_add_pop_p() -> Program {
        let instructions = vec![Push(1.into()), Push(2.into()), Add, Pop];
        Program::new(&instructions)
    }

    pub const HELLO_WORLD_1: &str = "
        push 10
        push 33
        push 100
        push 108
        push 114
        push 111
        push 87
        push 32
        push 44
        push 111
        push 108
        push 108
        push 101
        push 72

        write_io write_io write_io write_io write_io write_io write_io
        write_io write_io write_io write_io write_io write_io write_io
        ";

    pub const WRITE_42: &str = "
        push 42
        write_io
    ";

    pub const READ_WRITE_X3: &str = "
        read_io
        read_io
        read_io
        write_io
        write_io
        write_io
    ";

    pub const SUBTRACT: &str = "
        push 5
        push 18446744069414584320
        add
        halt
    ";

    pub const COUNTDOWN_FROM_10: &str = "
        push 10
        call 4
        push 18446744069414584320
        add
        dup0
        skiz
        recurse
        halt
    ";

    // leave the stack with the n first fibonacci numbers.  f_0 = 0; f_1 = 1
    // buttom-up approach
    pub const FIBONACCI_SOURCE: &str = "
    push 0
    push 1
    push n=6
    -- case: n==0 || n== 1
    dup0
    dup0
    dup0
    mul
    eq
    skiz
    call $basecase
    -- case: n>1
    call $nextline
    call $fib
    swap1 - n on top
    push 18446744069414584320
    add
    skiz
    recurse
    call $basecase
    dup0     :basecase
    push 0
    eq
    skiz
    pop
    pop - remove 1      :endone
    halt
";

    pub const FIBONACCI_VIT: &str = "
    push 0
    push 1
    push 7
    dup0
    dup0
    dup0
    mul
    eq
    skiz
    call 32
    call 19
    call 41
    swap1
    push 18446744069414584320
    add
    dup0
    skiz
    recurse
    call 39
    dup0
    push 0
    eq
    skiz
    pop
    pop
    halt
    dup2
    dup2
    add
    return
";

    pub const FIBONACCI_LT: &str = "
        push 0
        push 1
        push 7
        dup0
        push 2
        lt
        skiz
        call 29
        call 16
        call 38
        swap1
        push 18446744069414584320
        add
        dup0
        skiz
        recurse
        call 36
        dup0
        push 0
        eq
        skiz
        pop
        pop
        halt
        dup2
        dup2
        add
        return
    ";

    pub const GCD_X_Y: &str = "
        read_io
        read_io
        dup1
        dup1
        lt
        skiz
        swap1
        dup0
        push 0
        eq
        skiz
        call 31
        dup1
        dup1
        div
        swap2
        swap3
        pop
        pop
        call 10
        pop
        write_io
        halt
    ";

    //This cannot not print because we need to itoa() before write_io.
    pub const XGCD: &str = "
        push 1
        push 0
        push 0
        push 1
        push 240
        push 46
        dup1
        dup1
        lt
        skiz
        swap1
        dup0
        push 0
        eq
        skiz
        call 33
        dup1
        dup1
        div
        swap2
        swap3
        pop
        pop
        call 12
        pop
        halt
    ";

    pub const ALL_INSTRUCTIONS: &str = "
        pop push 42 pad dup0 dup1 dup2 dup3 swap1 swap2 swap3 swap4 skiz
        call 0 return recurse assert halt read_mem write_mem xlix clearall
        squeeze0 squeeze1 squeeze2 squeeze3 squeeze4 squeeze5 squeeze6
        squeeze7 squeeze8 squeeze9 squeeze10 squeeze11 squeeze12 squeeze13
        squeeze14 squeeze15 absorb0 absorb1 absorb2 absorb3 absorb4 absorb5
        absorb6 absorb7 absorb8 absorb9 absorb10 absorb11 absorb12 absorb13
        absorb14 absorb15 assert_digest add mul inv split eq lt and xor reverse
        div xxadd xxmul xinv xbmul read_io write_io
    ";

    pub fn all_instructions() -> Vec<Instruction> {
        vec![
            Pop,
            Push(42.into()),
            Pad,
            Dup(N0),
            Dup(N1),
            Dup(N2),
            Dup(N3),
            Swap(N0),
            Swap(N1),
            Swap(N2),
            Swap(N3),
            Skiz,
            Call(0.into()),
            Return,
            Recurse,
            Assert,
            Halt,
            ReadMem,
            WriteMem,
            Xlix,
            ClearAll,
            Squeeze(A0),
            Squeeze(A1),
            Squeeze(A2),
            Squeeze(A3),
            Squeeze(A4),
            Squeeze(A5),
            Squeeze(A6),
            Squeeze(A7),
            Squeeze(A8),
            Squeeze(A9),
            Squeeze(A10),
            Squeeze(A11),
            Squeeze(A12),
            Squeeze(A13),
            Squeeze(A14),
            Squeeze(A15),
            Absorb(A0),
            Absorb(A1),
            Absorb(A2),
            Absorb(A3),
            Absorb(A4),
            Absorb(A5),
            Absorb(A6),
            Absorb(A7),
            Absorb(A8),
            Absorb(A9),
            Absorb(A10),
            Absorb(A11),
            Absorb(A12),
            Absorb(A13),
            Absorb(A14),
            Absorb(A15),
            AssertDigest,
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
            XbMul,
            ReadIo,
            WriteIo,
        ]
    }

    pub fn all_instructions_displayed() -> Vec<String> {
        vec![
            "pop",
            "push 42",
            "pad",
            "dup0",
            "dup1",
            "dup2",
            "dup3",
            "swap1",
            "swap2",
            "swap3",
            "swap4",
            "skiz",
            "call 0",
            "return",
            "recurse",
            "assert",
            "halt",
            "read_mem",
            "write_mem",
            "xlix",
            "clearall",
            "squeeze0",
            "squeeze1",
            "squeeze2",
            "squeeze3",
            "squeeze4",
            "squeeze5",
            "squeeze6",
            "squeeze7",
            "squeeze8",
            "squeeze9",
            "squeeze10",
            "squeeze11",
            "squeeze12",
            "squeeze13",
            "squeeze14",
            "squeeze15",
            "absorb0",
            "absorb1",
            "absorb2",
            "absorb3",
            "absorb4",
            "absorb5",
            "absorb6",
            "absorb7",
            "absorb8",
            "absorb9",
            "absorb10",
            "absorb11",
            "absorb12",
            "absorb13",
            "absorb14",
            "absorb15",
            "assert_digest",
            "add",
            "mul",
            "inv",
            "split",
            "eq",
            "lt",
            "and",
            "xor",
            "reverse",
            "div",
            "xxadd",
            "xxmul",
            "xinv",
            "xbmul",
            "read_io",
            "write_io",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }
}

#[cfg(test)]
mod instruction_tests {
    use super::super::vm::Program;
    use super::parse;
    use super::sample_programs;

    #[test]
    fn parse_display_push_pop_test() {
        let pgm_expected = sample_programs::push_push_add_pop_p();
        let pgm_pretty = format!("{}", pgm_expected);
        let instructions = parse(&pgm_pretty).unwrap();
        let pgm_actual = Program::new(&instructions);

        println!("Expected:\n{}", pgm_expected);
        println!("Actual:\n{}", pgm_actual);

        assert_eq!(pgm_expected, pgm_actual);

        let pgm_text = sample_programs::PUSH_PUSH_ADD_POP_S;
        let instructions_2 = parse(&pgm_text).unwrap();
        let pgm_actual_2 = Program::new(&instructions_2);

        assert_eq!(pgm_expected, pgm_actual_2);
    }

    #[test]
    fn parse_and_display_each_instruction_test() {
        let all_instructions = parse(sample_programs::ALL_INSTRUCTIONS);
        assert!(all_instructions.is_ok());

        let actual = all_instructions.unwrap();
        let expected = sample_programs::all_instructions();

        assert_eq!(expected, actual);

        for (actual, expected) in actual
            .iter()
            .map(|instr| format!("{}", instr))
            .zip(sample_programs::all_instructions_displayed())
        {
            assert_eq!(expected, actual);
        }
    }
}
