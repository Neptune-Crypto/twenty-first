use super::error::{vm_fail, InstructionError::*};
use super::instruction::Ord4;
use crate::shared_math::b_field_element::BFieldElement;
use std::error::Error;

type BWord = BFieldElement;

#[derive(Debug, Clone)]
pub struct OpStack {
    stack: Vec<BWord>,
}

pub const OP_STACK_MIN: usize = 4;

impl Default for OpStack {
    fn default() -> Self {
        Self {
            stack: vec![0.into(); OP_STACK_MIN],
        }
    }
}

impl OpStack {
    pub fn push(&mut self, elem: BWord) {
        self.stack.push(elem);
    }

    pub fn pop(&mut self) -> Result<BFieldElement, Box<dyn Error>> {
        self.stack.pop().ok_or_else(|| vm_fail(OpStackTooShallow))
    }

    pub fn safe_peek(&self, arg: Ord4) -> BWord {
        let n: usize = arg.into();
        self.stack[n]
    }

    pub fn safe_swap(&mut self, a: Ord4, b: Ord4) {
        let n: usize = a.into();
        let m: usize = b.into();
        self.stack.swap(n, m);
    }

    pub fn is_too_shallow(&self) -> bool {
        self.stack.len() < OP_STACK_MIN
    }

    /// Get the i'th op-stack element

    pub fn st(&self, arg: Ord4) -> BWord {
        let n: usize = arg.into();
        self.stack[n]
    }
}
