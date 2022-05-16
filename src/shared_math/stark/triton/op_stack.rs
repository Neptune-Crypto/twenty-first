use super::error::{vm_fail, InstructionError::*};
use super::ord_n::{Ord8, Ord8::*};
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::traits::Inverse;
use crate::shared_math::x_field_element::XFieldElement;
use std::error::Error;

type BWord = BFieldElement;
type XWord = XFieldElement;

#[derive(Debug, Clone)]
pub struct OpStack {
    stack: Vec<BWord>,
}

/// The number of op-stack registers, and the internal index at which the
/// op-stack memory has index 0. This offset is used to adjust for the fact
/// that op-stack registers are stored in the same way as op-stack memory.
pub const OP_STACK_REG_COUNT: usize = 8;

impl Default for OpStack {
    fn default() -> Self {
        Self {
            stack: vec![0.into(); OP_STACK_REG_COUNT],
        }
    }
}

impl OpStack {
    pub fn push(&mut self, elem: BWord) {
        self.stack.push(elem);
    }

    pub fn pushx(&mut self, elem: XWord) {
        self.push(elem.coefficients[2]);
        self.push(elem.coefficients[1]);
        self.push(elem.coefficients[0]);
    }

    pub fn pop(&mut self) -> Result<BWord, Box<dyn Error>> {
        self.stack.pop().ok_or_else(|| vm_fail(OpStackTooShallow))
    }

    pub fn popx(&mut self) -> Result<XWord, Box<dyn Error>> {
        Ok(XWord::new([self.pop()?, self.pop()?, self.pop()?]))
    }

    pub fn safe_peek(&self, arg: Ord8) -> BWord {
        let n: usize = arg.into();
        let top = self.stack.len() - 1;
        self.stack[top - n]
    }

    pub fn safe_swap(&mut self, arg: Ord8) {
        let n: usize = arg.into();
        let top = self.stack.len() - 1;
        self.stack.swap(top, top - n + 1);
    }

    pub fn peek(&self, n: usize) -> Option<BWord> {
        let top = self.stack.len() - 1;
        self.stack.get(top - n).copied()
    }

    pub fn height(&self) -> usize {
        self.stack.len()
    }

    pub fn is_too_shallow(&self) -> bool {
        self.stack.len() < OP_STACK_REG_COUNT
    }

    /// Get the arg'th op-stack register value
    pub fn st(&self, arg: Ord8) -> BWord {
        let n: usize = arg.into();
        let top = self.stack.len() - 1;
        self.stack[top - n]
    }

    /// Operational stack pointer
    ///
    /// The first value in op-stack memory (that is, beyond the op-stack registers).
    ///
    /// Assumed to be 0 when op-stack memory is empty.
    pub fn osp(&self) -> BWord {
        if self.stack.len() <= OP_STACK_REG_COUNT {
            0.into()
        } else {
            let offset = OP_STACK_REG_COUNT + 1;
            let n = self.stack.len() - offset;
            BWord::new(n as u64)
        }
    }

    /// Operational stack value
    ///
    /// Assumed to be 0 when op-stack memory is empty.
    pub fn osv(&self) -> BWord {
        if self.stack.len() <= OP_STACK_REG_COUNT {
            0.into()
        } else {
            let n = self.stack.len() - OP_STACK_REG_COUNT;
            self.stack.get(n).copied().unwrap_or_else(|| 0.into())
        }
    }

    /// Inverse of st0, with the inverse of 0 being 0 instead of failure.
    ///
    /// This register is mainly intended for constraint polynomials.
    pub fn inv(&self) -> BWord {
        self.st(ST0).inverse()
    }
}
