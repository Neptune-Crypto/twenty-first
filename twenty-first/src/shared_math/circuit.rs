//! Constraint circuits are a way to represent constraint polynomials in a way that is amenable
//! to optimizations. The constraint circuit is a directed acyclic graph (DAG) of
//! [`CircuitExpression`]s, where each `CircuitExpression` is a node in the graph. The edges of the
//! graph are labeled with [`BinOp`]s. The leafs of the graph are the inputs to the constraint
//! polynomial, and the (multiple) roots of the graph are the outputs of all the
//! constraint polynomials, with each root corresponding to a different constraint polynomial.
//! Because the graph has multiple roots, it is called a “multitree.”

use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::iter::Sum;
use std::ops::Add;
use std::ops::Mul;
use std::ops::Sub;
use std::rc::Rc;

use num_traits::One;
use num_traits::WrappingAdd;
use num_traits::WrappingMul;
use num_traits::WrappingSub;
use num_traits::Zero;

use CircuitExpression::*;

#[derive(Debug, Clone, Copy, PartialEq, Hash, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
}

impl Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
        }
    }
}

/// A circuit expression is a recursive data structure.
/// It is a directed, acyclic graph of binary operations on the input variables.
/// It has multiple roots, making it a “multitree.”
///
/// The leafs of the tree are
/// - constants (T)
/// - input variables
///
/// An inner node, representing some binary operation, is either addition, multiplication, or
/// subtraction. The left and right children of the node are the operands of the binary operation.
/// The left and right children are not themselves `CircuitExpression`s, but rather
/// [`Circuit`]s, which is a wrapper around `CircuitExpression` that manages additional
/// bookkeeping information.
#[derive(Debug, Clone)]
pub enum CircuitExpression<
    T: Clone + Debug + Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Zero + One + Hash + Eq,
> {
    Constant(T),
    Input(usize),
    BinaryOperation(BinOp, Rc<RefCell<Circuit<T>>>, Rc<RefCell<Circuit<T>>>),
}

impl<
        T: Clone
            + Debug
            + Add<Output = T>
            + Mul<Output = T>
            + Sub<Output = T>
            + Zero
            + One
            + Hash
            + Eq,
    > Hash for CircuitExpression<T>
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Constant(i) => {
                "i".hash(state);
                i.hash(state);
            }
            Input(index) => {
                "input".hash(state);
                index.hash(state);
            }
            BinaryOperation(binop, lhs, rhs) => {
                "binop".hash(state);
                binop.hash(state);
                lhs.as_ref().borrow().hash(state);
                rhs.as_ref().borrow().hash(state);
            }
        }
    }
}

impl<
        T: Clone
            + Debug
            + Add<Output = T>
            + Mul<Output = T>
            + Sub<Output = T>
            + Zero
            + One
            + Hash
            + Eq,
    > PartialEq for CircuitExpression<T>
{
    fn eq(&self, other: &Self) -> bool {
        match self {
            Constant(self_i) => match other {
                Constant(other_i) => self_i == other_i,
                _ => false,
            },
            Input(self_input) => match other {
                Input(other_input) => self_input == other_input,
                _ => false,
            },
            BinaryOperation(binop_self, lhs_self, rhs_self) => {
                match other {
                    BinaryOperation(binop_other, lhs_other, rhs_other) => {
                        // a = b `op0` c,
                        // d = e `op1` f =>
                        // a = d <= op0 == op1 && b == e && c ==f
                        binop_self == binop_other && lhs_self == lhs_other && rhs_self == rhs_other
                    }

                    _ => false,
                }
            }
        }
    }
}

impl<
        T: Clone
            + Debug
            + Add<Output = T>
            + Mul<Output = T>
            + Sub<Output = T>
            + Zero
            + One
            + Hash
            + Eq,
    > Hash for Circuit<T>
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.expression.hash(state)
    }
}

impl<
        T: Clone
            + Debug
            + WrappingAdd<Output = T>
            + WrappingMul<Output = T>
            + WrappingSub<Output = T>
            + Zero
            + One
            + Hash
            + Eq,
    > Hash for CircuitMonad<T>
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.circuit.as_ref().borrow().hash(state)
    }
}

/// A wrapper around a [`CircuitExpression`] that manages additional bookkeeping information.
#[derive(Clone, Debug)]
pub struct Circuit<
    T: Clone + Debug + Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Zero + One + Hash + Eq,
> {
    pub id: usize,
    pub visited_counter: usize,
    pub expression: CircuitExpression<T>,
}

impl<
        T: Clone
            + Debug
            + Add<Output = T>
            + Mul<Output = T>
            + Sub<Output = T>
            + Zero
            + One
            + Hash
            + Eq,
    > Eq for Circuit<T>
{
}

impl<
        T: Clone
            + Debug
            + Add<Output = T>
            + Mul<Output = T>
            + Sub<Output = T>
            + Zero
            + One
            + Hash
            + Eq,
    > PartialEq for Circuit<T>
{
    /// Calculate equality of circuits. In particular, this function does *not* attempt to
    /// simplify or reduce neutral terms or products. So this comparison will return false for
    /// `a == a + 0`. It will also return false for `XFieldElement(7) == BFieldElement(7)`
    fn eq(&self, other: &Self) -> bool {
        self.expression == other.expression
    }
}

impl<
        T: Clone
            + Debug
            + Add<Output = T>
            + Mul<Output = T>
            + Sub<Output = T>
            + Zero
            + One
            + Hash
            + Eq,
    > Display for Circuit<T>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.expression {
            Constant(i) => {
                write!(f, "{i:?}")
            }
            Input(input) => write!(f, "{input} "),
            BinaryOperation(operation, lhs, rhs) => {
                write!(
                    f,
                    "({}) {} ({})",
                    lhs.as_ref().borrow(),
                    operation,
                    rhs.as_ref().borrow()
                )
            }
        }
    }
}

impl<
        T: Clone
            + Debug
            + WrappingAdd<Output = T>
            + WrappingMul<Output = T>
            + WrappingSub<Output = T>
            + Zero
            + One
            + Hash
            + Eq,
    > Circuit<T>
{
    /// Increment `visited_counter` by one for each reachable node
    fn traverse_single(&mut self) {
        self.visited_counter += 1;
        if let BinaryOperation(_, lhs, rhs) = self.expression.borrow_mut() {
            lhs.as_ref().borrow_mut().traverse_single();
            rhs.as_ref().borrow_mut().traverse_single();
        }
    }

    /// Count how many times each reachable node is reached when traversing from
    /// the starting points that are given as input. The result is stored in the
    /// `visited_counter` field in each node.
    pub fn traverse_multiple(ccs: &mut [Circuit<T>]) {
        for cc in ccs.iter_mut() {
            assert!(
                cc.visited_counter.is_zero(),
                "visited counter must be zero before starting count"
            );
            cc.traverse_single();
        }
    }

    /// Reset the visited counters for the entire subtree
    fn reset_visited_counters(&mut self) {
        self.visited_counter = 0;

        if let BinaryOperation(_, lhs, rhs) = &self.expression {
            lhs.as_ref().borrow_mut().reset_visited_counters();
            rhs.as_ref().borrow_mut().reset_visited_counters();
        }
    }

    /// Verify that all IDs in the subtree are unique. Panics otherwise.
    fn inner_has_unique_ids(&mut self, ids: &mut HashMap<usize, Circuit<T>>) {
        let node_with_repeated_id = ids.insert(self.id, self.clone());
        assert!(
            !self.visited_counter.is_zero() || node_with_repeated_id.is_none(),
            "ID = {} was repeated. Self was: {self:?}, other node: {:?}",
            self.id,
            node_with_repeated_id.unwrap(),
        );
        self.visited_counter += 1;
        if let BinaryOperation(_, lhs, rhs) = &self.expression {
            lhs.as_ref().borrow_mut().inner_has_unique_ids(ids);
            rhs.as_ref().borrow_mut().inner_has_unique_ids(ids);
        }
    }

    /// Verify that a multicircuit has unique IDs. Panics otherwise.
    pub fn assert_has_unique_ids(constraints: &mut [Circuit<T>]) {
        let mut ids: HashMap<usize, Circuit<T>> = HashMap::new();

        for circuit in constraints.iter_mut() {
            circuit.inner_has_unique_ids(&mut ids);
        }

        for circuit in constraints.iter_mut() {
            circuit.reset_visited_counters();
        }
    }

    /// Return all visited counters in the subtree
    pub fn get_all_visited_counters(&self) -> Vec<usize> {
        // Maybe this could be solved smarter with dynamic programming
        // but we probably don't need that as our circuits aren't too big.
        match &self.expression {
            // The highest number will always be in a leaf so we only
            // need to check those.
            BinaryOperation(_, lhs, rhs) => {
                let lhs_counters = lhs.as_ref().borrow().get_all_visited_counters();
                let rhs_counters = rhs.as_ref().borrow().get_all_visited_counters();
                let own_counter = self.visited_counter;
                let mut all = [vec![own_counter], lhs_counters, rhs_counters].concat();
                all.sort_unstable();
                all.dedup();
                all.reverse();
                all
            }
            _ => vec![self.visited_counter],
        }
    }

    /// Return true if the contained multivariate polynomial consists of only a single term. This
    /// means that it can be pretty-printed without parentheses.
    pub fn print_without_parentheses(&self) -> bool {
        !matches!(&self.expression, BinaryOperation(_, _, _))
    }

    /// Return true if this node represents a constant value of zero, does not catch composite
    /// expressions that will always evaluate to zero.
    pub fn is_zero(&self) -> bool {
        match &self.expression {
            Constant(i) => i.is_zero(),
            _ => false,
        }
    }

    /// Return true if this node represents a constant value of one, does not catch composite
    /// expressions that will always evaluate to one.
    pub fn is_one(&self) -> bool {
        match &self.expression {
            Constant(i) => i.is_one(),
            _ => false,
        }
    }

    /// Panics if two nodes evaluate to the same value
    pub fn assert_all_evaluate_different(circuits: &[Self], input: Vec<T>) {
        let mut evaluated_values = HashMap::default();
        for constraint in circuits.iter() {
            Self::evaluate_and_store_and_assert_unique(constraint, &input, &mut evaluated_values);
        }
    }

    /// Return own value and whether own value was seen before. Stores own value in hash map.
    fn evaluate_and_store_and_assert_unique(
        &self,
        inputs: &[T],
        evaluated_values: &mut HashMap<T, (usize, Circuit<T>)>,
    ) -> T {
        // assert_eq!(
        //     self.var_count,
        //     input.len(),
        //     "Input length match circuit's var count"
        // );
        let value = match self.expression.clone() {
            Constant(i) => i,
            Input(index) => inputs[index].clone(),
            BinaryOperation(binop, lhs, rhs) => {
                let lhs = lhs
                    .as_ref()
                    .borrow()
                    .evaluate_and_store_and_assert_unique(inputs, evaluated_values);
                let rhs = rhs
                    .as_ref()
                    .borrow()
                    .evaluate_and_store_and_assert_unique(inputs, evaluated_values);
                match binop {
                    BinOp::Add => lhs + rhs,
                    BinOp::Sub => lhs - rhs,
                    BinOp::Mul => lhs * rhs,
                }
            }
        };

        let self_evaluated_is_unique =
            evaluated_values.insert(value.clone(), (self.id.to_owned(), self.clone()));
        if let Some((collided_circuit_id, collided_circuit)) = self_evaluated_is_unique {
            let own_id = self.id.to_owned();
            if collided_circuit_id != self.id {
                panic!(
                    "Circuit ID {collided_circuit_id} and circuit ID {own_id} are not unique. \
                    Collission on:\n \
                    {collided_circuit_id}: {collided_circuit}\n {own_id}: {self}. \
                    Value was {value:?}",
                );
            }
        }
        value
    }

    pub fn evaluate(&self, inputs: &[T]) -> T {
        match self.clone().expression {
            Constant(i) => i,
            Input(index) => inputs[index].clone(),
            BinaryOperation(binop, lhs, rhs) => {
                let lhs_value = lhs.as_ref().borrow().evaluate(inputs);
                let rhs_value = rhs.as_ref().borrow().evaluate(inputs);
                match binop {
                    BinOp::Add => lhs_value.wrapping_add(&rhs_value),
                    BinOp::Sub => lhs_value.wrapping_sub(&rhs_value),
                    BinOp::Mul => lhs_value.wrapping_mul(&rhs_value),
                }
            }
        }
    }
}

/// Constraint expressions, with context needed to ensure that two equal nodes are not added to
/// the multicircuit.
#[derive(Clone)]
pub struct CircuitMonad<
    T: Clone
        + Debug
        + WrappingAdd<Output = T>
        + WrappingMul<Output = T>
        + WrappingSub<Output = T>
        + Zero
        + One
        + Hash
        + Eq,
> {
    pub circuit: Rc<RefCell<Circuit<T>>>,
    pub builder: CircuitBuilder<T>,
}

impl<
        T: Clone
            + Debug
            + WrappingAdd<Output = T>
            + WrappingMul<Output = T>
            + WrappingSub<Output = T>
            + Zero
            + One
            + Hash
            + Eq,
    > Debug for CircuitMonad<T>
{
    // We cannot derive `Debug` as `all_nodes` contains itself which a derived `Debug` will
    // attempt to print as well, thus leading to infinite recursion.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CircuitMonad")
            .field("id", &self.circuit)
            .field(
                "all_nodes length: ",
                &self.builder.all_nodes.as_ref().borrow().len(),
            )
            .field(
                "id_counter_ref value: ",
                &self.builder.id_counter.as_ref().borrow(),
            )
            .finish()
    }
}

impl<
        T: Clone
            + Debug
            + WrappingAdd<Output = T>
            + WrappingMul<Output = T>
            + WrappingSub<Output = T>
            + Zero
            + One
            + Hash
            + Eq,
    > Display for CircuitMonad<T>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.circuit.as_ref().borrow())
    }
}

impl<
        T: Clone
            + Debug
            + WrappingAdd<Output = T>
            + WrappingMul<Output = T>
            + WrappingSub<Output = T>
            + Zero
            + One
            + Hash
            + Eq,
    > PartialEq for CircuitMonad<T>
{
    // Equality for the CircuitMonad is defined by the circuit, not the
    // other metadata (e.g. ID) that it carries around.
    fn eq(&self, other: &Self) -> bool {
        self.circuit == other.circuit
    }
}

impl<
        T: Clone
            + Debug
            + WrappingAdd<Output = T>
            + WrappingMul<Output = T>
            + WrappingSub<Output = T>
            + Zero
            + One
            + Hash
            + Eq,
    > Eq for CircuitMonad<T>
{
}

impl<
        T: Clone
            + Debug
            + WrappingAdd<Output = T>
            + WrappingMul<Output = T>
            + WrappingSub<Output = T>
            + Zero
            + One
            + Hash
            + Eq,
    > WrappingAdd for CircuitMonad<T>
{
    fn wrapping_add(&self, v: &Self) -> Self {
        binop(BinOp::Add, self.clone(), v.clone())
    }
}

impl<
        T: Clone
            + Debug
            + WrappingAdd<Output = T>
            + WrappingMul<Output = T>
            + WrappingSub<Output = T>
            + Zero
            + One
            + Hash
            + Eq,
    > WrappingMul for CircuitMonad<T>
{
    fn wrapping_mul(&self, v: &Self) -> Self {
        binop(BinOp::Mul, self.clone(), v.clone())
    }
}

impl<
        T: Clone
            + Debug
            + WrappingAdd<Output = T>
            + WrappingMul<Output = T>
            + WrappingSub<Output = T>
            + Zero
            + One
            + Hash
            + Eq,
    > WrappingSub for CircuitMonad<T>
{
    fn wrapping_sub(&self, v: &Self) -> Self {
        binop(BinOp::Sub, self.clone(), v.clone())
    }
}

/// Helper function for binary operations that are used to generate new parent nodes in the
/// multitree that represents the algebraic circuit. Ensures that each newly created node has a
/// unique ID.
fn binop<
    T: Clone
        + Debug
        + WrappingAdd<Output = T>
        + WrappingMul<Output = T>
        + WrappingSub<Output = T>
        + Zero
        + One
        + Hash
        + Eq,
>(
    binop: BinOp,
    lhs: CircuitMonad<T>,
    rhs: CircuitMonad<T>,
) -> CircuitMonad<T> {
    // Get ID for the new node
    let new_index = lhs.builder.id_counter.as_ref().borrow().to_owned();
    let lhs = Rc::new(RefCell::new(lhs));
    let rhs = Rc::new(RefCell::new(rhs));

    let new_node = CircuitMonad {
        circuit: Rc::new(RefCell::new(Circuit {
            visited_counter: 0,
            expression: BinaryOperation(
                binop,
                Rc::clone(&lhs.as_ref().borrow().circuit),
                Rc::clone(&rhs.as_ref().borrow().circuit),
            ),
            id: new_index,
        })),
        builder: lhs.as_ref().borrow().builder.clone(),
    };

    // check if node already exists
    let contained = lhs
        .as_ref()
        .borrow()
        .builder
        .all_nodes
        .as_ref()
        .borrow()
        .contains(&new_node);
    if contained {
        let ret0 = &lhs.as_ref().borrow();
        let ret1 = ret0.builder.all_nodes.as_ref().borrow();
        let ret2 = &(*ret1.get(&new_node).as_ref().unwrap()).clone();
        return ret2.to_owned();
    }

    // If the operator commutes, check if the inverse node has already been constructed.
    // If it has, return this instead. Do not allow a new one to be built.
    if matches!(binop, BinOp::Add | BinOp::Mul) {
        let new_node_inverted = CircuitMonad {
            circuit: Rc::new(RefCell::new(Circuit {
                visited_counter: 0,
                expression: BinaryOperation(
                    binop,
                    // Switch rhs and lhs for symmetric operators to check membership in hash set
                    Rc::clone(&rhs.as_ref().borrow().circuit),
                    Rc::clone(&lhs.as_ref().borrow().circuit),
                ),
                id: new_index,
            })),
            builder: lhs.as_ref().borrow().builder.clone(),
        };

        // check if node already exists
        let inverted_contained = lhs
            .as_ref()
            .borrow()
            .builder
            .all_nodes
            .as_ref()
            .borrow()
            .contains(&new_node_inverted);
        if inverted_contained {
            let ret0 = &lhs.as_ref().borrow();
            let ret1 = ret0.builder.all_nodes.as_ref().borrow();
            let ret2 = &(*ret1.get(&new_node_inverted).as_ref().unwrap()).clone();
            return ret2.to_owned();
        }
    }

    // Increment counter index
    *lhs.as_ref()
        .borrow()
        .builder
        .id_counter
        .as_ref()
        .borrow_mut() = new_index + 1;

    // Store new node in HashSet
    let inserted_value_was_new = new_node
        .builder
        .all_nodes
        .as_ref()
        .borrow_mut()
        .insert(new_node.clone());
    assert!(inserted_value_was_new, "Binop-created value must be new");

    new_node
}

impl<
        T: Clone
            + Debug
            + WrappingAdd<Output = T>
            + WrappingMul<Output = T>
            + WrappingSub<Output = T>
            + Zero
            + One
            + Hash
            + Eq,
    > Add for CircuitMonad<T>
{
    type Output = CircuitMonad<T>;

    fn add(self, rhs: Self) -> Self::Output {
        binop(BinOp::Add, self, rhs)
    }
}

impl<
        T: Clone
            + Debug
            + WrappingAdd<Output = T>
            + WrappingMul<Output = T>
            + WrappingSub<Output = T>
            + Zero
            + One
            + Hash
            + Eq,
    > Sub for CircuitMonad<T>
{
    type Output = CircuitMonad<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        binop(BinOp::Sub, self, rhs)
    }
}

impl<
        T: Clone
            + Debug
            + WrappingAdd<Output = T>
            + WrappingMul<Output = T>
            + WrappingSub<Output = T>
            + Zero
            + One
            + Hash
            + Eq,
    > Mul for CircuitMonad<T>
{
    type Output = CircuitMonad<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        binop(BinOp::Mul, self, rhs)
    }
}

/// This will panic if the iterator is empty because the neutral element needs a unique ID, and
/// we have no way of getting that here.
impl<
        T: Clone
            + Debug
            + WrappingAdd<Output = T>
            + WrappingMul<Output = T>
            + WrappingSub<Output = T>
            + Zero
            + One
            + Hash
            + Eq,
    > Sum for CircuitMonad<T>
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|accum, item| accum + item)
            .expect("CircuitMonad Iterator was empty")
    }
}

impl<
        T: Clone
            + Debug
            + WrappingAdd<Output = T>
            + WrappingMul<Output = T>
            + WrappingSub<Output = T>
            + Zero
            + One
            + Hash
            + Eq,
    > CircuitMonad<T>
{
    /// Unwrap a CircuitMonad to reveal its inner Circuit
    pub fn consume(self) -> Circuit<T> {
        self.circuit.try_borrow().unwrap().to_owned()
    }

    pub fn max_id(&self) -> usize {
        let max_from_hash_map = self
            .builder
            .all_nodes
            .as_ref()
            .borrow()
            .iter()
            .map(|x| x.circuit.as_ref().borrow().id)
            .max()
            .unwrap();

        let id_ref_value = *self.builder.id_counter.borrow();
        assert_eq!(id_ref_value - 1, max_from_hash_map);
        max_from_hash_map
    }

    fn replace_references(&self, old_id: usize, new: Rc<RefCell<Circuit<T>>>) {
        for node in self.builder.all_nodes.as_ref().borrow().clone().into_iter() {
            if node.circuit.as_ref().borrow().id == old_id {
                continue;
            }

            if let BinaryOperation(_, ref mut lhs, ref mut rhs) =
                node.circuit.as_ref().borrow_mut().expression
            {
                if lhs.as_ref().borrow().id == old_id {
                    *lhs = new.clone();
                }
                if rhs.as_ref().borrow().id == old_id {
                    *rhs = new.clone();
                }
            }
        }
    }

    fn find_equivalent_expression(&self) -> Option<Rc<RefCell<Circuit<T>>>> {
        if let BinaryOperation(binop, lhs, rhs) = &self.circuit.as_ref().borrow().expression {
            // a + 0 = a ∧ a - 0 = a
            if matches!(binop, BinOp::Add | BinOp::Sub) && rhs.borrow().is_zero() {
                return Some(Rc::clone(lhs));
            }

            // 0 + a = a
            if *binop == BinOp::Add && lhs.borrow().is_zero() {
                return Some(Rc::clone(rhs));
            }

            if matches!(binop, BinOp::Mul) {
                // a * 1 = a
                if rhs.borrow().is_one() {
                    return Some(Rc::clone(lhs));
                }

                // 1 * a = a
                if lhs.borrow().is_one() {
                    return Some(Rc::clone(rhs));
                }

                // 0 * a = 0
                if lhs.borrow().is_zero() {
                    return Some(Rc::clone(lhs));
                }

                // a * 0 = 0
                if rhs.borrow().is_zero() {
                    return Some(Rc::clone(rhs));
                }
            }

            // if left and right hand sides are both constants
            if let Constant(lhs_i) = lhs.borrow().expression.clone() {
                if let Constant(rhs_i) = rhs.borrow().expression.clone() {
                    return match binop {
                        BinOp::Add => Some(Rc::new(RefCell::new(
                            self.builder
                                .make_leaf(Constant(lhs_i.wrapping_add(&rhs_i)))
                                .consume(),
                        ))),
                        BinOp::Sub => Some(Rc::new(RefCell::new(
                            self.builder
                                .make_leaf(Constant(lhs_i.wrapping_sub(&rhs_i)))
                                .consume(),
                        ))),
                        BinOp::Mul => Some(Rc::new(RefCell::new(
                            self.builder
                                .make_leaf(Constant(lhs_i.wrapping_mul(&rhs_i)))
                                .consume(),
                        ))),
                    };
                }
            }

            return None;
        }
        None
    }

    /// Map (e * c) * c' to e * (c*c')
    fn fold_uncles_inner(&mut self) -> (bool, Option<Rc<RefCell<Circuit<T>>>>) {
        let mut change_tracker = false;
        let mut new_root = None;
        let self_expr = self.circuit.as_ref().borrow().expression.clone();

        let mut update = None;

        if let BinaryOperation(outer_binop, original_lhs, original_rhs) = &self_expr {
            // fix subtrees
            let mut lhs_as_monadic_value = CircuitMonad {
                circuit: original_lhs.clone(),
                builder: self.builder.clone(),
            };
            let (lhs_was_changed, maybe_new_lhs) = lhs_as_monadic_value.fold_uncles_inner();
            change_tracker = change_tracker || lhs_was_changed;

            let mut rhs_as_monadic_value = CircuitMonad {
                circuit: original_rhs.clone(),
                builder: self.builder.clone(),
            };
            let (rhs_was_changed, maybe_new_rhs) = rhs_as_monadic_value.fold_uncles_inner();
            change_tracker = change_tracker || rhs_was_changed;

            // update left and right hand sides
            let outer_lhs = if let Some(new_lhs) = maybe_new_lhs {
                new_lhs
            } else {
                original_lhs.clone()
            };
            let outer_rhs = if let Some(new_rhs) = maybe_new_rhs {
                new_rhs
            } else {
                original_rhs.clone()
            };

            // match for (e * c) * c'
            if matches!(*outer_binop, BinOp::Mul) {
                if let CircuitExpression::Constant(outer_constant) =
                    &outer_rhs.as_ref().borrow().expression
                {
                    if let BinaryOperation(inner_binop, inner_lhs, inner_rhs) =
                        &outer_lhs.as_ref().borrow().expression
                    {
                        if matches!(inner_binop, BinOp::Mul) {
                            let inner_lhs_monad = CircuitMonad {
                                circuit: inner_lhs.clone(),
                                builder: self.builder.clone(),
                            };
                            if let CircuitExpression::Constant(inner_constant) =
                                &inner_rhs.as_ref().borrow().expression
                            {
                                change_tracker = true;
                                let new_circuit_monad = inner_lhs_monad
                                    * self
                                        .builder
                                        .constant(inner_constant.wrapping_mul(outer_constant));

                                update = Some((
                                    self.circuit.borrow().id,
                                    new_circuit_monad.circuit.clone(),
                                ));

                                new_root = Some(new_circuit_monad.circuit);
                            }
                        }
                    }
                }
            }
        }

        // apply update, if any
        if let Some((old_id, replacement_circuit)) = update {
            self.replace_references(old_id, replacement_circuit);
            self.builder.all_nodes.as_ref().borrow_mut().remove(self);
        }

        (change_tracker, new_root)
    }

    fn move_coefficients_right_inner(&mut self) -> bool {
        let mut change_tracker = false;
        let self_expr = self.circuit.as_ref().borrow().expression.clone();

        if let BinaryOperation(binop, lhs, rhs) = &self_expr {
            // fix subtrees
            let mut lhs_as_monadic_value = CircuitMonad {
                circuit: lhs.clone(),
                builder: self.builder.clone(),
            };
            let lhs_was_changed = lhs_as_monadic_value.move_coefficients_right_inner();
            change_tracker = change_tracker || lhs_was_changed;

            let mut rhs_as_monadic_value = CircuitMonad {
                circuit: rhs.clone(),
                builder: self.builder.clone(),
            };
            let rhs_was_changed = rhs_as_monadic_value.move_coefficients_right_inner();
            change_tracker = change_tracker || rhs_was_changed;

            // match for c * e
            if *binop == BinOp::Mul
                && matches!(lhs.borrow().expression, CircuitExpression::Constant(_))
                && !matches!(rhs.borrow().expression, CircuitExpression::Constant(_))
            {
                change_tracker = true;
                self.circuit.as_ref().borrow_mut().expression =
                    CircuitExpression::BinaryOperation(BinOp::Mul, rhs.clone(), lhs.clone());
            }
        }

        change_tracker
    }

    fn distribute_constants_inner(&mut self) -> (bool, Option<Rc<RefCell<Circuit<T>>>>) {
        let mut change_tracker = false;
        let mut new_root = None;
        let self_expr = self.circuit.as_ref().borrow().expression.clone();

        if let BinaryOperation(binop, lhs, rhs) = &self_expr {
            // fix subtrees
            let mut lhs_as_monadic_value = CircuitMonad {
                circuit: lhs.clone(),
                builder: self.builder.clone(),
            };
            let (lhs_was_changed, maybe_changed_lhs) =
                lhs_as_monadic_value.distribute_constants_inner();
            change_tracker = change_tracker || lhs_was_changed;
            let mut rhs_as_monadic_value = CircuitMonad {
                circuit: rhs.clone(),
                builder: self.builder.clone(),
            };
            let (rhs_was_changed, maybe_changed_rhs) =
                rhs_as_monadic_value.distribute_constants_inner();
            change_tracker = change_tracker || rhs_was_changed;

            // update references to lhs and rhs (if necessary)
            let lhs_expr = if let Some(changed_lhs) = maybe_changed_lhs {
                change_tracker = true;
                changed_lhs.as_ref().borrow().expression.clone()
            } else {
                lhs.as_ref().borrow().expression.clone()
            };
            let rhs_expr = if let Some(changed_rhs) = maybe_changed_rhs {
                change_tracker = true;
                changed_rhs.as_ref().borrow().expression.clone()
            } else {
                rhs.as_ref().borrow().expression.clone()
            };

            // match for c * (l + r) or (l + r) * c
            if *binop == BinOp::Mul {
                let matching = if let CircuitExpression::Constant(c) = lhs_expr {
                    if let CircuitExpression::BinaryOperation(
                        inner_binop,
                        inner_left,
                        inner_right,
                    ) = rhs_expr
                    {
                        if inner_binop == BinOp::Add || inner_binop == BinOp::Sub {
                            Some((c, inner_binop, inner_left, inner_right))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else if let CircuitExpression::Constant(c) = rhs_expr {
                    if let CircuitExpression::BinaryOperation(
                        inner_binop,
                        inner_left,
                        inner_right,
                    ) = lhs_expr
                    {
                        if inner_binop == BinOp::Add || inner_binop == BinOp::Sub {
                            Some((c, inner_binop, inner_left, inner_right))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                // if match, distribute constant
                if let Some((c, inner_binop, inner_left, inner_right)) = matching {
                    change_tracker = true;
                    let rhs_lhs_as_monadic_value = CircuitMonad {
                        circuit: inner_left,
                        builder: self.builder.clone(),
                    };
                    let rhs_rhs_as_monadic_value = CircuitMonad {
                        circuit: inner_right,
                        builder: self.builder.clone(),
                    };
                    let c_lhs = self.builder.constant(c.clone()) * rhs_lhs_as_monadic_value;
                    let c_rhs = self.builder.constant(c) * rhs_rhs_as_monadic_value;
                    let distributed_sum = match inner_binop {
                        BinOp::Add => c_lhs + c_rhs,
                        BinOp::Sub => c_lhs - c_rhs,
                        BinOp::Mul => unreachable!(),
                    };
                    let id_to_be_deleted = self.circuit.borrow().id;
                    self.replace_references(id_to_be_deleted, distributed_sum.circuit.clone());
                    self.builder.all_nodes.as_ref().borrow_mut().remove(self);

                    new_root = Some(distributed_sum.circuit);
                }
            }
        }
        (change_tracker, new_root)
    }

    /// Apply constant folding to simplify the (sub)tree.
    /// If the subtree is a leaf (terminal), no change.
    /// If the subtree is a binary operation on:
    ///
    ///  - one constant x one constant   => fold
    ///  - one constant x one expr       => can't
    ///  - one expr x one constant       => can't
    ///  - one expr x one expr           => can't
    ///
    /// This operation mutates self and returns true if a change was
    /// applied anywhere in the tree.
    fn constant_fold_inner(&mut self) -> (bool, Option<Rc<RefCell<Circuit<T>>>>) {
        let mut change_tracker = false;
        let self_expr = self.circuit.as_ref().borrow().expression.clone();

        // fix subtrees
        if let BinaryOperation(_, lhs, rhs) = &self_expr {
            let mut lhs_as_monadic_value = CircuitMonad {
                circuit: lhs.clone(),
                builder: self.builder.clone(),
            };
            let (change_in_lhs, _) = lhs_as_monadic_value.constant_fold_inner();
            change_tracker |= change_in_lhs;
            let mut rhs_as_monadic_value = CircuitMonad {
                circuit: rhs.clone(),
                builder: self.builder.clone(),
            };
            let (change_in_rhs, _) = rhs_as_monadic_value.constant_fold_inner();
            change_tracker |= change_in_rhs;
        }

        // perform folding
        let equivalent_circuit = self.find_equivalent_expression();
        change_tracker |= equivalent_circuit.is_some();

        if equivalent_circuit.is_some() {
            let equivalent_circuit = equivalent_circuit.as_ref().unwrap().clone();
            let id_of_node_to_be_deleted = self.circuit.borrow().id;
            self.replace_references(id_of_node_to_be_deleted, equivalent_circuit);
            self.builder.all_nodes.as_ref().borrow_mut().remove(self);
        }

        (change_tracker, equivalent_circuit)
    }

    /// Reduce size of multicircuit by simplifying constant expressions such as `1 * MPol(_,_)`
    pub fn constant_folding(circuits: &mut [CircuitMonad<T>]) {
        for circuit in circuits.iter_mut() {
            let mut mutated = true;
            while mutated {
                let (mutated_inner, maybe_new_root) = circuit.constant_fold_inner();
                mutated = mutated_inner;
                if let Some(new_root) = maybe_new_root {
                    *circuit = CircuitMonad {
                        circuit: new_root,
                        builder: circuit.builder.clone(),
                    };
                }
            }
        }
    }

    /// Apply distributivity law to subcircuits of the form c * (a + b).
    pub fn distribute_constants(circuits: &mut [CircuitMonad<T>]) {
        for circuit in circuits.iter_mut() {
            let mut mutated = true;
            while mutated {
                let (mutated_inner, maybe_new_root) = circuit.distribute_constants_inner();
                mutated = mutated_inner;
                if let Some(new_root) = maybe_new_root {
                    *circuit = CircuitMonad {
                        circuit: new_root,
                        builder: circuit.builder.clone(),
                    };
                }
            }
        }
    }

    /// Rearrange multiplication expressions so that the constant comes right
    pub fn move_coefficients_right(circuits: &mut [CircuitMonad<T>]) {
        for circuit in circuits.iter_mut() {
            let mut mutated = true;
            while mutated {
                mutated = circuit.move_coefficients_right_inner();
            }
        }
    }

    /// Map all instances of (e * c) *  c' to e * [c*c'] throughout the multicircuit
    pub fn fold_uncles(circuits: &mut [CircuitMonad<T>]) {
        for circuit in circuits.iter_mut() {
            let mut mutated = true;
            while mutated {
                let (mutated_inner, maybe_new_root) = circuit.fold_uncles_inner();
                mutated = mutated_inner;
                if let Some(new_root) = maybe_new_root {
                    *circuit = CircuitMonad {
                        circuit: new_root,
                        builder: circuit.builder.clone(),
                    };
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
/// Helper struct to construct new leaf nodes in the multicircuit. Ensures that each newly
/// created node gets a unique ID.
pub struct CircuitBuilder<
    T: Clone
        + Debug
        + WrappingAdd<Output = T>
        + WrappingMul<Output = T>
        + WrappingSub<Output = T>
        + Zero
        + One
        + Hash
        + Eq,
> {
    id_counter: Rc<RefCell<usize>>,
    all_nodes: Rc<RefCell<HashSet<CircuitMonad<T>>>>,
}

impl<
        T: Clone
            + Debug
            + WrappingAdd<Output = T>
            + WrappingMul<Output = T>
            + WrappingSub<Output = T>
            + Zero
            + One
            + Hash
            + Eq,
    > Default for CircuitBuilder<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
        T: Clone
            + Debug
            + WrappingAdd<Output = T>
            + WrappingMul<Output = T>
            + WrappingSub<Output = T>
            + Zero
            + One
            + Hash
            + Eq,
    > CircuitBuilder<T>
{
    pub fn new() -> Self {
        Self {
            id_counter: Rc::new(RefCell::new(0)),
            all_nodes: Rc::new(RefCell::new(HashSet::default())),
        }
    }

    pub fn get_node_by_id(&self, id: usize) -> Option<CircuitMonad<T>> {
        for node in self.all_nodes.as_ref().borrow().iter() {
            if node.circuit.as_ref().borrow().id == id {
                return Some(node.clone());
            }
        }
        None
    }

    /// Create constant leaf node.
    pub fn constant(&self, i: T) -> CircuitMonad<T> {
        let expression = Constant(i);
        self.make_leaf(expression)
    }

    /// Create deterministic input leaf node.
    pub fn input(&self, index: usize) -> CircuitMonad<T> {
        let expression = Input(index);
        self.make_leaf(expression)
    }

    fn make_leaf(&self, expression: CircuitExpression<T>) -> CircuitMonad<T> {
        let new_id = self.id_counter.as_ref().borrow().to_owned();
        let new_node = CircuitMonad {
            circuit: Rc::new(RefCell::new(Circuit {
                visited_counter: 0usize,
                expression,
                id: new_id,
            })),
            builder: self.clone(),
        };

        // Check if node already exists, return the existing one if it does
        let contained = self.all_nodes.as_ref().borrow().contains(&new_node);
        if contained {
            let ret0 = &self.all_nodes.as_ref().borrow();
            let ret1 = &(*ret0.get(&new_node).as_ref().unwrap()).clone();
            return ret1.to_owned();
        }

        // If node did not already exist, increment counter and insert node into hash set
        *self.id_counter.as_ref().borrow_mut() = new_id + 1;
        self.all_nodes
            .as_ref()
            .borrow_mut()
            .insert(new_node.clone());

        new_node
    }
}

#[cfg(test)]
mod constraint_circuit_tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    use rand::thread_rng;
    use rand::Rng;
    use rand::RngCore;

    use crate::shared_math::other::random_elements;

    use super::*;

    type T = u64;

    // fn node_counter_inner(constraint: &mut Circuit<T>, counter: &mut usize) {
    //     if constraint.visited_counter == 0 {
    //         *counter += 1;
    //         constraint.visited_counter = 1;

    //         if let BinaryOperation(_, lhs, rhs) = &constraint.expression {
    //             node_counter_inner(&mut lhs.as_ref().borrow_mut(), counter);
    //             node_counter_inner(&mut rhs.as_ref().borrow_mut(), counter);
    //         }
    //     }
    // }

    // /// Count the total number of nodes in cthe multicircuit
    // pub fn node_counter(constraints: &mut [Circuit<T>]) -> usize {
    //     let mut counter = 0;

    //     for constraint in constraints.iter_mut() {
    //         node_counter_inner(constraint, &mut counter);
    //     }

    //     for constraint in constraints.iter_mut() {
    //         Circuit::reset_visited_counters(constraint);
    //     }

    //     counter
    // }

    fn random_circuit_builder() -> (CircuitMonad<T>, CircuitBuilder<T>) {
        let mut rng = thread_rng();
        let num_inputs = 32;
        let num_constants = 51;
        let circuit_builder = CircuitBuilder::new();
        let constants: Vec<T> = random_elements(num_constants);
        let circuit_input = rng.gen_range(0..num_inputs);
        let mut ret_circuit = circuit_builder.input(circuit_input);
        for _ in 0..100 {
            let leaf = match rng.next_u64() % 3 {
                0 => {
                    // p(x, y, z) = x
                    circuit_builder.input(rng.next_u64() as usize % num_inputs)
                }
                1 => {
                    // p(x, y, z) = xfe
                    circuit_builder.constant(constants[rng.next_u64() as usize % num_constants])
                }
                2 => {
                    // p(x, y, z) = 0
                    circuit_builder.constant(0)
                }
                _ => unreachable!(),
            };
            match rng.next_u32() % 3 {
                0 => ret_circuit = ret_circuit * leaf,
                1 => ret_circuit = ret_circuit + leaf,
                2 => ret_circuit = ret_circuit - leaf,
                _ => unreachable!(),
            }
        }

        (ret_circuit, circuit_builder)
    }

    // Make a deep copy of a Multicircuit and return it as a CircuitMonad
    fn deep_copy_inner(val: &Circuit<T>, builder: &mut CircuitBuilder<T>) -> CircuitMonad<T> {
        match &val.expression {
            BinaryOperation(op, lhs, rhs) => {
                let lhs_ref = deep_copy_inner(&lhs.as_ref().borrow(), builder);
                let rhs_ref = deep_copy_inner(&rhs.as_ref().borrow(), builder);
                binop(*op, lhs_ref, rhs_ref)
            }
            Constant(i) => builder.constant(*i),
            Input(input_index) => builder.input(*input_index),
        }
    }

    fn deep_copy(val: &Circuit<T>) -> CircuitMonad<T> {
        let mut builder = CircuitBuilder::new();
        deep_copy_inner(val, &mut builder)
    }

    #[test]
    fn equality_and_hash_agree_test() {
        // The Multicircuits are put into a hash set. Hence, it is important that `Eq` and `Hash`
        // agree whether two nodes are equal: k1 == k2 => h(k1) == h(k2)
        for _ in 0..100 {
            let (circuit, circuit_builder) = random_circuit_builder();
            let mut hasher0 = DefaultHasher::new();
            circuit.hash(&mut hasher0);
            let hash0 = hasher0.finish();
            assert_eq!(circuit, circuit);

            let zero = circuit_builder.constant(0);
            let same_circuit = circuit.clone() + zero;
            let mut hasher1 = DefaultHasher::new();
            same_circuit.hash(&mut hasher1);
            let hash1 = hasher1.finish();
            let eq_eq = circuit == same_circuit;
            let hash_eq = hash0 == hash1;

            assert_eq!(eq_eq, hash_eq);
        }
    }

    #[test]
    fn multi_circuit_hash_is_unchanged_by_meta_data_test() {
        // From https://doc.rust-lang.org/std/collections/struct.HashSet.html
        // "It is a logic error for a key to be modified in such a way that the key’s hash, as
        // determined by the Hash trait, or its equality, as determined by the Eq trait, changes
        // while it is in the map. This is normally only possible through Cell, RefCell, global
        // state, I/O, or unsafe code. The behavior resulting from such a logic error is not
        // specified, but will be encapsulated to the HashSet that observed the logic error and not
        // result in undefined behavior. This could include panics, incorrect results, aborts,
        // memory leaks, and non-termination."
        // This means that the hash of a node may not depend on: `visited_counter`, `counter`,
        // `id_counter_ref`, or `all_nodes`. The reason for this constraint is that `all_nodes`
        // contains the digest of all nodes in the multi tree.
        let (circuit, _circuit_builder) = random_circuit_builder();
        let mut hasher0 = DefaultHasher::new();
        circuit.hash(&mut hasher0);
        let digest_prior = hasher0.finish();

        // Increase visited counter and verify digest is unchanged
        circuit.circuit.as_ref().borrow_mut().traverse_single();
        let mut hasher1 = DefaultHasher::new();
        circuit.hash(&mut hasher1);
        let digest_after = hasher1.finish();
        assert_eq!(
            digest_prior, digest_after,
            "Digest must be unchanged by traversal"
        );

        // id counter and verify digest is unchanged
        let _dummy = circuit.clone() + circuit.clone();
        let mut hasher2 = DefaultHasher::new();
        circuit.hash(&mut hasher2);
        let digest_after2 = hasher2.finish();
        assert_eq!(
            digest_prior, digest_after2,
            "Digest must be unchanged by Id counter increase"
        );
    }

    #[test]
    fn circuit_equality_check_and_constant_folding_test() {
        let circuit_builder: CircuitBuilder<T> = CircuitBuilder::new();
        let var_0 = circuit_builder.input(0);
        let var_4 = circuit_builder.input(4);
        let four = circuit_builder.constant(4);
        let one = circuit_builder.constant(1);
        let zero = circuit_builder.constant(0);

        assert_ne!(var_0, var_4);
        assert_ne!(var_0, four);
        assert_ne!(one, four);
        assert_ne!(one, zero);
        assert_ne!(zero, one);

        // Verify that constant folding can handle a = a * 1
        let var_0_copy_0 = deep_copy(&var_0.circuit.as_ref().borrow());
        let var_0_mul_one_0 = var_0_copy_0.clone() * one.clone();
        assert_ne!(var_0_copy_0, var_0_mul_one_0);
        let mut circuits = [var_0_copy_0, var_0_mul_one_0];
        CircuitMonad::constant_folding(&mut circuits);
        assert_eq!(
            circuits[0], circuits[1],
            "{} != {}",
            circuits[0], circuits[1]
        );
        assert_eq!(
            circuits[1], circuits[0],
            "{} != {}",
            circuits[1], circuits[0]
        );

        // Verify that constant folding can handle a = 1 * a
        let var_0_copy_1 = deep_copy(&var_0.circuit.as_ref().borrow());
        let var_0_one_mul_1 = one.clone() * var_0_copy_1.clone();
        assert_ne!(var_0_copy_1, var_0_one_mul_1);
        let mut circuits_ = [var_0_copy_1, var_0_one_mul_1];
        CircuitMonad::constant_folding(&mut circuits_);
        assert_eq!(
            circuits_[0], circuits_[1],
            "{} != {}",
            circuits_[0], circuits_[1]
        );
        assert_eq!(
            circuits_[1], circuits_[0],
            "{} != {}",
            circuits_[1], circuits_[0]
        );

        // Verify that constant folding can handle a = 1 * a * 1
        let var_0_copy_2 = deep_copy(&var_0.circuit.as_ref().borrow());
        let var_0_one_mul_2 = one.clone() * var_0_copy_2.clone() * one;
        assert_ne!(var_0_copy_2, var_0_one_mul_2);
        let mut circuits__ = [var_0_copy_2, var_0_one_mul_2];
        CircuitMonad::constant_folding(&mut circuits__);
        assert_eq!(
            circuits__[0], circuits__[1],
            "{} != {}",
            circuits__[0], circuits__[1]
        );
        assert_eq!(
            circuits__[1], circuits__[0],
            "{} != {}",
            circuits__[1], circuits__[0]
        );

        // Verify that constant folding handles a + 0 = a
        let var_0_copy_3 = deep_copy(&var_0.circuit.as_ref().borrow());
        let var_0_plus_zero_3 = var_0_copy_3.clone() + zero.clone();
        assert_ne!(var_0_copy_3, var_0_plus_zero_3);
        let mut circuits___ = [var_0_copy_3, var_0_plus_zero_3];
        CircuitMonad::constant_folding(&mut circuits___);
        assert_eq!(
            circuits___[0], circuits___[1],
            "{} != {}",
            circuits___[0], circuits___[1]
        );
        assert_eq!(
            circuits___[1], circuits___[0],
            "{} != {}",
            circuits___[1], circuits___[0]
        );

        // Verify that constant folding handles a + (a * 0) = a
        let var_0_copy_4 = deep_copy(&var_0.circuit.as_ref().borrow());
        let var_0_plus_zero_4 = var_0_copy_4.clone() + var_0_copy_4.clone() * zero.clone();
        assert_ne!(var_0_copy_4, var_0_plus_zero_4);
        let mut circuits_____ = [var_0_copy_4, var_0_plus_zero_4];
        CircuitMonad::constant_folding(&mut circuits_____);
        assert_eq!(
            circuits_____[0], circuits_____[1],
            "{} != {}",
            circuits_____[0], circuits_____[1]
        );
        assert_eq!(
            circuits_____[1], circuits_____[0],
            "{} != {}",
            circuits_____[1], circuits_____[0]
        );

        // Verify that constant folding does not equate `0 - a` with `a`
        let var_0_copy_5 = deep_copy(&var_0.circuit.as_ref().borrow());
        let zero_minus_var_0 = zero - var_0_copy_5.clone();
        assert_ne!(var_0_copy_5, zero_minus_var_0);
        let mut circuits______ = [var_0_copy_5, zero_minus_var_0];
        CircuitMonad::constant_folding(&mut circuits______);
        assert_ne!(
            circuits______[0], circuits______[1],
            "{} == {}",
            circuits______[0], circuits______[1]
        );
        assert_ne!(
            circuits______[1], circuits______[0],
            "{} == {}",
            circuits______[1], circuits______[0]
        );
    }

    #[test]
    fn constant_folding_pbt() {
        for _ in 0..200 {
            let (circuit, circuit_builder) = random_circuit_builder();
            let one = circuit_builder.constant(1);
            let zero = circuit_builder.constant(0);

            // Verify that constant folding can handle a = a * 1
            let copy_0 = deep_copy(&circuit.circuit.as_ref().borrow());
            let copy_0_alt = copy_0.clone() * one.clone();
            assert_ne!(copy_0, copy_0_alt);
            let mut circuits = [copy_0.clone(), copy_0_alt.clone()];
            CircuitMonad::constant_folding(&mut circuits);
            assert_eq!(
                circuits[0], circuits[1],
                "{} != {}",
                circuits[0], circuits[1]
            );
            assert_eq!(
                circuits[1], circuits[0],
                "{} != {}",
                circuits[1], circuits[0]
            );

            // Verify that constant folding can handle a = 1 * a
            let copy_1 = deep_copy(&circuit.circuit.as_ref().borrow());
            let copy_1_alt = one.clone() * copy_1.clone();
            assert_ne!(copy_1, copy_1_alt);
            let mut circuits_ = [copy_1, copy_1_alt];
            CircuitMonad::constant_folding(&mut circuits_);
            assert_eq!(
                circuits_[0], circuits_[1],
                "{} != {}",
                circuits_[0], circuits_[1]
            );
            assert_eq!(
                circuits_[1], circuits_[0],
                "{} != {}",
                circuits_[1], circuits_[0]
            );

            // Verify that constant folding can handle a = 1 * a * 1
            let copy_2 = deep_copy(&circuit.circuit.as_ref().borrow());
            let copy_2_alt = one.clone() * copy_2.clone() * one.clone();
            assert_ne!(copy_2, copy_2_alt);
            let mut circuits__ = [copy_2, copy_2_alt];
            CircuitMonad::constant_folding(&mut circuits__);
            assert_eq!(
                circuits__[0], circuits__[1],
                "{} != {}",
                circuits__[0], circuits__[1]
            );
            assert_eq!(
                circuits__[1], circuits__[0],
                "{} != {}",
                circuits__[1], circuits__[0]
            );

            // Verify that constant folding handles a + 0 = a
            let copy_3 = deep_copy(&circuit.circuit.as_ref().borrow());
            let copy_3_alt = copy_3.clone() + zero.clone();
            assert_ne!(copy_3, copy_3_alt);
            let mut circuits___ = [copy_3, copy_3_alt];
            CircuitMonad::constant_folding(&mut circuits___);
            assert_eq!(
                circuits___[0], circuits___[1],
                "{} != {}",
                circuits___[0], circuits___[1]
            );
            assert_eq!(
                circuits___[1], circuits___[0],
                "{} != {}",
                circuits___[1], circuits___[0]
            );

            // Verify that constant folding handles a + (a * 0) = a
            let copy_4 = deep_copy(&circuit.circuit.as_ref().borrow());
            let copy_4_alt = copy_4.clone() + copy_4.clone() * zero.clone();
            assert_ne!(copy_4, copy_4_alt);
            let mut circuits____ = [copy_4, copy_4_alt];
            CircuitMonad::constant_folding(&mut circuits____);
            assert_eq!(
                circuits____[0], circuits____[1],
                "{} != {}",
                circuits____[0], circuits____[1]
            );
            assert_eq!(
                circuits____[1], circuits____[0],
                "{} != {}",
                circuits____[1], circuits____[0]
            );

            // Verify that constant folding handles a + (0 * a) = a
            let copy_5 = deep_copy(&circuit.circuit.as_ref().borrow());
            let copy_5_alt = copy_5.clone() + copy_5.clone() * zero.clone();
            assert_ne!(copy_5, copy_5_alt);
            let mut circuits_____ = [copy_5, copy_5_alt];
            CircuitMonad::constant_folding(&mut circuits_____);
            assert_eq!(
                circuits_____[0], circuits_____[1],
                "{} != {}",
                circuits_____[0], circuits_____[1]
            );
            assert_eq!(
                circuits_____[1], circuits_____[0],
                "{} != {}",
                circuits_____[1], circuits_____[0]
            );

            // Verify that constant folding does not equate `0 - a` with `a`
            // But only if `a != 0`
            let copy_6 = deep_copy(&circuit.circuit.as_ref().borrow());
            let zero_minus_copy_6 = zero.clone() - copy_6.clone();
            assert_ne!(copy_6, zero_minus_copy_6);
            let mut circuits______ = [copy_6, zero_minus_copy_6];
            CircuitMonad::constant_folding(&mut circuits______);
            let copy_6_is_zero = circuits______[0].circuit.as_ref().borrow().is_zero();
            let copy_6_expr = circuits______[0]
                .circuit
                .as_ref()
                .borrow()
                .expression
                .clone();
            let zero_minus_copy_6_expr = circuits______[1]
                .circuit
                .as_ref()
                .borrow()
                .expression
                .clone();

            if copy_6_is_zero
                && (matches!(copy_6_expr, CircuitExpression::Constant(_))
                    && matches!(zero_minus_copy_6_expr, CircuitExpression::Constant(_)))
            {
                assert_eq!(
                    circuits______[0], circuits______[1],
                    "{} != {}",
                    circuits______[0], circuits______[1]
                );
                assert_eq!(
                    circuits______[1], circuits______[0],
                    "{} != {}",
                    circuits______[1], circuits______[0]
                );
            } else {
                assert_ne!(
                    circuits______[0], circuits______[1],
                    "{} == {}",
                    circuits______[0], circuits______[1]
                );
                assert_ne!(
                    circuits______[1], circuits______[0],
                    "{} == {}",
                    circuits______[1], circuits______[0]
                );
            }

            // Verify that constant folding handles a - 0 = a
            let copy_7 = deep_copy(&circuit.circuit.as_ref().borrow());
            let copy_7_alt = copy_7.clone() - zero.clone();
            assert_ne!(copy_7, copy_7_alt);
            let mut circuits_______ = [copy_7, copy_7_alt];
            CircuitMonad::constant_folding(&mut circuits_______);
            assert_eq!(
                circuits_______[0], circuits_______[1],
                "{} != {}",
                circuits_______[0], circuits_______[1]
            );
            assert_eq!(
                circuits_______[1], circuits_______[0],
                "{} != {}",
                circuits_______[1], circuits_______[0]
            );
        }
    }
}
