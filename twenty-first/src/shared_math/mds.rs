use std::collections::HashSet;
use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};

use itertools::Itertools;
use num_traits::{One, Zero};

use crate::shared_math::circuit::{CircuitBuilder, CircuitMonad};
use std::hash::Hash;

use super::circuit::{Circuit, CircuitExpression};

const CUTOFF: usize = 2;

fn schoolbook<T: Clone + Add<Output = T> + Mul<Output = T>>(
    a: &[T],
    b: &[T],
    n: usize,
    zero: T,
) -> Vec<T> {
    let mut res = vec![zero; 2 * n - 1];
    for i in 0..n {
        for j in 0..n {
            res[i + j] = res[i + j].clone() + a[i].clone() * b[j].clone();
        }
    }
    res
}

fn karatsuba<T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T>>(
    a: &[T],
    b: &[T],
    n: usize,
    zero: T,
) -> Vec<T> {
    debug_assert_eq!(n & (n - 1), 0);
    if n <= CUTOFF {
        return schoolbook(a, b, n, zero);
    }

    let half = n / 2;

    let alo = &a[0..half];
    let ahi = &a[half..];
    let asu = alo
        .iter()
        .zip(ahi.iter())
        .map(|(l, r)| l.to_owned() + r.to_owned())
        .collect_vec();

    let blo = &b[0..half];
    let bhi = &b[half..];
    let bsu = blo
        .iter()
        .zip(bhi.iter())
        .map(|(l, r)| l.to_owned() + r.to_owned())
        .collect_vec();

    let los = karatsuba(alo, blo, half, zero.clone());
    let his = karatsuba(ahi, bhi, half, zero.clone());
    let sus = karatsuba(&asu, &bsu, half, zero.clone());

    let mut c = vec![zero; 2 * n - 1];
    for i in 0..los.len() {
        c[i] = c[i].clone() + los[i].clone();
    }
    for i in 0..sus.len() {
        c[half + i] = c[half + i].clone() + sus[i].clone() - los[i].clone() - his[i].clone();
    }
    for i in 0..his.len() {
        c[n + i] = c[n + i].clone() + his[i].clone();
    }

    c
}

fn karatsuba_negacyclic_mul<
    T: Clone + Debug + Sub<Output = T> + Add<Output = T> + Mul<Output = T>,
>(
    a: &[T],
    b: &[T],
    n: usize,
    zero: T,
) -> Vec<T> {
    let prod = karatsuba(a, b, n, zero.clone());
    let mut res = vec![zero; n];
    for i in 0..n - 1 {
        res[i] = prod[i].clone() - prod[n + i].clone();
    }
    res[n - 1] = prod[n - 1].clone();
    res
}

fn quadratic_negacyclic_mul<
    T: Clone + Debug + Sub<Output = T> + Add<Output = T> + Mul<Output = T>,
>(
    a: &[T],
    b: &[T],
    n: usize,
    zero: T,
) -> Vec<T> {
    debug_assert_eq!(a.len(), n);
    debug_assert_eq!(b.len(), n);
    if n == 1 {
        return vec![a[0].clone() * b[0].clone()];
    }
    let mut res = vec![zero; n];
    let mut a_row: Vec<T> = a.to_vec();
    for i in 1..n {
        a_row[i] = a[n - i].clone();
    }
    for i in 0..n {
        for j in 0..n {
            let product = b[j].clone() * a_row[(n - i + j) % n].clone();
            if i < j {
                res[i] = res[i].clone() - product;
            } else {
                res[i] = res[i].clone() + product;
            }
        }
    }
    res
}

fn quadratic_cyclic_mul<T: Clone + Debug + Sub<Output = T> + Add<Output = T> + Mul<Output = T>>(
    a: &[T],
    b: &[T],
    n: usize,
    zero: T,
) -> Vec<T> {
    debug_assert_eq!(a.len(), n);
    debug_assert_eq!(b.len(), n);
    if n == 1 {
        return vec![a[0].clone() * b[0].clone()];
    }
    let mut a_row: Vec<T> = a.to_vec();
    for i in 1..n {
        a_row[i] = a[n - i].clone();
    }
    let mut res = vec![zero; n];
    for i in 0..n {
        for j in 0..n {
            let product = b[j].clone() * a_row[(n - i + j) % n].clone();
            res[i] = res[i].clone() + product;
        }
    }
    res
}

fn karatsuba_cyclic_mul<T: Clone + Sub<Output = T> + Add<Output = T> + Mul<Output = T>>(
    a: &[T],
    b: &[T],
    n: usize,
    zero: T,
) -> Vec<T> {
    let prod = karatsuba(a, b, n, zero.clone());
    let mut res = vec![zero; n];
    for i in 0..n - 1 {
        res[i] = prod[i].clone() + prod[n + i].clone();
    }
    res[n - 1] = prod[n - 1].clone();
    res
}

pub fn recursive_cyclic_mul<
    T: Clone + Debug + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
>(
    a: &[T],
    b: &[T],
    n: usize,
    zero: T,
    two: T,
) -> Vec<T> {
    debug_assert_eq!(n & (n - 1), 0);
    if n <= CUTOFF {
        let mut res = quadratic_cyclic_mul(a, b, n, zero);
        let mut nn = n;
        while nn > 1 {
            res = res.into_iter().map(|i| i * two.clone()).collect();
            nn >>= 1;
        }
        return res;
    }

    let half = n / 2;

    let alo = &a[0..half];
    let ahi = &a[half..];

    let blo = &b[0..half];
    let bhi = &b[half..];

    let asum = alo
        .iter()
        .zip(ahi.iter())
        .map(|(l, r)| l.to_owned() + r.to_owned())
        .collect_vec();

    let bsum = blo
        .iter()
        .zip(bhi.iter())
        .map(|(l, r)| l.to_owned() + r.to_owned())
        .collect_vec();

    let sums = recursive_cyclic_mul(&asum, &bsum, half, zero.clone(), two.clone());

    let adiff = alo
        .iter()
        .zip(ahi.iter())
        .map(|(l, r)| l.to_owned() - r.to_owned())
        .collect_vec();

    let bdiff = blo
        .iter()
        .zip(bhi.iter())
        .map(|(l, r)| l.to_owned() - r.to_owned())
        .collect_vec();

    let mut diffs = karatsuba_negacyclic_mul(&adiff, &bdiff, half, zero.clone());
    let mut nnn = half;
    while nnn > 1 {
        diffs = diffs.into_iter().map(|i| i * two.clone()).collect_vec();
        nnn >>= 1;
    }

    let mut res = vec![zero; n];
    for i in 0..half {
        res[i] = sums[i].clone() + diffs[i].clone();
        res[i + half] = sums[i].clone() - diffs[i].clone();
    }

    res
}

fn build_recursive_cyclic_mul_circuit() -> [Circuit<i64>; 16] {
    const STATE_SIZE: usize = 16;
    const MDS_MATRIX_FIRST_COLUMN: [i64; STATE_SIZE] = [
        61402, 1108, 28750, 33823, 7454, 43244, 53865, 12034, 56951, 27521, 41351, 40901, 12021,
        59689, 26798, 17845,
    ];
    type T = i64;
    let mut builder = CircuitBuilder::<T>::new();

    let inputs: [CircuitMonad<T>; STATE_SIZE] = (0..STATE_SIZE)
        .map(|i| builder.input(i))
        .collect_vec()
        .try_into()
        .unwrap();
    let mds_column = MDS_MATRIX_FIRST_COLUMN.map(|c| builder.constant(c));
    let mut outputs = recursive_cyclic_mul(
        &inputs,
        &mds_column,
        STATE_SIZE,
        builder.constant(0),
        builder.constant(2),
    );

    // CircuitMonad::constant_folding(&mut outputs);
    CircuitMonad::distribute_constants(&mut outputs);
    CircuitMonad::constant_folding(&mut outputs);
    CircuitMonad::move_coefficients_right(&mut outputs);

    let res = outputs
        .into_iter()
        .map(|o| o.consume())
        .collect_vec()
        .try_into()
        .unwrap();

    // Circuit::assert_all_evaluate_different(&res, input)

    res
}

/// Return a variable name for the node. Returns `point[n]` if node is just
/// a value from the codewords. Otherwise returns the ID of the circuit.
fn get_binding_name<T>(circuit: &Circuit<T>) -> String
where
    T: Clone
        + Debug
        + Add<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + Zero
        + One
        + Hash
        + PartialEq
        + Eq,
{
    match &circuit.expression {
        CircuitExpression::Constant(c) => format!("{:?}", c),
        CircuitExpression::Input(idx) => format!("input[{idx}]"),
        CircuitExpression::BinaryOperation(_, _, _) => format!("node_{}", circuit.id),
    }
}

/// Return (1) the code for evaluating a single node and (2) a list of symbols that this evaluation
/// depends on.
fn evaluate_single_node<T>(
    requested_visited_count: usize,
    circuit: &Circuit<T>,
    in_scope: &HashSet<usize>,
) -> (String, Vec<String>)
where
    T: Clone
        + Debug
        + Add<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + Zero
        + One
        + Hash
        + PartialEq
        + Eq,
{
    let mut output = String::default();
    // If this node has already been declared, or visit counter is higher than requested,
    // than the node value *must* be in scope, meaning that we can just reference it.
    if circuit.visited_counter > requested_visited_count || in_scope.contains(&circuit.id) {
        let binding_name = get_binding_name(circuit);
        output.push_str(&binding_name);
        return match &circuit.expression {
            CircuitExpression::BinaryOperation(_, _, _) => (output, vec![binding_name]),
            _ => (output, vec![]),
        };
    }

    // If variable is not already in scope, then we must generate the expression to evaluate it.
    let mut dependent_symbols = vec![];
    match &circuit.expression {
        CircuitExpression::BinaryOperation(binop, lhs, rhs) => {
            output.push('(');
            let (to_output_lhs, lhs_symbols) =
                evaluate_single_node(requested_visited_count, &lhs.as_ref().borrow(), in_scope);
            output.push_str(&to_output_lhs);
            output.push(')');
            output.push_str(&binop.to_string());
            output.push('(');
            let (to_output_rhs, rhs_symbols) =
                evaluate_single_node(requested_visited_count, &rhs.as_ref().borrow(), in_scope);
            output.push_str(&to_output_rhs);
            output.push(')');

            let ret_as_vec = vec![lhs_symbols, rhs_symbols].concat();
            let ret_as_hash_set: HashSet<String> = ret_as_vec.into_iter().collect();
            dependent_symbols = ret_as_hash_set.into_iter().collect_vec()
        }
        _ => output.push_str(&get_binding_name(circuit)),
    }

    (output, dependent_symbols)
}

fn declare_single_node_with_visit_count<T>(
    requested_visited_count: usize,
    circuit: &Circuit<T>,
    in_scope: &mut HashSet<usize>,
    output: &mut String,
) where
    T: Clone
        + Debug
        + Add<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + Zero
        + One
        + Hash
        + PartialEq
        + Eq,
{
    if circuit.visited_counter < requested_visited_count {
        // If the visited counter is not there yet, make a recursive call. We are
        // not yet ready to bind this node's ID to a value.
        if let CircuitExpression::<T>::BinaryOperation(_binop, lhs, rhs) = &circuit.expression {
            declare_single_node_with_visit_count(
                requested_visited_count,
                &lhs.as_ref().borrow(),
                in_scope,
                output,
            );
            declare_single_node_with_visit_count(
                requested_visited_count,
                &rhs.as_ref().borrow(),
                in_scope,
                output,
            );
        }
        return;
    }

    // If this node has already been declared, or visit counter is higher than requested,
    // then the node value *must* already be in scope. We should not redeclare it.
    // We also do not declare nodes that are e.g `row[3]` since they are already in scope
    // through the `points` input argument, and we do not declare constants.
    if circuit.visited_counter > requested_visited_count
        || in_scope.contains(&circuit.id)
        || !matches!(
            circuit.expression,
            CircuitExpression::BinaryOperation(_, _, _)
        )
    {
        return;
    }

    // If this line is met, it means that the visit count is as requested, and that
    // the value is not in scope. So it must be added to the scope. We find the
    // expression for the value, and then put it into scope through a let expression
    if circuit.visited_counter == requested_visited_count && !in_scope.contains(&circuit.id) {
        let binding_name = get_binding_name(circuit);
        output.push_str(&format!("let {binding_name} =\n"));
        let (to_output, _) = evaluate_single_node(requested_visited_count, circuit, in_scope);
        output.push_str(&to_output);
        output.push_str(";\n");

        let new_insertion = in_scope.insert(circuit.id);
        // sanity check: don't declare same node multiple times
        assert!(new_insertion);
    }
}

/// Produce the code to evaluate code for all nodes that share a value number of
/// times visited. A value for all nodes with a higher count than the provided are assumed
/// to be in scope.
fn declare_nodes_with_visit_count<T>(
    requested_visited_count: usize,
    circuits: &[Circuit<T>],
) -> String
where
    T: Clone
        + Debug
        + Add<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + Zero
        + One
        + Hash
        + PartialEq
        + Eq,
{
    let mut in_scope: HashSet<usize> = HashSet::new();
    let mut output = String::default();

    for circuit in circuits.iter() {
        declare_single_node_with_visit_count(
            requested_visited_count,
            circuit,
            &mut in_scope,
            &mut output,
        );
    }

    output
}

fn turn_circuits_into_string<T>(circuits: &mut [Circuit<T>]) -> String
where
    T: Clone
        + Debug
        + Add<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + Zero
        + One
        + Hash
        + PartialEq
        + Eq,
{
    // Assert that all node IDs are unique (sanity check)
    Circuit::assert_has_unique_ids(circuits);

    // Count number of times each node is visited
    Circuit::traverse_multiple(circuits);

    // Get all values for the visited counters in the entire multi-circuit
    let mut visited_counters = vec![];
    for constraint in circuits.iter() {
        visited_counters.append(&mut constraint.get_all_visited_counters());
    }

    visited_counters.sort_unstable();
    visited_counters.reverse();
    visited_counters.dedup();

    // Declare shared values
    // In the main function we predeclare all variables with a visit count of more than 1
    // These declarations must be made from the highest count number to the lowest, otherwise
    // the code will refer to bindings that have not yet been made
    let mut shared_evaluations: Vec<String> = vec![];
    for visited_counter in visited_counters {
        if visited_counter == 1 {
            continue;
        }
        shared_evaluations.push(declare_nodes_with_visit_count(visited_counter, circuits));
    }

    let shared_declarations = shared_evaluations.join("");

    let mut expressions: Vec<String> = vec![];
    for constraint in circuits.iter() {
        // Build code for expressions that evaluate to the constraints
        let (evaluation, _dependent_symbols) =
            evaluate_single_node(1, constraint, &HashSet::default());
        expressions.push(evaluation);
    }

    let evaluations_joined = expressions.join(",\n");

    format!(
        "{shared_declarations}

        let res = [{evaluations_joined}];

        res"
    )
}

pub fn generated_function(input: &[i64]) -> [i64; 16] {
    let node_444 = (((((2) * ((288592) * (input[0]))) + ((2) * ((288592) * (input[8]))))
        + (((2) * ((288592) * (input[4]))) + ((2) * ((288592) * (input[12])))))
        + ((((2) * ((288592) * (input[2]))) + ((2) * ((288592) * (input[10]))))
            + (((2) * ((288592) * (input[6]))) + ((2) * ((288592) * (input[14]))))))
        + (((((2) * ((236165) * (input[1]))) + ((2) * ((236165) * (input[9]))))
            + (((2) * ((236165) * (input[5]))) + ((2) * ((236165) * (input[13])))))
            + ((((2) * ((236165) * (input[3]))) + ((2) * ((236165) * (input[11]))))
                + (((2) * ((236165) * (input[7]))) + ((2) * ((236165) * (input[15]))))));
    let node_453 = (((((2) * ((-12936) * (input[0]))) + ((2) * ((-12936) * (input[8]))))
        + (((2) * ((-12936) * (input[4]))) + ((2) * ((-12936) * (input[12])))))
        - ((((2) * ((-12936) * (input[2]))) + ((2) * ((-12936) * (input[10]))))
            + (((2) * ((-12936) * (input[6]))) + ((2) * ((-12936) * (input[14]))))))
        - (((((2) * ((26959) * (input[1]))) + ((2) * ((26959) * (input[9]))))
            + (((2) * ((26959) * (input[5]))) + ((2) * ((26959) * (input[13])))))
            - ((((2) * ((26959) * (input[3]))) + ((2) * ((26959) * (input[11]))))
                + (((2) * ((26959) * (input[7]))) + ((2) * ((26959) * (input[15]))))));
    let node_735 = ((2) * ((2) * ((98878) * (input[0])))) + ((2) * ((2) * ((98878) * (input[8]))));
    let node_741 = ((2) * ((2) * ((98878) * (input[4])))) + ((2) * ((2) * ((98878) * (input[12]))));
    let node_963 =
        ((2) * ((2) * ((-74304) * (input[1])))) + ((2) * ((2) * ((-74304) * (input[9]))));
    let node_969 =
        ((2) * ((2) * ((-74304) * (input[5])))) + ((2) * ((2) * ((-74304) * (input[13]))));
    let node_903 = ((2) * ((2) * ((44845) * (input[3])))) + ((2) * ((2) * ((44845) * (input[11]))));
    let node_909 = ((2) * ((2) * ((44845) * (input[7])))) + ((2) * ((2) * ((44845) * (input[15]))));
    let node_861 =
        ((2) * ((2) * ((-10562) * (input[2])))) + ((2) * ((2) * ((-10562) * (input[10]))));
    let node_867 =
        ((2) * ((2) * ((-10562) * (input[6])))) + ((2) * ((2) * ((-10562) * (input[14]))));
    let node_612 = ((2) * ((2) * ((2) * ((4451) * (input[0])))))
        - ((2) * ((2) * ((2) * ((4451) * (input[8])))));
    let node_1173 = ((2) * ((2) * ((2) * ((-26413) * (input[1])))))
        - ((2) * ((2) * ((2) * ((-26413) * (input[9])))));
    let node_1101 = ((2) * ((2) * ((2) * ((-7078) * (input[3])))))
        - ((2) * ((2) * ((2) * ((-7078) * (input[11])))));
    let node_1005 = ((2) * ((2) * ((2) * ((-12601) * (input[2])))))
        - ((2) * ((2) * ((2) * ((-12601) * (input[10])))));
    let node_1119 = ((2) * ((2) * ((2) * ((-16445) * (input[5])))))
        - ((2) * ((2) * ((2) * ((-16445) * (input[13])))));
    let node_1023 = ((2) * ((2) * ((2) * ((-5811) * (input[7])))))
        - ((2) * ((2) * ((2) * ((-5811) * (input[15])))));
    let node_945 = ((2) * ((2) * ((2) * ((27067) * (input[6])))))
        - ((2) * ((2) * ((2) * ((27067) * (input[14])))));
    let node_795 = ((2) * ((2) * ((2) * ((-4567) * (input[4])))))
        - ((2) * ((2) * ((2) * ((-4567) * (input[12])))));
    let node_1290 = (((((2) * ((288592) * (input[1]))) + ((2) * ((288592) * (input[9]))))
        + (((2) * ((288592) * (input[5]))) + ((2) * ((288592) * (input[13])))))
        + ((((2) * ((288592) * (input[3]))) + ((2) * ((288592) * (input[11]))))
            + (((2) * ((288592) * (input[7]))) + ((2) * ((288592) * (input[15]))))))
        + (((((2) * ((236165) * (input[0]))) + ((2) * ((236165) * (input[8]))))
            + (((2) * ((236165) * (input[4]))) + ((2) * ((236165) * (input[12])))))
            + ((((2) * ((236165) * (input[2]))) + ((2) * ((236165) * (input[10]))))
                + (((2) * ((236165) * (input[6]))) + ((2) * ((236165) * (input[14]))))));
    let node_1299 = (((((2) * ((26959) * (input[0]))) + ((2) * ((26959) * (input[8]))))
        + (((2) * ((26959) * (input[4]))) + ((2) * ((26959) * (input[12])))))
        - ((((2) * ((26959) * (input[2]))) + ((2) * ((26959) * (input[10]))))
            + (((2) * ((26959) * (input[6]))) + ((2) * ((26959) * (input[14]))))))
        + (((((2) * ((-12936) * (input[1]))) + ((2) * ((-12936) * (input[9]))))
            + (((2) * ((-12936) * (input[5]))) + ((2) * ((-12936) * (input[13])))))
            - ((((2) * ((-12936) * (input[3]))) + ((2) * ((-12936) * (input[11]))))
                + (((2) * ((-12936) * (input[7]))) + ((2) * ((-12936) * (input[15]))))));
    let node_1644 =
        ((2) * ((2) * ((-74304) * (input[0])))) + ((2) * ((2) * ((-74304) * (input[8]))));
    let node_1650 =
        ((2) * ((2) * ((-74304) * (input[4])))) + ((2) * ((2) * ((-74304) * (input[12]))));
    let node_1656 = ((2) * ((2) * ((98878) * (input[1])))) + ((2) * ((2) * ((98878) * (input[9]))));
    let node_1662 =
        ((2) * ((2) * ((98878) * (input[5])))) + ((2) * ((2) * ((98878) * (input[13]))));
    let node_1668 =
        ((2) * ((2) * ((44845) * (input[2])))) + ((2) * ((2) * ((44845) * (input[10]))));
    let node_1674 =
        ((2) * ((2) * ((44845) * (input[6])))) + ((2) * ((2) * ((44845) * (input[14]))));
    let node_1680 =
        ((2) * ((2) * ((-10562) * (input[3])))) + ((2) * ((2) * ((-10562) * (input[11]))));
    let node_1686 =
        ((2) * ((2) * ((-10562) * (input[7])))) + ((2) * ((2) * ((-10562) * (input[15]))));
    let node_1563 = ((2) * ((2) * ((2) * ((-26413) * (input[0])))))
        - ((2) * ((2) * ((2) * ((-26413) * (input[8])))));
    let node_1572 = ((2) * ((2) * ((2) * ((4451) * (input[1])))))
        - ((2) * ((2) * ((2) * ((4451) * (input[9])))));
    let node_1785 = ((2) * ((2) * ((2) * ((-7078) * (input[2])))))
        - ((2) * ((2) * ((2) * ((-7078) * (input[10])))));
    let node_1794 = ((2) * ((2) * ((2) * ((-12601) * (input[3])))))
        - ((2) * ((2) * ((2) * ((-12601) * (input[11])))));
    let node_1749 = ((2) * ((2) * ((2) * ((-5811) * (input[6])))))
        - ((2) * ((2) * ((2) * ((-5811) * (input[14])))));
    let node_1758 = ((2) * ((2) * ((2) * ((27067) * (input[7])))))
        - ((2) * ((2) * ((2) * ((27067) * (input[15])))));
    let node_1713 = ((2) * ((2) * ((2) * ((-16445) * (input[4])))))
        - ((2) * ((2) * ((2) * ((-16445) * (input[12])))));
    let node_1722 = ((2) * ((2) * ((2) * ((-4567) * (input[5])))))
        - ((2) * ((2) * ((2) * ((-4567) * (input[13])))));
    let node_87 = (node_444) + (node_453);
    let node_474 = ((node_735) - (node_741))
        - ((((((((2) * ((2) * ((-29459) * (input[1]))))
            + ((2) * ((2) * ((-29459) * (input[9])))))
            - (((2) * ((2) * ((-29459) * (input[5]))))
                + ((2) * ((2) * ((-29459) * (input[13]))))))
            + ((((2) * ((2) * ((-29459) * (input[3]))))
                + ((2) * ((2) * ((-29459) * (input[11])))))
                - (((2) * ((2) * ((-29459) * (input[7]))))
                    + ((2) * ((2) * ((-29459) * (input[15])))))))
            - ((node_963) - (node_969)))
            - ((node_903) - (node_909)))
            + ((node_861) - (node_867)));
    let node_1218 = ((2) * ((2) * ((2) * ((-42858) * (input[1])))))
        - ((2) * ((2) * ((2) * ((-42858) * (input[9])))));
    let node_1227 = ((2) * ((2) * ((2) * ((-42858) * (input[5])))))
        - ((2) * ((2) * ((2) * ((-42858) * (input[13])))));
    let node_1146 = ((2) * ((2) * ((2) * ((-12889) * (input[3])))))
        - ((2) * ((2) * ((2) * ((-12889) * (input[11])))));
    let node_1155 = ((2) * ((2) * ((2) * ((-12889) * (input[7])))))
        - ((2) * ((2) * ((2) * ((-12889) * (input[15])))));
    let node_1074 = ((2) * ((2) * ((2) * ((14466) * (input[2])))))
        - ((2) * ((2) * ((2) * ((14466) * (input[10])))));
    let node_1083 = ((2) * ((2) * ((2) * ((14466) * (input[6])))))
        - ((2) * ((2) * ((2) * ((14466) * (input[14])))));
    let node_1236 = ((2) * ((2) * ((2) * ((-33491) * (input[1])))))
        - ((2) * ((2) * ((2) * ((-33491) * (input[9])))));
    let node_1245 = ((2) * ((2) * ((2) * ((-33491) * (input[3])))))
        - ((2) * ((2) * ((2) * ((-33491) * (input[11])))));
    let node_1182 = ((2) * ((2) * ((2) * ((-22256) * (input[5])))))
        - ((2) * ((2) * ((2) * ((-22256) * (input[13])))));
    let node_1191 = ((2) * ((2) * ((2) * ((-22256) * (input[7])))))
        - ((2) * ((2) * ((2) * ((-22256) * (input[15])))));
    let node_89 = (node_1290) + (node_1299);
    let node_1317 = (((node_1644) - (node_1650)) + ((node_1656) - (node_1662)))
        - (((node_1668) - (node_1674)) + ((node_1680) - (node_1686)));
    let node_1803 = ((2) * ((2) * ((2) * ((-12889) * (input[2])))))
        - ((2) * ((2) * ((2) * ((-12889) * (input[10])))));
    let node_1812 = ((2) * ((2) * ((2) * ((-12889) * (input[6])))))
        - ((2) * ((2) * ((2) * ((-12889) * (input[14])))));
    let node_1821 = ((2) * ((2) * ((2) * ((14466) * (input[3])))))
        - ((2) * ((2) * ((2) * ((14466) * (input[11])))));
    let node_1830 = ((2) * ((2) * ((2) * ((14466) * (input[7])))))
        - ((2) * ((2) * ((2) * ((14466) * (input[15])))));
    let node_88 = (node_444) - (node_453);
    let node_1839 = (((((node_963) - (node_969))
        + (((((2) * ((2) * ((88316) * (input[0])))) + ((2) * ((2) * ((88316) * (input[8])))))
            - (((2) * ((2) * ((88316) * (input[4]))))
                + ((2) * ((2) * ((88316) * (input[12]))))))
            + ((((2) * ((2) * ((88316) * (input[2]))))
                + ((2) * ((2) * ((88316) * (input[10])))))
                - (((2) * ((2) * ((88316) * (input[6]))))
                    + ((2) * ((2) * ((88316) * (input[14]))))))))
        - ((node_735) - (node_741)))
        - ((node_861) - (node_867)))
        - ((node_903) - (node_909));
    let node_2051 = ((2) * ((2) * ((2) * ((-8150) * (input[0])))))
        - ((2) * ((2) * ((2) * ((-8150) * (input[8])))));
    let node_2060 = ((2) * ((2) * ((2) * ((-8150) * (input[2])))))
        - ((2) * ((2) * ((2) * ((-8150) * (input[10])))));
    let node_2102 = ((2) * ((2) * ((2) * ((22500) * (input[4])))))
        - ((2) * ((2) * ((2) * ((22500) * (input[12])))));
    let node_2111 = ((2) * ((2) * ((2) * ((22500) * (input[6])))))
        - ((2) * ((2) * ((2) * ((22500) * (input[14])))));
    let node_90 = (node_1290) - (node_1299);
    let node_2122 = (((((((2) * ((2) * ((-29459) * (input[0]))))
        + ((2) * ((2) * ((-29459) * (input[8])))))
        - (((2) * ((2) * ((-29459) * (input[4])))) + ((2) * ((2) * ((-29459) * (input[12]))))))
        + ((((2) * ((2) * ((-29459) * (input[2])))) + ((2) * ((2) * ((-29459) * (input[10])))))
            - (((2) * ((2) * ((-29459) * (input[6]))))
                + ((2) * ((2) * ((-29459) * (input[14])))))))
        + (((((2) * ((2) * ((88316) * (input[1])))) + ((2) * ((2) * ((88316) * (input[9])))))
            - (((2) * ((2) * ((88316) * (input[5]))))
                + ((2) * ((2) * ((88316) * (input[13]))))))
            + ((((2) * ((2) * ((88316) * (input[3]))))
                + ((2) * ((2) * ((88316) * (input[11])))))
                - (((2) * ((2) * ((88316) * (input[7]))))
                    + ((2) * ((2) * ((88316) * (input[15]))))))))
        - (((node_1644) - (node_1650)) + ((node_1656) - (node_1662))))
        - (((node_1668) - (node_1674)) + ((node_1680) - (node_1686)));
    let node_2469 = ((2) * ((2) * ((2) * ((-33491) * (input[0])))))
        - ((2) * ((2) * ((2) * ((-33491) * (input[8])))));
    let node_2478 = ((2) * ((2) * ((2) * ((-33491) * (input[2])))))
        - ((2) * ((2) * ((2) * ((-33491) * (input[10])))));
    let node_2487 = ((2) * ((2) * ((2) * ((-8150) * (input[1])))))
        - ((2) * ((2) * ((2) * ((-8150) * (input[9])))));
    let node_2496 = ((2) * ((2) * ((2) * ((-8150) * (input[3])))))
        - ((2) * ((2) * ((2) * ((-8150) * (input[11])))));
    let node_2505 = ((2) * ((2) * ((2) * ((-22256) * (input[4])))))
        - ((2) * ((2) * ((2) * ((-22256) * (input[12])))));
    let node_2514 = ((2) * ((2) * ((2) * ((-22256) * (input[6])))))
        - ((2) * ((2) * ((2) * ((-22256) * (input[14])))));
    let node_2523 = ((2) * ((2) * ((2) * ((22500) * (input[5])))))
        - ((2) * ((2) * ((2) * ((22500) * (input[13])))));
    let node_2532 = ((2) * ((2) * ((2) * ((22500) * (input[7])))))
        - ((2) * ((2) * ((2) * ((22500) * (input[15])))));
    let node_2649 = ((2) * ((2) * ((2) * ((-116) * (input[0])))))
        - ((2) * ((2) * ((2) * ((-116) * (input[8])))));
    let node_2658 = ((2) * ((2) * ((2) * ((-116) * (input[4])))))
        - ((2) * ((2) * ((2) * ((-116) * (input[12])))));
    let node_2802 = ((2) * ((2) * ((2) * ((-42858) * (input[0])))))
        - ((2) * ((2) * ((2) * ((-42858) * (input[8])))));
    let node_2811 = ((2) * ((2) * ((2) * ((-42858) * (input[4])))))
        - ((2) * ((2) * ((2) * ((-42858) * (input[12])))));
    let node_2820 = ((2) * ((2) * ((2) * ((-116) * (input[1])))))
        - ((2) * ((2) * ((2) * ((-116) * (input[9])))));
    let node_2829 = ((2) * ((2) * ((2) * ((-116) * (input[5])))))
        - ((2) * ((2) * ((2) * ((-116) * (input[13])))));
    let node_153 = (node_87) + (node_474);
    let node_525 = (node_612)
        - (((((((((((2) * ((2) * ((2) * ((-55747) * (input[1])))))
            - ((2) * ((2) * ((2) * ((-55747) * (input[9]))))))
            + (((2) * ((2) * ((2) * ((-55747) * (input[5])))))
                - ((2) * ((2) * ((2) * ((-55747) * (input[13])))))))
            + ((((2) * ((2) * ((2) * ((-55747) * (input[3])))))
                - ((2) * ((2) * ((2) * ((-55747) * (input[11]))))))
                + (((2) * ((2) * ((2) * ((-55747) * (input[7])))))
                    - ((2) * ((2) * ((2) * ((-55747) * (input[15]))))))))
            - ((node_1218) + (node_1227)))
            - ((node_1146) + (node_1155)))
            + ((node_1074) + (node_1083)))
            - (((((node_1236) + (node_1245)) - (node_1173)) - (node_1101)) + (node_1005)))
            - (((((node_1182) + (node_1191)) - (node_1119)) - (node_1023)) + (node_945)))
            + (node_795));
    let node_155 = (node_89) + (node_1317);
    let node_1356 = ((node_1563) + (node_1572))
        - ((((((node_1803) + (node_1812)) + ((node_1821) + (node_1830)))
            - ((node_1785) + (node_1794)))
            - ((node_1749) + (node_1758)))
            + ((node_1713) + (node_1722)));
    let node_157 = (node_88) + (node_1839);
    let node_1854 = ((((node_1173) + ((node_2051) + (node_2060))) - (node_612)) - (node_1005))
        - (((((node_1146) + (node_1155)) - (node_1101)) - (node_1023))
            + ((((node_1119) + ((node_2102) + (node_2111))) - (node_795)) - (node_945)));
    let node_159 = (node_90) + (node_2122);
    let node_2143 = (((((node_2469) + (node_2478)) + ((node_2487) + (node_2496)))
        - ((node_1563) + (node_1572)))
        - ((node_1785) + (node_1794)))
        - (((((node_2505) + (node_2514)) + ((node_2523) + (node_2532)))
            - ((node_1713) + (node_1722)))
            - ((node_1749) + (node_1758)));
    let node_154 = (node_87) - (node_474);
    let node_2544 = ((((((((node_1236) + (node_1245)) - (node_1173)) - (node_1101))
        + (node_1005))
        + ((node_2649) + (node_2658)))
        - (node_612))
        - (node_795))
        - (((((node_1182) + (node_1191)) - (node_1119)) - (node_1023)) + (node_945));
    let node_156 = (node_89) - (node_1317);
    let node_2691 = (((((node_1785) + (node_1794))
        + (((node_2802) + (node_2811)) + ((node_2820) + (node_2829))))
        - ((node_1563) + (node_1572)))
        - ((node_1713) + (node_1722)))
        - ((node_1749) + (node_1758));
    let node_158 = (node_88) - (node_1839);
    let node_2841 = ((((node_1101)
        + (((((node_1218) + (node_1227))
            + (((((2) * ((2) * ((2) * ((14350) * (input[0])))))
                - ((2) * ((2) * ((2) * ((14350) * (input[8]))))))
                + (((2) * ((2) * ((2) * ((14350) * (input[4])))))
                    - ((2) * ((2) * ((2) * ((14350) * (input[12])))))))
                + ((((2) * ((2) * ((2) * ((14350) * (input[2])))))
                    - ((2) * ((2) * ((2) * ((14350) * (input[10]))))))
                    + (((2) * ((2) * ((2) * ((14350) * (input[6])))))
                        - ((2) * ((2) * ((2) * ((14350) * (input[14])))))))))
            - ((node_2649) + (node_2658)))
            - ((node_1074) + (node_1083))))
        - ((((node_1173) + ((node_2051) + (node_2060))) - (node_612)) - (node_1005)))
        - ((((node_1119) + ((node_2102) + (node_2111))) - (node_795)) - (node_945)))
        - (node_1023);
    let node_160 = (node_90) - (node_2122);
    let node_3089 = (((((((((2) * ((2) * ((2) * ((-55747) * (input[0])))))
        - ((2) * ((2) * ((2) * ((-55747) * (input[8]))))))
        + (((2) * ((2) * ((2) * ((-55747) * (input[4])))))
            - ((2) * ((2) * ((2) * ((-55747) * (input[12])))))))
        + ((((2) * ((2) * ((2) * ((-55747) * (input[2])))))
            - ((2) * ((2) * ((2) * ((-55747) * (input[10]))))))
            + (((2) * ((2) * ((2) * ((-55747) * (input[6])))))
                - ((2) * ((2) * ((2) * ((-55747) * (input[14]))))))))
        + (((((2) * ((2) * ((2) * ((14350) * (input[1])))))
            - ((2) * ((2) * ((2) * ((14350) * (input[9]))))))
            + (((2) * ((2) * ((2) * ((14350) * (input[5])))))
                - ((2) * ((2) * ((2) * ((14350) * (input[13])))))))
            + ((((2) * ((2) * ((2) * ((14350) * (input[3])))))
                - ((2) * ((2) * ((2) * ((14350) * (input[11]))))))
                + (((2) * ((2) * ((2) * ((14350) * (input[7])))))
                    - ((2) * ((2) * ((2) * ((14350) * (input[15])))))))))
        - (((node_2802) + (node_2811)) + ((node_2820) + (node_2829))))
        - (((node_1803) + (node_1812)) + ((node_1821) + (node_1830))))
        - (((((node_2469) + (node_2478)) + ((node_2487) + (node_2496)))
            - ((node_1563) + (node_1572)))
            - ((node_1785) + (node_1794))))
        - (((((node_2505) + (node_2514)) + ((node_2523) + (node_2532)))
            - ((node_1713) + (node_1722)))
            - ((node_1749) + (node_1758)));

    let res = [
        (node_153) + (node_525),
        (node_155) + (node_1356),
        (node_157) + (node_1854),
        (node_159) + (node_2143),
        (node_154) + (node_2544),
        (node_156) + (node_2691),
        (node_158) + (node_2841),
        (node_160) + (node_3089),
        (node_153) - (node_525),
        (node_155) - (node_1356),
        (node_157) - (node_1854),
        (node_159) - (node_2143),
        (node_154) - (node_2544),
        (node_156) - (node_2691),
        (node_158) - (node_2841),
        (node_160) - (node_3089),
    ];

    res
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::{thread_rng, RngCore};

    use super::*;

    #[test]
    fn test_karatsuba() {
        let mut rng = thread_rng();
        let n = 8;
        let a = (0..n)
            .map(|_| (rng.next_u32() % (1 << 20)) as u64)
            .collect_vec();
        let b = (0..n)
            .map(|_| (rng.next_u32() % (1 << 20)) as u64)
            .collect_vec();

        let c_schoolbook = schoolbook(&a, &b, n, 0);
        let c_karatsuba = karatsuba(&a, &b, n, 0);

        assert_eq!(c_schoolbook, c_karatsuba);

        println!("{}", c_schoolbook.iter().map(|x| x.to_string()).join(","));
        println!("{}", c_karatsuba.iter().map(|x| x.to_string()).join(","));
    }

    #[test]
    fn test_negacyclic_mul() {
        let mut rng = thread_rng();
        let n = 8;
        let a = (0..n)
            .map(|_| (rng.next_u32() % (1 << 20)) as u64)
            .collect_vec();
        let b = (0..n)
            .map(|_| (rng.next_u32() % (1 << 20)) as u64)
            .collect_vec();

        println!("a = {a:?}");
        println!("b = {b:?}");

        let c_quadratic = quadratic_negacyclic_mul(&a, &b, n, 0);
        let c_karatsuba = karatsuba_negacyclic_mul(&a, &b, n, 0);

        assert_eq!(c_quadratic, c_karatsuba);

        println!("{}", c_quadratic.iter().map(|x| x.to_string()).join(","));
        println!("{}", c_karatsuba.iter().map(|x| x.to_string()).join(","));
    }

    #[test]
    fn test_recursive_cyclic_mul() {
        let mut rng = thread_rng();
        let n = 16;
        let a = (0..n).map(|_| rng.next_u32() as u64).collect_vec();
        let b = (0..n)
            .map(|_| (rng.next_u32() % (1 << 20)) as u64)
            .collect_vec();

        println!("a = {a:?}");
        println!("b = {b:?}");

        let c_quadratic = quadratic_cyclic_mul(&a, &b, n, 0);
        let c_karatsuba = karatsuba_cyclic_mul(&a, &b, n, 0);
        let c_recursive = recursive_cyclic_mul(&a, &b, n, 0, 2)
            .iter()
            .map(|i| i / n as u64)
            .collect_vec();

        assert_eq!(c_quadratic, c_karatsuba);
        assert_eq!(c_karatsuba, c_recursive);
        assert_eq!(c_quadratic, c_recursive);

        println!("{}", c_quadratic.iter().map(|x| x.to_string()).join(","));
        println!("{}", c_karatsuba.iter().map(|x| x.to_string()).join(","));
        println!("{}", c_recursive.iter().map(|x| x.to_string()).join(","));
    }

    #[test]
    fn test_rcm_circuit() {
        const STATE_SIZE: usize = 16;
        const MDS_MATRIX_FIRST_COLUMN: [i64; STATE_SIZE] = [
            61402, 1108, 28750, 33823, 7454, 43244, 53865, 12034, 56951, 27521, 41351, 40901,
            12021, 59689, 26798, 17845,
        ];
        let mut rng = thread_rng();
        let circuit = build_recursive_cyclic_mul_circuit();
        let a = (0..STATE_SIZE).map(|_| rng.next_u32() as i64).collect_vec();

        // Circuit::assert_all_evaluate_different(&circuit, a.clone());
        let direct = recursive_cyclic_mul(&a, &MDS_MATRIX_FIRST_COLUMN, STATE_SIZE, 0, 2);
        let thru_circuit = circuit.iter().map(|c| c.evaluate(&a)).collect_vec();

        assert_eq!(direct, thru_circuit);
        assert_eq!(direct, generated_function(&a).to_vec());
    }

    #[test]
    fn test_rcm_circuit_into_string() {
        let mut circuits = build_recursive_cyclic_mul_circuit();
        let string = turn_circuits_into_string(&mut circuits);
        println!("{string}");
    }
}
