use std::collections::HashSet;
use std::fmt::Debug;

use itertools::Itertools;
use num_traits::{One, WrappingAdd, WrappingMul, WrappingSub, Zero};

use crate::shared_math::circuit::{CircuitBuilder, CircuitMonad};
use std::hash::Hash;

use super::circuit::{BinOp, Circuit, CircuitExpression};

const KARATSUBA_CUTOFF: usize = 2;
const RCM_CUTOFF: usize = 1;

fn schoolbook<T: Clone + WrappingAdd<Output = T> + WrappingMul<Output = T>>(
    a: &[T],
    b: &[T],
    n: usize,
    zero: T,
) -> Vec<T> {
    let mut res = vec![zero; 2 * n - 1];
    for i in 0..n {
        for j in 0..n {
            res[i + j] = res[i + j].wrapping_add(&a[i].wrapping_mul(&b[j]));
        }
    }
    res
}

fn karatsuba<
    T: Clone + WrappingAdd<Output = T> + WrappingSub<Output = T> + WrappingMul<Output = T>,
>(
    a: &[T],
    b: &[T],
    n: usize,
    zero: T,
) -> Vec<T> {
    debug_assert_eq!(n & (n - 1), 0);
    if n <= KARATSUBA_CUTOFF {
        return schoolbook(a, b, n, zero);
    }

    let half = n / 2;

    let alo = &a[0..half];
    let ahi = &a[half..];
    let asu = alo
        .iter()
        .zip(ahi.iter())
        .map(|(l, r)| l.wrapping_add(r))
        .collect_vec();

    let blo = &b[0..half];
    let bhi = &b[half..];
    let bsu = blo
        .iter()
        .zip(bhi.iter())
        .map(|(l, r)| l.wrapping_add(r))
        .collect_vec();

    let los = karatsuba(alo, blo, half, zero.clone());
    let his = karatsuba(ahi, bhi, half, zero.clone());
    let sus = karatsuba(&asu, &bsu, half, zero.clone());

    let mut c = vec![zero; 2 * n - 1];
    for i in 0..los.len() {
        c[i] = c[i].wrapping_add(&los[i]);
    }
    for i in 0..sus.len() {
        c[half + i] = c[half + i]
            .wrapping_add(&sus[i])
            .wrapping_sub(&los[i])
            .wrapping_sub(&his[i]);
    }
    for i in 0..his.len() {
        c[n + i] = c[n + i].wrapping_add(&his[i]);
    }

    c
}

fn karatsuba_negacyclic_mul<
    T: Clone + Debug + WrappingSub<Output = T> + WrappingAdd<Output = T> + WrappingMul<Output = T>,
>(
    a: &[T],
    b: &[T],
    n: usize,
    zero: T,
) -> Vec<T> {
    let prod = karatsuba(a, b, n, zero.clone());
    let mut res = vec![zero; n];
    for i in 0..n - 1 {
        res[i] = prod[i].wrapping_sub(&prod[n + i]);
    }
    res[n - 1] = prod[n - 1].clone();
    res
}

#[allow(dead_code)]
fn quadratic_negacyclic_mul<
    T: Clone + Debug + WrappingSub<Output = T> + WrappingAdd<Output = T> + WrappingMul<Output = T>,
>(
    a: &[T],
    b: &[T],
    n: usize,
    zero: T,
) -> Vec<T> {
    debug_assert_eq!(a.len(), n);
    debug_assert_eq!(b.len(), n);
    if n == 1 {
        return vec![a[0].wrapping_mul(&b[0])];
    }
    let mut res = vec![zero; n];
    let mut a_row: Vec<T> = a.to_vec();
    for i in 1..n {
        a_row[i] = a[n - i].clone();
    }
    for i in 0..n {
        for j in 0..n {
            let product = b[j].wrapping_mul(&a_row[(n - i + j) % n]);
            if i < j {
                res[i] = res[i].wrapping_sub(&product);
            } else {
                res[i] = res[i].wrapping_add(&product);
            }
        }
    }
    res
}

fn quadratic_cyclic_mul<
    T: Clone + Debug + WrappingSub<Output = T> + WrappingAdd<Output = T> + WrappingMul<Output = T>,
>(
    a: &[T],
    b: &[T],
    n: usize,
    zero: T,
) -> Vec<T> {
    debug_assert_eq!(a.len(), n);
    debug_assert_eq!(b.len(), n);
    if n == 1 {
        return vec![a[0].wrapping_mul(&b[0])];
    }
    let mut a_row: Vec<T> = a.to_vec();
    for i in 1..n {
        a_row[i] = a[n - i].clone();
    }
    let mut res = vec![zero; n];
    for i in 0..n {
        for j in 0..n {
            let product = b[j].wrapping_mul(&a_row[(n - i + j) % n]);
            res[i] = res[i].wrapping_add(&product);
        }
    }
    res
}

#[allow(dead_code)]
fn karatsuba_cyclic_mul<
    T: Clone + WrappingSub<Output = T> + WrappingAdd<Output = T> + WrappingMul<Output = T>,
>(
    a: &[T],
    b: &[T],
    n: usize,
    zero: T,
) -> Vec<T> {
    let prod = karatsuba(a, b, n, zero.clone());
    let mut res = vec![zero; n];
    for i in 0..n - 1 {
        res[i] = prod[i].wrapping_add(&prod[n + i]);
    }
    res[n - 1] = prod[n - 1].clone();
    res
}

pub fn recursive_cyclic_mul<
    T: Clone + Debug + WrappingAdd<Output = T> + WrappingMul<Output = T> + WrappingSub<Output = T>,
>(
    a: &[T],
    b: &[T],
    n: usize,
    zero: T,
    two: T,
) -> Vec<T> {
    debug_assert_eq!(n & (n - 1), 0);
    if n <= RCM_CUTOFF {
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
        .map(|(l, r)| l.wrapping_sub(r))
        .collect_vec();

    let bdiff = blo
        .iter()
        .zip(bhi.iter())
        .map(|(l, r)| l.wrapping_sub(r))
        .collect_vec();

    let mut diffs = karatsuba_negacyclic_mul(&adiff, &bdiff, half, zero.clone());
    let mut nnn = half;
    while nnn > 1 {
        diffs = diffs
            .into_iter()
            .map(|i| i.wrapping_mul(&two))
            .collect_vec();
        nnn >>= 1;
    }

    let mut res = vec![zero; n];
    for i in 0..half {
        res[i] = sums[i].wrapping_add(&diffs[i]);
        res[i + half] = sums[i].wrapping_sub(&diffs[i]);
    }

    res
}

#[allow(dead_code)]
fn build_recursive_cyclic_mul_circuit() -> [Circuit<u64>; 16] {
    const STATE_SIZE: usize = 16;
    const MDS_MATRIX_FIRST_COLUMN: [u64; STATE_SIZE] = [
        61402, 1108, 28750, 33823, 7454, 43244, 53865, 12034, 56951, 27521, 41351, 40901, 12021,
        59689, 26798, 17845,
    ];
    type T = u64;
    let builder = CircuitBuilder::<T>::new();

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
    CircuitMonad::fold_uncles(&mut outputs);

    outputs
        .into_iter()
        .map(|o| o.consume())
        .collect_vec()
        .try_into()
        .unwrap()
}

/// Return a variable name for the node. Returns `point[n]` if node is just
/// a value from the codewords. Otherwise returns the ID of the circuit.
#[allow(dead_code)]
fn get_binding_name<T>(circuit: &Circuit<T>) -> String
where
    T: Clone
        + Debug
        + WrappingAdd<Output = T>
        + WrappingMul<Output = T>
        + WrappingSub<Output = T>
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
#[allow(dead_code)]
fn evaluate_single_node<T>(
    requested_visited_count: usize,
    circuit: &Circuit<T>,
    in_scope: &HashSet<usize>,
) -> (String, Vec<String>)
where
    T: Clone
        + Debug
        + WrappingAdd<Output = T>
        + WrappingMul<Output = T>
        + WrappingSub<Output = T>
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
            let binop_function = match binop {
                BinOp::Add => "wrapping_add",
                BinOp::Sub => "wrapping_sub",
                BinOp::Mul => "wrapping_mul",
            };
            output.push('(');
            let (to_output_lhs, lhs_symbols) =
                evaluate_single_node(requested_visited_count, &lhs.as_ref().borrow(), in_scope);
            output.push_str(&to_output_lhs);
            output.push_str(").");
            output.push_str(binop_function);
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

#[allow(dead_code)]
fn declare_single_node_with_visit_count<T>(
    requested_visited_count: usize,
    circuit: &Circuit<T>,
    in_scope: &mut HashSet<usize>,
    output: &mut String,
) where
    T: Clone
        + Debug
        + WrappingAdd<Output = T>
        + WrappingMul<Output = T>
        + WrappingSub<Output = T>
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
#[allow(dead_code)]
fn declare_nodes_with_visit_count<T>(
    requested_visited_count: usize,
    circuits: &[Circuit<T>],
) -> String
where
    T: Clone
        + Debug
        + WrappingAdd<Output = T>
        + WrappingMul<Output = T>
        + WrappingSub<Output = T>
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

#[allow(dead_code)]
fn turn_circuits_into_string<T>(circuits: &mut [Circuit<T>]) -> String
where
    T: Clone
        + Debug
        + WrappingAdd<Output = T>
        + WrappingMul<Output = T>
        + WrappingSub<Output = T>
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

#[inline(always)]
pub fn generated_function(input: &[u64]) -> [u64; 16] {
    let node_34 = (input[0]).wrapping_add(input[8]);
    let node_38 = (input[4]).wrapping_add(input[12]);
    let node_36 = (input[2]).wrapping_add(input[10]);
    let node_40 = (input[6]).wrapping_add(input[14]);
    let node_35 = (input[1]).wrapping_add(input[9]);
    let node_39 = (input[5]).wrapping_add(input[13]);
    let node_37 = (input[3]).wrapping_add(input[11]);
    let node_41 = (input[7]).wrapping_add(input[15]);
    let node_50 = (node_34).wrapping_add(node_38);
    let node_52 = (node_36).wrapping_add(node_40);
    let node_51 = (node_35).wrapping_add(node_39);
    let node_53 = (node_37).wrapping_add(node_41);
    let node_160 = (input[0]).wrapping_sub(input[8]);
    let node_161 = (input[1]).wrapping_sub(input[9]);
    let node_165 = (input[5]).wrapping_sub(input[13]);
    let node_163 = (input[3]).wrapping_sub(input[11]);
    let node_167 = (input[7]).wrapping_sub(input[15]);
    let node_162 = (input[2]).wrapping_sub(input[10]);
    let node_166 = (input[6]).wrapping_sub(input[14]);
    let node_164 = (input[4]).wrapping_sub(input[12]);
    let node_58 = (node_50).wrapping_add(node_52);
    let node_59 = (node_51).wrapping_add(node_53);
    let node_90 = (node_34).wrapping_sub(node_38);
    let node_91 = (node_35).wrapping_sub(node_39);
    let node_93 = (node_37).wrapping_sub(node_41);
    let node_92 = (node_36).wrapping_sub(node_40);
    let node_64 = ((node_58).wrapping_add(node_59)).wrapping_mul(524757);
    let node_67 = ((node_58).wrapping_sub(node_59)).wrapping_mul(52427);
    let node_71 = (node_50).wrapping_sub(node_52);
    let node_72 = (node_51).wrapping_sub(node_53);
    let node_177 = (node_161).wrapping_add(node_165);
    let node_179 = (node_163).wrapping_add(node_167);
    let node_178 = (node_162).wrapping_add(node_166);
    let node_176 = (node_160).wrapping_add(node_164);
    let node_69 = (node_64).wrapping_add(node_67);
    let node_397 =
        ((node_71).wrapping_mul(18446744073709525744)).wrapping_sub((node_72).wrapping_mul(53918));
    let node_1857 = (node_90).wrapping_mul(395512);
    let node_99 = (node_91).wrapping_add(node_93);
    let node_1865 = (node_91).wrapping_mul(18446744073709254400);
    let node_1869 = (node_93).wrapping_mul(179380);
    let node_1873 = (node_92).wrapping_mul(18446744073709509368);
    let node_1879 = (node_160).wrapping_mul(35608);
    let node_185 = (node_161).wrapping_add(node_163);
    let node_1915 = (node_161).wrapping_mul(18446744073709340312);
    let node_1921 = (node_163).wrapping_mul(18446744073709494992);
    let node_1927 = (node_162).wrapping_mul(18446744073709450808);
    let node_228 = (node_165).wrapping_add(node_167);
    let node_1939 = (node_165).wrapping_mul(18446744073709420056);
    let node_1945 = (node_167).wrapping_mul(18446744073709505128);
    let node_1951 = (node_166).wrapping_mul(216536);
    let node_1957 = (node_164).wrapping_mul(18446744073709515080);
    let node_70 = (node_64).wrapping_sub(node_67);
    let node_702 =
        ((node_71).wrapping_mul(53918)).wrapping_add((node_72).wrapping_mul(18446744073709525744));
    let node_1961 = (node_90).wrapping_mul(18446744073709254400);
    let node_1963 = (node_91).wrapping_mul(395512);
    let node_1965 = (node_92).wrapping_mul(179380);
    let node_1967 = (node_93).wrapping_mul(18446744073709509368);
    let node_1970 = (node_160).wrapping_mul(18446744073709340312);
    let node_1973 = (node_161).wrapping_mul(35608);
    let node_1982 = (node_162).wrapping_mul(18446744073709494992);
    let node_1985 = (node_163).wrapping_mul(18446744073709450808);
    let node_1988 = (node_166).wrapping_mul(18446744073709505128);
    let node_1991 = (node_167).wrapping_mul(216536);
    let node_1994 = (node_164).wrapping_mul(18446744073709420056);
    let node_1997 = (node_165).wrapping_mul(18446744073709515080);
    let node_98 = (node_90).wrapping_add(node_92);
    let node_184 = (node_160).wrapping_add(node_162);
    let node_227 = (node_164).wrapping_add(node_166);
    let node_86 = (node_69).wrapping_add(node_397);
    let node_403 = (node_1857).wrapping_sub(
        ((((node_99).wrapping_mul(18446744073709433780)).wrapping_sub(node_1865))
            .wrapping_sub(node_1869))
        .wrapping_add(node_1873),
    );
    let node_271 = (node_177).wrapping_add(node_179);
    let node_1891 = (node_177).wrapping_mul(18446744073709208752);
    let node_1897 = (node_179).wrapping_mul(18446744073709448504);
    let node_1903 = (node_178).wrapping_mul(115728);
    let node_1909 = (node_185).wrapping_mul(18446744073709283688);
    let node_1933 = (node_228).wrapping_mul(18446744073709373568);
    let node_88 = (node_70).wrapping_add(node_702);
    let node_708 =
        ((node_1961).wrapping_add(node_1963)).wrapping_sub((node_1965).wrapping_add(node_1967));
    let node_1976 = (node_178).wrapping_mul(18446744073709448504);
    let node_1979 = (node_179).wrapping_mul(115728);
    let node_87 = (node_69).wrapping_sub(node_397);
    let node_897 = ((((node_1865).wrapping_add((node_98).wrapping_mul(353264)))
        .wrapping_sub(node_1857))
    .wrapping_sub(node_1873))
    .wrapping_sub(node_1869);
    let node_2007 = (node_184).wrapping_mul(18446744073709486416);
    let node_2013 = (node_227).wrapping_mul(180000);
    let node_89 = (node_70).wrapping_sub(node_702);
    let node_1077 = ((((node_98).wrapping_mul(18446744073709433780))
        .wrapping_add((node_99).wrapping_mul(353264)))
    .wrapping_sub((node_1961).wrapping_add(node_1963)))
    .wrapping_sub((node_1965).wrapping_add(node_1967));
    let node_2020 = (node_184).wrapping_mul(18446744073709283688);
    let node_2023 = (node_185).wrapping_mul(18446744073709486416);
    let node_2026 = (node_227).wrapping_mul(18446744073709373568);
    let node_2029 = (node_228).wrapping_mul(180000);
    let node_2035 = (node_176).wrapping_mul(18446744073709550688);
    let node_2038 = (node_176).wrapping_mul(18446744073709208752);
    let node_2041 = (node_177).wrapping_mul(18446744073709550688);
    let node_270 = (node_176).wrapping_add(node_178);
    let node_152 = (node_86).wrapping_add(node_403);
    let node_412 = (node_1879).wrapping_sub(
        (((((((node_271).wrapping_mul(18446744073709105640)).wrapping_sub(node_1891))
            .wrapping_sub(node_1897))
        .wrapping_add(node_1903))
        .wrapping_sub(
            (((node_1909).wrapping_sub(node_1915)).wrapping_sub(node_1921)).wrapping_add(node_1927),
        ))
        .wrapping_sub(
            (((node_1933).wrapping_sub(node_1939)).wrapping_sub(node_1945)).wrapping_add(node_1951),
        ))
        .wrapping_add(node_1957),
    );
    let node_154 = (node_88).wrapping_add(node_708);
    let node_717 = ((node_1970).wrapping_add(node_1973)).wrapping_sub(
        ((((node_1976).wrapping_add(node_1979)).wrapping_sub((node_1982).wrapping_add(node_1985)))
            .wrapping_sub((node_1988).wrapping_add(node_1991)))
        .wrapping_add((node_1994).wrapping_add(node_1997)),
    );
    let node_156 = (node_87).wrapping_add(node_897);
    let node_906 = ((((node_1915).wrapping_add(node_2007)).wrapping_sub(node_1879))
        .wrapping_sub(node_1927))
    .wrapping_sub(
        (((node_1897).wrapping_sub(node_1921)).wrapping_sub(node_1945)).wrapping_add(
            (((node_1939).wrapping_add(node_2013)).wrapping_sub(node_1957)).wrapping_sub(node_1951),
        ),
    );
    let node_158 = (node_89).wrapping_add(node_1077);
    let node_1086 = ((((node_2020).wrapping_add(node_2023))
        .wrapping_sub((node_1970).wrapping_add(node_1973)))
    .wrapping_sub((node_1982).wrapping_add(node_1985)))
    .wrapping_sub(
        (((node_2026).wrapping_add(node_2029)).wrapping_sub((node_1994).wrapping_add(node_1997)))
            .wrapping_sub((node_1988).wrapping_add(node_1991)),
    );
    let node_153 = (node_86).wrapping_sub(node_403);
    let node_1237 = (((((((node_1909).wrapping_sub(node_1915)).wrapping_sub(node_1921))
        .wrapping_add(node_1927))
    .wrapping_add(node_2035))
    .wrapping_sub(node_1879))
    .wrapping_sub(node_1957))
    .wrapping_sub(
        (((node_1933).wrapping_sub(node_1939)).wrapping_sub(node_1945)).wrapping_add(node_1951),
    );
    let node_155 = (node_88).wrapping_sub(node_708);
    let node_1375 = (((((node_1982).wrapping_add(node_1985))
        .wrapping_add((node_2038).wrapping_add(node_2041)))
    .wrapping_sub((node_1970).wrapping_add(node_1973)))
    .wrapping_sub((node_1994).wrapping_add(node_1997)))
    .wrapping_sub((node_1988).wrapping_add(node_1991));
    let node_157 = (node_87).wrapping_sub(node_897);
    let node_1492 = ((((node_1921).wrapping_add(
        (((node_1891).wrapping_add((node_270).wrapping_mul(114800))).wrapping_sub(node_2035))
            .wrapping_sub(node_1903),
    ))
    .wrapping_sub(
        (((node_1915).wrapping_add(node_2007)).wrapping_sub(node_1879)).wrapping_sub(node_1927),
    ))
    .wrapping_sub(
        (((node_1939).wrapping_add(node_2013)).wrapping_sub(node_1957)).wrapping_sub(node_1951),
    ))
    .wrapping_sub(node_1945);
    let node_159 = (node_89).wrapping_sub(node_1077);
    let node_1657 = ((((((node_270).wrapping_mul(18446744073709105640))
        .wrapping_add((node_271).wrapping_mul(114800)))
    .wrapping_sub((node_2038).wrapping_add(node_2041)))
    .wrapping_sub((node_1976).wrapping_add(node_1979)))
    .wrapping_sub(
        (((node_2020).wrapping_add(node_2023)).wrapping_sub((node_1970).wrapping_add(node_1973)))
            .wrapping_sub((node_1982).wrapping_add(node_1985)),
    ))
    .wrapping_sub(
        (((node_2026).wrapping_add(node_2029)).wrapping_sub((node_1994).wrapping_add(node_1997)))
            .wrapping_sub((node_1988).wrapping_add(node_1991)),
    );

    [
        (node_152).wrapping_add(node_412),
        (node_154).wrapping_add(node_717),
        (node_156).wrapping_add(node_906),
        (node_158).wrapping_add(node_1086),
        (node_153).wrapping_add(node_1237),
        (node_155).wrapping_add(node_1375),
        (node_157).wrapping_add(node_1492),
        (node_159).wrapping_add(node_1657),
        (node_152).wrapping_sub(node_412),
        (node_154).wrapping_sub(node_717),
        (node_156).wrapping_sub(node_906),
        (node_158).wrapping_sub(node_1086),
        (node_153).wrapping_sub(node_1237),
        (node_155).wrapping_sub(node_1375),
        (node_157).wrapping_sub(node_1492),
        (node_159).wrapping_sub(node_1657),
    ]
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
        const MDS_MATRIX_FIRST_COLUMN: [u64; STATE_SIZE] = [
            61402, 1108, 28750, 33823, 7454, 43244, 53865, 12034, 56951, 27521, 41351, 40901,
            12021, 59689, 26798, 17845,
        ];
        let mut rng = thread_rng();
        let circuit = build_recursive_cyclic_mul_circuit();
        let a = (0..STATE_SIZE).map(|_| rng.next_u32() as u64).collect_vec();

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
