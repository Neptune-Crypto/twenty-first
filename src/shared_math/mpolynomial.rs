use crate::shared_math::polynomial::Polynomial;
use crate::shared_math::traits::{IdentityValues, ModPowU64};
use crate::timing_reporter::TimingReporter;
use crate::util_types::tree_m_ary::Node;
use itertools::Itertools;
use num_bigint::BigInt;
use num_traits::Zero;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::rc::Rc;
use std::{cmp, fmt};

type MCoefficients<T> = HashMap<Vec<u64>, T>;

const EDMONDS_WEIGHT_CUTOFF_FACTOR: u64 = 2;

#[derive(Debug, Clone)]
pub struct PolynomialEvaluationDataNode {
    diff_exponents: Vec<u64>,
    diff_sum: u64,
    abs_exponents: Vec<u64>,
    single_point: Option<usize>,
    x_powers: usize,
    // index: usize,
}

impl<'a, T: Sized> Node<T> {
    fn traverse_tree<
        U: Add<Output = U>
            + Div<Output = U>
            + Mul<Output = U>
            + Sub<Output = U>
            + IdentityValues
            + Clone
            + Display
            + Debug
            + PartialEq
            + Eq,
    >(
        nodes: Vec<Rc<RefCell<Node<PolynomialEvaluationDataNode>>>>,
        point: &[Polynomial<U>],
        one: U,
        polynomium_products: &mut HashMap<Vec<u64>, Polynomial<U>>,
    ) {
        let zero = point[0].coefficients[0].ring_zero();

        // We might be able to avoid the `clone()` of exponents_list elements here, if we are smart
        polynomium_products.insert(
            nodes[0].borrow().data.abs_exponents.clone(),
            Polynomial::from_constant(one.clone()),
        );

        // Consider if there might be a speedup in sorting `point` values
        // after complexity (degree)
        for node in nodes {
            for child_item in node.borrow().children.iter() {
                let child = child_item.borrow();
                let PolynomialEvaluationDataNode {
                    diff_exponents: child_diff_exponents,
                    abs_exponents: child_abs_exponents,
                    single_point,
                    diff_sum: _,
                    x_powers,
                } = &child.data;

                if child_diff_exponents.iter().all(|e| *e == 0) {
                    // println!(
                    //     "Hit x-power optimization for child with diff_exponents: {:?}",
                    //     child_diff_exponents
                    // );
                    // println!("child_abs_exponents: {:?}", child_abs_exponents);
                    // println!("x_powers: {}", x_powers);
                    let mut res = polynomium_products[&node.borrow().data.abs_exponents].clone();
                    res.shift_coefficients_mut(*x_powers, zero.clone());
                    polynomium_products.insert(child_abs_exponents.clone(), res);
                    continue;
                }

                let mul = if single_point.is_some() {
                    point[single_point.unwrap()].clone()
                } else if polynomium_products.contains_key(child_diff_exponents) {
                    // if count < 100 {
                    //     // println!("Caught {:?}", child_diff_exponents);
                    //     count += 1;
                    // }
                    polynomium_products[child_diff_exponents].clone()
                } else {
                    // if count < 100 {
                    //     // println!("Missed {:?}", child_diff_exponents);
                    //     count += 1;
                    // }
                    let mut intermediate_mul: Polynomial<U> =
                        Polynomial::from_constant(one.clone());
                    let mut intermediate_exponents: Vec<u64> = vec![0; point.len()];
                    let mut remaining_exponents: Vec<u64> = child_diff_exponents.clone();
                    let mut mod_pow_exponents: Vec<u64> = vec![0; point.len()];
                    // Would it be faster to traverse this list in random order?
                    for (i, diff_exponent) in child_diff_exponents
                        .iter()
                        .enumerate()
                        .filter(|(_i, d)| **d != 0)
                    {
                        // TODO: Insert both each intermediate mul *and* each mod_pow in tree here
                        // println!("Hi! i = {}, diff_exponent = {}", i, diff_exponent);
                        // println!("intermediate_exponents = {:?}", intermediate_exponents);
                        // println!("remaining_exponents = {:?}", remaining_exponents);
                        // println!("mod_pow_exponents = {:?}", mod_pow_exponents);

                        if polynomium_products.contains_key(&remaining_exponents) {
                            // TODO: Consider fast multiplication here
                            intermediate_mul = intermediate_mul
                                * polynomium_products[&remaining_exponents].clone();
                            break;
                        }

                        mod_pow_exponents[i] = *diff_exponent;
                        let mod_pow = if polynomium_products.contains_key(&mod_pow_exponents) {
                            polynomium_products[&mod_pow_exponents].clone()
                        } else {
                            // println!("Calculating mod_pow");
                            let mut mod_pow_intermediate: Option<Polynomial<U>> = None;
                            let mut mod_pow_reduced = mod_pow_exponents.clone();
                            while mod_pow_reduced[i] > 2 {
                                mod_pow_reduced[i] -= 1;
                                // println!("looking for {:?}", mod_pow_reduced);
                                if polynomium_products.contains_key(&mod_pow_reduced) {
                                    // println!("Found result for {:?}", mod_pow_reduced);
                                    mod_pow_intermediate =
                                        Some(polynomium_products[&mod_pow_reduced].clone());
                                    break;
                                }
                            }

                            match mod_pow_intermediate {
                                None => {
                                    // println!("Missed reduced mod_pow result!");
                                    let mod_pow_intermediate =
                                        point[i].mod_pow((*diff_exponent).into(), one.clone());
                                    polynomium_products.insert(
                                        mod_pow_exponents.clone(),
                                        mod_pow_intermediate.clone(),
                                    );
                                    mod_pow_intermediate
                                }
                                Some(res) => {
                                    // println!("Found reduced mod_pow result!");
                                    point[i].mod_pow(
                                        (*diff_exponent - mod_pow_reduced[i]).into(),
                                        one.clone(),
                                    ) * res
                                }
                            }
                        };

                        // TODO: Consider fast multiplication here
                        intermediate_mul = intermediate_mul * mod_pow;
                        intermediate_exponents[i] = *diff_exponent;

                        if polynomium_products.contains_key(&intermediate_exponents) {
                            // TODO: Consider what this branch means!
                            // println!("Contained!");
                        } else {
                            polynomium_products
                                .insert(intermediate_exponents.clone(), intermediate_mul.clone());
                        }
                        remaining_exponents[i] = 0;
                        mod_pow_exponents[i] = 0;
                    }
                    intermediate_mul
                };

                // TODO: Add fast multiplication (with NTT) here
                let mut res = mul * polynomium_products[&node.borrow().data.abs_exponents].clone();
                res.shift_coefficients_mut(*x_powers, zero.clone());
                polynomium_products.insert(child_abs_exponents.clone(), res);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct MPolynomial<
    T: Add + Div + Mul + Sub + IdentityValues + Clone + PartialEq + Eq + Hash + Display + Debug,
> {
    // Multivariate polynomials are represented as hash maps with exponent vectors
    // as keys and coefficients as values. E.g.:
    // f(x,y,z) = 17 + 2xy + 42z - 19x^6*y^3*z^12 is represented as:
    // {
    //     [0,0,0] => 17,
    //     [1,1,0] => 2,
    //     [0,0,1] => 42,
    //     [6,3,12] => -19,
    // }
    pub variable_count: usize,
    pub coefficients: HashMap<Vec<u64>, T>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum PrecalculationError {
    EmptyInput,
    LengthMismatch,
    ZeroDegreeInput,
}

impl Error for PrecalculationError {}

impl fmt::Display for PrecalculationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl<
        U: Add<Output = U>
            + Div<Output = U>
            + Mul<Output = U>
            + Sub<Output = U>
            + Neg<Output = U>
            + ModPowU64
            + IdentityValues
            + Clone
            + PartialEq
            + Eq
            + Hash
            + Display
            + Send
            + Sync
            + Debug,
    > Display for MPolynomial<U>
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let output;
        if self.is_zero() {
            output = "0".to_string();
        } else {
            let mut term_strings = self
                .coefficients
                .iter()
                .sorted_by_key(|x| x.0[0])
                .map(|(k, v)| Self::term_print(k, v));
            output = term_strings.join("\n+ ");
        }

        write!(f, "\n  {}", output)
    }
}

impl<
        U: Add<Output = U>
            + Div<Output = U>
            + Mul<Output = U>
            + Sub<Output = U>
            + IdentityValues
            + Clone
            + PartialEq
            + Eq
            + Hash
            + Display
            + Debug,
    > PartialEq for MPolynomial<U>
{
    fn eq(&self, other: &Self) -> bool {
        let (shortest, var_count, longest) = if self.variable_count > other.variable_count {
            (
                other.coefficients.clone(),
                self.variable_count,
                self.coefficients.clone(),
            )
        } else {
            (
                self.coefficients.clone(),
                other.variable_count,
                other.coefficients.clone(),
            )
        };

        let mut padded: HashMap<Vec<u64>, U> = HashMap::new();
        for (k, v) in shortest.iter() {
            let mut pad = k.clone();
            pad.resize_with(var_count, || 0);
            padded.insert(pad, v.clone());
        }

        for (fst, snd) in [(padded.clone(), longest.clone()), (longest, padded)] {
            for (k, v) in fst.iter() {
                if !v.is_zero() {
                    if !snd.contains_key(k) {
                        return false;
                    }
                    if snd[k] != *v {
                        return false;
                    }
                }
            }
        }

        true
    }
}

impl<
        U: Add<Output = U>
            + Div<Output = U>
            + Mul<Output = U>
            + Sub<Output = U>
            + IdentityValues
            + Clone
            + PartialEq
            + Eq
            + Hash
            + Display
            + Debug,
    > Eq for MPolynomial<U>
{
}

impl<
        U: Add<Output = U>
            + Div<Output = U>
            + Mul<Output = U>
            + Sub<Output = U>
            + Neg<Output = U>
            + IdentityValues
            + ModPowU64
            + Clone
            + Display
            + Debug
            + PartialEq
            + Eq
            + Hash
            + Send
            + Sync,
    > MPolynomial<U>
{
    fn term_print(exponents: &[u64], coefficient: &U) -> String {
        if coefficient.is_zero() {
            return "".to_string();
        }

        let mut str_elems: Vec<String> = vec![];
        if !coefficient.is_one() {
            str_elems.push(coefficient.to_string());
        }

        for (i, exponent) in exponents.iter().enumerate() {
            if *exponent == 0 {
                continue;
            }
            let factor_str = if *exponent == 1 {
                format!("x_{}", i)
            } else {
                format!("x_{}^{}", i, exponent)
            };
            str_elems.push(factor_str);
        }

        str_elems.join("*")
    }

    pub fn zero(variable_count: usize) -> Self {
        Self {
            coefficients: HashMap::new(),
            variable_count,
        }
    }

    pub fn is_zero(&self) -> bool {
        if self.coefficients.is_empty() {
            return true;
        }

        for (_, v) in self.coefficients.iter() {
            if !v.is_zero() {
                return false;
            }
        }

        true
    }

    pub fn from_constant(element: U, variable_count: usize) -> Self {
        let mut cs: MCoefficients<U> = HashMap::new();
        cs.insert(vec![0; variable_count], element);
        Self {
            variable_count,
            coefficients: cs,
        }
    }

    // Returns the multivariate polynomials representing each indeterminates linear function
    // with a leading coefficient of one. For three indeterminates, returns:
    // [f(x,y,z) = x, f(x,y,z) = y, f(x,y,z) = z]
    pub fn variables(variable_count: usize, one: U) -> Vec<Self> {
        assert!(one.is_one(), "Provided one must be one");
        let mut res: Vec<Self> = vec![];
        for i in 0..variable_count {
            let mut exponent = vec![0u64; variable_count];
            exponent[i] = 1;
            let mut coefficients: MCoefficients<U> = HashMap::new();
            coefficients.insert(exponent, one.clone());
            res.push(Self {
                variable_count,
                coefficients,
            });
        }

        res
    }

    pub fn evaluate(&self, point: &[U]) -> U {
        assert_eq!(
            self.variable_count,
            point.len(),
            "Dimensionality of multivariate polynomial and point must agree in evaluate"
        );
        let mut acc = point[0].ring_zero();
        for (k, v) in self.coefficients.iter() {
            let mut prod = v.clone();
            for i in 0..k.len() {
                prod = prod.clone() * point[i].mod_pow_u64(k[i]);
            }
            acc = acc + prod;
        }

        acc
    }

    pub fn precalculate_exponents_memoization(
        mpols: &[Self],
        point: &[Polynomial<U>],
        exponents_memoization: &mut HashMap<Vec<u64>, Polynomial<U>>,
    ) -> Result<(), Box<dyn Error>> {
        let mut timer = TimingReporter::start();
        if mpols.is_empty() || point.is_empty() {
            return Err(Box::new(PrecalculationError::EmptyInput));
        }

        let variable_count = mpols[0].variable_count;

        if point.iter().any(|p| p.degree() <= 0) {
            return Err(Box::new(PrecalculationError::ZeroDegreeInput));
        }

        if point.len() != variable_count {
            return Err(Box::new(PrecalculationError::LengthMismatch));
        }

        timer.elapsed("init stuff");
        let one: U = point[0].coefficients[0].ring_one(); // guaranteed to exist because of above checks
        let exponents_set: HashSet<Vec<u64>> = mpols
            .iter()
            .map(|mpol| mpol.coefficients.keys().map(|x| x.to_owned()))
            .flatten()
            .collect();
        timer.elapsed("calculated exponents_set");
        let mut exponents_list: Vec<Vec<u64>> = if exponents_set.contains(&vec![0; variable_count])
        {
            vec![]
        } else {
            vec![vec![0; variable_count]]
        };
        exponents_list.append(&mut exponents_set.into_iter().collect());
        timer.elapsed("calculated exponents_list");

        for i in 0..variable_count {
            exponents_list.sort_by_cached_key(|exponents| exponents[i]);
        }
        timer.elapsed("sorted exponents_list");

        let points_are_x: Vec<bool> = point.iter().map(|x| x.is_x()).collect();
        let x_point_indices: Vec<usize> = points_are_x
            .iter()
            .enumerate()
            .filter(|(_i, is_x)| **is_x)
            .map(|(i, _)| i)
            .collect();

        println!("choosing edges");

        // Calculate the relevant weight for making a calculation from one list of exponents another
        // Use these weights to pick the minimal edges.
        // This algorithm i a variation of Edmond's algorithm for finding the minimal spanning tree
        // in a directed graph. Only, we don't calculate all edges but instead look for the ones that
        // are minimal while calculating their weights.
        let mut chosen_edges: Vec<(u64, u64)> = vec![(0, 0); exponents_list.len()];
        'outer: for i in 1..exponents_list.len() {
            let mut min_weight = u64::MAX;
            'middle: for j in 1..=i {
                let index = i - j;

                // Check if calculation from `index` to `i` can be made.
                // If just one of the `exponents_list[i][k]`s is bigger than the `exponents_list[index][k]` it cannot be made
                // through multiplication but would have to be constructed through division. This would never be worthwhile
                // so we simply discard that possibility.
                '_inner: for k in 0..variable_count {
                    if exponents_list[i][k] < exponents_list[index][k] {
                        continue 'middle;
                    }
                }

                // let mut diff = 0;
                // diff += exponents_list[i][k] - exponents_list[index][k];
                let diff: u64 = exponents_list[i]
                    .iter()
                    .zip(exponents_list[index].iter())
                    .map(|(ei, ej)| *ei - *ej)
                    .sum();

                if diff < min_weight {
                    min_weight = diff;
                    chosen_edges[i].0 = index as u64;
                    chosen_edges[i].1 = min_weight;
                    // println!("(min_weight, index) = {:?}", (min_weight, index));

                    // If we found a minimum possible weight, corresponding to the multiplication
                    // with *one* polynomial with the lowest degree, then we are done with the middle
                    // loop and can continue to the next iteration in the outer loop.
                    // Good enough
                    if min_weight <= EDMONDS_WEIGHT_CUTOFF_FACTOR {
                        // println!("continuing");
                        continue 'outer;
                    }
                }
            }
        }

        println!("chose edges");
        timer.elapsed("chose edges");

        // data: (diff, abs)
        let nodes: Vec<Rc<RefCell<Node<PolynomialEvaluationDataNode>>>> = exponents_list
            .into_iter()
            .enumerate()
            .map(|(_i, exponents)| {
                // .map(|exponents| {
                Rc::new(RefCell::new(Node {
                    children: vec![],
                    data: PolynomialEvaluationDataNode {
                        abs_exponents: exponents,
                        diff_exponents: vec![0; variable_count],
                        single_point: None,
                        diff_sum: 0,
                        x_powers: 0,
                        // index: i,
                    },
                }))
            })
            .collect();
        timer.elapsed("initialized nodes");
        for (end, (start, weight)) in chosen_edges.into_iter().enumerate().skip(1) {
            // println!("({} => {})", end, start);

            nodes[start as usize]
                .borrow_mut()
                .children
                .push(nodes[end].clone());
            let mut diff_exponents: Vec<u64> = nodes[end as usize]
                .borrow()
                .data
                .abs_exponents
                .iter()
                .zip(nodes[start as usize].borrow().data.abs_exponents.iter())
                .map(|(end_exponent, start_exponent)| end_exponent - start_exponent)
                .collect();
            let mut x_power = 0;
            for x_point_index in x_point_indices.iter() {
                x_power += diff_exponents[*x_point_index];
                diff_exponents[*x_point_index] = 0;
            }
            if diff_exponents.iter().sum::<u64>() == 1u64 {
                nodes[end as usize].borrow_mut().data.single_point =
                    Some(diff_exponents.iter().position(|&x| x == 1).unwrap());
            }
            nodes[end as usize].borrow_mut().data.diff_exponents = diff_exponents;
            nodes[end as usize].borrow_mut().data.diff_sum = weight;
            nodes[end as usize].borrow_mut().data.x_powers = x_power as usize;
        }
        timer.elapsed("built nodes");

        Node::<PolynomialEvaluationDataNode>::traverse_tree::<U>(
            nodes,
            point,
            one,
            exponents_memoization,
        );
        timer.elapsed("traversed tree");
        let report = timer.finish();
        println!("{}", report);

        Ok(())
    }

    // Substitute the variables in a multivariate polynomial with univariate polynomials in parallel.
    // All "intermediate results" **must** be present in `exponents_memoization` or this function
    // will panic.
    pub fn evaluate_symbolic_with_memoization_precalculated(
        &self,
        point: &[Polynomial<U>],
        exponents_memoization: &mut HashMap<Vec<u64>, Polynomial<U>>,
    ) -> Polynomial<U> {
        assert_eq!(
            self.variable_count,
            point.len(),
            "Dimensionality of multivariate polynomial and point must agree in evaluate_symbolic"
        );
        let acc = self
            .coefficients
            .par_iter()
            .map(|(k, v)| exponents_memoization[k].clone().scalar_mul(v.clone()))
            .reduce(|| Polynomial::ring_zero(), |a, b| a + b);

        acc
    }

    // Substitute the variables in a multivariate polynomial with univariate polynomials, fast
    #[allow(clippy::map_entry)]
    #[allow(clippy::type_complexity)]
    pub fn evaluate_symbolic_with_memoization(
        &self,
        point: &[Polynomial<U>],
        mod_pow_memoization: &mut HashMap<(usize, u64), Polynomial<U>>,
        mul_memoization: &mut HashMap<(Polynomial<U>, (usize, u64)), Polynomial<U>>,
        exponents_memoization: &mut HashMap<Vec<u64>, Polynomial<U>>,
    ) -> Polynomial<U> {
        // Notice that the `exponents_memoization` only gives a speedup if this function is evaluated multiple
        // times for the same `point` input. This condition holds when evaluating the AIR constraints
        // symbolically in a generic STARK prover.
        assert_eq!(
            self.variable_count,
            point.len(),
            "Dimensionality of multivariate polynomial and point must agree in evaluate_symbolic"
        );
        let points_are_x: Vec<bool> = point.iter().map(|p| p.is_x()).collect();
        let mut acc: Polynomial<U> = Polynomial::ring_zero();
        // Sort k after complexity
        // let mut ks: Vec<(Vec<u64>, U)> = self.coefficients.clone().into_iter().collect();
        // ks.sort_by_key(|k: (Vec<u64>, U)| k.0.iter().sum());
        // ks.sort_by_cached_key::<u64, _>(|k| {
        //     k.0.iter()
        //         .enumerate()
        //         .filter(|(i, _)| !points_are_x[*i])
        //         .map(|(_, x)| x)
        //         .sum()
        // });

        for (k, v) in self.coefficients.iter() {
            // for (k, v) in ks.iter() {
            let mut prod: Polynomial<U>;
            if exponents_memoization.contains_key(k) {
                prod = exponents_memoization[k].clone();
            } else {
                println!("Missed!");
                prod = Polynomial::from_constant(v.ring_one());
                let mut k_sorted: Vec<(usize, u64)> = k.clone().into_iter().enumerate().collect();
                k_sorted.sort_by_key(|k| k.1);
                let mut x_pow_mul = 0;
                for (i, ki) in k_sorted.into_iter() {
                    // calculate prod * point[i].mod_pow(k[i].into(), v.ring_one()) with some optimizations,
                    // mainly memoization.
                    // prod = prod * point[i].mod_pow(k[i].into(), v.ring_one());

                    if ki == 0 {
                        // This should be the common (branch-predicted) case for the early iterations of the inner loop
                        continue;
                    }

                    // Decrease the number of `mul_memoization` misses by doing all powers of x after this inner loop
                    if points_are_x[i] {
                        x_pow_mul += ki;
                        continue;
                    }

                    // With this `mul_key` to lookup previous multiplications there is no risk that we miss
                    // already done calculations in terms of the commutation of multiplication. Since the `ki`
                    // values are sorted there is no risk that we do double work by first calculating
                    // `point[2].mod_pow(2) * point[4].mod_pow(3)` (1) and then
                    // `point[4].mod_pow(3) * point[2].mod_pow(2)` (2). The sorting of the `ki`s ensure that
                    // the calculation will always be done as (1) and never as (2).
                    let mul_key = (prod.clone(), (i, ki));
                    prod = if mul_memoization.contains_key(&mul_key) {
                        // This should be the common case for the late iterations of the inner loop
                        mul_memoization[&mul_key].clone()
                    } else if ki == 1 {
                        let mul_res = prod.clone() * point[i].clone();
                        mul_memoization.insert(mul_key, mul_res.clone());
                        mul_res
                    } else {
                        // Check if we have already done multiplications with a lower power of `point[i]`
                        // than what we are looking for. If we have, then we use this multiplication
                        // as a starting point to calculation the next.

                        let mut reduced_mul_result: Option<Polynomial<U>> = None;
                        let mut reduced_mul_key = (prod.clone(), (i, ki));
                        for j in 1..ki - 1 {
                            reduced_mul_key.1 .1 = ki - j;
                            if mul_memoization.contains_key(&reduced_mul_key) {
                                reduced_mul_result =
                                    Some(mul_memoization[&reduced_mul_key].clone());
                                break;
                            }
                        }

                        let mod_pow_key = match reduced_mul_result {
                            None => (i, ki),
                            // i = 1, ki = 5, found reduced result for (i=1, ki = 2), need mod_pow_key = (i = 1, ki = 3)
                            Some(_) => (i, ki - reduced_mul_key.1 .1),
                        };
                        let mod_pow = if mod_pow_key.1 == 1 {
                            point[i].clone()
                        } else if mod_pow_memoization.contains_key(&mod_pow_key) {
                            mod_pow_memoization[&mod_pow_key].clone()
                        } else {
                            // With precalculation of `mod_pow_memoization`, this should never happen
                            println!("missed mod_pow_memoization!");
                            let mod_pow_res = point[i].mod_pow(mod_pow_key.1.into(), v.ring_one());
                            mod_pow_memoization.insert(mod_pow_key, mod_pow_res.clone());
                            mod_pow_res
                        };
                        let mul_res = match reduced_mul_result {
                            Some(reduced) => reduced * mod_pow,
                            None => prod.clone() * mod_pow,
                        };
                        mul_memoization.insert(mul_key, mul_res.clone());
                        mul_res
                    }
                }

                prod.shift_coefficients_mut(x_pow_mul as usize, v.ring_zero());
                exponents_memoization.insert(k.to_vec(), prod.clone());
            }
            prod.scalar_mul_mut(v.clone());
            acc += prod;
        }

        acc
    }

    // Substitute the variables in a multivariate polynomial with univariate polynomials
    pub fn evaluate_symbolic(&self, point: &[Polynomial<U>]) -> Polynomial<U> {
        assert_eq!(
            self.variable_count,
            point.len(),
            "Dimensionality of multivariate polynomial and point must agree in evaluate_symbolic"
        );
        let mut acc: Polynomial<U> = Polynomial::ring_zero();
        for (k, v) in self.coefficients.iter() {
            let mut prod = Polynomial::from_constant(v.clone());
            for i in 0..k.len() {
                // calculate prod * point[i].mod_pow(k[i].into(), v.ring_one()) with some small optimizations
                // prod = prod * point[i].mod_pow(k[i].into(), v.ring_one());
                prod = if k[i] == 0 {
                    prod
                } else if point[i].is_x() {
                    prod * point[i].shift_coefficients(k[i] as usize - 1, v.ring_zero())
                } else {
                    prod * point[i].mod_pow(k[i].into(), v.ring_one())
                };
            }
            acc += prod;
        }

        acc
    }

    pub fn lift(
        univariate_polynomial: Polynomial<U>,
        variable_index: usize,
        variable_count: usize,
    ) -> Self {
        assert!(
            variable_count > variable_index,
            "number of variables must be at least one larger than the variable index"
        );
        if univariate_polynomial.is_zero() {
            return Self::zero(variable_count);
        }

        let one = univariate_polynomial.coefficients[0].ring_one();
        let mut coefficients: MCoefficients<U> = HashMap::new();
        let mut key = vec![0u64; variable_count];
        key[variable_index] = 1;
        coefficients.insert(key, one.clone());
        let indeterminate: MPolynomial<U> = Self {
            variable_count,
            coefficients,
        };

        let mut acc = MPolynomial::<U>::zero(variable_count);
        for i in 0..univariate_polynomial.coefficients.len() {
            acc += MPolynomial::from_constant(
                univariate_polynomial.coefficients[i].clone(),
                variable_count,
            ) * indeterminate.mod_pow(i.into(), one.clone());
        }

        acc
    }

    pub fn scalar_mul(&self, factor: U) -> Self {
        if self.is_zero() {
            return Self::zero(self.variable_count);
        }

        let mut output_coefficients: MCoefficients<U> = HashMap::new();
        for (k, v) in self.coefficients.iter() {
            output_coefficients.insert(k.to_vec(), v.clone() * factor.clone());
        }

        Self {
            variable_count: self.variable_count,
            coefficients: output_coefficients,
        }
    }

    pub fn scalar_mul_mut(&mut self, factor: U) {
        if self.is_zero() || factor.is_one() {
            return;
        }

        for (_k, v) in self.coefficients.iter_mut() {
            *v = v.to_owned() * factor.clone();
        }
    }

    pub fn mod_pow(&self, pow: BigInt, one: U) -> Self {
        // Handle special case of 0^0
        if pow.is_zero() {
            let mut coefficients: MCoefficients<U> = HashMap::new();
            coefficients.insert(vec![0; self.variable_count], one);
            return MPolynomial {
                variable_count: self.variable_count,
                coefficients,
            };
        }

        // Handle 0^n for n > 0
        if self.is_zero() {
            return Self::zero(self.variable_count);
        }

        let one = self.coefficients.values().last().unwrap().ring_one();
        let exp = vec![0u64; self.variable_count];
        let mut acc_coefficients_init: MCoefficients<U> = HashMap::new();
        acc_coefficients_init.insert(exp, one);
        let mut acc: MPolynomial<U> = Self {
            variable_count: self.variable_count,
            coefficients: acc_coefficients_init,
        };
        let bit_length: u64 = pow.bits();
        for i in 0..bit_length {
            acc = acc.square();
            let set: bool =
                !(pow.clone() & Into::<BigInt>::into(1u128 << (bit_length - 1 - i))).is_zero();
            if set {
                acc = acc * self.clone();
            }
        }

        acc
    }

    pub fn square(&self) -> Self {
        if self.is_zero() {
            return Self::zero(self.variable_count);
        }

        let mut output_coefficients: MCoefficients<U> = HashMap::new();
        let exponents = self.coefficients.keys().collect::<Vec<&Vec<u64>>>();
        let c0 = self.coefficients.values().next().unwrap();
        let two = c0.ring_one() + c0.ring_one();

        for i in 0..exponents.len() {
            let ki = exponents[i];
            let v0 = self.coefficients[ki].clone();
            let mut new_exponents = Vec::with_capacity(self.variable_count);
            for exponent in ki {
                new_exponents.push(exponent * 2);
            }
            if output_coefficients.contains_key(&new_exponents) {
                output_coefficients.insert(
                    new_exponents.to_vec(),
                    v0.to_owned() * v0.to_owned() + output_coefficients[&new_exponents].clone(),
                );
            } else {
                output_coefficients.insert(new_exponents.to_vec(), v0.to_owned() * v0.to_owned());
            }

            for kj in exponents.iter().skip(i + 1) {
                let mut new_exponents = Vec::with_capacity(self.variable_count);
                for k in 0..self.variable_count {
                    // TODO: Can overflow.
                    let exponent = ki[k] + kj[k];
                    new_exponents.push(exponent);
                }
                let v1 = self.coefficients[*kj].clone();
                if output_coefficients.contains_key(&new_exponents) {
                    output_coefficients.insert(
                        new_exponents.to_vec(),
                        two.clone() * v0.to_owned() * v1.to_owned()
                            + output_coefficients[&new_exponents].clone(),
                    );
                } else {
                    output_coefficients.insert(
                        new_exponents.to_vec(),
                        two.clone() * v0.to_owned() * v1.to_owned(),
                    );
                }
            }
        }

        Self {
            coefficients: output_coefficients,
            variable_count: self.variable_count,
        }
    }

    pub fn degree(&self) -> u64 {
        self.coefficients
            .keys()
            .map(|coefficients| coefficients.iter().sum::<u64>())
            .max()
            .unwrap_or(0) as u64
    }
}

impl<
        U: Add<Output = U>
            + Div<Output = U>
            + Mul<Output = U>
            + Sub<Output = U>
            + Neg<Output = U>
            + IdentityValues
            + ModPowU64
            + Clone
            + PartialEq
            + Eq
            + Hash
            + Display
            + Send
            + Sync
            + Debug,
    > Add for MPolynomial<U>
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let variable_count: usize = cmp::max(self.variable_count, other.variable_count);
        if self.is_zero() && other.is_zero() {
            return Self::zero(variable_count);
        }

        let mut output_coefficients: MCoefficients<U> = HashMap::new();
        for (k, v) in self.coefficients.iter() {
            let mut pad = k.clone();
            pad.resize_with(variable_count, || 0);
            output_coefficients.insert(pad, v.clone());
        }
        for (k, v) in other.coefficients.iter() {
            let mut pad = k.clone();
            pad.resize_with(variable_count, || 0);

            // TODO: This can probably be done smarter
            if output_coefficients.contains_key(&pad) {
                output_coefficients.insert(
                    pad.clone(),
                    v.to_owned() + output_coefficients[&pad].clone(),
                );
            } else {
                output_coefficients.insert(pad.to_vec(), v.to_owned());
            }
        }

        Self {
            coefficients: output_coefficients,
            variable_count,
        }
    }
}

impl<
        U: Add<Output = U>
            + Div<Output = U>
            + Mul<Output = U>
            + Sub<Output = U>
            + Neg<Output = U>
            + IdentityValues
            + ModPowU64
            + Clone
            + PartialEq
            + Eq
            + Hash
            + Display
            + Send
            + Sync
            + Debug,
    > AddAssign for MPolynomial<U>
{
    fn add_assign(&mut self, rhs: Self) {
        if self.variable_count != rhs.variable_count {
            let result = self.clone() + rhs;
            self.variable_count = result.variable_count;
            self.coefficients = result.coefficients;
            return;
        }

        for (k, v1) in rhs.coefficients.iter() {
            if self.coefficients.contains_key(k) {
                let v0 = self.coefficients[k].clone();
                self.coefficients.insert(k.clone(), v0 + v1.to_owned());
            } else {
                self.coefficients.insert(k.clone(), v1.to_owned());
            }
        }
    }
}

impl<
        U: Add<Output = U>
            + Div<Output = U>
            + Mul<Output = U>
            + Sub<Output = U>
            + Neg<Output = U>
            + IdentityValues
            + ModPowU64
            + Clone
            + PartialEq
            + Eq
            + Hash
            + Display
            + Send
            + Sync
            + Debug,
    > Sub for MPolynomial<U>
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let variable_count: usize = cmp::max(self.variable_count, other.variable_count);
        if self.is_zero() && other.is_zero() {
            return Self::zero(variable_count);
        }

        let mut output_coefficients: MCoefficients<U> = HashMap::new();
        for (k, v) in self.coefficients.iter() {
            let mut pad = k.clone();
            pad.resize_with(variable_count, || 0);
            output_coefficients.insert(pad, v.clone());
        }
        for (k, v) in other.coefficients.iter() {
            let mut pad = k.clone();
            pad.resize_with(variable_count, || 0);

            // TODO: This can probably be done smarter
            if output_coefficients.contains_key(&pad) {
                output_coefficients.insert(
                    pad.to_vec(),
                    output_coefficients[&pad].clone() - v.to_owned(),
                );
            } else {
                output_coefficients.insert(pad.to_vec(), -v.to_owned());
            }
        }

        Self {
            coefficients: output_coefficients,
            variable_count,
        }
    }
}

impl<
        U: Add<Output = U>
            + Div<Output = U>
            + Mul<Output = U>
            + Sub<Output = U>
            + Neg<Output = U>
            + IdentityValues
            + ModPowU64
            + Clone
            + PartialEq
            + Eq
            + Hash
            + Display
            + Debug,
    > Neg for MPolynomial<U>
{
    type Output = Self;

    fn neg(self) -> Self {
        let mut output_coefficients: MCoefficients<U> = HashMap::new();
        for (k, v) in self.coefficients.iter() {
            output_coefficients.insert(k.to_vec(), -v.clone());
        }

        Self {
            variable_count: self.variable_count,
            coefficients: output_coefficients,
        }
    }
}

impl<
        U: Add<Output = U>
            + Div<Output = U>
            + Mul<Output = U>
            + Sub<Output = U>
            + Neg<Output = U>
            + IdentityValues
            + ModPowU64
            + Clone
            + PartialEq
            + Eq
            + Hash
            + Display
            + Send
            + Sync
            + Debug,
    > Mul for MPolynomial<U>
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let variable_count: usize = cmp::max(self.variable_count, other.variable_count);
        if self.is_zero() || other.is_zero() {
            return Self::zero(variable_count);
        }

        let mut output_coefficients: MCoefficients<U> = HashMap::new();
        for (k0, v0) in self.coefficients.iter() {
            for (k1, v1) in other.coefficients.iter() {
                let mut exponent = vec![0u64; variable_count];
                for k in 0..self.variable_count {
                    exponent[k] += k0[k];
                }
                for k in 0..other.variable_count {
                    exponent[k] += k1[k];
                }
                if output_coefficients.contains_key(&exponent) {
                    output_coefficients.insert(
                        exponent.to_vec(),
                        v0.to_owned() * v1.to_owned() + output_coefficients[&exponent].clone(),
                    );
                } else {
                    output_coefficients.insert(exponent.to_vec(), v0.to_owned() * v1.to_owned());
                }
            }
        }
        Self {
            coefficients: output_coefficients,
            variable_count,
        }
    }
}

#[cfg(test)]
mod test_mpolynomials {
    #![allow(clippy::just_underscores_and_digits)]
    use super::*;
    use crate::shared_math::b_field_element::BFieldElement;
    use crate::shared_math::prime_field_element_flexible::PrimeFieldElementFlexible;
    use crate::utils::generate_random_numbers_u128;
    use primitive_types::U256;
    use rand::RngCore;
    use std::collections::HashSet;

    fn pfb(n: i64, q: u64) -> PrimeFieldElementFlexible {
        let q_u256: U256 = q.into();
        if n < 0 {
            let positive_n: U256 = (-n).into();
            let field_element_n: U256 = positive_n % q_u256;

            -PrimeFieldElementFlexible::new(field_element_n, q_u256)
        } else {
            let positive_n: U256 = n.into();
            let field_element_n: U256 = positive_n % q_u256;
            PrimeFieldElementFlexible::new(field_element_n, q_u256)
        }
    }

    fn get_x(q: u64) -> MPolynomial<PrimeFieldElementFlexible> {
        let mut coefficients: HashMap<Vec<u64>, PrimeFieldElementFlexible> = HashMap::new();
        coefficients.insert(vec![1, 0, 0], pfb(1, q));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_x_squared(q: u64) -> MPolynomial<PrimeFieldElementFlexible> {
        let mut coefficients: HashMap<Vec<u64>, PrimeFieldElementFlexible> = HashMap::new();
        coefficients.insert(vec![2, 0, 0], pfb(1, q));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_x_quartic(q: u64) -> MPolynomial<PrimeFieldElementFlexible> {
        let mut coefficients: HashMap<Vec<u64>, PrimeFieldElementFlexible> = HashMap::new();
        coefficients.insert(vec![4, 0, 0], pfb(1, q));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_y(q: u64) -> MPolynomial<PrimeFieldElementFlexible> {
        let mut coefficients: HashMap<Vec<u64>, PrimeFieldElementFlexible> = HashMap::new();
        coefficients.insert(vec![0, 1, 0], pfb(1, q));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_z(q: u64) -> MPolynomial<PrimeFieldElementFlexible> {
        let mut coefficients: HashMap<Vec<u64>, PrimeFieldElementFlexible> = HashMap::new();
        coefficients.insert(vec![0, 0, 1], pfb(1, q));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_xz(q: u64) -> MPolynomial<PrimeFieldElementFlexible> {
        let mut coefficients: HashMap<Vec<u64>, PrimeFieldElementFlexible> = HashMap::new();
        coefficients.insert(vec![1, 0, 1], pfb(1, q));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_x_squared_z_squared(q: u64) -> MPolynomial<PrimeFieldElementFlexible> {
        let mut coefficients: HashMap<Vec<u64>, PrimeFieldElementFlexible> = HashMap::new();
        coefficients.insert(vec![2, 0, 2], pfb(1, q));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_xyz(q: u64) -> MPolynomial<PrimeFieldElementFlexible> {
        let mut coefficients: HashMap<Vec<u64>, PrimeFieldElementFlexible> = HashMap::new();
        coefficients.insert(vec![1, 1, 1], pfb(1, q));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_x_plus_xz(q: u64) -> MPolynomial<PrimeFieldElementFlexible> {
        let mut coefficients: HashMap<Vec<u64>, PrimeFieldElementFlexible> = HashMap::new();
        coefficients.insert(vec![1, 0, 1], pfb(1, q));
        coefficients.insert(vec![1, 0, 0], pfb(1, q));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_x_minus_xz(q: u64) -> MPolynomial<PrimeFieldElementFlexible> {
        let mut coefficients: HashMap<Vec<u64>, PrimeFieldElementFlexible> = HashMap::new();
        coefficients.insert(vec![1, 0, 1], pfb(-1, q));
        coefficients.insert(vec![1, 0, 0], pfb(1, q));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_minus_17y(q: u64) -> MPolynomial<PrimeFieldElementFlexible> {
        let mut coefficients: HashMap<Vec<u64>, PrimeFieldElementFlexible> = HashMap::new();
        coefficients.insert(vec![0, 1, 0], pfb(-17, q));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_x_plus_xz_minus_17y(q: u64) -> MPolynomial<PrimeFieldElementFlexible> {
        let mut coefficients: HashMap<Vec<u64>, PrimeFieldElementFlexible> = HashMap::new();
        coefficients.insert(vec![1, 0, 1], pfb(1, q));
        coefficients.insert(vec![1, 0, 0], pfb(1, q));
        coefficients.insert(vec![0, 1, 0], pfb(-17, q));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_big_mpol(q: u64) -> MPolynomial<PrimeFieldElementFlexible> {
        let mut big_c: HashMap<Vec<u64>, PrimeFieldElementFlexible> = HashMap::new();
        big_c.insert(vec![0, 0, 1, 0, 0], pfb(1, q));
        big_c.insert(vec![0, 1, 0, 0, 0], pfb(1, q));
        big_c.insert(vec![10, 3, 8, 0, 3], pfb(-9, q));
        big_c.insert(vec![2, 3, 4, 0, 0], pfb(12, q));
        big_c.insert(vec![5, 5, 5, 0, 8], pfb(-4, q));
        big_c.insert(vec![0, 6, 0, 0, 1], pfb(3, q));
        big_c.insert(vec![1, 4, 11, 0, 0], pfb(10, q));
        big_c.insert(vec![1, 0, 12, 0, 2], pfb(2, q));
        MPolynomial {
            coefficients: big_c,
            variable_count: 5,
        }
    }

    fn get_big_mpol_extra_variabel(q: u64) -> MPolynomial<PrimeFieldElementFlexible> {
        let mut big_c: HashMap<Vec<u64>, PrimeFieldElementFlexible> = HashMap::new();
        big_c.insert(vec![0, 0, 1, 0, 0, 0], pfb(1, q));
        big_c.insert(vec![0, 1, 0, 0, 0, 0], pfb(1, q));
        big_c.insert(vec![10, 3, 8, 0, 3, 0], pfb(-9, q));
        big_c.insert(vec![2, 3, 4, 0, 0, 0], pfb(12, q));
        big_c.insert(vec![5, 5, 5, 0, 8, 0], pfb(-4, q));
        big_c.insert(vec![0, 6, 0, 0, 1, 0], pfb(3, q));
        big_c.insert(vec![1, 4, 11, 0, 0, 0], pfb(10, q));
        big_c.insert(vec![1, 0, 12, 0, 2, 0], pfb(2, q));
        MPolynomial {
            coefficients: big_c,
            variable_count: 6,
        }
    }

    #[test]
    fn equality_test() {
        let q = 23;
        assert_eq!(get_big_mpol(q), get_big_mpol_extra_variabel(q));
        assert_ne!(get_big_mpol(q), get_big_mpol_extra_variabel(q) + get_x(q));
    }

    #[test]
    fn simple_add_test() {
        let q = 23;
        let x = get_x(q);
        let xz = get_xz(q);
        let x_plus_xz = get_x_plus_xz(q);
        assert_eq!(x_plus_xz, x.clone() + xz.clone());

        let minus_17y = get_minus_17y(q);
        let x_plus_xz_minus_17_y = get_x_plus_xz_minus_17y(q);
        assert_eq!(x_plus_xz_minus_17_y, x + xz + minus_17y);
    }

    #[test]
    fn simple_sub_test() {
        let q = 65537;
        let x = get_x(q);
        let xz = get_xz(q);
        let x_minus_xz = get_x_minus_xz(q);
        assert_eq!(x_minus_xz, x.clone() - xz.clone());

        let big = get_big_mpol(q);
        assert_eq!(big.clone(), big.clone() - x.clone() + x.clone());
        assert_eq!(big.clone(), big.clone() - xz.clone() + xz.clone());
        assert_eq!(big.clone(), big.clone() - big.clone() + big.clone());
        assert_eq!(
            big.clone(),
            big.clone() - x_minus_xz.clone() + x_minus_xz.clone()
        );

        // Catch error fixed in sub where similar exponents in both terms of
        // `a(x,y) - b(x,y)` were calculated as `c_b - c_a` instead of as `c_a - c_b`,
        // as it should be.
        let _0 = MPolynomial::from_constant(pfb(0, q), 3);
        let _2 = MPolynomial::from_constant(pfb(2, q), 3);
        let _3 = MPolynomial::from_constant(pfb(3, q), 3);
        let _4 = MPolynomial::from_constant(pfb(4, q), 3);
        let _6 = MPolynomial::from_constant(pfb(6, q), 3);
        let _8 = MPolynomial::from_constant(pfb(8, q), 3);
        let _16 = MPolynomial::from_constant(pfb(16, q), 3);
        assert_eq!(_0, _2.clone() - _2.clone());
        assert_eq!(_0, _4.clone() - _4.clone());
        assert_eq!(_6, _8.clone() - _2.clone());
        assert_eq!(_4, _6.clone() - _2.clone());
        assert_eq!(_2, _4.clone() - _2.clone());
        assert_eq!(_6, _4.clone() + _2.clone());
        assert_eq!(_16, _8.clone() + _8.clone());
    }

    #[test]
    fn simple_mul_test() {
        let q = 13;
        let x = get_x(q);
        let z = get_z(q);
        let x_squared = get_x_squared(q);
        let xz = get_xz(q);
        assert_eq!(x_squared, x.clone() * x.clone());
        assert_eq!(xz, x.clone() * z.clone());
    }

    #[test]
    fn simple_modpow_test() {
        let q = 13;
        let one = pfb(1, q);
        let x = get_x(q);
        let x_squared = get_x_squared(q);
        let x_quartic = get_x_quartic(q);
        assert_eq!(x_squared, x.mod_pow(2.into(), one.clone()));
        assert_eq!(x_quartic, x.mod_pow(4.into(), one.clone()));
        assert_eq!(x_quartic, x_squared.mod_pow(2.into(), one.clone()));
        assert_eq!(
            get_x_squared_z_squared(q),
            get_xz(q).mod_pow(2.into(), one.clone())
        );

        assert_eq!(
            x_squared.scalar_mul(pfb(9, q)),
            x.scalar_mul(pfb(3, q)).mod_pow(2.into(), one.clone())
        );
        assert_eq!(
            x_squared.scalar_mul(pfb(16, q)),
            x.scalar_mul(pfb(4, q)).mod_pow(2.into(), one.clone())
        );
        assert_eq!(
            x_quartic.scalar_mul(pfb(16, q)),
            x.scalar_mul(pfb(2, q)).mod_pow(4.into(), one.clone())
        );
        assert_eq!(x_quartic, x.mod_pow(4.into(), one.clone()));
        assert_eq!(x_quartic, x_squared.mod_pow(2.into(), one.clone()));
        assert_eq!(
            get_x_squared_z_squared(q),
            get_xz(q).mod_pow(2.into(), one.clone())
        );
        assert_eq!(
            get_x_squared_z_squared(q).scalar_mul(pfb(25, q)),
            get_xz(q)
                .scalar_mul(pfb(5, q))
                .mod_pow(2.into(), one.clone())
        );
        assert_eq!(
            get_big_mpol(q) * get_big_mpol(q),
            get_big_mpol(q).mod_pow(2.into(), one.clone())
        );
        assert_eq!(
            get_big_mpol(q).scalar_mul(pfb(25, q)) * get_big_mpol(q),
            get_big_mpol(q)
                .scalar_mul(pfb(5, q))
                .mod_pow(2.into(), one.clone())
        );
    }

    #[test]
    fn variables_test() {
        let q = 13;
        let one = pfb(1, q);
        let vars_1 = MPolynomial::variables(1, one.clone());
        assert_eq!(1usize, vars_1.len());
        assert_eq!(get_x(q), vars_1[0]);
        let vars_3 = MPolynomial::variables(3, one);
        assert_eq!(3usize, vars_3.len());
        assert_eq!(get_x(q), vars_3[0]);
        assert_eq!(get_y(q), vars_3[1]);
        assert_eq!(get_z(q), vars_3[2]);
    }

    #[test]
    fn evaluate_symbolic_test() {
        let empty_intermediate_results: HashMap<Vec<u64>, Polynomial<PrimeFieldElementFlexible>> =
            HashMap::new();
        let empty_mod_pow_memoization: HashMap<
            (usize, u64),
            Polynomial<PrimeFieldElementFlexible>,
        > = HashMap::new();
        let empty_mul_memoization: HashMap<
            (Polynomial<PrimeFieldElementFlexible>, (usize, u64)),
            Polynomial<PrimeFieldElementFlexible>,
        > = HashMap::new();

        let q = 13;
        let zero = pfb(0.into(), q);
        let one = pfb(1.into(), q);
        let two = pfb(1.into(), q);
        let seven = pfb(7.into(), q);
        let xyz_m = get_xyz(q);
        let x: Polynomial<PrimeFieldElementFlexible> =
            Polynomial::from_constant(one.clone()).shift_coefficients(1, zero.clone());

        let mut precalculated_intermediate_results: HashMap<
            Vec<u64>,
            Polynomial<PrimeFieldElementFlexible>,
        > = HashMap::new();
        let precalculation_result = MPolynomial::precalculate_exponents_memoization(
            &[xyz_m.clone()],
            &vec![x.clone(), x.clone(), x.clone()],
            &mut precalculated_intermediate_results,
        );
        match precalculation_result {
            Ok(_) => (),
            Err(e) => panic!("error: {}", e),
        };

        let x_cubed: Polynomial<PrimeFieldElementFlexible> =
            Polynomial::from_constant(one.clone()).shift_coefficients(3, zero.clone());
        assert_eq!(
            x_cubed,
            xyz_m.evaluate_symbolic(&vec![x.clone(), x.clone(), x.clone()])
        );
        assert_eq!(
            x_cubed,
            xyz_m.evaluate_symbolic_with_memoization(
                &vec![x.clone(), x.clone(), x.clone()],
                &mut empty_mod_pow_memoization.clone(),
                &mut empty_mul_memoization.clone(),
                &mut empty_intermediate_results.clone()
            )
        );
        assert_eq!(
            x_cubed,
            xyz_m.evaluate_symbolic_with_memoization(
                &vec![x.clone(), x.clone(), x.clone()],
                &mut empty_mod_pow_memoization.clone(),
                &mut empty_mul_memoization.clone(),
                &mut precalculated_intermediate_results.clone()
            )
        );
        assert_eq!(
            x_cubed,
            xyz_m.evaluate_symbolic_with_memoization_precalculated(
                &vec![x.clone(), x.clone(), x],
                &mut precalculated_intermediate_results.clone()
            )
        );

        // More complex
        let univariate_pol_1 = Polynomial {
            coefficients: vec![
                one.clone(),
                seven.clone(),
                one.clone(),
                seven.clone(),
                seven.clone(),
                zero.clone(),
            ],
        };
        let univariate_pol_2 = Polynomial {
            coefficients: vec![
                one.clone(),
                seven.clone(),
                one.clone(),
                seven.clone(),
                zero.clone(),
                seven.clone(),
                seven.clone(),
                one.clone(),
                two.clone(),
            ],
        };
        let pol_m = get_x_plus_xz_minus_17y(q);
        let evaluated_pol_u = pol_m.evaluate_symbolic(&vec![
            univariate_pol_1.clone(),
            univariate_pol_1.clone(),
            univariate_pol_2.clone(),
        ]);

        // Calculated on Wolfram Alpha
        let expected_result = Polynomial {
            coefficients: vec![11, 6, 9, 7, 7, 5, 8, 2, 12, 2, 5, 1, 7]
                .iter()
                .map(|&x| pfb(x, q))
                .collect(),
        };

        assert_eq!(expected_result, evaluated_pol_u);
        // evaluate_symbolic_with_memoization_precalculated
        assert_eq!(
            expected_result,
            pol_m.evaluate_symbolic_with_memoization(
                &vec![
                    univariate_pol_1.clone(),
                    univariate_pol_1.clone(),
                    univariate_pol_2.clone(),
                ],
                &mut empty_mod_pow_memoization.clone(),
                &mut empty_mul_memoization.clone(),
                &mut empty_intermediate_results.clone()
            )
        );

        // Verify symbolic evaluation function with precalculated "intermediate results"
        let mut new_precalculated_intermediate_results: HashMap<
            Vec<u64>,
            Polynomial<PrimeFieldElementFlexible>,
        > = HashMap::new();
        let precalculation_result = MPolynomial::precalculate_exponents_memoization(
            &[pol_m.clone()],
            &vec![
                univariate_pol_1.clone(),
                univariate_pol_1.clone(),
                univariate_pol_2.clone(),
            ],
            &mut new_precalculated_intermediate_results,
        );
        match precalculation_result {
            Ok(_) => (),
            Err(e) => panic!("error: {}", e),
        };
        assert_eq!(
            expected_result,
            pol_m.evaluate_symbolic_with_memoization_precalculated(
                &vec![univariate_pol_1.clone(), univariate_pol_1, univariate_pol_2,],
                &mut new_precalculated_intermediate_results
            )
        );
    }

    #[test]
    fn evaluate_symbolic_with_zeros_test() {
        let q = 13;
        let one = pfb(1, q);
        let zero = pfb(0, q);
        let xm = get_x(q);
        let xu: Polynomial<PrimeFieldElementFlexible> =
            Polynomial::from_constant(one).shift_coefficients(1, zero);
        let zero_upol: Polynomial<PrimeFieldElementFlexible> = Polynomial::ring_zero();
        assert_eq!(
            xu,
            xm.evaluate_symbolic(&vec![xu.clone(), zero_upol.clone(), zero_upol.clone()])
        );

        let empty_intermediate_results: HashMap<Vec<u64>, Polynomial<PrimeFieldElementFlexible>> =
            HashMap::new();
        let empty_mod_pow_memoization: HashMap<
            (usize, u64),
            Polynomial<PrimeFieldElementFlexible>,
        > = HashMap::new();
        let empty_mul_memoization: HashMap<
            (Polynomial<PrimeFieldElementFlexible>, (usize, u64)),
            Polynomial<PrimeFieldElementFlexible>,
        > = HashMap::new();
        assert_eq!(
            xu,
            xm.evaluate_symbolic_with_memoization(
                &vec![xu.clone(), zero_upol.clone(), zero_upol],
                &mut empty_mod_pow_memoization.clone(),
                &mut empty_mul_memoization.clone(),
                &mut empty_intermediate_results.clone()
            )
        );
    }

    #[test]
    fn evaluate_test() {
        let q = 13;
        let x = get_x(q);
        assert_eq!(
            pfb(12, q),
            x.evaluate(&vec![pfb(12, q), pfb(0, q), pfb(0, q)])
        );
        assert_eq!(
            pfb(12, q),
            x.evaluate(&vec![pfb(12, q), pfb(12, q), pfb(12, q)])
        );

        let xszs = get_x_squared_z_squared(q);
        assert_eq!(
            pfb(1, q),
            xszs.evaluate(&vec![pfb(12, q), pfb(0, q), pfb(1, q)])
        );
        assert_eq!(
            pfb(1, q),
            xszs.evaluate(&vec![pfb(12, q), pfb(12, q), pfb(12, q)])
        );
        assert_eq!(
            pfb(3, q),
            xszs.evaluate(&vec![pfb(6, q), pfb(3, q), pfb(8, q)])
        );
        assert_eq!(
            pfb(9, q),
            xszs.evaluate(&vec![pfb(8, q), pfb(12, q), pfb(2, q)])
        );
        assert_eq!(
            pfb(3, q),
            xszs.evaluate(&vec![pfb(4, q), pfb(8, q), pfb(1, q)])
        );
        assert_eq!(
            pfb(12, q),
            xszs.evaluate(&vec![pfb(4, q), pfb(9, q), pfb(11, q)])
        );
        assert_eq!(
            pfb(4, q),
            xszs.evaluate(&vec![pfb(1, q), pfb(0, q), pfb(11, q)])
        );
        assert_eq!(
            pfb(0, q),
            xszs.evaluate(&vec![pfb(1, q), pfb(11, q), pfb(0, q)])
        );
        assert_eq!(
            pfb(4, q),
            xszs.evaluate(&vec![pfb(11, q), pfb(0, q), pfb(1, q)])
        );
    }

    #[test]
    fn lift_test() {
        let q = 13;
        let xm = get_x(q);
        let zm = get_z(q);
        let xs = Polynomial {
            coefficients: vec![pfb(0, q), pfb(1, q)],
        };
        assert_eq!(xm, MPolynomial::lift(xs.clone(), 0, 3));
        assert_eq!(zm, MPolynomial::lift(xs.clone(), 2, 3));

        let seven_s = Polynomial {
            coefficients: vec![pfb(7, q)],
        };
        assert_eq!(
            MPolynomial::from_constant(pfb(7, q), 3),
            MPolynomial::lift(seven_s.clone(), 0, 3)
        );
        assert_ne!(
            MPolynomial::from_constant(pfb(8, q), 3),
            MPolynomial::lift(seven_s, 0, 3)
        );

        let x_quartic_s = Polynomial {
            coefficients: vec![pfb(0, q), pfb(0, q), pfb(0, q), pfb(0, q), pfb(1, q)],
        };
        assert_eq!(
            get_x_quartic(q),
            MPolynomial::lift(x_quartic_s.clone(), 0, 3)
        );
        assert_eq!(
            get_x_quartic(q).scalar_mul(pfb(5, q)),
            MPolynomial::lift(x_quartic_s.scalar_mul(pfb(5, q)).clone(), 0, 3)
        );

        let x_squared_s = Polynomial {
            coefficients: vec![pfb(0, q), pfb(0, q), pfb(1, q)],
        };
        assert_eq!(
            get_x_quartic(q) + get_x_squared(q) + get_x(q),
            MPolynomial::lift(x_quartic_s.clone() + x_squared_s.clone() + xs.clone(), 0, 3)
        );
        assert_eq!(
            get_x_quartic(q).scalar_mul(pfb(5, q))
                + get_x_squared(q).scalar_mul(pfb(4, q))
                + get_x(q).scalar_mul(pfb(3, q)),
            MPolynomial::lift(
                x_quartic_s.scalar_mul(pfb(5, q))
                    + x_squared_s.scalar_mul(pfb(4, q))
                    + xs.scalar_mul(pfb(3, q)),
                0,
                3
            )
        );
    }

    #[test]
    fn add_assign_simple_test() {
        for i in 0..10 {
            let mut a = gen_mpolynomial(i, i, 14, u64::MAX);
            let a_clone = a.clone();
            let mut b = gen_mpolynomial(i, i, 140, u64::MAX);
            let b_clone = b.clone();
            a += b_clone.clone();
            assert_eq!(a_clone.clone() + b_clone.clone(), a);
            b += a_clone.clone();
            assert_eq!(a_clone + b_clone, b);
        }
    }

    #[test]
    fn square_test_simple() {
        let q = 13;
        let xz = get_xz(q);
        let xz_squared = get_x_squared_z_squared(q);
        assert_eq!(xz_squared, xz.square());
    }

    #[test]
    fn square_test() {
        for i in 0..10 {
            let poly = gen_mpolynomial(i, i, 7, u64::MAX);
            let actual = poly.square();
            let expected = poly.clone() * poly;
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn mul_commutative_test() {
        let a = gen_mpolynomial(40, 40, 100, u64::MAX);
        let b = gen_mpolynomial(20, 20, 1000, u64::MAX);
        let ab = a.clone() * b.clone();
        let ba = b.clone() * a.clone();
        assert_eq!(ab, ba);
    }

    #[test]
    fn mod_pow_test() {
        let a = gen_mpolynomial(4, 6, 2, 20);
        let mut acc = MPolynomial::from_constant(BFieldElement::ring_one(), 4);
        for i in 0..10 {
            let mod_pow = a.mod_pow(i.into(), BFieldElement::ring_one());
            println!(
                "mod_pow.coefficients.len() = {}",
                mod_pow.coefficients.len()
            );
            assert!(unique_exponent_vectors(&mod_pow));
            assert_eq!(acc, mod_pow);
            acc = acc.clone() * a.clone();
        }
    }

    #[test]
    fn precalculate_exponents_memoization_test() {
        for _ in 0..30 {
            let variable_count = 6;
            let a = gen_mpolynomial(variable_count, 12, 7, BFieldElement::MAX as u64);
            let b = gen_mpolynomial(variable_count, 12, 7, BFieldElement::MAX as u64);
            let c = gen_mpolynomial(variable_count, 12, 7, BFieldElement::MAX as u64);
            let mpolynomials = vec![a, b, c];
            let mut point = gen_upolynomials(variable_count - 1, 5, BFieldElement::MAX);

            // Add an x-value to the list of polynomials to verify that I didn't mess up the optimization
            // used for x-polynomials
            point.push(Polynomial {
                coefficients: vec![BFieldElement::ring_zero(), BFieldElement::ring_one()],
            });

            let mut precalculated_intermediate_results: HashMap<
                Vec<u64>,
                Polynomial<BFieldElement>,
            > = HashMap::new();
            let precalculation_result = MPolynomial::precalculate_exponents_memoization(
                &mpolynomials,
                &point,
                &mut precalculated_intermediate_results,
            );
            match precalculation_result {
                Ok(_) => (),
                Err(e) => panic!("error: {}", e),
            };

            // Verify precalculation results
            // println!("************** precalculation_result **************");
            for (k, v) in precalculated_intermediate_results.iter() {
                let mut expected_result = Polynomial::from_constant(BFieldElement::ring_one());
                for (i, &exponent) in k.iter().enumerate() {
                    expected_result = expected_result
                        * point[i].mod_pow(exponent.into(), BFieldElement::ring_one())
                }
                // println!("k = {:?}", k);
                assert_eq!(&expected_result, v);
            }

            // Verify that function gets the same result with and without precalculated values
            let mut empty_intermediate_results: HashMap<Vec<u64>, Polynomial<BFieldElement>> =
                HashMap::new();
            let mut empty_mod_pow_memoization: HashMap<(usize, u64), Polynomial<BFieldElement>> =
                HashMap::new();
            let mut empty_mul_memoization: HashMap<
                (Polynomial<BFieldElement>, (usize, u64)),
                Polynomial<BFieldElement>,
            > = HashMap::new();

            for mpolynomial in mpolynomials {
                let with_precalculation = mpolynomial.evaluate_symbolic_with_memoization(
                    &point,
                    &mut empty_mod_pow_memoization,
                    &mut empty_mul_memoization,
                    &mut precalculated_intermediate_results,
                );
                let with_precalculation_parallel = mpolynomial
                    .evaluate_symbolic_with_memoization_precalculated(
                        &point,
                        &mut precalculated_intermediate_results,
                    );
                let without_precalculation = mpolynomial.evaluate_symbolic_with_memoization(
                    &point,
                    &mut empty_mod_pow_memoization.clone(),
                    &mut empty_mul_memoization.clone(),
                    &mut empty_intermediate_results,
                );

                assert_eq!(with_precalculation, without_precalculation);
                assert_eq!(with_precalculation, with_precalculation_parallel);
            }
        }
    }

    fn unique_exponent_vectors(input: &MPolynomial<BFieldElement>) -> bool {
        let mut hashset: HashSet<Vec<u64>> = HashSet::new();

        input
            .coefficients
            .iter()
            .all(|(k, _v)| hashset.insert(k.clone()))
    }

    fn gen_upolynomials(
        degree: usize,
        count: usize,
        coefficient_limit: u128,
    ) -> Vec<Polynomial<BFieldElement>> {
        let mut ret: Vec<Polynomial<BFieldElement>> = vec![];
        for _ in 0..count {
            let coefficients: Vec<BFieldElement> = generate_random_numbers_u128(degree + 1, None)
                .into_iter()
                .map(|x| BFieldElement::new(x % coefficient_limit + 1))
                .collect();
            ret.push(Polynomial { coefficients });
        }

        ret
    }

    fn gen_mpolynomial(
        variable_count: usize,
        term_count: usize,
        exponenent_limit: u128,
        coefficient_limit: u64,
    ) -> MPolynomial<BFieldElement> {
        let mut coefficients: HashMap<Vec<u64>, BFieldElement> = HashMap::new();

        for _ in 0..term_count {
            let key = generate_random_numbers_u128(variable_count, None)
                .iter()
                .map(|x| (*x % exponenent_limit) as u64)
                .collect::<Vec<u64>>();
            let value = gen_bfield_element(coefficient_limit);
            coefficients.insert(key, value);
        }

        MPolynomial {
            variable_count,
            coefficients,
        }
    }

    fn gen_bfield_element(limit: u64) -> BFieldElement {
        let mut rng = rand::thread_rng();

        // adding 1 prevents us from building multivariate polynomial containing zero-coefficients
        let elem = rng.next_u64() % limit + 1;
        BFieldElement::new(elem as u128)
    }
}
