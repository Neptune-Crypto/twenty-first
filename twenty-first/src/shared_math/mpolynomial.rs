use super::traits::FiniteField;
use crate::shared_math::polynomial::Polynomial;
use crate::timing_reporter::TimingReporter;
use crate::util_types::tree_m_ary::Node;
use itertools::{izip, Itertools};
use num_traits::{One, Zero};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use std::cell::RefCell;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::error::Error;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};
use std::rc::Rc;
use std::{cmp, fmt};

type MCoefficients<T> = HashMap<Vec<u8>, T>;
pub type Degree = i64;

const EDMONDS_WEIGHT_CUTOFF_FACTOR: u64 = 2;

/// This is the data contained in each node in the tree. It contains
/// the information needed to calculate values for the
/// `polynomium_products` hash map. The keys in this hash map are
/// `abs_exponents` and the values are
/// `polynomium_products[parent] * prod_{i=0}^N(point[i]^diff_exponents[i])`.
#[derive(Debug, Clone)]
pub struct PolynomialEvaluationDataNode {
    diff_exponents: Vec<u8>,
    diff_sum: u64,
    abs_exponents: Vec<u8>,
    single_point: Option<usize>,
    x_powers: usize,
    // index: usize,
}

impl<T: Sized> Node<T> {
    fn traverse_tree<FF: FiniteField>(
        nodes: Vec<Rc<RefCell<Node<PolynomialEvaluationDataNode>>>>,
        point: &[Polynomial<FF>],
        one: FF,
        polynomium_products: &mut HashMap<Vec<u8>, Polynomial<FF>>,
    ) {
        let zero = FF::zero();

        // We might be able to avoid the `clone()` of exponents_list elements here, if we are smart
        polynomium_products.insert(
            nodes[0].borrow().data.abs_exponents.clone(),
            Polynomial::from_constant(one),
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
                    res.shift_coefficients_mut(*x_powers, zero);
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
                    let mut intermediate_mul: Polynomial<FF> = Polynomial::from_constant(one);
                    let mut intermediate_exponents: Vec<u8> = vec![0; point.len()];
                    let mut remaining_exponents: Vec<u8> = child_diff_exponents.clone();
                    let mut mod_pow_exponents: Vec<u8> = vec![0; point.len()];
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
                            let mut mod_pow_intermediate_lookup: Option<Polynomial<FF>> = None;
                            let mut mod_pow_reduced = mod_pow_exponents.clone();
                            while mod_pow_reduced[i] > 2 {
                                mod_pow_reduced[i] -= 1;
                                // println!("looking for {:?}", mod_pow_reduced);
                                if polynomium_products.contains_key(&mod_pow_reduced) {
                                    // println!("Found result for {:?}", mod_pow_reduced);
                                    mod_pow_intermediate_lookup =
                                        Some(polynomium_products[&mod_pow_reduced].clone());
                                    break;
                                }
                            }

                            match mod_pow_intermediate_lookup {
                                None => {
                                    // println!("Missed reduced mod_pow result!");
                                    let mod_pow_intermediate =
                                        point[i].mod_pow((*diff_exponent).into());
                                    polynomium_products.insert(
                                        mod_pow_exponents.clone(),
                                        mod_pow_intermediate.clone(),
                                    );
                                    mod_pow_intermediate
                                }
                                Some(res) => {
                                    // println!("Found reduced mod_pow result!");
                                    point[i].mod_pow((*diff_exponent - mod_pow_reduced[i]).into())
                                        * res
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
                res.shift_coefficients_mut(*x_powers, zero);
                polynomium_products.insert(child_abs_exponents.clone(), res);
            }
        }
    }
}

pub struct MPolynomial<T: FiniteField> {
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
    // Notice that the exponent values may not exceed 0xFF = 255 = u8::MAX.
    pub coefficients: HashMap<Vec<u8>, T>,
}

impl<T: FiniteField> Clone for MPolynomial<T> {
    fn clone(&self) -> Self {
        Self {
            variable_count: self.variable_count,
            coefficients: self.coefficients.clone(),
        }
    }
}

impl<T: FiniteField> Debug for MPolynomial<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MPolynomial")
            .field("variable_count", &self.variable_count)
            .field("coefficients", &self.coefficients)
            .finish()
    }
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
        write!(f, "{self:?}")
    }
}

impl<FF: FiniteField> Display for MPolynomial<FF> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let output = if self.is_zero() {
            "0".to_string()
        } else {
            // Stable sort once per variable. This implementation is implicit “invlex” monomial
            // order printed in reverse, i.e., constant term comes first, high-degree monomials are
            // printed last.
            let mut coefficients_iter = self.coefficients.iter().sorted_by_key(|x| x.0[0]);
            for i in 1..self.variable_count {
                coefficients_iter = coefficients_iter.sorted_by_key(|x| x.0[i]);
            }
            coefficients_iter
                .map(|(k, v)| Self::term_print(k, v))
                .filter(|s| !s.is_empty())
                .join(" + ")
        };

        write!(f, "{output}")
    }
}

impl<FF: FiniteField> Hash for MPolynomial<FF> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.variable_count.hash(state);
        let mut coeffs_list_sorted = self
            .coefficients
            .iter()
            .filter(|x| !x.1.is_zero())
            .collect_vec();
        coeffs_list_sorted.sort_by(|a, b| a.0.partial_cmp(b.0).unwrap());
        coeffs_list_sorted.hash(state);
    }
}

impl<FF: FiniteField> PartialEq for MPolynomial<FF> {
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

        let mut padded: HashMap<Vec<u8>, FF> = HashMap::new();
        for (k, v) in shortest.iter() {
            let mut pad = k.clone();
            pad.resize_with(var_count, || 0);
            padded.insert(pad, *v);
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

impl<FF: FiniteField> Eq for MPolynomial<FF> {}

impl<FF: FiniteField> MPolynomial<FF> {
    fn term_print(exponents: &[u8], coefficient: &FF) -> String {
        if coefficient.is_zero() {
            return "".to_string();
        }

        let mut str_elems: Vec<String> = vec![];
        if !coefficient.is_one() || exponents.iter().all(|&x| x.is_zero()) {
            str_elems.push(coefficient.to_string());
        }

        for (i, exponent) in exponents.iter().enumerate() {
            if *exponent == 0 {
                continue;
            }
            let factor_str = if *exponent == 1 {
                format!("x_{i}")
            } else {
                format!("x_{i}^{exponent}")
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

    pub fn is_one(&self) -> bool {
        if self.coefficients.is_empty() {
            return false;
        }

        for (k, v) in self.coefficients.iter() {
            if *k == vec![0u8; self.variable_count] {
                if !v.is_one() {
                    return false;
                } else {
                    continue;
                }
            }

            if !v.is_zero() {
                return false;
            }
        }

        true
    }

    /// Returns an `MPolynomial` instance over `variable_count` variables
    /// that evaluates to `element` everywhere.
    /// I.e.
    ///     P(x,y..,z) = element
    ///
    /// Note that in this encoding
    ///     P(x,y) == P(x,w)
    /// but
    ///     P(x,y) != P(x,y,z).
    pub fn from_constant(element: FF, variable_count: usize) -> Self {
        // Potential guarantee: assert!(!element.zero);
        let mut cs: MCoefficients<FF> = HashMap::new();
        cs.insert(vec![0; variable_count], element);
        Self {
            variable_count,
            coefficients: cs,
        }
    }

    /// Returns a vector of multivariate polynomials that each represent a
    /// function of one indeterminate variable to the first power. For example,
    /// `let p = variables(3, 1.into())` represents the functions
    ///
    /// - `p[0] =` $f(x,y,z) = x$
    /// - `p[1] =`$f(x,y,z) = y$
    /// - `p[2] =`$f(x,y,z) = z$
    ///
    /// and `let p = variables(5, 1.into())` represents the functions
    ///
    /// - `p[0] =` $p(a,b,c,d,e) = a$
    /// - `p[1] =` $p(a,b,c,d,e) = b$
    /// - `p[2] =` $p(a,b,c,d,e) = c$
    /// - `p[3] =` $p(a,b,c,d,e) = d$
    /// - `p[4] =` $p(a,b,c,d,e) = e$
    pub fn variables(variable_count: usize) -> Vec<Self> {
        let one = FF::one();
        let mut res: Vec<Self> = vec![];
        for i in 0..variable_count {
            let mut exponent = vec![0u8; variable_count];
            exponent[i] = 1;
            let mut coefficients: MCoefficients<FF> = HashMap::new();
            coefficients.insert(exponent, one);
            res.push(Self {
                variable_count,
                coefficients,
            });
        }

        res
    }

    pub fn precalculate_symbolic_exponents(
        mpols: &[Self],
        point: &[Polynomial<FF>],
        exponents_memoization: &mut HashMap<Vec<u8>, Polynomial<FF>>,
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
        let one: FF = FF::one(); // guaranteed to exist because of above checks

        let mut exponents_list: Vec<Vec<u8>> = Self::extract_exponents_list(mpols)?;
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
                    .map(|(ei, ej)| (*ei - *ej) as u64)
                    .sum::<u64>();

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
            let mut diff_exponents: Vec<u8> = nodes[end]
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
            if diff_exponents.iter().sum::<u8>() == 1u8 {
                nodes[end].borrow_mut().data.single_point =
                    Some(diff_exponents.iter().position(|&x| x == 1).unwrap());
            }
            nodes[end].borrow_mut().data.diff_exponents = diff_exponents;
            nodes[end].borrow_mut().data.diff_sum = weight;
            nodes[end].borrow_mut().data.x_powers = x_power as usize;
        }
        timer.elapsed("built nodes");

        Node::<PolynomialEvaluationDataNode>::traverse_tree::<FF>(
            nodes,
            point,
            one,
            exponents_memoization,
        );
        timer.elapsed("traversed tree");
        let report = timer.finish();
        println!("{report}");

        Ok(())
    }

    // Simple evaluation, without precalculated results
    pub fn evaluate(&self, point: &[FF]) -> FF {
        assert_eq!(
            self.variable_count,
            point.len(),
            "Dimensionality of multivariate polynomial, {}, and dimensionality of point, {}, must agree in evaluate", self.variable_count, point.len()
        );

        let mut acc = FF::zero();
        for (k, v) in self.coefficients.iter() {
            let mut prod = FF::one();
            for i in 0..k.len() {
                // If the exponent is zero, multiplying with this factor is the identity operator.
                if k[i] == 0 {
                    continue;
                }
                // FIXME: We really don't want to cast to 'u32' here, but
                // refactoring k: Vec<u32> is a bit of a task, too. See issue ...
                prod *= point[i].mod_pow_u32(k[i] as u32);
            }
            prod *= *v;
            acc += prod;
        }

        acc
    }

    // Warning: may not be used for any un-verified user input as it's easy to force to
    // run slow for anyone who gets to decide the input
    // Get the exponents list of a list of multivariate polynomials. This is just a list of
    // all the exponent vectors present in a list of multivariate polynomials.
    // We precalculate this to make the verifier faster, as the exponents list is the
    // same across all calls to the evaluater
    pub fn extract_exponents_list(mpols: &[Self]) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
        if mpols.is_empty() {
            return Err(Box::new(PrecalculationError::EmptyInput));
        }

        // We assume that the input is well-formed, meaning that the `variable_count`
        // in all input `mpols` are the same.
        // We include the 0-exponent list as this is needed when doing symbolic evaluation
        // in the STARK prover.
        // It's easy to find collissions for the `hashbrown` hasher, so this function
        // is not secure to
        // Warning: it's easy to find collissions for the `hashbrown` hasher, so
        // this function and this hasher and hash set may not be used to handle any user-
        // generated input.
        // Pre-allocating the hash set with the right capacity *seemed* to give a speedup,
        // but I might be wrong about that. That speedup is the reason I included the
        // below "unused" assignment.
        #[allow(unused_assignments)]
        let mut exponents_set: hashbrown::hash_set::HashSet<Vec<u8>> =
            hashbrown::hash_set::HashSet::with_capacity(mpols[0].coefficients.len());
        exponents_set = mpols
            .iter()
            .flat_map(|mpol| mpol.coefficients.keys().map(|x| x.to_owned()))
            .collect();
        exponents_set.insert(vec![0; mpols[0].variable_count]);

        Ok(exponents_set.into_iter().collect())
    }

    // This does relate to multivariate polynomials since it is used as a step to do faster
    // scalar evaluation. For this reason, this function is placed here.
    /// Get a hash map with precalculated values for `point[i]^j`
    /// Only exponents 2 and above are stored.
    pub fn precalculate_scalar_mod_pows(limit: u8, point: &[FF]) -> HashMap<(usize, u8), FF> {
        let mut hash_map: HashMap<(usize, u8), FF> = HashMap::new();

        // TODO: Would runing this in parallel give a speedup?
        for (i, coordinate) in point.iter().enumerate() {
            let mut acc = *coordinate;
            for k in 2..=limit {
                acc *= *coordinate;
                hash_map.insert((i, k), acc);
            }
        }

        hash_map
    }

    // Precalculater for scalar evaluation
    // Assumes that all point[i]^j that are needed already exist in
    // `precalculated_mod_pows`
    pub fn precalculate_scalar_exponents(
        point: &[FF],
        precalculated_mod_pows: &HashMap<(usize, u8), FF>,
        exponents_list: &[Vec<u8>],
    ) -> Result<HashMap<Vec<u8>, FF>, Box<dyn Error>> {
        if point.is_empty() {
            return Err(Box::new(PrecalculationError::EmptyInput));
        }

        // We assume all exponents_list elements have same length
        let variable_count = exponents_list[0].len();

        if point.len() != variable_count {
            return Err(Box::new(PrecalculationError::LengthMismatch));
        }

        // Perform parallel computation of all intermediate results
        // which constitute calculations on the form
        // `prod_i^N(point[i]^e_i)
        let mut intermediate_results_hash_map: HashMap<Vec<u8>, FF> = HashMap::new();
        let one: FF = FF::one();
        let intermediate_results: Vec<FF> = exponents_list
            .par_iter()
            .map(|exponents| {
                let mut acc = one;
                for (i, e) in exponents.iter().enumerate() {
                    if e.is_zero() {
                        continue;
                    } else if e.is_one() {
                        acc *= point[i];
                    } else {
                        acc *= precalculated_mod_pows[&(i, *e)];
                    }
                }
                acc
            })
            .collect();

        for (exponents, result) in izip!(exponents_list, intermediate_results) {
            intermediate_results_hash_map.insert(exponents.to_vec(), result);
        }

        Ok(intermediate_results_hash_map)
    }

    pub fn evaluate_with_precalculation(
        &self,
        point: &[FF],
        intermediate_results: &HashMap<Vec<u8>, FF>,
    ) -> FF {
        assert_eq!(
            self.variable_count,
            point.len(),
            "Dimensionality of multivariate polynomial and point must agree in evaluate"
        );
        let zero: FF = FF::zero();
        let acc = self
            .coefficients
            .par_iter()
            .map(|(k, v)| intermediate_results[k] * *v)
            .reduce(|| zero, |a, b| a + b);
        acc
    }

    // Substitute the variables in a multivariate polynomial with univariate polynomials in parallel.
    // All "intermediate results" **must** be present in `exponents_memoization` or this function
    // will panic.
    pub fn evaluate_symbolic_with_memoization_precalculated(
        &self,
        point: &[Polynomial<FF>],
        exponents_memoization: &mut HashMap<Vec<u8>, Polynomial<FF>>,
    ) -> Polynomial<FF> {
        assert_eq!(
            self.variable_count,
            point.len(),
            "Dimensionality of multivariate polynomial and point must agree in evaluate_symbolic"
        );
        let acc = self
            .coefficients
            .par_iter()
            .map(|(k, v)| exponents_memoization[k].scalar_mul(*v))
            .reduce(Polynomial::zero, |a, b| a + b);

        acc
    }

    // Substitute the variables in a multivariate polynomial with univariate polynomials, fast
    #[allow(clippy::map_entry)]
    #[allow(clippy::type_complexity)]
    pub fn evaluate_symbolic_with_memoization(
        &self,
        point: &[Polynomial<FF>],
        mod_pow_memoization: &mut HashMap<(usize, u8), Polynomial<FF>>,
        mul_memoization: &mut HashMap<(Polynomial<FF>, (usize, u8)), Polynomial<FF>>,
        exponents_memoization: &mut HashMap<Vec<u8>, Polynomial<FF>>,
    ) -> Polynomial<FF> {
        // Notice that the `exponents_memoization` only gives a speedup if this function is evaluated multiple
        // times for the same `point` input. This condition holds when evaluating the AIR constraints
        // symbolically in a generic STARK prover.
        assert_eq!(
            self.variable_count,
            point.len(),
            "Dimensionality of multivariate polynomial and point must agree in evaluate_symbolic"
        );
        let points_are_x: Vec<bool> = point.iter().map(|p| p.is_x()).collect();
        let mut acc: Polynomial<FF> = Polynomial::zero();
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
            let mut prod: Polynomial<FF>;
            if exponents_memoization.contains_key(k) {
                prod = exponents_memoization[k].clone();
            } else {
                println!("Missed!");
                prod = Polynomial::from_constant(FF::one());
                let mut k_sorted: Vec<(usize, u8)> = k.clone().into_iter().enumerate().collect();
                k_sorted.sort_by_key(|e| e.1);
                let mut x_pow_mul = 0;
                for (i, ki) in k_sorted.into_iter() {
                    // calculate prod * point[i].mod_pow(k[i].into(), v.one()) with some optimizations,
                    // mainly memoization.
                    // prod = prod * point[i].mod_pow(k[i].into(), v.one());

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

                        let mut reduced_mul_result: Option<Polynomial<FF>> = None;
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
                            let mod_pow_res = point[i].mod_pow(mod_pow_key.1.into());
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

                prod.shift_coefficients_mut(x_pow_mul as usize, FF::zero());
                exponents_memoization.insert(k.to_vec(), prod.clone());
            }
            prod.scalar_mul_mut(*v);
            acc += prod;
        }

        acc
    }

    // Substitute the variables in a multivariate polynomial with univariate polynomials
    pub fn evaluate_symbolic(&self, point: &[Polynomial<FF>]) -> Polynomial<FF> {
        assert_eq!(
            self.variable_count,
            point.len(),
            "Dimensionality of multivariate polynomial and point must agree in evaluate_symbolic"
        );
        let mut acc: Polynomial<FF> = Polynomial::zero();
        for (k, v) in self.coefficients.iter() {
            let mut prod = Polynomial::from_constant(*v);
            for i in 0..k.len() {
                // calculate prod * point[i].mod_pow(k[i].into(), v.one()) with some small optimizations
                // prod = prod * point[i].mod_pow(k[i].into(), v.one());
                prod = if k[i] == 0 {
                    prod
                } else if point[i].is_x() {
                    prod * point[i].shift_coefficients(k[i] as usize - 1)
                } else {
                    prod * point[i].mod_pow(k[i].into())
                };
            }
            acc += prod;
        }

        acc
    }

    /// lift
    /// Creates a multivariate polynomial from a univariate one.
    pub fn lift(
        univariate_polynomial: Polynomial<FF>,
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

        let one = FF::one();
        let mut coefficients: MCoefficients<FF> = HashMap::new();
        let mut key = vec![0u8; variable_count];
        key[variable_index] = 1;
        coefficients.insert(key, one);
        let indeterminate: MPolynomial<FF> = Self {
            variable_count,
            coefficients,
        };

        let mut acc = MPolynomial::<FF>::zero(variable_count);
        for i in 0..univariate_polynomial.coefficients.len() {
            acc +=
                MPolynomial::from_constant(univariate_polynomial.coefficients[i], variable_count)
                    * indeterminate.pow(i as u8);
        }

        acc
    }

    #[must_use]
    pub fn scalar_mul(&self, factor: FF) -> Self {
        if self.is_zero() {
            return Self::zero(self.variable_count);
        }

        let mut output_coefficients: MCoefficients<FF> = HashMap::new();
        for (k, &v) in self.coefficients.iter() {
            output_coefficients.insert(k.to_vec(), v * factor);
        }

        Self {
            variable_count: self.variable_count,
            coefficients: output_coefficients,
        }
    }

    pub fn scalar_mul_mut(&mut self, factor: FF) {
        if self.is_zero() || factor.is_one() {
            return;
        }

        for (_k, v) in self.coefficients.iter_mut() {
            *v *= factor;
        }
    }

    #[must_use]
    pub fn pow(&self, pow: u8) -> Self {
        // Handle special case of 0^0
        if pow.is_zero() {
            let mut coefficients: MCoefficients<FF> = HashMap::new();
            coefficients.insert(vec![0; self.variable_count], FF::one());
            return MPolynomial {
                variable_count: self.variable_count,
                coefficients,
            };
        }

        // Handle 0^n for n > 0
        if self.is_zero() {
            return Self::zero(self.variable_count);
        }

        // create object, to be populated
        let exp = vec![0u8; self.variable_count];
        let mut acc_coefficients_init: MCoefficients<FF> = HashMap::new();
        acc_coefficients_init.insert(exp, FF::one());
        let mut acc: MPolynomial<FF> = Self {
            variable_count: self.variable_count,
            coefficients: acc_coefficients_init,
        };

        // calculate bit length
        let mut bit_length = 0;
        let mut pow_ = pow;
        while pow_ != 0 {
            pow_ >>= 1;
            bit_length += 1;
        }

        // square and multiply
        for i in (0..=bit_length).rev() {
            acc = acc.square();
            if pow & 1 << i != 0 {
                acc *= self.clone();
            }
        }

        acc
    }

    #[must_use]
    pub fn square(&self) -> Self {
        if self.is_zero() {
            return Self::zero(self.variable_count);
        }

        let mut output_coefficients: MCoefficients<FF> = HashMap::new();
        let exponents = self.coefficients.keys().collect::<Vec<&Vec<u8>>>();
        let two = FF::one() + FF::one();

        for i in 0..exponents.len() {
            let ki = exponents[i];
            let v0 = self.coefficients[ki];
            let mut diagonal_exponents = Vec::with_capacity(self.variable_count);
            for exponent in ki {
                diagonal_exponents.push(exponent * 2);
            }
            if output_coefficients.contains_key(&diagonal_exponents) {
                output_coefficients.insert(
                    diagonal_exponents.to_vec(),
                    v0 * v0 + output_coefficients[&diagonal_exponents],
                );
            } else {
                output_coefficients.insert(diagonal_exponents.to_vec(), v0 * v0);
            }

            for kj in exponents.iter().skip(i + 1) {
                let mut non_diagonal_exponents = Vec::with_capacity(self.variable_count);
                for k in 0..self.variable_count {
                    // TODO: Can overflow.
                    let exponent = ki[k] + kj[k];
                    non_diagonal_exponents.push(exponent);
                }
                let v1 = self.coefficients[*kj];
                if output_coefficients.contains_key(&non_diagonal_exponents) {
                    output_coefficients.insert(
                        non_diagonal_exponents.to_vec(),
                        two * v0 * v1 + output_coefficients[&non_diagonal_exponents],
                    );
                } else {
                    output_coefficients.insert(non_diagonal_exponents.to_vec(), two * v0 * v1);
                }
            }
        }

        Self {
            coefficients: output_coefficients,
            variable_count: self.variable_count,
        }
    }

    /// Return the highest number present in the list of list of exponents
    /// For `P(x,y) = x^4*y^3`, `4` would be returned.
    pub fn max_exponent(&self) -> u8 {
        *self
            .coefficients
            .keys()
            .map(|exponents| exponents.iter().max().unwrap_or(&0))
            .max()
            .unwrap_or(&0)
    }

    /// Calculate the "total degree" of a multivariate polynomial.
    ///
    /// The total degree is defined as the highest combined degree of any
    /// term where the combined degree is the sum of all the term's variable
    /// exponents.
    ///
    /// As a convention, the polynomial f(x) = 0 has degree -1.
    pub fn degree(&self) -> Degree {
        if self.is_zero() {
            // The zero polynomial has degree -1.
            return -1;
        };

        let total_degree: Degree = self
            .coefficients
            .iter()
            .filter(|(_, coefficient)| !coefficient.is_zero())
            .map(|(exponent, _)| exponent.iter().map(|&x| x as Degree).sum())
            .max()
            .unwrap_or(0);

        total_degree
    }

    /// Removes exponents whose coefficients are 0.
    pub fn normalize(&mut self) {
        let mut spurious_exponents: Vec<Vec<u8>> = vec![];
        for (exponent, coefficient) in self.coefficients.iter() {
            if coefficient.is_zero() {
                spurious_exponents.push(exponent.to_owned());
            }
        }
        for se in spurious_exponents {
            self.coefficients.remove(&se);
        }
    }

    /// During symbolic evaluation, i.e., when substituting a univariate polynomial for one of the
    /// variables, the total degree of the resulting polynomial can be upper bounded.  This bound
    /// is the `total_degree_bound`, and can be calculated across all terms.  Only the constant
    /// zero polynomial `P(x,..) = 0` has a negative degree and it is always -1.  All other
    /// constant polynomials have degree 0.
    ///
    /// - `max_degrees`:  the max degrees for each of the univariate polynomials.
    /// - `total_degree_bound`:  the max resulting degree from the substitution.
    pub fn symbolic_degree_bound(&self, max_degrees: &[i64]) -> Degree {
        assert_eq!(max_degrees.len(), self.variable_count);
        let mut total_degree_bound: i64 = -1;
        for (exponents, coefficients) in self.coefficients.iter() {
            if coefficients.is_zero() {
                continue;
            }

            let signed_exponents = exponents.iter().map(|e| {
                let res = i64::try_from(*e);
                assert!(res.is_ok());
                res.unwrap()
            });

            let term_degree_bound = max_degrees
                .iter()
                .zip(signed_exponents)
                .map(|(md, exp)| md * exp)
                .sum();
            total_degree_bound = cmp::max(total_degree_bound, term_degree_bound);
        }
        total_degree_bound
    }
}

impl<FF: FiniteField> Add for MPolynomial<FF> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let variable_count: usize = cmp::max(self.variable_count, other.variable_count);
        if self.is_zero() && other.is_zero() {
            return Self::zero(variable_count);
        }

        let mut output_coefficients: MCoefficients<FF> = HashMap::new();
        for (k, &v) in self.coefficients.iter() {
            let mut pad = k.clone();
            pad.resize_with(variable_count, || 0);
            output_coefficients.insert(pad, v);
        }
        for (k, v) in other.coefficients.iter() {
            let mut pad = k.clone();
            pad.resize_with(variable_count, || 0);

            // TODO: This can probably be done smarter
            if output_coefficients.contains_key(&pad) {
                output_coefficients.insert(pad.clone(), v.to_owned() + output_coefficients[&pad]);
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

impl<FF: FiniteField> AddAssign for MPolynomial<FF> {
    fn add_assign(&mut self, rhs: Self) {
        if self.variable_count != rhs.variable_count {
            let result = self.clone() + rhs;
            self.variable_count = result.variable_count;
            self.coefficients = result.coefficients;
            return;
        }

        for (k, v1) in rhs.coefficients.iter() {
            if self.coefficients.contains_key(k) {
                let v0 = self.coefficients[k];
                self.coefficients.insert(k.clone(), v0 + v1.to_owned());
            } else {
                self.coefficients.insert(k.clone(), v1.to_owned());
            }
        }
    }
}

impl<FF: FiniteField> Sum for MPolynomial<FF> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(MPolynomial::zero(0), |a, b| a + b)
    }
}

impl<FF: FiniteField> Sub for MPolynomial<FF> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let variable_count: usize = cmp::max(self.variable_count, other.variable_count);
        if self.is_zero() && other.is_zero() {
            return Self::zero(variable_count);
        }

        let mut output_coefficients: MCoefficients<FF> = HashMap::new();
        for (k, &v) in self.coefficients.iter() {
            let mut pad = k.clone();
            pad.resize_with(variable_count, || 0);
            output_coefficients.insert(pad, v);
        }
        for (k, &v) in other.coefficients.iter() {
            let mut pad = k.clone();
            pad.resize_with(variable_count, || 0);

            // TODO: This can probably be done smarter
            if output_coefficients.contains_key(&pad) {
                output_coefficients.insert(pad.to_vec(), output_coefficients[&pad] - v);
            } else {
                output_coefficients.insert(pad.to_vec(), -v);
            }
        }

        Self {
            coefficients: output_coefficients,
            variable_count,
        }
    }
}

impl<FF: FiniteField> Neg for MPolynomial<FF> {
    type Output = Self;

    fn neg(self) -> Self {
        let mut output_coefficients: MCoefficients<FF> = HashMap::new();
        for (k, &v) in self.coefficients.iter() {
            output_coefficients.insert(k.to_vec(), -v);
        }

        Self {
            variable_count: self.variable_count,
            coefficients: output_coefficients,
        }
    }
}

impl<FF: FiniteField> Mul for MPolynomial<FF> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let variable_count: usize = cmp::max(self.variable_count, other.variable_count);
        if self.is_zero() || other.is_zero() {
            return Self::zero(variable_count);
        }

        if self.is_one() {
            return other;
        } else if other.is_one() {
            return self;
        }

        let mut output_coefficients: MCoefficients<FF> = HashMap::new();
        for (k0, &v0) in self.coefficients.iter() {
            for (k1, &v1) in other.coefficients.iter() {
                let mut exponent = vec![0u8; variable_count];
                for k in 0..self.variable_count {
                    exponent[k] += k0[k];
                }
                for k in 0..other.variable_count {
                    exponent[k] += k1[k];
                }
                if output_coefficients.contains_key(&exponent) {
                    output_coefficients
                        .insert(exponent.to_vec(), v0 * v1 + output_coefficients[&exponent]);
                } else {
                    output_coefficients.insert(exponent.to_vec(), v0 * v1);
                }
            }
        }
        Self {
            coefficients: output_coefficients,
            variable_count,
        }
    }
}

impl<FF: FiniteField> MulAssign for MPolynomial<FF> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs
    }
}

#[cfg(test)]
mod test_mpolynomials {
    #![allow(clippy::just_underscores_and_digits)]

    use num_traits::One;
    use num_traits::Pow;
    use rand::Rng;
    use std::collections::hash_map::DefaultHasher;
    use std::collections::HashSet;
    use std::hash::Hasher;

    use crate::shared_math::b_field_element::BFieldElement;
    use crate::shared_math::other::random_elements;
    use crate::shared_math::other::random_elements_range;
    use crate::shared_math::x_field_element::XFieldElement;

    use super::*;

    fn get_x() -> MPolynomial<BFieldElement> {
        let mut coefficients: HashMap<Vec<u8>, BFieldElement> = HashMap::new();
        coefficients.insert(vec![1, 0, 0], BFieldElement::from(1u64));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_x_squared() -> MPolynomial<BFieldElement> {
        let mut coefficients: HashMap<Vec<u8>, BFieldElement> = HashMap::new();
        coefficients.insert(vec![2, 0, 0], BFieldElement::from(1u64));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_x_quartic() -> MPolynomial<BFieldElement> {
        let mut coefficients: HashMap<Vec<u8>, BFieldElement> = HashMap::new();
        coefficients.insert(vec![4, 0, 0], BFieldElement::from(1u64));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_y() -> MPolynomial<BFieldElement> {
        let mut coefficients: HashMap<Vec<u8>, BFieldElement> = HashMap::new();
        coefficients.insert(vec![0, 1, 0], BFieldElement::from(1u64));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_z() -> MPolynomial<BFieldElement> {
        let mut coefficients: HashMap<Vec<u8>, BFieldElement> = HashMap::new();
        coefficients.insert(vec![0, 0, 1], BFieldElement::from(1u64));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_xz() -> MPolynomial<BFieldElement> {
        let mut coefficients: HashMap<Vec<u8>, BFieldElement> = HashMap::new();
        coefficients.insert(vec![1, 0, 1], BFieldElement::from(1u64));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_x_squared_z_squared() -> MPolynomial<BFieldElement> {
        let mut coefficients: HashMap<Vec<u8>, BFieldElement> = HashMap::new();
        coefficients.insert(vec![2, 0, 2], BFieldElement::from(1u64));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_xyz() -> MPolynomial<BFieldElement> {
        let mut coefficients: HashMap<Vec<u8>, BFieldElement> = HashMap::new();
        coefficients.insert(vec![1, 1, 1], BFieldElement::from(1u64));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_x_plus_xz() -> MPolynomial<BFieldElement> {
        let mut coefficients: HashMap<Vec<u8>, BFieldElement> = HashMap::new();
        coefficients.insert(vec![1, 0, 1], BFieldElement::from(1u64));
        coefficients.insert(vec![1, 0, 0], BFieldElement::from(1u64));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_x_minus_xz() -> MPolynomial<BFieldElement> {
        let mut coefficients: HashMap<Vec<u8>, BFieldElement> = HashMap::new();
        coefficients.insert(vec![1, 0, 1], -BFieldElement::from(1u64));
        coefficients.insert(vec![1, 0, 0], BFieldElement::from(1u64));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_minus_17y() -> MPolynomial<BFieldElement> {
        let mut coefficients: HashMap<Vec<u8>, BFieldElement> = HashMap::new();
        coefficients.insert(vec![0, 1, 0], -BFieldElement::from(17u64));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_x_plus_xz_minus_17y() -> MPolynomial<BFieldElement> {
        let mut coefficients: HashMap<Vec<u8>, BFieldElement> = HashMap::new();
        coefficients.insert(vec![1, 0, 1], BFieldElement::from(1u64));
        coefficients.insert(vec![1, 0, 0], BFieldElement::from(1u64));
        coefficients.insert(vec![0, 1, 0], -BFieldElement::from(17u64));
        MPolynomial {
            coefficients,
            variable_count: 3,
        }
    }

    fn get_big_mpol() -> MPolynomial<BFieldElement> {
        let mut big_c: HashMap<Vec<u8>, BFieldElement> = HashMap::new();
        big_c.insert(vec![0, 0, 1, 0, 0], BFieldElement::from(1u64));
        big_c.insert(vec![0, 1, 0, 0, 0], BFieldElement::from(1u64));
        big_c.insert(vec![10, 3, 8, 0, 3], -BFieldElement::from(9u64));
        big_c.insert(vec![2, 3, 4, 0, 0], BFieldElement::from(12u64));
        big_c.insert(vec![5, 5, 5, 0, 8], -BFieldElement::from(4u64));
        big_c.insert(vec![0, 6, 0, 0, 1], BFieldElement::from(3u64));
        big_c.insert(vec![1, 4, 11, 0, 0], BFieldElement::from(10u64));
        big_c.insert(vec![1, 0, 12, 0, 2], BFieldElement::from(2u64));
        MPolynomial {
            coefficients: big_c,
            variable_count: 5,
        }
    }

    fn get_big_mpol_extra_variabel() -> MPolynomial<BFieldElement> {
        let mut big_c: HashMap<Vec<u8>, BFieldElement> = HashMap::new();
        big_c.insert(vec![0, 0, 1, 0, 0, 0], BFieldElement::from(1u64));
        big_c.insert(vec![0, 1, 0, 0, 0, 0], BFieldElement::from(1u64));
        big_c.insert(vec![10, 3, 8, 0, 3, 0], -BFieldElement::from(9u64));
        big_c.insert(vec![2, 3, 4, 0, 0, 0], BFieldElement::from(12u64));
        big_c.insert(vec![5, 5, 5, 0, 8, 0], -BFieldElement::from(4u64));
        big_c.insert(vec![0, 6, 0, 0, 1, 0], BFieldElement::from(3u64));
        big_c.insert(vec![1, 4, 11, 0, 0, 0], BFieldElement::from(10u64));
        big_c.insert(vec![1, 0, 12, 0, 2, 0], BFieldElement::from(2u64));
        MPolynomial {
            coefficients: big_c,
            variable_count: 6,
        }
    }

    #[test]
    fn equality_test() {
        assert_eq!(get_big_mpol(), get_big_mpol_extra_variabel());
        assert_ne!(get_big_mpol(), get_big_mpol_extra_variabel() + get_x());
    }

    #[test]
    fn simple_add_test() {
        let x = get_x();
        let xz = get_xz();
        let x_plus_xz = get_x_plus_xz();
        assert_eq!(x_plus_xz, x.clone() + xz.clone());

        let minus_17y = get_minus_17y();
        let x_plus_xz_minus_17_y = get_x_plus_xz_minus_17y();
        assert_eq!(x_plus_xz_minus_17_y, x + xz + minus_17y);
    }

    #[test]
    fn simple_sub_test() {
        let x = get_x();
        let xz = get_xz();
        let x_minus_xz = get_x_minus_xz();
        assert_eq!(x_minus_xz, x.clone() - xz.clone());

        let big = get_big_mpol();
        assert_eq!(big, big.clone() - x.clone() + x);
        assert_eq!(big, big.clone() - xz.clone() + xz);
        assert_eq!(big, big.clone() - big.clone() + big.clone());
        assert_eq!(big, big.clone() - x_minus_xz.clone() + x_minus_xz);

        // Catch error fixed in sub where similar exponents in both terms of
        // `a(x,y) - b(x,y)` were calculated as `c_b - c_a` instead of as `c_a - c_b`,
        // as it should be.
        let _0: MPolynomial<BFieldElement> =
            MPolynomial::from_constant(BFieldElement::from(0u64), 3);
        let _2: MPolynomial<BFieldElement> =
            MPolynomial::from_constant(BFieldElement::from(2u64), 3);
        let _3: MPolynomial<BFieldElement> =
            MPolynomial::from_constant(BFieldElement::from(3u64), 3);
        let _4: MPolynomial<BFieldElement> =
            MPolynomial::from_constant(BFieldElement::from(4u64), 3);
        let _6: MPolynomial<BFieldElement> =
            MPolynomial::from_constant(BFieldElement::from(6u64), 3);
        let _8: MPolynomial<BFieldElement> =
            MPolynomial::from_constant(BFieldElement::from(8u64), 3);
        let _16 = MPolynomial::from_constant(BFieldElement::from(16u64), 3);
        assert_eq!(_0, _2.clone() - _2.clone());
        assert_eq!(_0, _4.clone() - _4.clone());
        assert_eq!(_6, _8.clone() - _2.clone());
        assert_eq!(_4, _6.clone() - _2.clone());
        assert_eq!(_2, _4.clone() - _2.clone());
        assert_eq!(_6, _4 + _2);
        assert_eq!(_16, _8.clone() + _8);
    }

    #[test]
    fn simple_mul_test() {
        let x = get_x();
        let z = get_z();
        let x_squared = get_x_squared();
        let xz = get_xz();
        assert_eq!(x_squared, x.clone() * x.clone());
        assert_eq!(xz, x * z);

        // Multiplying with one must be the identity operator
        let one: MPolynomial<BFieldElement> =
            MPolynomial::from_constant(BFieldElement::from(1u64), 3);
        assert_eq!(x_squared, x_squared.clone() * one.clone());
        assert_eq!(x_squared, one.clone() * x_squared.clone());
        assert_eq!(xz, one.clone() * xz.clone());
        assert_eq!(xz, xz.clone() * one);
    }

    #[test]
    fn is_one_test() {
        let b_two = BFieldElement::new(2);
        let b_three = BFieldElement::new(3);
        let zero = MPolynomial::<BFieldElement>::from_constant(BFieldElement::zero(), 10);
        let one = MPolynomial::<BFieldElement>::from_constant(BFieldElement::one(), 10);
        let two = MPolynomial::<BFieldElement>::from_constant(b_two, 10);
        let three = MPolynomial::<BFieldElement>::from_constant(b_three, 10);
        assert!(!zero.is_one());
        assert!(one.is_one());
        assert!(!two.is_one());
        assert!(!three.is_one());

        let mut mcoef_one_alt: MCoefficients<BFieldElement> = HashMap::new();
        mcoef_one_alt.insert(vec![0, 0, 0], BFieldElement::new(1));
        mcoef_one_alt.insert(vec![1, 0, 0], BFieldElement::new(0));
        mcoef_one_alt.insert(vec![1, 188, 3], BFieldElement::new(0));
        let one_alt = MPolynomial::<BFieldElement> {
            variable_count: 3,
            coefficients: mcoef_one_alt.clone(),
        };
        assert!(one_alt.is_one());
        assert!(!one_alt.scalar_mul(b_two).is_one()); // $P(x, y, z) = 2$ is not one

        let mut mcoef: MCoefficients<BFieldElement> = HashMap::new();
        mcoef.insert(vec![1, 0, 0], BFieldElement::new(1));
        let x = MPolynomial::<BFieldElement> {
            variable_count: 3,
            coefficients: mcoef.clone(),
        };
        assert!(!x.is_one()); // $P(x, y, z) = x$ is not one
        assert!(!x.scalar_mul(b_two).is_one()); // $P(x, y, z) = 2*x$ is not one

        // x + one is not one
        assert!(!(x + one).is_one());
    }

    #[test]
    fn simple_pow_test() {
        let x = get_x();
        let x_squared = get_x_squared();
        let x_quartic = get_x_quartic();
        assert_eq!(x_squared, x.pow(2));
        assert_eq!(x_quartic, x.pow(4));
        assert_eq!(x_quartic, x_squared.pow(2));
        assert_eq!(get_x_squared_z_squared(), get_xz().pow(2));

        assert_eq!(
            x_squared.scalar_mul(BFieldElement::from(9u64)),
            x.scalar_mul(BFieldElement::from(3u64)).pow(2)
        );
        assert_eq!(
            x_squared.scalar_mul(BFieldElement::from(16u64)),
            x.scalar_mul(BFieldElement::from(4u64)).pow(2)
        );
        assert_eq!(
            x_quartic.scalar_mul(BFieldElement::from(16u64)),
            x.scalar_mul(BFieldElement::from(2u64)).pow(4)
        );
        assert_eq!(x_quartic, x.pow(4));
        assert_eq!(x_quartic, x_squared.pow(2));
        assert_eq!(get_x_squared_z_squared(), get_xz().pow(2));
        assert_eq!(
            get_x_squared_z_squared().scalar_mul(BFieldElement::from(25u64)),
            get_xz().scalar_mul(BFieldElement::from(5u64)).pow(2)
        );
        assert_eq!(get_big_mpol() * get_big_mpol(), get_big_mpol().pow(2));
        assert_eq!(
            get_big_mpol().scalar_mul(BFieldElement::from(25u64)) * get_big_mpol(),
            get_big_mpol().scalar_mul(BFieldElement::from(5u64)).pow(2)
        );
    }

    #[test]
    fn variables_test() {
        let vars_1 = MPolynomial::variables(1);
        assert_eq!(1usize, vars_1.len());
        assert_eq!(get_x(), vars_1[0]);
        let vars_3 = MPolynomial::variables(3);
        assert_eq!(3usize, vars_3.len());
        assert_eq!(get_x(), vars_3[0]);
        assert_eq!(get_y(), vars_3[1]);
        assert_eq!(get_z(), vars_3[2]);
    }

    #[test]
    fn evaluate_symbolic_test() {
        let mut empty_intermediate_results: HashMap<Vec<u8>, Polynomial<BFieldElement>> =
            HashMap::new();
        let mut empty_mod_pow_memoization: HashMap<(usize, u8), Polynomial<BFieldElement>> =
            HashMap::new();
        #[allow(clippy::type_complexity)]
        let mut empty_mul_memoization: HashMap<
            (Polynomial<BFieldElement>, (usize, u8)),
            Polynomial<BFieldElement>,
        > = HashMap::new();

        let one = BFieldElement::from(1u64);
        let xyz_m = get_xyz();
        let x: Polynomial<BFieldElement> = Polynomial::from_constant(one).shift_coefficients(1);

        let mut precalculated_intermediate_results: HashMap<Vec<u8>, Polynomial<BFieldElement>> =
            HashMap::new();
        let precalculation_result = MPolynomial::precalculate_symbolic_exponents(
            &[xyz_m.clone()],
            &[x.clone(), x.clone(), x.clone()],
            &mut precalculated_intermediate_results,
        );
        match precalculation_result {
            Ok(_) => (),
            Err(e) => panic!("error: {e}"),
        };

        let x_cubed: Polynomial<BFieldElement> =
            Polynomial::from_constant(one).shift_coefficients(3);
        assert_eq!(
            x_cubed,
            xyz_m.evaluate_symbolic(&[x.clone(), x.clone(), x.clone()])
        );
        assert_eq!(
            x_cubed,
            xyz_m.evaluate_symbolic_with_memoization(
                &[x.clone(), x.clone(), x.clone()],
                &mut empty_mod_pow_memoization.clone(),
                &mut empty_mul_memoization.clone(),
                &mut empty_intermediate_results,
            )
        );
        assert_eq!(
            x_cubed,
            xyz_m.evaluate_symbolic_with_memoization(
                &[x.clone(), x.clone(), x.clone()],
                &mut empty_mod_pow_memoization,
                &mut empty_mul_memoization,
                &mut precalculated_intermediate_results.clone(),
            )
        );
        assert_eq!(
            x_cubed,
            xyz_m.evaluate_symbolic_with_memoization_precalculated(
                &[x.clone(), x.clone(), x],
                &mut precalculated_intermediate_results.clone(),
            )
        );
    }

    #[test]
    fn evaluate_symbolic_with_zeros_test() {
        let one = BFieldElement::from(1u64);
        let xm = get_x();
        let xu: Polynomial<BFieldElement> = Polynomial::from_constant(one).shift_coefficients(1);
        let zero_upol: Polynomial<BFieldElement> = Polynomial::zero();
        assert_eq!(
            xu,
            xm.evaluate_symbolic(&[xu.clone(), zero_upol.clone(), zero_upol.clone()])
        );

        let mut empty_intermediate_results: HashMap<Vec<u8>, Polynomial<BFieldElement>> =
            HashMap::new();
        let mut empty_mod_pow_memoization: HashMap<(usize, u8), Polynomial<BFieldElement>> =
            HashMap::new();
        #[allow(clippy::type_complexity)]
        let mut empty_mul_memoization: HashMap<
            (Polynomial<BFieldElement>, (usize, u8)),
            Polynomial<BFieldElement>,
        > = HashMap::new();
        assert_eq!(
            xu,
            xm.evaluate_symbolic_with_memoization(
                &[xu.clone(), zero_upol.clone(), zero_upol],
                &mut empty_mod_pow_memoization,
                &mut empty_mul_memoization,
                &mut empty_intermediate_results,
            )
        );
    }

    #[test]
    fn evaluate_test() {
        let x = get_x();
        assert_eq!(
            BFieldElement::from(12u64),
            x.evaluate(&[
                BFieldElement::from(12u64),
                BFieldElement::from(0u64),
                BFieldElement::from(0u64)
            ])
        );
        assert_eq!(
            BFieldElement::from(12u64),
            x.evaluate(&[
                BFieldElement::from(12u64),
                BFieldElement::from(12u64),
                BFieldElement::from(12u64)
            ])
        );

        let xszs = get_x_squared_z_squared();
        assert_eq!(
            BFieldElement::from(16u64),
            xszs.evaluate(&[
                BFieldElement::from(2u64),
                BFieldElement::from(0u64),
                BFieldElement::from(2u64)
            ])
        );
        assert_eq!(
            BFieldElement::from(16u64),
            xszs.evaluate(&[
                BFieldElement::from(2u64),
                BFieldElement::from(2u64),
                BFieldElement::from(2u64)
            ])
        );
        assert_eq!(
            BFieldElement::from(0u64),
            xszs.evaluate(&[
                BFieldElement::from(0u64),
                BFieldElement::from(3u64),
                BFieldElement::from(8u64)
            ])
        );
        assert_eq!(
            BFieldElement::from(0u64),
            xszs.evaluate(&[
                BFieldElement::from(1u64),
                BFieldElement::from(11u64),
                BFieldElement::from(0u64)
            ])
        );
    }

    #[test]
    fn lift_test() {
        let xm = get_x();
        let zm = get_z();
        let xs = Polynomial {
            coefficients: vec![BFieldElement::from(0u64), BFieldElement::from(1u64)],
        };
        assert_eq!(xm, MPolynomial::lift(xs.clone(), 0, 3));
        assert_eq!(zm, MPolynomial::lift(xs.clone(), 2, 3));

        let seven_s: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![BFieldElement::from(7u64)],
        };
        assert_eq!(
            MPolynomial::from_constant(BFieldElement::from(7u64), 3),
            MPolynomial::lift(seven_s.clone(), 0, 3)
        );
        assert_ne!(
            MPolynomial::from_constant(BFieldElement::from(8u64), 3),
            MPolynomial::lift(seven_s, 0, 3)
        );

        let x_quartic_s = Polynomial {
            coefficients: vec![
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
                BFieldElement::from(1u64),
            ],
        };
        assert_eq!(
            get_x_quartic(),
            MPolynomial::lift(x_quartic_s.clone(), 0, 3)
        );
        assert_eq!(
            get_x_quartic().scalar_mul(BFieldElement::from(5u64)),
            MPolynomial::lift(x_quartic_s.scalar_mul(BFieldElement::from(5u64)), 0, 3)
        );

        let x_squared_s = Polynomial {
            coefficients: vec![
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
                BFieldElement::from(1u64),
            ],
        };
        assert_eq!(
            get_x_quartic() + get_x_squared() + get_x(),
            MPolynomial::lift(x_quartic_s.clone() + x_squared_s.clone() + xs.clone(), 0, 3)
        );
        assert_eq!(
            get_x_quartic().scalar_mul(BFieldElement::from(5u64))
                + get_x_squared().scalar_mul(BFieldElement::from(4u64))
                + get_x().scalar_mul(BFieldElement::from(3u64)),
            MPolynomial::lift(
                x_quartic_s.scalar_mul(BFieldElement::from(5u64))
                    + x_squared_s.scalar_mul(BFieldElement::from(4u64))
                    + xs.scalar_mul(BFieldElement::from(3u64)),
                0,
                3,
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
        let xz = get_xz();
        let xz_squared = get_x_squared_z_squared();
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
        let a = gen_mpolynomial(40, 40, 80, u64::MAX);
        let b = gen_mpolynomial(20, 20, 150, u64::MAX);
        let ab = a.clone() * b.clone();
        let ba = b * a;
        assert_eq!(ab, ba);
    }

    #[test]
    fn mut_with_one_test() {
        let one = MPolynomial::<BFieldElement>::from_constant(BFieldElement::one(), 3);
        for _ in 0..10 {
            let a = gen_mpolynomial(3, 5, 10, u64::MAX);
            assert_eq!(a, a.clone() * one.clone());
            assert_eq!(a, one.clone() * a.clone());
        }
    }

    #[test]
    fn precalculate_scalar_exponents_simple_test() {
        let a: MPolynomial<BFieldElement> = MPolynomial {
            variable_count: 2,
            coefficients: HashMap::from([
                (vec![0, 1], BFieldElement::new(2)),
                (vec![0, 2], BFieldElement::new(3)),
            ]),
        };
        let b: MPolynomial<BFieldElement> = MPolynomial {
            variable_count: 2,
            coefficients: HashMap::from([
                (vec![1, 1], BFieldElement::new(2)),
                (vec![3, 4], BFieldElement::new(5)),
                (vec![0, 1], BFieldElement::new(10)),
            ]),
        };
        let exponents_list: Vec<Vec<u8>> =
            MPolynomial::extract_exponents_list(&[a.clone(), b.clone()]).unwrap();

        // The `exponents_list` returns the present exponents *plus* [0, 0] in this case
        assert_eq!(5, exponents_list.len());

        let point: Vec<BFieldElement> = vec![BFieldElement::new(2), BFieldElement::new(3)];

        // Verify mod_pow precalculation
        let mod_pow_precalculations: HashMap<(usize, u8), BFieldElement> =
            MPolynomial::<BFieldElement>::precalculate_scalar_mod_pows(4, &point);
        assert_eq!(6, mod_pow_precalculations.len()); // `6` because only powers [2,3,4] are present
        for ((i, k), v) in mod_pow_precalculations.iter() {
            assert_eq!(point[*i].mod_pow(*k as u64), *v);
        }

        let intermediate_results_res: Result<HashMap<Vec<u8>, BFieldElement>, Box<dyn Error>> =
            MPolynomial::<BFieldElement>::precalculate_scalar_exponents(
                &point,
                &mod_pow_precalculations,
                &exponents_list,
            );
        let intermediate_results = match intermediate_results_res {
            Ok(res) => res,
            Err(e) => panic!("{}", e),
        };
        assert_eq!(5, intermediate_results.len());

        // point = [2,3]
        // [0, 0] = 0
        // [0, 1] = 3
        // [0, 2] = 9
        // [1, 1] = 6
        // [3, 4] = 8 * 81 = 648
        let expected_intermediate_results: Vec<BFieldElement> = vec![
            BFieldElement::new(1),
            BFieldElement::new(3),
            BFieldElement::new(9),
            BFieldElement::new(6),
            BFieldElement::new(648),
        ];
        for (i, k) in [vec![0, 0], vec![0, 1], vec![0, 2], vec![1, 1], vec![3, 4]]
            .iter()
            .enumerate()
        {
            assert_eq!(expected_intermediate_results[i], intermediate_results[k]);
        }

        // Use the intermediate precalculated result to get the evaluation.
        // Then compare this to naive evaluation.
        assert_eq!(
            a.evaluate(&point),
            a.evaluate_with_precalculation(&point, &intermediate_results)
        );
        assert_eq!(
            BFieldElement::new(33),
            a.evaluate_with_precalculation(&point, &intermediate_results)
        );
        assert_eq!(
            b.evaluate(&point),
            b.evaluate_with_precalculation(&point, &intermediate_results)
        );
        assert_eq!(
            BFieldElement::new(3282),
            b.evaluate_with_precalculation(&point, &intermediate_results)
        );

        // Test max_exponent
        assert_eq!(2, a.max_exponent());
        assert_eq!(4, b.max_exponent());
    }

    #[test]
    fn extract_exponents_list_test() {
        // Funky property-based test to verify that the `extract_exponents_list`
        // behaves as expected
        let a = gen_mpolynomial(40, 40, 30, u64::MAX);
        let b = gen_mpolynomial(20, 20, 130, u64::MAX);
        let mut exponents: Vec<Vec<u8>> =
            MPolynomial::extract_exponents_list(&[a.clone(), b.clone()]).unwrap();
        exponents.sort();
        let exponents_a: Vec<Vec<u8>> = a.coefficients.keys().map(|x| x.to_owned()).collect();
        let exponents_b: Vec<Vec<u8>> = b.coefficients.keys().map(|x| x.to_owned()).collect();
        let mut expected_exponents_set: HashSet<Vec<u8>> = [exponents_a, exponents_b]
            .iter()
            .flat_map(|mpol| mpol.iter().map(|x| x.to_owned()))
            .collect();
        expected_exponents_set.insert(vec![0; a.variable_count]);
        let mut expected_exponents_list: Vec<Vec<u8>> =
            expected_exponents_set.into_iter().collect();
        expected_exponents_list.sort();
        assert_eq!(expected_exponents_list, exponents);
    }

    #[test]
    fn pow_test() {
        let a = gen_mpolynomial(4, 6, 2, 20);
        let mut acc = MPolynomial::from_constant(BFieldElement::one(), 4);
        for i in 0..10 {
            let mod_pow = a.pow(i);
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
            let a = gen_mpolynomial(variable_count, 12, 7, BFieldElement::MAX);
            let b = gen_mpolynomial(variable_count, 12, 7, BFieldElement::MAX);
            let c = gen_mpolynomial(variable_count, 12, 7, BFieldElement::MAX);
            let mpolynomials = vec![a, b, c];
            let mut point = gen_upolynomials(variable_count - 1, 5, BFieldElement::MAX);

            // Add an x-value to the list of polynomials to verify that I didn't mess up the optimization
            // used for x-polynomials
            point.push(Polynomial {
                coefficients: vec![BFieldElement::zero(), BFieldElement::one()],
            });

            let mut precalculated_intermediate_results: HashMap<
                Vec<u8>,
                Polynomial<BFieldElement>,
            > = HashMap::new();
            let precalculation_result = MPolynomial::precalculate_symbolic_exponents(
                &mpolynomials,
                &point,
                &mut precalculated_intermediate_results,
            );
            match precalculation_result {
                Ok(_) => (),
                Err(e) => panic!("error: {e}"),
            };

            // Verify precalculation results
            // println!("************** precalculation_result **************");
            for (k, v) in precalculated_intermediate_results.iter() {
                let mut expected_result = Polynomial::from_constant(BFieldElement::one());
                for (i, &exponent) in k.iter().enumerate() {
                    expected_result = expected_result * point[i].mod_pow(exponent.into())
                }
                // println!("k = {:?}", k);
                assert_eq!(&expected_result, v);
            }

            // Verify that function gets the same result with and without precalculated values
            let mut empty_intermediate_results: HashMap<Vec<u8>, Polynomial<BFieldElement>> =
                HashMap::new();
            let mut empty_mod_pow_memoization: HashMap<(usize, u8), Polynomial<BFieldElement>> =
                HashMap::new();
            #[allow(clippy::type_complexity)]
            let mut empty_mul_memoization: HashMap<
                (Polynomial<BFieldElement>, (usize, u8)),
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
                    &mut empty_mod_pow_memoization,
                    &mut empty_mul_memoization,
                    &mut empty_intermediate_results,
                );

                assert_eq!(with_precalculation, without_precalculation);
                assert_eq!(with_precalculation, with_precalculation_parallel);
            }
        }
    }

    fn unique_exponent_vectors(input: &MPolynomial<BFieldElement>) -> bool {
        let mut hashset: HashSet<Vec<u8>> = HashSet::new();

        input
            .coefficients
            .iter()
            .all(|(k, _v)| hashset.insert(k.clone()))
    }

    fn gen_upolynomials(
        degree: usize,
        count: usize,
        coefficient_limit: u64,
    ) -> Vec<Polynomial<BFieldElement>> {
        let mut ret: Vec<Polynomial<BFieldElement>> = vec![];
        for _ in 0..count {
            let coefficients: Vec<BFieldElement> = random_elements::<u64>(degree + 1)
                .into_iter()
                .map(|value| BFieldElement::new(value % coefficient_limit + 1))
                .collect();
            ret.push(Polynomial { coefficients });
        }

        ret
    }

    fn gen_mpolynomial(
        variable_count: usize,
        term_count: usize,
        exponent_limit_inclusive: u8,
        coefficient_limit_inclusive: u64,
    ) -> MPolynomial<BFieldElement> {
        let mut coefficients: HashMap<Vec<u8>, BFieldElement> = HashMap::new();
        let mut rng = rand::thread_rng();

        for _ in 0..term_count {
            let key = random_elements_range(variable_count, 0..=exponent_limit_inclusive);

            // Don't build multivariate polynomial with zero coefficients
            let value = BFieldElement::new(rng.gen_range(1..=coefficient_limit_inclusive));
            coefficients.insert(key, value);
        }

        MPolynomial {
            variable_count,
            coefficients,
        }
    }

    #[test]
    fn symbolic_degree_bound_zero() {
        let n = 0;
        let zero_poly: MPolynomial<BFieldElement> = MPolynomial::zero(n);
        let max_degrees = vec![];
        let degree_zero = zero_poly.symbolic_degree_bound(&max_degrees);

        assert_eq!(degree_zero, -1)
    }

    #[test]
    fn symbolic_degree_bound_simple() {
        // mpoly(x,y,z) := 3y^2z + 2z

        let mut mcoef: MCoefficients<BFieldElement> = HashMap::new();

        mcoef.insert(vec![0, 2, 1], BFieldElement::new(3));
        mcoef.insert(vec![0, 0, 1], BFieldElement::new(1));

        let mpoly = MPolynomial::<BFieldElement> {
            variable_count: 3,
            coefficients: mcoef.clone(),
        };

        let max_degrees = vec![3, 5, 7];
        let degree_poly = mpoly.symbolic_degree_bound(&max_degrees);

        #[allow(clippy::erasing_op, clippy::identity_op)]
        let expected = cmp::max(0 * 3 + 2 * 5 + 1 * 7, 0 * 3 + 0 * 5 + 1 * 7);
        assert_eq!(degree_poly, expected);

        // Verify that a zero-coefficient does not alter the result
        let new_key = vec![4, 2, 1];
        mcoef.insert(new_key.clone(), BFieldElement::new(0));
        let mpoly_alt = MPolynomial::<BFieldElement> {
            variable_count: 3,
            coefficients: mcoef.clone(),
        };
        assert_eq!(expected, mpoly_alt.symbolic_degree_bound(&max_degrees));

        // Verify that the result is different when the coefficient is non-zero
        mcoef.insert(new_key, BFieldElement::new(4562));
        let mpoly_alt_alt = MPolynomial::<BFieldElement> {
            variable_count: 3,
            coefficients: mcoef,
        };
        assert_ne!(expected, mpoly_alt_alt.symbolic_degree_bound(&max_degrees));
    }

    #[test]
    fn symbolic_degree_bound_simple2() {
        let mut mcoef: MCoefficients<BFieldElement> = HashMap::new();

        mcoef.insert(vec![11, 13, 29], BFieldElement::new(41));
        mcoef.insert(vec![19, 23, 17], BFieldElement::new(47));

        let mpoly = MPolynomial::<BFieldElement> {
            variable_count: 3,
            coefficients: mcoef,
        };

        let max_degrees = vec![3, 5, 7];
        let degree_poly = mpoly.symbolic_degree_bound(&max_degrees);

        let expected = cmp::max(11 * 3 + 13 * 5 + 29 * 7, 19 * 3 + 23 * 5 + 17 * 7);
        assert_eq!(degree_poly, expected);
    }

    #[test]
    fn symbolic_degree_bound_zeroes() {
        let mut mcoef: MCoefficients<BFieldElement> = HashMap::new();

        mcoef.insert(vec![1, 2, 58], BFieldElement::new(76));
        mcoef.insert(vec![1, 4, 5], BFieldElement::new(3));
        mcoef.insert(vec![11, 13, 29], BFieldElement::new(41));
        mcoef.insert(vec![19, 23, 17], BFieldElement::new(47));

        let mpoly = MPolynomial::<BFieldElement> {
            variable_count: 3,
            coefficients: mcoef,
        };

        let max_degrees = vec![0, 0, 0];
        let degree_poly = mpoly.symbolic_degree_bound(&max_degrees);

        let expected = 0;
        assert_eq!(degree_poly, expected);
    }

    #[test]
    fn symbolic_degree_bound_ones() {
        let mut mcoef: MCoefficients<BFieldElement> = HashMap::new();

        mcoef.insert(vec![1, 2, 58], BFieldElement::new(76));
        mcoef.insert(vec![1, 4, 5], BFieldElement::new(3));
        mcoef.insert(vec![11, 13, 29], BFieldElement::new(41));
        mcoef.insert(vec![19, 23, 17], BFieldElement::new(47));

        let mpoly = MPolynomial::<BFieldElement> {
            variable_count: 3,
            coefficients: mcoef,
        };

        let max_degrees = vec![1, 1, 1];
        let degree_poly = mpoly.symbolic_degree_bound(&max_degrees);

        let expected = 1 + 2 + 58;
        assert_eq!(degree_poly, expected);
    }

    #[test]
    fn symbolic_degree_bound_random() {
        let variable_count = 3;
        let term_count = 5;
        let exponent_limit = 7;
        let coefficient_limit = 11;

        let rnd_mvpoly = gen_mpolynomial(
            variable_count,
            term_count,
            exponent_limit,
            coefficient_limit,
        );

        let max_degrees: Vec<Degree> =
            random_elements_range(variable_count, 1..=(exponent_limit as Degree));
        let degree_poly = rnd_mvpoly.symbolic_degree_bound(&max_degrees[..]);

        assert!(
            degree_poly
                <= (variable_count as u64
                    * (exponent_limit + 1) as u64
                    * (exponent_limit + 1) as u64) as Degree,
            "The total degree is the max of the sums of the exponents in any term.",
        )
    }

    #[test]
    fn degree_bounds_big() {
        // Verify that we can calculate degree bounds larger than u8::MAX
        // We don't want to overflow on these calculations
        let mut mcoef: MCoefficients<BFieldElement> = HashMap::new();
        mcoef.insert(vec![100, 200, 255], BFieldElement::new(76));
        mcoef.insert(vec![243, 99, 255], BFieldElement::new(3));
        mcoef.insert(vec![110, 180, 254], BFieldElement::new(41));
        mcoef.insert(vec![190, 231, 255], BFieldElement::new(47));
        let mpoly = MPolynomial::<BFieldElement> {
            variable_count: 3,
            coefficients: mcoef,
        };

        let max_degrees = vec![101, 101, 101];
        assert_eq!(
            101 * 190 + 101 * 231 + 101 * 255,
            mpoly.symbolic_degree_bound(&max_degrees)
        );

        assert_eq!(676, mpoly.degree());
    }

    fn symbolic_degree_bound_prop_gen() {
        let variable_count = 4;
        let term_count = 5;
        let exponent_limit = 7;
        let coefficient_limit = 12;

        // Generate one MPoly.
        let mvpoly: MPolynomial<BFieldElement> = gen_mpolynomial(
            variable_count,
            term_count,
            exponent_limit,
            coefficient_limit,
        );

        // Generate one UPoly for each variable in MPoly.
        let uvpolys: Vec<Polynomial<BFieldElement>> = (0..variable_count)
            .map(|_| Polynomial {
                coefficients: random_elements(term_count),
            })
            .collect();

        // Track A
        let sym_eval: Degree = mvpoly.evaluate_symbolic(&uvpolys[..]).degree() as Degree;

        // Track B
        let max_degrees: Vec<Degree> = uvpolys.iter().map(|uvp| uvp.degree() as Degree).collect();

        let sym_bound: Degree = mvpoly.symbolic_degree_bound(&max_degrees[..]);

        assert_eq!(sym_eval, sym_bound)
    }

    #[test]
    fn symbolic_degree_bound_prop() {
        let runs = 100;

        for _ in 0..runs {
            symbolic_degree_bound_prop_gen();
        }
    }

    #[test]
    fn print_display_bfield_test() {
        let mut mcoef: MCoefficients<BFieldElement> = HashMap::new();

        mcoef.insert(vec![0, 0], BFieldElement::new(1));
        mcoef.insert(vec![0, 1], BFieldElement::new(5));
        mcoef.insert(vec![2, 0], BFieldElement::new(6));
        mcoef.insert(vec![3, 4], BFieldElement::new(7));

        let mpoly = MPolynomial {
            variable_count: 2,
            coefficients: mcoef,
        };

        let expected = "1 + 6*x_0^2 + 5*x_1 + 7*x_0^3*x_1^4";
        assert_eq!(expected, format!("{mpoly}"));
    }

    #[test]
    fn print_display_xfield_test() {
        let mut mcoef: MCoefficients<XFieldElement> = HashMap::new();

        mcoef.insert(vec![0, 0], XFieldElement::new_u64([5, 6, 7]));
        mcoef.insert(vec![0, 1], XFieldElement::new_u64([8, 9, 10]));
        mcoef.insert(vec![2, 0], XFieldElement::new_u64([11, 12, 13]));
        mcoef.insert(vec![3, 4], XFieldElement::new_u64([14, 15, 16]));

        let mpoly = MPolynomial {
            variable_count: 2,
            coefficients: mcoef,
        };

        let expected = [
            "(7·x² + 6·x + 5) + ",
            "(13·x² + 12·x + 11)*x_0^2 + ",
            "(10·x² + 9·x + 8)*x_1 + ",
            "(16·x² + 15·x + 14)*x_0^3*x_1^4",
        ]
        .concat();
        assert_eq!(expected, format!("{mpoly}"));
    }

    #[test]
    fn ignore_zero_coefficients_for_total_degree_computation_test() {
        let mut exp_and_coeff = HashMap::new();
        exp_and_coeff.insert(vec![9, 9, 9], XFieldElement::zero());
        exp_and_coeff.insert(vec![2, 2, 2], XFieldElement::one());
        let polynomial = MPolynomial {
            variable_count: 3,
            coefficients: exp_and_coeff,
        };
        assert_eq!(6, polynomial.degree());
    }

    #[test]
    fn mpoly_hashing_is_deterministic_and_unaffected_by_zero_coefficients_test() {
        let variable_count = 10;
        let mut rnd_mvpoly = gen_mpolynomial(variable_count, 10, 15, 540);
        for _ in 0..10 {
            let mut hasher0 = DefaultHasher::new();
            rnd_mvpoly.hash(&mut hasher0);
            let hash0 = hasher0.finish();
            let mut hasher1 = DefaultHasher::new();
            rnd_mvpoly.hash(&mut hasher1);
            let hash1 = hasher1.finish();

            assert_eq!(hash0, hash1, "hashing must be deterministic");

            // Hashing is unchanged by 0-coefficient
            rnd_mvpoly
                .coefficients
                .insert(vec![9; variable_count], BFieldElement::zero());
            let mut hasher2 = DefaultHasher::new();
            rnd_mvpoly.hash(&mut hasher2);
            let hash2 = hasher2.finish();
            assert_eq!(
                hash0, hash2,
                "hashing must be invariant under 0-coefficients"
            );

            // Digest must change if mpoly changes (non-zero coefficient term)
            rnd_mvpoly
                .coefficients
                .insert(vec![2; variable_count], BFieldElement::one());
            let mut hasher3 = DefaultHasher::new();
            rnd_mvpoly.hash(&mut hasher3);
            let hash3 = hasher3.finish();
            assert_ne!(hash0, hash3, "hash digest must change if mpoly changes");

            // Digest must change if coefficient changes
            rnd_mvpoly.scalar_mul_mut(BFieldElement::new(4));
            let mut hasher4 = DefaultHasher::new();
            rnd_mvpoly.hash(&mut hasher4);
            let hash4 = hasher4.finish();
            assert_ne!(hash3, hash4, "hashing must change if mpoly changes");
        }
    }

    #[test]
    fn mpolys_in_hash_set_test() {
        let variable_count = 20;
        let mut hash_set: HashSet<MPolynomial<BFieldElement>> = HashSet::new();
        for i in 0..10 {
            let mut a = gen_mpolynomial(variable_count, 20, 10, Pow::pow(10u32, 2u32) as u64);
            let b = gen_mpolynomial(variable_count, 20, 10, Pow::pow(10u32, 2u32) as u64);
            let mut a_was_new = hash_set.insert(a.clone());
            let mut b_was_new = hash_set.insert(b.clone());
            assert!(
                a_was_new,
                "newly generated mpoly must be new in hash set a: iteration {i}"
            );
            assert!(
                b_was_new,
                "newly generated mpoly must be new in hash set b: iteration {i}"
            );

            a_was_new = hash_set.insert(a.clone());
            b_was_new = hash_set.insert(b);
            assert!(
                !a_was_new,
                "Already inserted mpoly must already exist in hash set a: iteration {i}"
            );
            assert!(
                !b_was_new,
                "Already inserted mpoly must already exist in hash set b: iteration {i}"
            );

            a.coefficients
                .insert(vec![9; variable_count], BFieldElement::zero());
            a_was_new = hash_set.insert(a);
            assert!(
                !a_was_new,
                "Already inserted mpoly must already exist in hash set a, zero coefficient: iteration {i}"
            );
        }

        assert_eq!(20, hash_set.len());
    }
}
