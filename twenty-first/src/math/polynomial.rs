//! Univariate polynomials over [finite fields](FiniteField).

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Neg;
use std::ops::Rem;
use std::ops::Sub;

use arbitrary::Arbitrary;
use arbitrary::MaxRecursionReached;
use arbitrary::Unstructured;
use itertools::EitherOrBoth;
use itertools::Itertools;
use num_traits::ConstOne;
use num_traits::ConstZero;
use num_traits::One;
use num_traits::Zero;
use rayon::current_num_threads;
use rayon::prelude::*;

use super::traits::PrimitiveRootOfUnity;
use super::zerofier_tree::Branch;
use super::zerofier_tree::ZerofierTree;
use crate::math::ntt::intt;
use crate::math::ntt::ntt;
use crate::math::ntt::par_intt;
use crate::math::ntt::par_ntt;
use crate::math::ntt::par_scaled_zero_padded_ntt;
use crate::math::ntt::scaled_zero_padded_ntt;
use crate::math::traits::FiniteField;
use crate::math::traits::ModPowU32;
use crate::prelude::BFieldElement;
use crate::prelude::Inverse;
use crate::prelude::XFieldElement;

impl<FF: FiniteField> Zero for Polynomial<'static, FF> {
    fn zero() -> Self {
        Self::new(vec![])
    }

    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl<FF: FiniteField> One for Polynomial<'static, FF> {
    fn one() -> Self {
        Self::new(vec![FF::ONE])
    }

    fn is_one(&self) -> bool {
        self.degree() == 0 && self.coefficients[0].is_one()
    }
}

/// Data produced by the preprocessing phase of a batch modular interpolation.
/// Marked `pub` for benchmarking purposes. Not part of the public API.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct ModularInterpolationPreprocessingData<'coeffs, FF: FiniteField> {
    pub even_zerofiers: Vec<Polynomial<'coeffs, FF>>,
    pub odd_zerofiers: Vec<Polynomial<'coeffs, FF>>,
    pub shift_coefficients: Vec<FF>,
    pub tail_length: usize,
}

/// A univariate polynomial with coefficients in a [finite field](FiniteField),
/// in monomial form.
///
/// The polynomial can either own its coefficients ([Polynomial::new]) or borrow
/// them ([Polynomial::new_borrowed]). The former is usually more convenient,
/// but might require an expensive duplication of the polynomial's coefficients.
#[derive(Clone)]
pub struct Polynomial<'coeffs, FF: FiniteField> {
    /// The polynomial's coefficients, in order of increasing degree. That is,
    /// the leading coefficient is `coefficients.last()`. See
    /// [`Polynomial::normalize`] and [`Polynomial::coefficients`] for caveats
    /// of that statement.
    coefficients: Cow<'coeffs, [FF]>,
}

impl<'a, FF> Arbitrary<'a> for Polynomial<'static, FF>
where
    FF: FiniteField + Arbitrary<'a>,
{
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self::new(u.arbitrary()?))
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        Self::try_size_hint(depth).unwrap_or_default()
    }

    fn try_size_hint(
        depth: usize,
    ) -> arbitrary::Result<(usize, Option<usize>), MaxRecursionReached> {
        arbitrary::size_hint::try_recursion_guard(depth, <Cow<[FF]>>::try_size_hint)
    }
}

impl<FF: FiniteField> Debug for Polynomial<'_, FF> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Polynomial")
            .field("coefficients", &self.coefficients)
            .finish()
    }
}

// Not derived because `PartialEq` is also not derived.
impl<FF: FiniteField> Hash for Polynomial<'_, FF> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.coefficients.hash(state);
    }
}

impl<FF: FiniteField> Display for Polynomial<'_, FF> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        let degree = match self.degree() {
            -1 => return write!(f, "0"),
            d => d as usize,
        };

        for pow in (0..=degree).rev() {
            let coeff = self.coefficients[pow];
            if coeff.is_zero() {
                continue;
            }

            if pow != degree {
                write!(f, " + ")?;
            }
            if !coeff.is_one() || pow == 0 {
                write!(f, "{coeff}")?;
            }
            match pow {
                0 => (),
                1 => write!(f, "x")?,
                _ => write!(f, "x^{pow}")?,
            }
        }

        Ok(())
    }
}

// Manually implemented to correctly handle leading zeros.
impl<FF: FiniteField> PartialEq<Polynomial<'_, FF>> for Polynomial<'_, FF> {
    fn eq(&self, other: &Polynomial<'_, FF>) -> bool {
        if self.degree() != other.degree() {
            return false;
        }

        self.coefficients
            .iter()
            .zip(other.coefficients.iter())
            .all(|(x, y)| x == y)
    }
}

impl<FF: FiniteField> Eq for Polynomial<'_, FF> {}

impl<FF> Polynomial<'_, FF>
where
    FF: FiniteField,
{
    /// The degree of the polynomial, with -1 for the zero-polynomial.
    ///
    /// ```
    /// # use num_traits::Zero;
    /// # use twenty_first::prelude::*;
    /// assert_eq!(2, Polynomial::new(bfe_vec![2, 3, 4]).degree());
    /// assert_eq!(0, Polynomial::new(xfe_vec![42]).degree());
    ///
    /// // special treatment for the zero-polynomial
    /// assert_eq!(-1, Polynomial::<BFieldElement>::zero().degree());
    /// ```
    pub fn degree(&self) -> isize {
        let mut deg = self.coefficients.len() as isize - 1;
        while deg >= 0 && self.coefficients[deg as usize].is_zero() {
            deg -= 1;
        }

        deg // -1 for the zero polynomial
    }

    /// The polynomial's coefficients, in order of increasing degree. That is,
    /// the leading coefficient is the slice's last element.
    ///
    /// The leading coefficient is guaranteed to be non-zero. Consequently, the
    /// zero-polynomial is the empty slice.
    ///
    /// See also [`into_coefficients()`][Self::into_coefficients].
    pub fn coefficients(&self) -> &[FF] {
        let coefficients = self.coefficients.as_ref();

        let Some(leading_coeff_idx) = coefficients.iter().rposition(|&c| !c.is_zero()) else {
            // `coefficients` contains no elements or only zeroes
            return &[];
        };

        &coefficients[0..=leading_coeff_idx]
    }

    /// Like [`coefficients()`][Self::coefficients], but consumes `self`.
    ///
    /// Only clones the underlying coefficients if they are not already owned.
    pub fn into_coefficients(mut self) -> Vec<FF> {
        self.normalize();
        self.coefficients.into_owned()
    }

    /// Remove any leading coefficients that are 0.
    ///
    /// Notably, does _not_ make `self` monic.
    fn normalize(&mut self) {
        while self.coefficients.last().is_some_and(Zero::is_zero) {
            self.coefficients.to_mut().pop();
        }
    }

    /// The coefficient of the polynomial's term of highest power. `None` if
    /// (and only if) `self` [is zero](Self::is_zero).
    ///
    /// Furthermore, is never `Some(FF::ZERO)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use twenty_first::prelude::*;
    /// # use num_traits::Zero;
    /// let f = Polynomial::new(bfe_vec![1, 2, 3]);
    /// assert_eq!(Some(bfe!(3)), f.leading_coefficient());
    /// assert_eq!(None, Polynomial::<XFieldElement>::zero().leading_coefficient());
    /// ```
    pub fn leading_coefficient(&self) -> Option<FF> {
        match self.degree() {
            -1 => None,
            n => Some(self.coefficients[n as usize]),
        }
    }

    /// Whether `self` is equal to the single monomial `x` with coefficient 1.
    ///
    /// ```
    /// # use twenty_first::prelude::*;
    /// // the easiest way to get `x`
    /// assert!(Polynomial::<BFieldElement>::x_to_the(1).is_x());
    /// assert!(!Polynomial::<BFieldElement>::x_to_the(2).is_x());
    ///
    /// // it's also possible to get `x` more manually
    /// let coefficients_for_x = bfe_vec![0, 1];
    /// assert!(Polynomial::new_borrowed(&coefficients_for_x).is_x());
    /// ```
    pub fn is_x(&self) -> bool {
        self.degree() == 1 && self.coefficients[0].is_zero() && self.coefficients[1].is_one()
    }

    /// The [formal derivative](https://en.wikipedia.org/wiki/Formal_derivative)
    /// of this polynomial.
    ///
    /// ```
    /// # use twenty_first::prelude::*;
    /// // 1 + 2·x¹ + 3·x² + 4·x³
    /// let polynomial = Polynomial::new(bfe_vec![1, 2, 3, 4]);
    ///
    /// // 2 + 6·x¹ + 12·x²
    /// let derivative = polynomial.formal_derivative();
    ///
    /// assert_eq!(bfe_vec![2, 6, 12], derivative.into_coefficients());
    /// ```
    pub fn formal_derivative(&self) -> Polynomial<'static, FF> {
        // not `enumerate()`ing: `FiniteField` is trait-bound to `From<u64>` but
        // not `From<usize>`
        let coefficients = (0..)
            .zip(self.coefficients.iter())
            .map(|(i, &coefficient)| FF::from(i) * coefficient)
            .skip(1)
            .collect();

        Polynomial::new(coefficients)
    }

    /// Parallel version of [`evaluate`](Self::evaluate) for long polynomials.
    /// The coefficients are split into chunks, each chunk is evaluated with
    /// Horner's method, and the chunks' values are combined with powers of the
    /// indeterminate. Short polynomials are evaluated sequentially.
    pub fn par_evaluate<Ind, Eval>(&self, x: Ind) -> Eval
    where
        Ind: Clone + One + Send + Sync,
        Eval: Mul<Ind, Output = Eval> + Add<FF, Output = Eval> + Zero + Send,
    {
        // Large enough to amortize the per-chunk overhead, small enough to
        // keep all threads busy for polynomials of a few hundred thousand
        // coefficients.
        const CHUNK_LEN: usize = 1 << 11;

        if self.coefficients.len() <= CHUNK_LEN {
            return self.evaluate(x);
        }

        let x_to_the_chunk_len = generic_pow(x.clone(), CHUNK_LEN);
        let chunk_values = self
            .coefficients
            .par_chunks(CHUNK_LEN)
            .map(|chunk| Polynomial::new_borrowed(chunk).evaluate::<Ind, Eval>(x.clone()))
            .collect::<Vec<_>>();
        let mut acc = Eval::zero();
        for value in chunk_values.into_iter().rev() {
            acc = acc * x_to_the_chunk_len.clone() + value;
        }
        acc
    }

    /// Evaluate `self` in an indeterminate.
    ///
    /// The indeterminate must come from a field that is compatible with the
    /// field over which the polynomial is defined, but it does not have to be
    /// the same field.
    ///
    /// For a specialized version, with fewer type annotations needed, see
    /// [`Self::evaluate_in_same_field`].
    ///
    /// ```
    /// # use twenty_first::prelude::*;
    /// // 2 + 5·x + 12·x²
    /// let polynomial = Polynomial::<BFieldElement>::new(bfe_vec![2, 5, 12]);
    /// let at_1: BFieldElement = polynomial.evaluate(bfe![1]);
    /// assert_eq!(bfe!(19), at_1);
    ///
    /// // `at_2` is an element of the extension field even though `polynomial`
    /// // is defined over the base field and the indeterminate is an element
    /// // of the base field
    /// let at_2: XFieldElement = polynomial.evaluate(bfe![2]);
    /// assert_eq!(xfe!(60), at_2);
    /// ```
    pub fn evaluate<Ind, Eval>(&self, x: Ind) -> Eval
    where
        Ind: Clone,
        Eval: Mul<Ind, Output = Eval> + Add<FF, Output = Eval> + Zero,
    {
        let mut acc = Eval::zero();
        for &c in self.coefficients.iter().rev() {
            acc = acc * x.clone() + c;
        }

        acc
    }
    /// Evaluate `self` in an indeterminate.
    ///
    /// For a generalized version, with more type annotations needed, see
    /// [`Self::evaluate`].
    // todo: try to remove this once specialization is stabilized; see
    //  https://rust-lang.github.io/rfcs/1210-impl-specialization.html
    pub fn evaluate_in_same_field(&self, x: FF) -> FF {
        self.evaluate::<FF, FF>(x)
    }

    /// Check whether all the given points lie on the same line.
    ///
    /// A point is a tuple of (x, y)-coordinates.
    ///
    /// Returns `false` if any two points lie on a line that is parallel to the
    /// y-axis.
    ///
    /// ```
    /// # use twenty_first::prelude::*;
    /// let to_bfe_tuple = |(x, y)| (bfe!(x), bfe!(y));
    ///
    /// let on_line = [(0, 0), (1, 1), (2, 2)].map(to_bfe_tuple);
    /// assert!(Polynomial::are_colinear(&on_line));
    ///
    /// let off_line = [(0, 0), (1, 1), (2, 3)].map(to_bfe_tuple);
    /// assert!(!Polynomial::are_colinear(&off_line));
    ///```
    pub fn are_colinear(points: &[(FF, FF)]) -> bool {
        if points.len() < 3 {
            return false;
        }

        if !points.iter().map(|(x, _)| x).all_unique() {
            return false;
        }

        // Find 1st degree polynomial through first two points
        let (p0_x, p0_y) = points[0];
        let (p1_x, p1_y) = points[1];
        let a = (p0_y - p1_y) / (p0_x - p1_x);
        let b = p0_y - a * p0_x;

        points.iter().skip(2).all(|&(x, y)| a * x + b == y)
    }

    /// Given two points and a third x-coordinate, return the corresponding
    /// y-coordinate such that the third point lies on the line defined by the
    /// first two points.
    ///
    /// ### Panics
    ///
    /// Panics if any two x-coordinates are identical.
    ///
    /// ```
    /// # use twenty_first::prelude::*;
    /// let point_0 = (bfe!(0), bfe!(0));
    /// let point_1 = (bfe!(2), bfe!(4));
    /// let point_2_x = bfe!(1);
    ///
    /// let point_2_y = Polynomial::get_colinear_y(point_0, point_1, point_2_x);
    /// assert_eq!(bfe!(2), point_2_y);
    ///
    /// let point_2 = (point_2_x, point_2_y);
    /// assert!(Polynomial::are_colinear(&[point_0, point_1, point_2]));
    /// ```
    pub fn get_colinear_y(p0: (FF, FF), p1: (FF, FF), p2_x: FF) -> FF {
        assert_ne!(p0.0, p1.0, "Line must not be parallel to y-axis");
        let dy = p0.1 - p1.1;
        let dx = p0.0 - p1.0;
        let p2_y_times_dx = dy * (p2_x - p0.0) + dx * p0.1;

        // Can we implement this without division?
        p2_y_times_dx / dx
    }

    /// Slow square implementation that does not use NTT.
    ///
    /// If your trait bounds allow it, use the faster [Polynomial::square]
    /// instead.
    #[must_use]
    pub fn slow_square(&self) -> Polynomial<'static, FF> {
        if self.degree() < 0 {
            return Polynomial::zero();
        }

        let squared_coefficient_len = self.degree() as usize * 2 + 1;
        let mut squared_coefficients = vec![FF::ZERO; squared_coefficient_len];

        let two = FF::ONE + FF::ONE;
        for i in 0..self.coefficients.len() {
            let ci = self.coefficients[i];
            squared_coefficients[2 * i] += ci * ci;

            for j in i + 1..self.coefficients.len() {
                let cj = self.coefficients[j];
                squared_coefficients[i + j] += two * ci * cj;
            }
        }

        Polynomial::new(squared_coefficients)
    }

    /// Only `pub` to allow benchmarking; not considered part of the public API.
    #[doc(hidden)]
    pub fn naive_multiply<FF2>(
        &self,
        other: &Polynomial<FF2>,
    ) -> Polynomial<'static, <FF as Mul<FF2>>::Output>
    where
        FF: Mul<FF2>,
        FF2: FiniteField,
        <FF as Mul<FF2>>::Output: FiniteField,
    {
        let Ok(degree_lhs) = usize::try_from(self.degree()) else {
            return Polynomial::zero();
        };
        let Ok(degree_rhs) = usize::try_from(other.degree()) else {
            return Polynomial::zero();
        };

        let mut product = vec![<FF as Mul<FF2>>::Output::ZERO; degree_lhs + degree_rhs + 1];
        for i in 0..=degree_lhs {
            for j in 0..=degree_rhs {
                product[i + j] += self.coefficients[i] * other.coefficients[j];
            }
        }

        Polynomial::new(product)
    }

    /// Multiply `self` with itself `pow` times.
    ///
    /// Similar to [`Self::fast_pow`], but slower and slightly more general.
    #[must_use]
    pub fn pow(&self, pow: u32) -> Polynomial<'static, FF> {
        // special case: 0^0 = 1
        let Some(bit_length) = pow.checked_ilog2() else {
            return Polynomial::one();
        };

        if self.degree() < 0 {
            return Polynomial::zero();
        }

        // square-and-multiply
        let mut acc = Polynomial::one();
        for i in 0..=bit_length {
            acc = acc.slow_square();
            let bit_is_set = (pow >> (bit_length - i)) & 1 == 1;
            if bit_is_set {
                acc = acc * self.clone();
            }
        }

        acc
    }

    /// Multiply a polynomial with x^power
    #[must_use]
    pub fn shift_coefficients(self, power: usize) -> Polynomial<'static, FF> {
        let mut coefficients = self.coefficients.into_owned();
        coefficients.splice(0..0, vec![FF::ZERO; power]);
        Polynomial::new(coefficients)
    }

    /// Multiply a polynomial with a scalar, _i.e._, compute `scalar · self(x)`.
    ///
    /// Slightly faster but slightly less general than [`Self::scalar_mul`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use twenty_first::prelude::*;
    /// let mut f = Polynomial::new(bfe_vec![1, 2, 3]);
    /// f.scalar_mul_mut(bfe!(2));
    /// assert_eq!(Polynomial::new(bfe_vec![2, 4, 6]), f);
    /// ```
    pub fn scalar_mul_mut<S>(&mut self, scalar: S)
    where
        S: Clone,
        FF: MulAssign<S>,
    {
        let mut coefficients = std::mem::take(&mut self.coefficients).into_owned();
        for coefficient in &mut coefficients {
            *coefficient *= scalar.clone();
        }
        self.coefficients = Cow::Owned(coefficients);
    }

    /// Multiply a polynomial with a scalar, _i.e._, compute `scalar · self(x)`.
    ///
    /// Slightly slower but slightly more general than [`Self::scalar_mul_mut`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use twenty_first::prelude::*;
    /// let f = Polynomial::new(bfe_vec![1, 2, 3]);
    /// let g = f.scalar_mul(bfe!(2));
    /// assert_eq!(Polynomial::new(bfe_vec![2, 4, 6]), g);
    /// ```
    #[must_use]
    pub fn scalar_mul<S, FF2>(&self, scalar: S) -> Polynomial<'static, FF2>
    where
        S: Clone,
        FF: Mul<S, Output = FF2>,
        FF2: FiniteField,
    {
        let coeff_iter = self.coefficients.iter();
        let new_coeffs = coeff_iter.map(|&c| c * scalar.clone()).collect();
        Polynomial::new(new_coeffs)
    }

    /// Divide `self` by some `divisor`, returning (`quotient`, `remainder`).
    ///
    /// # Panics
    ///
    /// Panics if the `divisor` is zero.
    pub fn divide(
        &self,
        divisor: &Polynomial<'_, FF>,
    ) -> (Polynomial<'static, FF>, Polynomial<'static, FF>) {
        // There is an NTT-based division algorithm, but for no practical
        // parameter set is it faster than long division.
        self.naive_divide(divisor)
    }

    /// Return (quotient, remainder).
    ///
    /// Only `pub` to allow benchmarking; not considered part of the public API.
    #[doc(hidden)]
    pub fn naive_divide(
        &self,
        divisor: &Polynomial<'_, FF>,
    ) -> (Polynomial<'static, FF>, Polynomial<'static, FF>) {
        let divisor_lc_inv = divisor
            .leading_coefficient()
            .expect("divisor should be non-zero")
            .inverse();

        let Ok(quotient_degree) = usize::try_from(self.degree() - divisor.degree()) else {
            // self.degree() < divisor.degree()
            return (Polynomial::zero(), self.clone().into_owned());
        };
        debug_assert!(self.degree() >= 0);

        // quotient is built from back to front, must be reversed later
        let mut rev_quotient = Vec::with_capacity(quotient_degree + 1);
        let mut remainder = self.clone();
        remainder.normalize();

        // The divisor is also iterated back to front.
        // It is normalized manually to avoid it being a `&mut` argument.
        let rev_divisor = divisor.coefficients.iter().rev();
        let normal_rev_divisor = rev_divisor.skip_while(|c| c.is_zero());

        let mut remainder_coefficients = remainder.coefficients.into_owned();
        for _ in 0..=quotient_degree {
            let remainder_lc = remainder_coefficients.pop().unwrap();
            let quotient_coeff = remainder_lc * divisor_lc_inv;
            rev_quotient.push(quotient_coeff);

            if quotient_coeff.is_zero() {
                continue;
            }

            let remainder_degree = remainder_coefficients.len().saturating_sub(1);

            // skip divisor's leading coefficient: it has already been dealt with
            for (i, &divisor_coeff) in normal_rev_divisor.clone().skip(1).enumerate() {
                remainder_coefficients[remainder_degree - i] -= quotient_coeff * divisor_coeff;
            }
        }

        rev_quotient.reverse();

        let quot = Polynomial::new(rev_quotient);
        let rem = Polynomial::new(remainder_coefficients);
        (quot, rem)
    }

    /// Extended Euclidean algorithm with polynomials. Computes the greatest
    /// common divisor `gcd` as a monic polynomial, as well as the corresponding
    /// Bézout coefficients `a` and `b`, satisfying `gcd = a·x + b·y`
    ///
    /// # Example
    ///
    /// ```
    /// # use twenty_first::prelude::Polynomial;
    /// # use twenty_first::prelude::BFieldElement;
    /// let x = Polynomial::<BFieldElement>::from([1, 0, 1]);
    /// let y = Polynomial::<BFieldElement>::from([1, 1]);
    /// let (gcd, a, b) = Polynomial::xgcd(x.clone(), y.clone());
    /// assert_eq!(gcd, a * x + b * y);
    /// ```
    pub fn xgcd(
        x: Self,
        y: Polynomial<'_, FF>,
    ) -> (
        Polynomial<'static, FF>,
        Polynomial<'static, FF>,
        Polynomial<'static, FF>,
    ) {
        let mut x = x.into_owned();
        let mut y = y.into_owned();
        let (mut a_factor, mut a1) = (Polynomial::one(), Polynomial::zero());
        let (mut b_factor, mut b1) = (Polynomial::zero(), Polynomial::one());

        while !y.is_zero() {
            let (quotient, remainder) = x.naive_divide(&y);
            let c = a_factor - quotient.clone() * a1.clone();
            let d = b_factor - quotient * b1.clone();

            x = y;
            y = remainder;
            a_factor = a1;
            a1 = c;
            b_factor = b1;
            b1 = d;
        }

        // normalize result to ensure the gcd, _i.e._, `x` has leading
        // coefficient 1
        let lc = x.leading_coefficient().unwrap_or(FF::ONE);
        let normalize = |poly: Self| poly.scalar_mul(lc.inverse());

        let [x, a, b] = [x, a_factor, b_factor].map(normalize);
        (x, a, b)
    }

    pub(crate) fn reverse(&self) -> Polynomial<'static, FF> {
        let degree = self.degree();
        let new_coefficients = self
            .coefficients
            .iter()
            .take((degree + 1) as usize)
            .copied()
            .rev()
            .collect_vec();
        Polynomial::new(new_coefficients)
    }

    /// Return a polynomial that owns its coefficients. Clones the coefficients
    /// if they are not already owned.
    pub fn into_owned(self) -> Polynomial<'static, FF> {
        Polynomial::new(self.coefficients.into_owned())
    }
}

/// The largest number of independent, concurrently running tasks for which
/// each task's transforms should still be parallelized internally. Beyond
/// that, the tasks themselves provide the parallelism, and nested parallel
/// transforms only contend for the threads.
pub(crate) const MAX_CONCURRENT_PAR_NTTS: usize = 3;

/// Whether a transform of the given length, one of `num_concurrent` running
/// concurrently, should be computed in parallel.
pub(crate) fn should_par_ntt(len: usize, num_concurrent: usize) -> bool {
    len >= 1 << crate::math::ntt::par_min_log_2_len() && num_concurrent <= MAX_CONCURRENT_PAR_NTTS
}

/// The [NTT](ntt), [in parallel](par_ntt) if requested.
pub(crate) fn ntt_maybe_par<FF: FiniteField + MulAssign<BFieldElement>>(x: &mut [FF], par: bool) {
    if par { par_ntt(x) } else { ntt(x) }
}

/// A vector of the given length whose `i`-th element is `f(i)`, initialized
/// in parallel if requested. Parallel initialization also distributes the
/// page faults of large, fresh allocations across threads.
pub(crate) fn init_maybe_par<FF: FiniteField>(
    len: usize,
    par: bool,
    f: impl Fn(usize) -> FF + Sync + Send,
) -> Vec<FF> {
    let mut vec = crate::memory::vec_with_capacity(len);
    if par {
        vec.par_extend((0..len).into_par_iter().map(f));
    } else {
        vec.extend((0..len).map(f));
    }
    vec
}

/// A copy of `coefficients`, zero-padded to `len`. See [`init_maybe_par`].
pub(crate) fn zero_padded_maybe_par<FF: FiniteField>(
    coefficients: &[FF],
    len: usize,
    par: bool,
) -> Vec<FF> {
    debug_assert!(coefficients.len() <= len);
    if !par {
        let mut vec = crate::memory::vec_with_capacity(len);
        vec.extend_from_slice(coefficients);
        vec.resize(len, FF::ZERO);
        return vec;
    }
    init_maybe_par(len, true, |i| {
        coefficients.get(i).copied().unwrap_or(FF::ZERO)
    })
}

/// Element-wise `lhs[i] *= rhs[i]`, in parallel if requested.
pub(crate) fn hadamard_product_maybe_par<FF: FiniteField>(lhs: &mut [FF], rhs: &[FF], par: bool) {
    if par {
        lhs.par_iter_mut().zip(rhs).for_each(|(l, r)| *l *= *r);
    } else {
        for (l, r) in lhs.iter_mut().zip(rhs) {
            *l *= *r;
        }
    }
}

/// The [inverse NTT](intt), [in parallel](par_intt) if requested.
pub(crate) fn intt_maybe_par<FF: FiniteField + MulAssign<BFieldElement>>(x: &mut [FF], par: bool) {
    if par { par_intt(x) } else { intt(x) }
}

impl<FF> Polynomial<'_, FF>
where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    /// [Fast multiplication](Self::multiply) is slower than [naïve multiplication](Self::mul)
    /// for polynomials of degree less than this threshold.
    ///
    /// Extracted from `cargo bench --bench poly_mul` on mjolnir.
    const FAST_MULTIPLY_CUTOFF_THRESHOLD: isize = 1 << 8;

    /// [Fast interpolation](Self::fast_interpolate) is slower than
    /// [Lagrange interpolation](Self::lagrange_interpolate) below this
    /// threshold.
    ///
    /// Extracted from `cargo bench --bench interpolation` on mjolnir.
    const FAST_INTERPOLATE_CUTOFF_THRESHOLD_SEQUENTIAL: usize = 1 << 12;

    /// [Parallel Fast interpolation](Self::par_fast_interpolate) is slower than
    /// [Lagrange interpolation](Self::lagrange_interpolate) below this
    /// threshold.
    ///
    /// Extracted from `cargo bench --bench interpolation` on mjolnir.
    const FAST_INTERPOLATE_CUTOFF_THRESHOLD_PARALLEL: usize = 1 << 8;

    /// Regulates the recursion depth at which
    /// [Fast modular coset interpolation](Self::fast_modular_coset_interpolate)
    /// is slower and switches to
    /// [Lagrange interpolation](Self::lagrange_interpolate).
    const FAST_MODULAR_COSET_INTERPOLATE_CUTOFF_THRESHOLD_PREFER_LAGRANGE: usize = 1 << 8;

    /// Regulates the recursion depth at which
    /// [Fast modular coset interpolation](Self::fast_modular_coset_interpolate)
    /// is slower and switches to [INTT][intt]-then-[reduce](Self::reduce).
    const FAST_MODULAR_COSET_INTERPOLATE_CUTOFF_THRESHOLD_PREFER_INTT: usize = 1 << 17;

    /// Regulates when to prefer the [Fast coset extrapolation](Self::fast_coset_extrapolate)
    /// over the [naïve method](Self::naive_coset_extrapolate). Threshold found
    /// using `cargo criterion --bench extrapolation`.
    const FAST_COSET_EXTRAPOLATE_THRESHOLD: usize = 100;

    /// Inside `formal_power_series_inverse`, when to multiply naively and when
    /// to use NTT-based multiplication. Use benchmark
    /// `formal_power_series_inverse` to find the optimum. Based on benchmarks,
    /// the optimum probably lies somewhere between 2^5 and 2^9.
    const FORMAL_POWER_SERIES_INVERSE_CUTOFF: isize = 1 << 8;

    /// Modular reduction is made fast by first finding a multiple of the
    /// denominator that allows for chunk-wise reduction, and then finishing off
    /// by reducing by the plain denominator using plain long division. The
    /// "fast"ness comes from using NTT-based multiplication in the chunk-wise
    /// reduction step. This const regulates the chunk size and thus the domain
    /// size of the NTT.
    const FAST_REDUCE_CUTOFF_THRESHOLD: usize = 1 << 8;

    /// Below this product of a modulus' degree and the precision of a
    /// reduction, i.e., below roughly this many base field multiplications,
    /// quadratic algorithms (long division, coefficient-wise power series
    /// inversion) beat their quasi-linear counterparts, which have larger
    /// constants.
    const QUADRATIC_REDUCTION_CUTOFF: usize = 1 << 14;

    /// When doing batch evaluation, sometimes it makes sense to reduce the
    /// polynomial modulo the zerofier of the domain first. This const regulates
    /// when.
    const REDUCE_BEFORE_EVALUATE_THRESHOLD_RATIO: isize = 4;

    /// Return the polynomial which corresponds to the transformation `x → α·x`.
    ///
    /// Given a polynomial P(x), produce P'(x) := P(α·x). Evaluating P'(x) then
    /// corresponds to evaluating P(α·x).
    #[must_use]
    pub fn scale<S, XF>(&self, alpha: S) -> Polynomial<'static, XF>
    where
        S: Clone + One,
        FF: Mul<S, Output = XF>,
        XF: FiniteField,
    {
        Polynomial::new(self.scaled_coefficients(alpha, self.coefficients.len()))
    }

    /// The coefficients of [`scale`](Self::scale), in a vector with (at least)
    /// the given capacity. See also [`par_scaled_coefficients`][par].
    ///
    /// [par]: Self::par_scaled_coefficients
    fn scaled_coefficients<S, XF>(&self, alpha: S, capacity: usize) -> Vec<XF>
    where
        S: Clone + One,
        FF: Mul<S, Output = XF>,
        XF: FiniteField,
    {
        let mut power_of_alpha = S::one();
        let mut return_coefficients =
            crate::memory::vec_with_capacity(capacity.max(self.coefficients.len()));
        for &coefficient in self.coefficients.iter() {
            return_coefficients.push(coefficient * power_of_alpha.clone());
            power_of_alpha = power_of_alpha * alpha.clone();
        }
        return_coefficients
    }

    /// Parallel version of [`scale`](Self::scale).
    #[must_use]
    pub fn par_scale<S, XF>(&self, alpha: S) -> Polynomial<'static, XF>
    where
        S: Clone + One + Send + Sync,
        FF: Mul<S, Output = XF>,
        XF: FiniteField,
    {
        Polynomial::new(self.par_scaled_coefficients(alpha, self.coefficients.len()))
    }

    /// The coefficients of [`par_scale`](Self::par_scale), in a vector with
    /// (at least) the given capacity. Allocating the final capacity up front
    /// lets callers extend the vector, e.g., by padding it to the length of an
    /// NTT domain, without reallocating.
    fn par_scaled_coefficients<S, XF>(&self, alpha: S, capacity: usize) -> Vec<XF>
    where
        S: Clone + One + Send + Sync,
        FF: Mul<S, Output = XF>,
        XF: FiniteField,
    {
        // Large enough to amortize computing the chunk's first power of α
        // by square-and-multiply, small enough to keep all threads busy.
        const CHUNK_SIZE: usize = 1 << 12;

        // Writing into pre-allocated chunks is considerably faster than
        // collecting an unindexed parallel iterator. Not initializing the
        // memory up front saves a full pass over it; the chunks are written
        // to in parallel, which also spreads the page faults across threads.
        let num_coefficients = self.coefficients.len();
        let mut return_coefficients =
            crate::memory::vec_with_capacity(capacity.max(num_coefficients));
        return_coefficients
            .spare_capacity_mut()
            .par_chunks_mut(CHUNK_SIZE)
            .zip(self.coefficients.par_chunks(CHUNK_SIZE))
            .enumerate()
            .for_each(|(chunk_index, (scaled_chunk, chunk))| {
                let mut power_of_alpha = generic_pow(alpha.clone(), chunk_index * CHUNK_SIZE);
                for (scaled_coefficient, &coefficient) in scaled_chunk.iter_mut().zip(chunk) {
                    scaled_coefficient.write(coefficient * power_of_alpha.clone());
                    power_of_alpha = power_of_alpha * alpha.clone();
                }
            });
        // SAFETY:
        // 1. The capacity is at least `num_coefficients`.
        // 2. The chunks of the spare capacity and of the coefficients are
        //    zipped in lockstep and have identical lengths, so exactly the
        //    first `num_coefficients` elements were written to, and every
        //    one of them was.
        unsafe { return_coefficients.set_len(num_coefficients) };
        return_coefficients
    }

    /// Square `self`.
    ///
    /// It is the caller's responsibility that this function is called with
    /// sufficiently large input to be faster than [`Polynomial::square`].
    #[must_use]
    pub fn fast_square(&self) -> Polynomial<'static, FF> {
        let result_degree = match self.degree() {
            -1 => return Polynomial::zero(),
            0 => return Polynomial::from_constant(self.coefficients[0] * self.coefficients[0]),
            d => 2 * d as u64 + 1,
        };
        let ntt_size = result_degree.next_power_of_two();

        let mut coefficients = self.coefficients.to_vec();
        coefficients.resize(ntt_size as usize, FF::ZERO);
        ntt(&mut coefficients);
        for element in &mut coefficients {
            *element = *element * *element;
        }
        intt(&mut coefficients);
        coefficients.truncate(result_degree as usize);

        Polynomial::new(coefficients)
    }

    /// Square `self`.
    ///
    /// This is the recommended method for polynomial squaring.
    ///
    /// ```
    /// # use twenty_first::prelude::*;
    /// let polynomial = Polynomial::new(bfe_vec![2, 3]);
    /// let square = Polynomial::new(bfe_vec![4, 12, 9]);
    /// assert_eq!(square, polynomial.square());
    /// ```
    #[must_use]
    pub fn square(&self) -> Polynomial<'static, FF> {
        if self.degree() == -1 {
            return Polynomial::zero();
        }

        // A benchmark run on sword_smith's PC revealed that `fast_square` was
        // faster when the input size exceeds a length of 64.
        let squared_coefficient_len = self.degree() as usize * 2 + 1;
        if squared_coefficient_len > 64 {
            return self.fast_square();
        }

        let zero = FF::ZERO;
        let one = FF::ONE;
        let two = one + one;
        let mut squared_coefficients = vec![zero; squared_coefficient_len];

        for i in 0..self.coefficients.len() {
            let ci = self.coefficients[i];
            squared_coefficients[2 * i] += ci * ci;

            for j in i + 1..self.coefficients.len() {
                let cj = self.coefficients[j];
                squared_coefficients[i + j] += two * ci * cj;
            }
        }

        Polynomial::new(squared_coefficients)
    }

    /// Multiply `self` with itself `pow` times.
    ///
    /// Similar to [`Self::pow`], but faster and slightly less general.
    #[must_use]
    pub fn fast_pow(&self, pow: u32) -> Polynomial<'static, FF> {
        // special case: 0^0 = 1
        let Some(bit_length) = pow.checked_ilog2() else {
            return Polynomial::one();
        };

        if self.degree() < 0 {
            return Polynomial::zero();
        }

        // square-and-multiply
        let mut acc = Polynomial::one();
        for i in 0..=bit_length {
            acc = acc.square();
            let bit_is_set = (pow >> (bit_length - i)) & 1 == 1;
            if bit_is_set {
                acc = self.multiply(&acc);
            }
        }

        acc
    }

    /// Multiply `self` by `other`.
    ///
    /// Prefer this over [`self * other`](Self::mul) since it chooses the
    /// fastest multiplication strategy.
    #[must_use]
    pub fn multiply<FF2>(
        &self,
        other: &Polynomial<'_, FF2>,
    ) -> Polynomial<'static, <FF as Mul<FF2>>::Output>
    where
        FF: Mul<FF2>,
        FF2: FiniteField + MulAssign<BFieldElement>,
        <FF as Mul<FF2>>::Output: FiniteField + MulAssign<BFieldElement>,
    {
        if self.degree() + other.degree() < Self::FAST_MULTIPLY_CUTOFF_THRESHOLD {
            self.naive_multiply(other)
        } else {
            self.fast_multiply(other)
        }
    }

    /// Use [Self::multiply] instead. Only `pub` to allow benchmarking; not
    /// considered part of the public API.
    ///
    /// This method is asymptotically faster than [naive
    /// multiplication](Self::naive_multiply). For
    /// small instances, _i.e._, polynomials of low degree, it is slower.
    ///
    /// The time complexity of this method is in O(n·log(n)), where `n` is the
    /// sum of the degrees of the operands. The time complexity of the naive
    /// multiplication is in O(n^2).
    #[doc(hidden)]
    pub fn fast_multiply<FF2>(
        &self,
        other: &Polynomial<FF2>,
    ) -> Polynomial<'static, <FF as Mul<FF2>>::Output>
    where
        FF: Mul<FF2>,
        FF2: FiniteField + MulAssign<BFieldElement>,
        <FF as Mul<FF2>>::Output: FiniteField + MulAssign<BFieldElement>,
    {
        let Ok(degree) = usize::try_from(self.degree() + other.degree()) else {
            return Polynomial::zero();
        };
        let order = (degree + 1).next_power_of_two();

        let mut lhs_coefficients = self.coefficients.to_vec();
        let mut rhs_coefficients = other.coefficients.to_vec();

        lhs_coefficients.resize(order, FF::ZERO);
        rhs_coefficients.resize(order, FF2::ZERO);

        ntt(&mut lhs_coefficients);
        ntt(&mut rhs_coefficients);

        let mut hadamard_product = lhs_coefficients
            .into_iter()
            .zip(rhs_coefficients)
            .map(|(l, r)| l * r)
            .collect_vec();

        intt(&mut hadamard_product);
        hadamard_product.truncate(degree + 1);
        Polynomial::new(hadamard_product)
    }

    /// Parallel version of [`fast_multiply`](Self::fast_multiply).
    pub fn par_fast_multiply<FF2>(
        &self,
        other: &Polynomial<FF2>,
    ) -> Polynomial<'static, <FF as Mul<FF2>>::Output>
    where
        FF: Mul<FF2>,
        FF2: FiniteField + MulAssign<BFieldElement>,
        <FF as Mul<FF2>>::Output: FiniteField + MulAssign<BFieldElement>,
    {
        let Ok(degree) = usize::try_from(self.degree() + other.degree()) else {
            return Polynomial::zero();
        };
        let order = (degree + 1).next_power_of_two();

        let par = should_par_ntt(order, 1);
        let mut lhs_coefficients = zero_padded_maybe_par(&self.coefficients, order, par);
        let mut rhs_coefficients = zero_padded_maybe_par(&other.coefficients, order, par);

        rayon::join(
            || par_ntt(&mut lhs_coefficients),
            || par_ntt(&mut rhs_coefficients),
        );

        let mut hadamard_product = lhs_coefficients
            .into_par_iter()
            .zip(rhs_coefficients)
            .map(|(l, r)| l * r)
            .collect::<Vec<_>>();

        par_intt(&mut hadamard_product);
        hadamard_product.truncate(degree + 1);
        Polynomial::new(hadamard_product)
    }

    /// [`multiply`](Self::multiply), using the [parallel][par] NTT for large
    /// operands.
    ///
    /// [par]: Self::par_fast_multiply
    pub(crate) fn multiply_maybe_par(&self, other: &Polynomial<FF>) -> Polynomial<'static, FF> {
        const PAR_MULTIPLY_CUTOFF_THRESHOLD: isize = 1 << 15;

        if self.degree() + other.degree() < PAR_MULTIPLY_CUTOFF_THRESHOLD {
            self.multiply(other)
        } else {
            self.par_fast_multiply(other)
        }
    }

    /// The product of two monic polynomials, using a cyclic convolution of
    /// the smallest power-of-two length that is at least the product's
    /// degree. If that length equals the degree, the leading coefficient
    /// wraps around onto the constant term, where it is known and can be
    /// undone. Compared to [`multiply`](Self::multiply), this halves the
    /// transform lengths whenever the product's degree is a power of two.
    ///
    /// The transforms are parallelized if `par` is set and the product is
    /// long enough; see [`should_par_ntt`].
    ///
    /// # Panics
    ///
    /// Panics if either polynomial is not monic.
    pub(crate) fn multiply_monic(
        &self,
        other: &Polynomial<FF>,
        par: bool,
    ) -> Polynomial<'static, FF> {
        let self_degree = usize::try_from(self.degree()).expect("monic polynomial is non-zero");
        let other_degree = usize::try_from(other.degree()).expect("monic polynomial is non-zero");
        assert_eq!(FF::ONE, self.coefficients[self_degree], "must be monic");
        assert_eq!(FF::ONE, other.coefficients[other_degree], "must be monic");
        if self_degree == 0 {
            return other.clone().into_owned();
        }
        if other_degree == 0 {
            return self.clone().into_owned();
        }

        let degree = self_degree + other_degree;
        if (degree as isize) < Self::FAST_MULTIPLY_CUTOFF_THRESHOLD {
            return self.naive_multiply(other);
        }

        let len = degree.next_power_of_two();
        let par = par && should_par_ntt(len, 1);
        let mut lhs = zero_padded_maybe_par(&self.coefficients[..=self_degree], len, par);
        let mut rhs = zero_padded_maybe_par(&other.coefficients[..=other_degree], len, par);
        if par {
            rayon::join(|| par_ntt(&mut lhs), || par_ntt(&mut rhs));
        } else {
            ntt(&mut lhs);
            ntt(&mut rhs);
        }
        hadamard_product_maybe_par(&mut lhs, &rhs, par);
        intt_maybe_par(&mut lhs, par);

        let mut product = lhs;
        if len == degree {
            product[0] -= FF::ONE;
            product.push(FF::ONE);
        } else {
            product.truncate(degree + 1);
        }
        Polynomial::new(product)
    }

    /// Given a polynomial f(X), find the polynomial g(X) of degree at most n
    /// such that f(X) * g(X) = 1 mod X^{n+1} where n is the precision.
    /// # Panics
    ///
    /// Panics if f(X) does not have an inverse in the formal power series
    /// ring, _i.e._ if its constant coefficient is zero.
    fn formal_power_series_inverse_minimal(&self, precision: usize) -> Polynomial<'static, FF> {
        let lc_inv = self.coefficients.first().unwrap().inverse();
        let mut g = vec![lc_inv];

        // invariant: product[i] = 0
        for _ in 1..(precision + 1) {
            let inner_product = self
                .coefficients
                .iter()
                .skip(1)
                .take(g.len())
                .zip(g.iter().rev())
                .map(|(l, r)| *l * *r)
                .fold(FF::ZERO, |l, r| l + r);
            g.push(-inner_product * lc_inv);
        }

        Polynomial::new(g)
    }

    /// `self mod x^n`, as an owned polynomial. See also
    /// [`mod_x_to_the_n`](Polynomial::mod_x_to_the_n).
    fn truncated(&self, n: usize) -> Polynomial<'static, FF> {
        let num_coefficients_to_retain = n.min(self.coefficients.len());
        Polynomial::new(self.coefficients[..num_coefficients_to_retain].to_vec())
    }

    /// The inverse of `self` as a formal power series, modulo `x^precision`.
    /// That is, the returned polynomial `g` of degree less than `precision`
    /// satisfies `self · g ≡ 1 (mod x^precision)`.
    ///
    /// Uses Newton iteration, doubling the precision in every step, at a
    /// total cost of a few multiplications of polynomials of degree
    /// `precision`.
    ///
    /// # Panics
    ///
    /// Panics if `self`'s constant term is zero, or if `precision` is zero.
    pub(crate) fn power_series_inverse(&self, precision: usize) -> Polynomial<'static, FF> {
        // Below this precision, the quadratic algorithm is used to bootstrap
        // the Newton iteration.
        const NEWTON_CUTOFF_PRECISION: usize = 1 << 7;

        assert!(precision > 0, "precision must be positive");
        let constant_term = self.coefficients.first().copied().unwrap_or(FF::ZERO);
        assert!(!constant_term.is_zero(), "constant term must be invertible");

        let bootstrap_precision = precision.min(NEWTON_CUTOFF_PRECISION);
        let mut inverse = crate::memory::vec_with_capacity(precision.next_power_of_two());
        inverse.extend(
            self.formal_power_series_inverse_minimal(bootstrap_precision - 1)
                .into_coefficients(),
        );
        inverse.resize(bootstrap_precision, FF::ZERO);

        // Newton iteration g ← g + g·(1 - f·g), doubling the precision p in
        // every step. Since f·g ≡ 1 (mod x^p), the term (1 - f·g) mod x^(2p)
        // is x^p times the negated coefficients p through 2p-1 of f·g, and the
        // update is x^p · (g · h mod x^p) with h those coefficients. Both
        // products are computed with cyclic convolutions of length 2p: the
        // first because only its upper half is needed and the wrapped
        // coefficients land in the lower half, the second because it does not
        // wrap at all. The transform of g is shared between them.
        let mut current_precision = bootstrap_precision;
        while current_precision < precision {
            let len = 2 * current_precision;
            let par = should_par_ntt(len, 1);

            let mut inverse_ntt = zero_padded_maybe_par(&inverse, len, par);
            let num_own_coefficients = len.min(self.coefficients.len());
            let mut self_times_inverse =
                zero_padded_maybe_par(&self.coefficients[..num_own_coefficients], len, par);
            rayon::join(
                || ntt_maybe_par(&mut inverse_ntt, par),
                || ntt_maybe_par(&mut self_times_inverse, par),
            );
            hadamard_product_maybe_par(&mut self_times_inverse, &inverse_ntt, par);
            intt_maybe_par(&mut self_times_inverse, par);

            // the update's coefficients, negated and shifted down by p
            let mut update = init_maybe_par(len, par, |i| {
                if i < current_precision {
                    -self_times_inverse[current_precision + i]
                } else {
                    FF::ZERO
                }
            });
            drop(self_times_inverse);
            ntt_maybe_par(&mut update, par);
            hadamard_product_maybe_par(&mut update, &inverse_ntt, par);
            intt_maybe_par(&mut update, par);

            update.truncate(current_precision);
            if par {
                inverse.par_extend(update);
            } else {
                inverse.extend(update);
            }
            current_precision = len;
        }
        inverse.truncate(precision);

        Polynomial::new(inverse)
    }

    /// `self mod modulus`, given the inverse of the reversed modulus as a
    /// formal power series to a precision of at least
    /// `self.degree() - modulus.degree() + 1`. See
    /// [`power_series_inverse`](Self::power_series_inverse).
    ///
    /// This is fast division: the quotient's reversal is the product of the
    /// reversed dividend and the reversed divisor's inverse, and the remainder
    /// follows from the quotient.
    pub(crate) fn reduce_with_reversed_inverse(
        &self,
        modulus: &Polynomial<FF>,
        reversed_modulus_inverse: &Polynomial<FF>,
    ) -> Polynomial<'static, FF> {
        let modulus_degree = usize::try_from(modulus.degree()).expect("modulus must not be zero");
        let Ok(self_degree) = usize::try_from(self.degree()) else {
            return Polynomial::zero();
        };
        if self_degree < modulus_degree {
            return self.clone().into_owned();
        }
        let quotient_degree = self_degree - modulus_degree;

        let reversed_self = self.reverse().truncated(quotient_degree + 1);
        let reversed_inverse = reversed_modulus_inverse.truncated(quotient_degree + 1);
        let reversed_quotient = reversed_self
            .multiply_maybe_par(&reversed_inverse)
            .truncated(quotient_degree + 1);
        let quotient_coefficients = (0..=quotient_degree)
            .map(|i| {
                reversed_quotient
                    .coefficients
                    .get(quotient_degree - i)
                    .copied()
                    .unwrap_or(FF::ZERO)
            })
            .collect_vec();
        let quotient = Polynomial::new(quotient_coefficients);

        // The remainder has degree less than the modulus; the higher
        // coefficients of the difference are zero by construction.
        let product = quotient.multiply_maybe_par(modulus);
        let remainder_coefficients = self
            .coefficients
            .iter()
            .take(modulus_degree)
            .zip_longest(product.coefficients.iter().take(modulus_degree))
            .map(|pair| match pair {
                EitherOrBoth::Both(&s, &p) => s - p,
                EitherOrBoth::Left(&s) => s,
                EitherOrBoth::Right(&p) => -p,
            })
            .collect_vec();
        Polynomial::new(remainder_coefficients)
    }

    /// Multiply a bunch of polynomials together.
    pub fn batch_multiply(factors: &[Self]) -> Polynomial<'static, FF> {
        // Build a tree-like structure of multiplications to keep the degrees of
        // the factors roughly equal throughout the process. This makes
        // efficient use of the `.multiply()` dispatcher.
        // In contrast, using a simple `.reduce()`, the accumulator polynomial
        // would have a much higher degree than the individual factors.
        // todo: benchmark the current approach against the “reduce” approach.

        if factors.is_empty() {
            return Polynomial::one();
        }
        let mut products = factors.to_vec();
        while products.len() != 1 {
            products = products
                .chunks(2)
                .map(|chunk| match chunk.len() {
                    2 => chunk[0].multiply(&chunk[1]),
                    1 => chunk[0].clone(),
                    _ => unreachable!(),
                })
                .collect();
        }

        // If any multiplications happened, `into_owned()` will not clone
        // anything. If no multiplications happened,
        //   a) what is the caller doing?
        //   b) a `'static` lifetime needs to be guaranteed, requiring
        //      `into_owned()`.
        let product_coeffs = products.pop().unwrap().coefficients.into_owned();
        Polynomial::new(product_coeffs)
    }

    /// Parallel version of [`batch_multiply`](Self::batch_multiply).
    pub fn par_batch_multiply(factors: &[Self]) -> Polynomial<'static, FF> {
        if factors.is_empty() {
            return Polynomial::one();
        }
        let num_threads = current_num_threads().max(1);
        let mut products = factors.to_vec();
        while products.len() != 1 {
            let chunk_size = usize::max(2, products.len() / num_threads);
            products = products
                .par_chunks(chunk_size)
                .map(Self::batch_multiply)
                .collect();
        }

        let product_coeffs = products.pop().unwrap().coefficients.into_owned();
        Polynomial::new(product_coeffs)
    }

    /// Divide (with remainder) and throw away the quotient. Note that the self
    /// object is the numerator and the argument is the denominator (or
    /// modulus).
    pub fn reduce(&self, modulus: &Polynomial<'_, FF>) -> Polynomial<'static, FF> {
        const FAST_REDUCE_MAKES_SENSE_MULTIPLE: isize = 4;
        if modulus.degree() < 0 {
            panic!("Cannot divide by zero; needed for reduce.");
        } else if modulus.degree() == 0 {
            Polynomial::zero()
        } else if self.degree() < modulus.degree() {
            self.clone().into_owned()
        } else if self.degree() > FAST_REDUCE_MAKES_SENSE_MULTIPLE * modulus.degree() {
            self.fast_reduce(modulus)
        } else {
            self.reduce_long_division(modulus)
        }
    }

    /// Compute the remainder after division of one polynomial by another. This
    /// method first reduces the numerator by a multiple of the denominator that
    /// was constructed to enable NTT-based chunk-wise reduction, before
    /// invoking the standard long division based algorithm to finalize. As a
    /// result, it works best for large numerators being reduced by small
    /// denominators.
    pub fn fast_reduce(&self, modulus: &Self) -> Polynomial<'static, FF> {
        if modulus.degree() == 0 {
            return Polynomial::zero();
        }
        if self.degree() < modulus.degree() {
            return self.clone().into_owned();
        }

        // 1. Chunk-wise reduction in NTT domain.
        // We generate a structured multiple of the modulus of the form
        // 1, (many zeros), *, *, *, *, *; where
        //                  -------------
        //                        |- m coefficients
        //    ---------------------------
        //               |- n=2^k coefficients.
        // This allows us to reduce the numerator's coefficients in chunks of
        // n-m using NTT-based multiplication over a domain of size n = 2^k.

        let (shift_factor_ntt, tail_size) = modulus.shift_factor_ntt_with_tail_length();
        let intermediate_remainder =
            self.reduce_by_ntt_friendly_modulus(&shift_factor_ntt, tail_size);

        // 2. Reduction of the intermediate remainder, which has degree at
        // most a small multiple of the modulus' degree. Long division costs
        // about (deg r - deg m) · deg m multiplications and has no overhead,
        // which beats fast division for small moduli.
        let modulus_degree = usize::try_from(modulus.degree()).expect("modulus is non-zero");
        let Ok(remainder_degree) = usize::try_from(intermediate_remainder.degree()) else {
            return Polynomial::zero();
        };
        if remainder_degree < modulus_degree {
            return intermediate_remainder;
        }
        let precision = remainder_degree - modulus_degree + 1;
        if modulus_degree * precision <= Self::QUADRATIC_REDUCTION_CUTOFF {
            return intermediate_remainder.reduce_long_division(modulus);
        }
        let reversed_modulus_inverse = modulus.reverse().power_series_inverse(precision);
        intermediate_remainder.reduce_with_reversed_inverse(modulus, &reversed_modulus_inverse)
    }

    /// Only marked `pub` for benchmarking purposes. Not considered part of the
    /// public API.
    #[doc(hidden)]
    pub fn shift_factor_ntt_with_tail_length(&self) -> (Vec<FF>, usize)
    where
        FF: 'static,
    {
        let n = usize::max(
            Self::FAST_REDUCE_CUTOFF_THRESHOLD,
            self.degree() as usize * 2,
        )
        .next_power_of_two();
        let ntt_friendly_multiple = self.structured_multiple_of_degree(n);

        // m = 1 + degree(ntt_friendly_multiple - leading term)
        let m = 1 + ntt_friendly_multiple
            .coefficients
            .iter()
            .enumerate()
            .rev()
            .skip(1)
            .find_map(|(i, c)| if !c.is_zero() { Some(i) } else { None })
            .unwrap_or(0);
        let mut shift_factor_ntt = ntt_friendly_multiple.coefficients[..n].to_vec();
        ntt(&mut shift_factor_ntt);
        (shift_factor_ntt, m)
    }

    /// Reduces f(X) by a structured modulus, which is of the form
    /// X^{m+n} + (something of degree less than m). When the modulus has this
    /// form, polynomial modular reductions can be computed faster than in the
    /// generic case.
    ///
    /// This method uses NTT-based multiplication, meaning that the unstructured
    /// part of the structured multiple must be given in NTT-domain.
    ///
    /// This function is marked `pub` for benchmarking. Not considered part of
    /// the public API
    #[doc(hidden)]
    pub fn reduce_by_ntt_friendly_modulus(
        &self,
        shift_ntt: &[FF],
        tail_length: usize,
    ) -> Polynomial<'static, FF> {
        let domain_length = shift_ntt.len();
        assert!(domain_length.is_power_of_two());
        let chunk_size = domain_length - tail_length;

        if self.coefficients.len() < chunk_size + tail_length {
            return self.clone().into_owned();
        }
        let num_reducible_chunks =
            (self.coefficients.len() - (tail_length + chunk_size)).div_ceil(chunk_size);

        let range_start = num_reducible_chunks * chunk_size;
        let mut working_window = if range_start >= self.coefficients.len() {
            vec![FF::ZERO; chunk_size + tail_length]
        } else {
            self.coefficients[range_start..].to_vec()
        };
        working_window.resize(chunk_size + tail_length, FF::ZERO);

        for chunk_index in (0..num_reducible_chunks).rev() {
            let mut product = [
                working_window[tail_length..].to_vec(),
                vec![FF::ZERO; tail_length],
            ]
            .concat();
            ntt(&mut product);
            product
                .iter_mut()
                .zip(shift_ntt.iter())
                .for_each(|(l, r)| *l *= *r);
            intt(&mut product);

            working_window = [
                vec![FF::ZERO; chunk_size],
                working_window[0..tail_length].to_vec(),
            ]
            .concat();
            for (i, wwi) in working_window.iter_mut().enumerate().take(chunk_size) {
                *wwi = self.coefficients[chunk_index * chunk_size + i];
            }

            for (i, wwi) in working_window
                .iter_mut()
                .enumerate()
                .take(chunk_size + tail_length)
            {
                *wwi -= product[i];
            }
        }

        Polynomial::new(working_window)
    }

    /// Given a polynomial f(X) and an integer n, find a multiple of f(X) of the
    /// form X^n + (something of much smaller degree).
    ///
    /// # Panics
    ///
    /// Panics if the polynomial is zero, or if its degree is larger than n
    pub fn structured_multiple_of_degree(&self, n: usize) -> Polynomial<'static, FF> {
        let Ok(degree) = usize::try_from(self.degree()) else {
            panic!("cannot compute multiples of zero");
        };
        assert!(degree <= n, "cannot compute multiple of smaller degree.");
        if degree == 0 {
            return Polynomial::new(
                [vec![FF::ZERO; n], vec![self.coefficients[0].inverse()]].concat(),
            );
        }

        let reverse = self.reverse();

        // The next function gives back a polynomial g(X) of degree at most arg,
        // such that f(X) * g(X) = 1 mod X^arg.
        // Without modular reduction, the degree of the product f(X) * g(X) is
        // deg(f) + arg -- even after coefficient reversal. So n = deg(f) + arg
        // and arg = n - deg(f).
        // For a modulus of small degree, the quadratic algorithm is cheaper
        // than Newton iteration; see also `fast_reduce`.
        let precision = (n - degree).max(1);
        let inverse_reverse = if degree * precision <= Self::QUADRATIC_REDUCTION_CUTOFF {
            reverse.formal_power_series_inverse_minimal(precision)
        } else {
            reverse.power_series_inverse(precision)
        };
        let product_reverse = reverse.multiply(&inverse_reverse);
        let product = product_reverse.reverse();

        // Coefficient reversal drops trailing zero. Correct for that.
        let product_degree = product.degree() as usize;
        product.shift_coefficients(n - product_degree)
    }

    fn reduce_long_division(&self, modulus: &Polynomial<'_, FF>) -> Polynomial<'static, FF> {
        let (_quotient, remainder) = self.divide(modulus);
        remainder
    }

    /// Compute a polynomial g(X) from a given polynomial f(X) such that
    /// g(X) * f(X) = 1 mod X^n , where n is the precision.
    ///
    /// In formal terms, g(X) is the approximate multiplicative inverse in
    /// the formal power series ring, where elements obey the same
    /// algebraic rules as polynomials do but can have an infinite number of
    /// coefficients. To represent these elements on a computer, one has to
    /// truncate the coefficient vectors somewhere. The resulting truncation
    /// error is considered "small" when it lives on large powers of X. This
    /// function works by applying Newton's method in this ring.
    ///
    /// # Example
    ///
    /// ```
    /// # use num_traits::One;
    /// # use twenty_first::prelude::*;
    /// let precision = 8;
    /// let f = Polynomial::new(bfe_vec![42; precision]);
    /// let g = f.clone().formal_power_series_inverse_newton(precision);
    /// let x_to_the_n = Polynomial::one().shift_coefficients(precision);
    /// let (_quotient, remainder) = g.multiply(&f).divide(&x_to_the_n);
    /// assert!(remainder.is_one());
    /// ```
    /// # Panics
    ///
    /// Panics when f(X) is not invertible in the formal power series ring,
    /// _i.e._, when its constant coefficient is zero.
    pub fn formal_power_series_inverse_newton(self, precision: usize) -> Polynomial<'static, FF> {
        // polynomials of degree zero are non-zero and have an exact inverse
        let self_degree = self.degree();
        if self_degree == 0 {
            return Polynomial::from_constant(self.coefficients[0].inverse());
        }

        // otherwise we need to run some iterations of Newton's method
        let num_rounds = precision.next_power_of_two().ilog2();

        // for small polynomials we use standard multiplication,
        // but for larger ones we want to stay in the ntt domain
        let switch_point = if Self::FORMAL_POWER_SERIES_INVERSE_CUTOFF < self_degree {
            0
        } else {
            (Self::FORMAL_POWER_SERIES_INVERSE_CUTOFF / self_degree).ilog2()
        };

        let cc = self.coefficients[0];

        // standard part
        let mut f = Polynomial::from_constant(cc.inverse());
        for _ in 0..u32::min(num_rounds, switch_point) {
            let sub = f.multiply(&f).multiply(&self);
            f.scalar_mul_mut(FF::from(2));
            f = f - sub;
        }

        // if we already have the required precision, terminate early
        if switch_point >= num_rounds {
            return f;
        }

        // ntt-based multiplication from here on out

        // final NTT domain
        let full_domain_length =
            ((1 << (num_rounds + 1)) * self_degree as usize).next_power_of_two();

        let mut self_ntt = self.coefficients.into_owned();
        self_ntt.resize(full_domain_length, FF::ZERO);
        ntt(&mut self_ntt);

        // while possible, we calculate over a smaller domain
        let mut current_domain_length = f.coefficients.len().next_power_of_two();

        // migrate to a larger domain as necessary
        let lde = |v: &mut [FF], old_domain_length: usize, new_domain_length: usize| {
            intt(&mut v[..old_domain_length]);
            ntt(&mut v[..new_domain_length]);
        };

        // use degree to track when domain-changes are necessary
        let mut f_degree = f.degree();

        // allocate enough space for f and set initial values of elements used
        // later to zero
        let mut f_ntt = f.coefficients.into_owned();
        f_ntt.resize(full_domain_length, FF::ZERO);
        ntt(&mut f_ntt[..current_domain_length]);

        for _ in switch_point..num_rounds {
            f_degree = 2 * f_degree + self_degree;
            if f_degree as usize >= current_domain_length {
                let next_domain_length = (1 + f_degree as usize).next_power_of_two();
                lde(&mut f_ntt, current_domain_length, next_domain_length);
                current_domain_length = next_domain_length;
            }
            f_ntt
                .iter_mut()
                .zip(
                    self_ntt
                        .iter()
                        .step_by(full_domain_length / current_domain_length),
                )
                .for_each(|(ff, dd)| *ff = FF::from(2) * *ff - *ff * *ff * *dd);
        }

        intt(&mut f_ntt[..current_domain_length]);
        Polynomial::new(f_ntt)
    }

    /// Fast evaluate on a coset domain, which is the group generated by
    /// `generator^i * offset`.
    ///
    /// # Performance
    ///
    /// If possible, use a [base field element](BFieldElement) as the offset.
    ///
    /// # Panics
    ///
    /// Panics if the order of the domain generated by the `generator` is
    /// smaller than or equal to the degree of `self`.
    pub fn fast_coset_evaluate<S>(&self, offset: S, order: usize) -> Vec<FF>
    where
        S: Clone + One,
        FF: Mul<S, Output = FF> + 'static,
    {
        // NTT's input and output are of the same size. For domains of an order
        // that is larger than or equal to the number of coefficients of the
        // polynomial, padding with leading zeros (a no-op to the polynomial)
        // achieves this requirement. However, if the order is smaller than the
        // number of coefficients in the polynomial, this would mean chopping
        // off leading coefficients, which changes the polynomial. Therefore,
        // this method is currently limited to domain orders greater than the
        // degree of the polynomial.
        // todo: move Triton VM's solution for above issue in here
        assert!(
            (order as isize) > self.degree(),
            "`Polynomial::fast_coset_evaluate` is currently limited to domains of order \
            greater than the degree of the polynomial."
        );

        let mut coefficients = self.scaled_coefficients(offset, order);
        coefficients.resize(order, FF::ZERO);
        ntt(&mut coefficients);

        coefficients
    }

    /// [`fast_coset_evaluate`](Self::fast_coset_evaluate), writing the
    /// codeword into the given, possibly uninitialized memory instead of
    /// allocating. On return, every element of `codeword` is initialized.
    ///
    /// Besides avoiding the allocation, this fuses the scaling and
    /// zero-padding with the transform's first passes over memory; see
    /// [`scaled_zero_padded_ntt`]. For large codewords, this is considerably
    /// faster than [`fast_coset_evaluate`](Self::fast_coset_evaluate).
    ///
    /// # Panics
    ///
    /// Panics if the codeword's length is not a power of two, or if the
    /// polynomial has more coefficients than the codeword is long.
    pub fn fast_coset_evaluate_into(
        &self,
        offset: BFieldElement,
        codeword: &mut [std::mem::MaybeUninit<FF>],
    ) where
        FF: Mul<BFieldElement, Output = FF>,
    {
        scaled_zero_padded_ntt(&self.coefficients, offset, codeword);
    }

    /// Parallel version of
    /// [`fast_coset_evaluate_into`](Self::fast_coset_evaluate_into). Use this
    /// for a single, large evaluation; see [`par_scaled_zero_padded_ntt`].
    ///
    /// # Panics
    ///
    /// See [`fast_coset_evaluate_into`](Self::fast_coset_evaluate_into).
    pub fn par_fast_coset_evaluate_into(
        &self,
        offset: BFieldElement,
        codeword: &mut [std::mem::MaybeUninit<FF>],
    ) where
        FF: Mul<BFieldElement, Output = FF>,
    {
        par_scaled_zero_padded_ntt(&self.coefficients, offset, codeword);
    }

    /// Parallel version of [`fast_coset_evaluate`](Self::fast_coset_evaluate).
    ///
    /// Use this for a single, large evaluation. If many polynomials are to be
    /// evaluated, it is generally more efficient to evaluate them in parallel
    /// using the serial version for each.
    ///
    /// # Panics
    ///
    /// See [`fast_coset_evaluate`](Self::fast_coset_evaluate).
    pub fn par_fast_coset_evaluate<S>(&self, offset: S, order: usize) -> Vec<FF>
    where
        S: Clone + One + Send + Sync,
        FF: Mul<S, Output = FF> + 'static,
    {
        assert!(
            (order as isize) > self.degree(),
            "`Polynomial::par_fast_coset_evaluate` is currently limited to domains of order \
            greater than the degree of the polynomial."
        );

        let mut coefficients = self.par_scaled_coefficients(offset, order);
        coefficients.resize(order, FF::ZERO);
        par_ntt(&mut coefficients);

        coefficients
    }
}

/// `base^exponent` by square-and-multiply, for any multiplicative monoid.
fn generic_pow<S: Clone + One>(base: S, mut exponent: usize) -> S {
    let mut result = S::one();
    let mut base = base;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = result * base.clone();
        }
        base = base.clone() * base;
        exponent >>= 1;
    }
    result
}

impl<FF> Polynomial<'static, FF>
where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    /// Computing the [fast zerofier][fast] is slower than computing the
    /// [smart zerofier][smart] for domain sizes smaller than this threshold.
    /// The [naïve zerofier][naive] is always slower to compute than the
    /// [smart zerofier][smart] for domain sizes smaller than the threshold.
    ///
    /// Extracted from `cargo bench --bench zerofier`.
    ///
    /// [naive]: Self::naive_zerofier
    /// [smart]: Self::smart_zerofier
    /// [fast]: Self::fast_zerofier
    const FAST_ZEROFIER_CUTOFF_THRESHOLD: usize = 100;

    /// Compute the lowest degree polynomial with the provided roots.
    /// Also known as “vanishing polynomial.”
    ///
    /// # Example
    ///
    /// ```
    /// # use num_traits::Zero;
    /// # use twenty_first::prelude::*;
    /// let roots = bfe_array![2, 4, 6];
    /// let zerofier = Polynomial::zerofier(&roots);
    ///
    /// assert_eq!(3, zerofier.degree());
    /// assert_eq!(bfe_vec![0, 0, 0], zerofier.batch_evaluate(&roots));
    ///
    /// let  non_roots = bfe_vec![0, 1, 3, 5];
    /// assert!(zerofier.batch_evaluate(&non_roots).iter().all(|x| !x.is_zero()));
    /// ```
    pub fn zerofier(roots: &[FF]) -> Self {
        if roots.len() < Self::FAST_ZEROFIER_CUTOFF_THRESHOLD {
            Self::smart_zerofier(roots)
        } else {
            Self::fast_zerofier(roots)
        }
    }

    /// Parallel version of [`zerofier`](Self::zerofier).
    pub fn par_zerofier(roots: &[FF]) -> Self {
        if roots.is_empty() {
            return Polynomial::one();
        }
        let num_threads = current_num_threads().max(1);
        let chunk_size = roots
            .len()
            .div_ceil(num_threads)
            .max(Self::FAST_ZEROFIER_CUTOFF_THRESHOLD);
        let factors = roots
            .par_chunks(chunk_size)
            .map(|chunk| Self::zerofier(chunk))
            .collect::<Vec<_>>();
        Polynomial::par_batch_multiply(&factors)
    }

    /// Only `pub` to allow benchmarking; not considered part of the public API.
    #[doc(hidden)]
    pub fn smart_zerofier(roots: &[FF]) -> Self {
        let mut zerofier = vec![FF::ZERO; roots.len() + 1];
        zerofier[0] = FF::ONE;
        for (num_coeffs, &root) in (1..).zip(roots) {
            for k in (1..=num_coeffs).rev() {
                zerofier[k] = zerofier[k - 1] - root * zerofier[k];
            }
            zerofier[0] = -root * zerofier[0];
        }
        Self::new(zerofier)
    }

    /// Only `pub` to allow benchmarking; not considered part of the public API.
    #[doc(hidden)]
    pub fn fast_zerofier(roots: &[FF]) -> Self {
        let mid_point = roots.len() / 2;
        let left = Self::zerofier(&roots[..mid_point]);
        let right = Self::zerofier(&roots[mid_point..]);

        left.multiply(&right)
    }

    /// Construct the lowest-degree polynomial interpolating the given points.
    ///
    /// ```
    /// # use twenty_first::prelude::*;
    /// let domain = bfe_vec![0, 1, 2, 3];
    /// let values = bfe_vec![1, 3, 5, 7];
    /// let polynomial = Polynomial::interpolate(&domain, &values);
    ///
    /// assert_eq!(1, polynomial.degree());
    /// assert_eq!(bfe!(9), polynomial.evaluate(bfe!(4)));
    /// ```
    ///
    /// # Panics
    ///
    /// - Panics if the provided domain is empty.
    /// - Panics if the provided domain and values are not of the same length.
    pub fn interpolate(domain: &[FF], values: &[FF]) -> Self {
        assert!(
            !domain.is_empty(),
            "interpolation must happen through more than zero points"
        );
        assert_eq!(
            domain.len(),
            values.len(),
            "The domain and values lists have to be of equal length."
        );

        if domain.len() <= Self::FAST_INTERPOLATE_CUTOFF_THRESHOLD_SEQUENTIAL {
            Self::lagrange_interpolate(domain, values)
        } else {
            Self::fast_interpolate(domain, values)
        }
    }

    /// Parallel version of [`interpolate`](Self::interpolate).
    ///
    /// # Panics
    ///
    /// See [`interpolate`](Self::interpolate).
    pub fn par_interpolate(domain: &[FF], values: &[FF]) -> Self {
        assert!(
            !domain.is_empty(),
            "interpolation must happen through more than zero points"
        );
        assert_eq!(
            domain.len(),
            values.len(),
            "The domain and values lists have to be of equal length."
        );

        // Reuse sequential threshold. We don't know how speed up this task with
        // parallelism below this threshold.
        if domain.len() <= Self::FAST_INTERPOLATE_CUTOFF_THRESHOLD_PARALLEL {
            Self::lagrange_interpolate(domain, values)
        } else {
            Self::par_fast_interpolate(domain, values)
        }
    }

    /// Any fast interpolation will use NTT, so this is mainly used for
    /// testing & integrity purposes. This also means that it is not pivotal
    /// that this function has an optimal runtime.
    #[doc(hidden)]
    pub fn lagrange_interpolate_zipped(points: &[(FF, FF)]) -> Self {
        assert!(
            !points.is_empty(),
            "interpolation must happen through more than zero points"
        );
        assert!(
            points.iter().map(|x| x.0).all_unique(),
            "Repeated x values received. Got: {points:?}",
        );

        let xs: Vec<FF> = points.iter().map(|x| x.0.to_owned()).collect();
        let ys: Vec<FF> = points.iter().map(|x| x.1.to_owned()).collect();
        Self::lagrange_interpolate(&xs, &ys)
    }

    #[doc(hidden)]
    pub fn lagrange_interpolate(domain: &[FF], values: &[FF]) -> Self {
        debug_assert!(
            !domain.is_empty(),
            "interpolation domain cannot have zero points"
        );
        debug_assert_eq!(domain.len(), values.len());

        let zero = FF::ZERO;
        let zerofier = Self::zerofier(domain).coefficients;

        // In each iteration of this loop, accumulate into the sum one
        // polynomial that evaluates to some abscis (y-value) in the given
        // ordinate (domain point), and to zero in all other ordinates.
        let mut lagrange_sum_array = vec![zero; domain.len()];
        let mut summand_array = vec![zero; domain.len()];
        for (i, &abscis) in values.iter().enumerate() {
            // divide (X - domain[i]) out of zerofier to get unweighted summand
            let mut leading_coefficient = zerofier[domain.len()];
            let mut supporting_coefficient = zerofier[domain.len() - 1];
            let mut summand_eval = zero;
            for j in (1..domain.len()).rev() {
                summand_array[j] = leading_coefficient;
                summand_eval = summand_eval * domain[i] + leading_coefficient;
                leading_coefficient = supporting_coefficient + leading_coefficient * domain[i];
                supporting_coefficient = zerofier[j - 1];
            }

            // avoid `j - 1` for j == 0 in the loop above
            summand_array[0] = leading_coefficient;
            summand_eval = summand_eval * domain[i] + leading_coefficient;

            // summand does not necessarily evaluate to 1 in domain[i]: correct
            // for this value
            let corrected_abscis = abscis / summand_eval;

            // accumulate term
            for j in 0..domain.len() {
                lagrange_sum_array[j] += corrected_abscis * summand_array[j];
            }
        }

        Self::new(lagrange_sum_array)
    }

    /// Only `pub` to allow benchmarking; not considered part of the public API.
    #[doc(hidden)]
    pub fn fast_interpolate(domain: &[FF], values: &[FF]) -> Self {
        debug_assert!(
            !domain.is_empty(),
            "interpolation domain cannot have zero points"
        );
        debug_assert_eq!(domain.len(), values.len());

        // prevent edge case failure where the left half would be empty
        if domain.len() == 1 {
            return Self::from_constant(values[0]);
        }

        let mid_point = domain.len() / 2;
        let left_domain_half = &domain[..mid_point];
        let left_values_half = &values[..mid_point];
        let right_domain_half = &domain[mid_point..];
        let right_values_half = &values[mid_point..];

        let left_zerofier = Self::zerofier(left_domain_half);
        let right_zerofier = Self::zerofier(right_domain_half);

        let left_offset = right_zerofier.batch_evaluate(left_domain_half);
        let right_offset = left_zerofier.batch_evaluate(right_domain_half);

        let hadamard_mul = |x: &[_], y: Vec<_>| x.iter().zip(y).map(|(&n, d)| n * d).collect_vec();
        let interpolate_half = |offset, domain_half, values_half| {
            let offset_inverse = FF::batch_inversion(offset);
            let targets = hadamard_mul(values_half, offset_inverse);
            Self::interpolate(domain_half, &targets)
        };
        let (left_interpolant, right_interpolant) = (
            interpolate_half(left_offset, left_domain_half, left_values_half),
            interpolate_half(right_offset, right_domain_half, right_values_half),
        );

        let (left_term, right_term) = (
            left_interpolant.multiply(&right_zerofier),
            right_interpolant.multiply(&left_zerofier),
        );

        left_term + right_term
    }

    /// Only `pub` to allow benchmarking; not considered part of the public API.
    #[doc(hidden)]
    pub fn par_fast_interpolate(domain: &[FF], values: &[FF]) -> Self {
        debug_assert!(
            !domain.is_empty(),
            "interpolation domain cannot have zero points"
        );
        debug_assert_eq!(domain.len(), values.len());

        if domain.len() == 1 {
            return Self::from_constant(values[0]);
        }

        let zerofier_tree = ZerofierTree::par_new_from_domain(domain);
        Self::par_interpolate_with_zerofier_tree(&zerofier_tree, values)
    }

    /// Like [`par_fast_interpolate`](Self::par_fast_interpolate), but for a
    /// domain given as its [zerofier tree](ZerofierTree), which can then be
    /// shared with other computations over the same domain, like
    /// [`par_divide_and_conquer_batch_evaluate`][eval].
    ///
    /// # Panics
    ///
    /// Panics if the number of values does not equal the tree's number of
    /// points, or if the tree's points are not distinct.
    ///
    /// [eval]: Self::par_divide_and_conquer_batch_evaluate
    pub fn par_interpolate_with_zerofier_tree(
        zerofier_tree: &ZerofierTree<FF>,
        values: &[FF],
    ) -> Self {
        assert_eq!(zerofier_tree.num_points(), values.len());
        if zerofier_tree.num_points() == 1 {
            return Self::from_constant(values[0]);
        }

        // Lagrange's formula: with z the zerofier of the domain and z' its
        // formal derivative, the interpolant is Σ_i y_i / z'(x_i) · z / (x - x_i).
        // The sums over subsets of the points are combined along the zerofier
        // tree: the sum over a branch is the sum over the left child times
        // the right zerofier, plus vice versa. All steps are parallel, and
        // the total work is quasi-linear in the number of points.
        let zerofier_derivative = zerofier_tree.zerofier_view().formal_derivative();
        let derivative_in_domain =
            zerofier_derivative.par_divide_and_conquer_batch_evaluate(zerofier_tree);
        let derivative_inverses = FF::par_batch_inversion(derivative_in_domain);
        let weights = values
            .par_iter()
            .zip(derivative_inverses)
            .map(|(&value, inverse)| value * inverse)
            .collect::<Vec<_>>();

        Self::par_interpolate_with_zerofier_tree_and_weights(zerofier_tree, &weights)
    }

    /// Like [`par_interpolate_with_zerofier_tree`][interpolate], but for
    /// Lagrange weights instead of values. With `z` the zerofier of the
    /// tree's points `x_i`, the returned polynomial is `Σ_i weights[i] · z /
    /// (x - x_i)`. It interpolates the values `y_i` if `weights[i] = y_i /
    /// z'(x_i)`, where `z'` is the formal derivative of `z`.
    ///
    /// Prefer this over [`par_interpolate_with_zerofier_tree`][interpolate]
    /// when the evaluations of `z'` in the points, or their inverses, are
    /// already known: computing them is the bulk of the work.
    ///
    /// # Panics
    ///
    /// Panics if the number of weights does not equal the tree's number of
    /// points.
    ///
    /// [interpolate]: Self::par_interpolate_with_zerofier_tree
    pub fn par_interpolate_with_zerofier_tree_and_weights(
        zerofier_tree: &ZerofierTree<FF>,
        weights: &[FF],
    ) -> Self {
        assert_eq!(zerofier_tree.num_points(), weights.len());
        Self::interpolate_with_zerofier_tree(zerofier_tree, weights)
    }

    /// The polynomial `Σ_i weights[i] · z / (x - x_i)`, where `z` is the
    /// zerofier of the tree's points `x_i`. See
    /// [`par_fast_interpolate`](Self::par_fast_interpolate).
    ///
    /// The tree is traversed level by level, from the leafs up, with all
    /// nodes of a level processed in parallel. The (few, large) nodes near
    /// the root use parallel transforms internally; the (many, small) nodes
    /// further down do not.
    fn interpolate_with_zerofier_tree(
        zerofier_tree: &ZerofierTree<FF>,
        weights: &[FF],
    ) -> Polynomial<'static, FF> {
        // Every level lists its nodes from left to right, each with the
        // index of its first point.
        let mut levels = vec![vec![(zerofier_tree, 0)]];
        loop {
            let next_level = levels
                .last()
                .unwrap()
                .iter()
                .flat_map(|&(node, first_point)| match node {
                    ZerofierTree::Branch(branch) => {
                        let right_first_point = first_point + branch.left.num_points();
                        vec![
                            (&branch.left, first_point),
                            (&branch.right, right_first_point),
                        ]
                    }
                    _ => vec![],
                })
                .collect_vec();
            if next_level.is_empty() {
                break;
            }
            levels.push(next_level);
        }

        let interpolate_leaf_or_padding = |node: &ZerofierTree<FF>, first_point: usize| match node {
            ZerofierTree::Leaf(leaf) => {
                let weights = &weights[first_point..first_point + leaf.points.len()];
                Self::interpolate_leaf(node, &leaf.points, weights)
            }
            ZerofierTree::Padding => Polynomial::zero(),
            ZerofierTree::Branch(_) => unreachable!("branches are handled separately"),
        };

        let mut interpolants_below: Vec<Polynomial<'static, FF>> = vec![];
        for level in levels.iter().rev() {
            // Hand every branch its two children's interpolants. The level
            // below lists them in the same order as the branches here.
            let mut interpolants_below_iter = interpolants_below.into_iter();
            let children = level
                .iter()
                .map(|(node, _)| match node {
                    ZerofierTree::Branch(_) => {
                        let left = interpolants_below_iter.next().unwrap();
                        let right = interpolants_below_iter.next().unwrap();
                        Some((left, right))
                    }
                    _ => None,
                })
                .collect_vec();
            debug_assert!(interpolants_below_iter.next().is_none());

            interpolants_below = level
                .par_iter()
                .zip(children)
                .map(|(&(node, first_point), children)| match (node, children) {
                    (ZerofierTree::Branch(branch), Some((left, right))) => {
                        Self::interpolate_branch(branch, left, right, level.len())
                    }
                    (node, None) => interpolate_leaf_or_padding(node, first_point),
                    (_, Some(_)) => unreachable!("only branches have children"),
                })
                .collect();
        }

        interpolants_below.pop().unwrap()
    }

    /// The leaf step of [`interpolate_with_zerofier_tree`][interp]: for each
    /// point, synthetic division of the leaf's zerofier by (x - x_i) gives
    /// z / (x - x_i).
    ///
    /// [interp]: Self::interpolate_with_zerofier_tree
    fn interpolate_leaf(
        leaf: &ZerofierTree<FF>,
        points: &[FF],
        weights: &[FF],
    ) -> Polynomial<'static, FF> {
        let zerofier = leaf.zerofier_view();
        let z = zerofier.coefficients();
        let num_points = points.len();
        debug_assert_eq!(num_points, weights.len());
        let mut accumulator = vec![FF::ZERO; num_points];
        for (&x_i, &weight) in points.iter().zip(weights) {
            let mut quotient_coefficient = FF::ONE;
            accumulator[num_points - 1] += weight * quotient_coefficient;
            for k in (1..num_points).rev() {
                quotient_coefficient = z[k] + x_i * quotient_coefficient;
                accumulator[k - 1] += weight * quotient_coefficient;
            }
        }
        Polynomial::new(accumulator)
    }

    /// The branch step of [`interpolate_with_zerofier_tree`][interp]:
    /// `left · z_right + right · z_left`, which has degree less than the
    /// branch's number of points and thus fits into a cyclic convolution of
    /// the next power of two.
    ///
    /// [interp]: Self::interpolate_with_zerofier_tree
    fn interpolate_branch(
        branch: &Branch<FF>,
        left: Polynomial<'static, FF>,
        right: Polynomial<'static, FF>,
        num_concurrent: usize,
    ) -> Polynomial<'static, FF> {
        if branch.right.num_points() == 0 {
            return left;
        }

        let num_points = branch.num_points;
        let len = num_points.next_power_of_two();
        let par = should_par_ntt(len, num_concurrent);
        let transform = |coefficients: &[FF]| {
            let mut transformed = zero_padded_maybe_par(coefficients, len, par);
            ntt_maybe_par(&mut transformed, par);
            transformed
        };
        let left_zerofier = branch.left.zerofier_view();
        let right_zerofier = branch.right.zerofier_view();

        // For small nodes, the level's many nodes already saturate the
        // threads; spawning more tasks only adds stealing overhead.
        let ((mut left, right_zerofier), (right, left_zerofier)) = if par {
            rayon::join(
                || {
                    rayon::join(
                        || transform(left.coefficients()),
                        || transform(right_zerofier.coefficients()),
                    )
                },
                || {
                    rayon::join(
                        || transform(right.coefficients()),
                        || transform(left_zerofier.coefficients()),
                    )
                },
            )
        } else {
            (
                (
                    transform(left.coefficients()),
                    transform(right_zerofier.coefficients()),
                ),
                (
                    transform(right.coefficients()),
                    transform(left_zerofier.coefficients()),
                ),
            )
        };

        let combine = |(((l, zr), r), zl): (((&mut FF, &FF), &FF), &FF)| *l = *l * *zr + *r * *zl;
        if par {
            left.par_iter_mut()
                .zip(&right_zerofier)
                .zip(&right)
                .zip(&left_zerofier)
                .for_each(combine);
        } else {
            left.iter_mut()
                .zip(&right_zerofier)
                .zip(&right)
                .zip(&left_zerofier)
                .for_each(combine);
        }
        intt_maybe_par(&mut left, par);
        left.truncate(num_points);
        Polynomial::new(left)
    }

    pub fn batch_fast_interpolate(
        domain: &[FF],
        values_matrix: &[Vec<FF>],
        primitive_root: BFieldElement,
        root_order: usize,
    ) -> Vec<Self> {
        debug_assert_eq!(
            primitive_root.mod_pow_u32(root_order as u32),
            BFieldElement::ONE,
            "Supplied element “primitive_root” must have supplied order.\
            Supplied element was: {primitive_root:?}\
            Supplied order was: {root_order:?}"
        );

        assert!(
            !domain.is_empty(),
            "Cannot fast interpolate through zero points.",
        );

        let mut zerofier_dictionary: HashMap<(FF, FF), Polynomial<FF>> = HashMap::default();
        let mut offset_inverse_dictionary: HashMap<(FF, FF), Vec<FF>> = HashMap::default();

        Self::batch_fast_interpolate_with_memoization(
            domain,
            values_matrix,
            &mut zerofier_dictionary,
            &mut offset_inverse_dictionary,
        )
    }

    fn batch_fast_interpolate_with_memoization(
        domain: &[FF],
        values_matrix: &[Vec<FF>],
        zerofier_dictionary: &mut HashMap<(FF, FF), Polynomial<'static, FF>>,
        offset_inverse_dictionary: &mut HashMap<(FF, FF), Vec<FF>>,
    ) -> Vec<Self> {
        // This value of 16 was found to be optimal through a benchmark on
        // sword_smith's machine.
        const OPTIMAL_CUTOFF_POINT_FOR_BATCHED_INTERPOLATION: usize = 16;
        if domain.len() < OPTIMAL_CUTOFF_POINT_FOR_BATCHED_INTERPOLATION {
            return values_matrix
                .iter()
                .map(|values| Self::lagrange_interpolate(domain, values))
                .collect();
        }

        // calculate everything related to the domain
        let half = domain.len() / 2;

        let left_key = (domain[0], domain[half - 1]);
        let left_zerofier = match zerofier_dictionary.get(&left_key) {
            Some(z) => z.to_owned(),
            None => {
                let left_zerofier = Self::zerofier(&domain[..half]);
                zerofier_dictionary.insert(left_key, left_zerofier.clone());
                left_zerofier
            }
        };
        let right_key = (domain[half], *domain.last().unwrap());
        let right_zerofier = match zerofier_dictionary.get(&right_key) {
            Some(z) => z.to_owned(),
            None => {
                let right_zerofier = Self::zerofier(&domain[half..]);
                zerofier_dictionary.insert(right_key, right_zerofier.clone());
                right_zerofier
            }
        };

        let left_offset_inverse = match offset_inverse_dictionary.get(&left_key) {
            Some(vector) => vector.to_owned(),
            None => {
                let left_offset: Vec<FF> = Self::batch_evaluate(&right_zerofier, &domain[..half]);
                let left_offset_inverse = FF::batch_inversion(left_offset);
                offset_inverse_dictionary.insert(left_key, left_offset_inverse.clone());
                left_offset_inverse
            }
        };
        let right_offset_inverse = match offset_inverse_dictionary.get(&right_key) {
            Some(vector) => vector.to_owned(),
            None => {
                let right_offset: Vec<FF> = Self::batch_evaluate(&left_zerofier, &domain[half..]);
                let right_offset_inverse = FF::batch_inversion(right_offset);
                offset_inverse_dictionary.insert(right_key, right_offset_inverse.clone());
                right_offset_inverse
            }
        };

        // prepare target matrices
        let all_left_targets: Vec<_> = values_matrix
            .par_iter()
            .map(|values| {
                values[..half]
                    .iter()
                    .zip(left_offset_inverse.iter())
                    .map(|(n, d)| n.to_owned() * *d)
                    .collect()
            })
            .collect();
        let all_right_targets: Vec<_> = values_matrix
            .par_iter()
            .map(|values| {
                values[half..]
                    .par_iter()
                    .zip(right_offset_inverse.par_iter())
                    .map(|(n, d)| n.to_owned() * *d)
                    .collect()
            })
            .collect();

        // recurse
        let left_interpolants = Self::batch_fast_interpolate_with_memoization(
            &domain[..half],
            &all_left_targets,
            zerofier_dictionary,
            offset_inverse_dictionary,
        );
        let right_interpolants = Self::batch_fast_interpolate_with_memoization(
            &domain[half..],
            &all_right_targets,
            zerofier_dictionary,
            offset_inverse_dictionary,
        );

        // add vectors of polynomials
        left_interpolants
            .par_iter()
            .zip(right_interpolants.par_iter())
            .map(|(left_interpolant, right_interpolant)| {
                let left_term = left_interpolant.multiply(&right_zerofier);
                let right_term = right_interpolant.multiply(&left_zerofier);

                left_term + right_term
            })
            .collect()
    }

    /// Evaluate the polynomial on a batch of points.
    pub fn batch_evaluate(&self, domain: &[FF]) -> Vec<FF> {
        if self.is_zero() {
            vec![FF::ZERO; domain.len()]
        } else if self.degree()
            >= Self::REDUCE_BEFORE_EVALUATE_THRESHOLD_RATIO * (domain.len() as isize)
        {
            self.reduce_then_batch_evaluate(domain)
        } else {
            let zerofier_tree = ZerofierTree::new_from_domain(domain);
            self.divide_and_conquer_batch_evaluate(&zerofier_tree)
        }
    }

    fn reduce_then_batch_evaluate(&self, domain: &[FF]) -> Vec<FF> {
        let zerofier_tree = ZerofierTree::new_from_domain(domain);
        let zerofier = zerofier_tree.zerofier();
        let remainder = self.fast_reduce(&zerofier);
        remainder.divide_and_conquer_batch_evaluate(&zerofier_tree)
    }

    /// Parallel version of [`batch_evaluate`](Self::batch_evaluate).
    pub fn par_batch_evaluate(&self, domain: &[FF]) -> Vec<FF> {
        // For few points, the zerofier tree is not worth building.
        const ITERATIVE_EVALUATION_THRESHOLD: usize = 1 << 5;

        if domain.is_empty() || self.is_zero() {
            return vec![FF::ZERO; domain.len()];
        }
        if domain.len() < ITERATIVE_EVALUATION_THRESHOLD
            && self.degree() < 4 * (domain.len() as isize)
        {
            return self.iterative_batch_evaluate(domain);
        }

        let zerofier_tree = ZerofierTree::par_new_from_domain(domain);
        self.par_divide_and_conquer_batch_evaluate(&zerofier_tree)
    }

    /// Parallel version of
    /// [`divide_and_conquer_batch_evaluate`](Self::divide_and_conquer_batch_evaluate).
    /// Unlike the sequential version, the total work is quasi-linear in the
    /// number of points (plus the degree of `self`).
    ///
    /// Uses a scaled remainder tree, after [Bernstein][srt]: instead of the
    /// remainders of `self` modulo the nodes' zerofiers, the tree is traversed
    /// with the _scaled_ remainders `(self mod z) / z`, expanded as power
    /// series in `1/x`. Passing from a node to a child only requires
    /// multiplication with the sibling's zerofier, and the leafs' remainders
    /// follow from their scaled remainders with one more multiplication. The
    /// only division is the one at the root.
    ///
    /// [srt]: https://cr.yp.to/arith/scaledmod-20040820.pdf
    pub fn par_divide_and_conquer_batch_evaluate(
        &self,
        zerofier_tree: &ZerofierTree<FF>,
    ) -> Vec<FF> {
        let num_points = zerofier_tree.num_points();
        if num_points == 0 {
            return vec![];
        }
        let Ok(degree) = usize::try_from(self.degree()) else {
            return vec![FF::ZERO; num_points];
        };

        let zerofier = zerofier_tree.zerofier_view();
        let reversed_zerofier = zerofier.reverse();
        if degree >= num_points {
            // Reduce modulo the zerofier of all points first, so that the
            // scaled remainder tree starts from a polynomial of degree less
            // than the number of points. For a degree much larger than the
            // number of points, chunk-wise reduction is the better fit;
            // otherwise, fast division is.
            let degree_ratio = Self::REDUCE_BEFORE_EVALUATE_THRESHOLD_RATIO as usize;
            let reduced = if degree >= degree_ratio * num_points {
                self.fast_reduce(&zerofier)
            } else {
                let quotient_degree = degree - num_points;
                let reversed_zerofier_inverse =
                    reversed_zerofier.power_series_inverse(quotient_degree + 1);
                self.reduce_with_reversed_inverse(&zerofier, &reversed_zerofier_inverse)
            };
            return reduced.par_divide_and_conquer_batch_evaluate(zerofier_tree);
        }

        // With n the number of points and y = 1/x, the expansion of self / z
        // in y is y · rev(self) / rev(z), where rev(self) is the reversal of
        // self with respect to degree n - 1. Its first n coefficients are the
        // root's scaled remainder.
        let reversed_zerofier_inverse = reversed_zerofier.power_series_inverse(num_points);
        let par = should_par_ntt(num_points, 1);
        let reversed_self = init_maybe_par(num_points, par, |i| {
            let j = num_points - 1 - i;
            if j <= degree {
                self.coefficients[j]
            } else {
                FF::ZERO
            }
        });
        let mut scaled_remainder = Polynomial::new(reversed_self)
            .multiply_maybe_par(&reversed_zerofier_inverse)
            .into_coefficients();
        scaled_remainder.resize(num_points, FF::ZERO);
        Self::evaluate_scaled_remainder_tree(zerofier_tree, scaled_remainder)
    }

    /// Evaluate the polynomial `r` of degree less than the tree's number of
    /// points `n` in the tree's points, given its scaled remainder: the
    /// coefficients `s_1, …, s_n` of the expansion `r / z = Σ_k s_k · x^(-k)`
    /// where `z` is the tree's zerofier. See
    /// [`par_divide_and_conquer_batch_evaluate`][eval].
    ///
    /// The tree is traversed level by level, with all nodes of a level
    /// processed in parallel. The (few, large) nodes near the root use
    /// parallel transforms internally; the (many, small) nodes further down
    /// do not.
    ///
    /// [eval]: Self::par_divide_and_conquer_batch_evaluate
    fn evaluate_scaled_remainder_tree(
        zerofier_tree: &ZerofierTree<FF>,
        scaled_remainder: Vec<FF>,
    ) -> Vec<FF> {
        enum Item<'tree, 'coeffs, FF: FiniteField + MulAssign<BFieldElement>> {
            Pending(&'tree ZerofierTree<'coeffs, FF>, Vec<FF>),
            Evaluated(Vec<FF>),
        }

        debug_assert_eq!(zerofier_tree.num_points(), scaled_remainder.len());
        let mut items = vec![Item::Pending(zerofier_tree, scaled_remainder)];
        loop {
            let num_pending = items
                .iter()
                .filter(|item| matches!(item, Item::Pending(..)))
                .count();
            if num_pending == 0 {
                break;
            }

            // Every item turns into at most two items for the next level.
            let successors = items
                .into_par_iter()
                .map(|item| match item {
                    Item::Evaluated(evaluations) => [Some(Item::Evaluated(evaluations)), None],
                    Item::Pending(ZerofierTree::Padding, _) => [None, None],
                    Item::Pending(tree @ ZerofierTree::Leaf(leaf), remainder) => {
                        let evaluations =
                            Self::evaluate_scaled_remainder_leaf(tree, &leaf.points, &remainder);
                        [Some(Item::Evaluated(evaluations)), None]
                    }
                    Item::Pending(ZerofierTree::Branch(branch), remainder) => {
                        if branch.right.num_points() == 0 {
                            return [Some(Item::Pending(&branch.left, remainder)), None];
                        }
                        let (left, right) =
                            Self::scaled_remainders_of_children(branch, remainder, num_pending);
                        [
                            Some(Item::Pending(&branch.left, left)),
                            Some(Item::Pending(&branch.right, right)),
                        ]
                    }
                })
                .collect::<Vec<_>>();
            items = successors.into_iter().flatten().flatten().collect();
        }

        items
            .into_iter()
            .flat_map(|item| match item {
                Item::Evaluated(evaluations) => evaluations,
                Item::Pending(..) => unreachable!(),
            })
            .collect()
    }

    /// The leaf step of [`evaluate_scaled_remainder_tree`][srt].
    ///
    /// [srt]: Self::evaluate_scaled_remainder_tree
    fn evaluate_scaled_remainder_leaf(
        leaf: &ZerofierTree<FF>,
        points: &[FF],
        scaled_remainder: &[FF],
    ) -> Vec<FF> {
        // The remainder is the polynomial part of z · Σ_k s_k · x^(-k),
        // i.e., its j-th coefficient is Σ_k z_(j+k) · s_k.
        let num_points = points.len();
        let zerofier = leaf.zerofier_view();
        let z = zerofier.coefficients();
        let remainder = (0..num_points)
            .map(|j| {
                (1..=num_points - j)
                    .map(|k| z[j + k] * scaled_remainder[k - 1])
                    .fold(FF::ZERO, |acc, term| acc + term)
            })
            .collect_vec();
        Polynomial::new(remainder).iterative_batch_evaluate(points)
    }

    /// The branch step of [`evaluate_scaled_remainder_tree`][srt]: the scaled
    /// remainders of both children, given the branch's.
    ///
    /// The scaled remainder of a child is that of the parent times the
    /// sibling's zerofier, retaining only the negative powers of x. In terms
    /// of y = 1/x, the sibling's zerofier of degree m is y^(-m) · rev(z_sibling),
    /// and the child's coefficients are the coefficients of y^m through
    /// y^(n-1) of the product of the parent's scaled remainder and
    /// rev(z_sibling). A cyclic convolution of length ≥ n suffices: the
    /// wrapped coefficients land strictly below index m.
    ///
    /// [srt]: Self::evaluate_scaled_remainder_tree
    fn scaled_remainders_of_children(
        branch: &Branch<FF>,
        scaled_remainder: Vec<FF>,
        num_concurrent: usize,
    ) -> (Vec<FF>, Vec<FF>) {
        let num_points = branch.num_points;
        let len = num_points.next_power_of_two();
        let par = should_par_ntt(len, num_concurrent);

        let mut scaled_remainder_ntt = zero_padded_maybe_par(&scaled_remainder, len, par);
        drop(scaled_remainder);
        ntt_maybe_par(&mut scaled_remainder_ntt, par);

        let child_scaled_remainder = |sibling: &ZerofierTree<FF>| {
            let sibling_zerofier = sibling.zerofier_view();
            let sibling_num_points = sibling.num_points();
            let z = sibling_zerofier.coefficients();
            let mut product = init_maybe_par(len, par, |i| {
                if i <= sibling_num_points {
                    z[sibling_num_points - i]
                } else {
                    FF::ZERO
                }
            });
            ntt_maybe_par(&mut product, par);
            hadamard_product_maybe_par(&mut product, &scaled_remainder_ntt, par);
            intt_maybe_par(&mut product, par);
            product.truncate(num_points);
            product.drain(..sibling_num_points);
            product
        };

        // For small nodes, the level's many nodes already saturate the
        // threads; spawning more tasks only adds stealing overhead.
        if par {
            rayon::join(
                || child_scaled_remainder(&branch.right),
                || child_scaled_remainder(&branch.left),
            )
        } else {
            (
                child_scaled_remainder(&branch.right),
                child_scaled_remainder(&branch.left),
            )
        }
    }

    /// Only marked `pub` for benchmarking; not considered part of the public
    /// API.
    #[doc(hidden)]
    pub fn iterative_batch_evaluate(&self, domain: &[FF]) -> Vec<FF> {
        domain.iter().map(|&p| self.evaluate(p)).collect()
    }

    /// Only `pub` to allow benchmarking; not considered part of the public API.
    #[doc(hidden)]
    pub fn divide_and_conquer_batch_evaluate(&self, zerofier_tree: &ZerofierTree<FF>) -> Vec<FF> {
        match zerofier_tree {
            ZerofierTree::Leaf(leaf) => self
                .reduce(&zerofier_tree.zerofier())
                .iterative_batch_evaluate(&leaf.points),
            ZerofierTree::Branch(branch) => [
                self.divide_and_conquer_batch_evaluate(&branch.left),
                self.divide_and_conquer_batch_evaluate(&branch.right),
            ]
            .concat(),
            ZerofierTree::Padding => vec![],
        }
    }

    /// The inverse of [`Self::fast_coset_evaluate`].
    ///
    /// # Performance
    ///
    /// If possible, use a [base field element](BFieldElement) as the offset.
    ///
    /// # Panics
    ///
    /// Panics if the length of `values` is
    /// - not a power of 2
    /// - larger than [`u32::MAX`]
    pub fn fast_coset_interpolate<S>(offset: S, values: &[FF]) -> Self
    where
        S: Clone + One + Inverse,
        FF: Mul<S, Output = FF>,
    {
        let mut coefficients = crate::memory::vec_with_capacity(values.len());
        coefficients.extend_from_slice(values);

        intt(&mut coefficients);
        Self::scale_in_place(&mut coefficients, offset.inverse());
        Polynomial::new(coefficients)
    }

    /// Replace every `coefficients[i]` by `coefficients[i] · alpha^i`, i.e.,
    /// [`scale`](Self::scale) without allocating.
    fn scale_in_place<S>(coefficients: &mut [FF], alpha: S)
    where
        S: Clone + One,
        FF: Mul<S, Output = FF>,
    {
        let mut power_of_alpha = S::one();
        for coefficient in coefficients {
            *coefficient = *coefficient * power_of_alpha.clone();
            power_of_alpha = power_of_alpha * alpha.clone();
        }
    }

    /// Parallel version of [`scale_in_place`](Self::scale_in_place).
    fn par_scale_in_place<S>(coefficients: &mut [FF], alpha: S)
    where
        S: Clone + One + Send + Sync,
        FF: Mul<S, Output = FF>,
    {
        // Large enough to amortize computing the chunk's first power of α
        // by square-and-multiply, small enough to keep all threads busy.
        const CHUNK_SIZE: usize = 1 << 12;

        coefficients
            .par_chunks_mut(CHUNK_SIZE)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                let mut power_of_alpha = generic_pow(alpha.clone(), chunk_index * CHUNK_SIZE);
                for coefficient in chunk {
                    *coefficient = *coefficient * power_of_alpha.clone();
                    power_of_alpha = power_of_alpha * alpha.clone();
                }
            });
    }

    /// Parallel version of
    /// [`fast_coset_interpolate`](Self::fast_coset_interpolate). See also
    /// [`par_fast_coset_evaluate`](Self::par_fast_coset_evaluate).
    ///
    /// # Panics
    ///
    /// See [`fast_coset_interpolate`](Self::fast_coset_interpolate).
    pub fn par_fast_coset_interpolate<S>(offset: S, values: &[FF]) -> Self
    where
        S: Clone + One + Inverse + Send + Sync,
        FF: Mul<S, Output = FF>,
    {
        let par = should_par_ntt(values.len(), 1);
        let mut coefficients = zero_padded_maybe_par(values, values.len(), par);
        par_intt(&mut coefficients);
        if par {
            Self::par_scale_in_place(&mut coefficients, offset.inverse());
        } else {
            Self::scale_in_place(&mut coefficients, offset.inverse());
        }
        Polynomial::new(coefficients)
    }

    /// The degree-`k` polynomial with the same `k + 1` leading coefficients as
    /// `self`.
    ///
    /// To be more precise: The degree of the result will be the minimum of `k`
    /// and [`Self::degree()`]. This implies, among other things, that if `self`
    /// [is zero](Self::is_zero()), the result will also be zero, independent
    /// of `k`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use twenty_first::prelude::*;
    /// let f = Polynomial::new(bfe_vec![0, 1, 2, 3, 4]); // 4x⁴ + 3x³ + 2x² + 1x¹ + 0
    /// let g = f.truncate(2);                            // 4x² + 3x¹ + 2
    /// assert_eq!(Polynomial::new(bfe_vec![2, 3, 4]), g);
    /// ```
    pub fn truncate(&self, k: usize) -> Self {
        let coefficients = self.coefficients.iter().copied();
        let coefficients = coefficients.rev().take(k + 1).rev().collect();
        Self::new(coefficients)
    }

    /// `self % x^n`
    ///
    /// A special case of [Self::rem], and faster.
    ///
    /// # Examples
    ///
    /// ```
    /// # use twenty_first::prelude::*;
    /// let f = Polynomial::new(bfe_vec![0, 1, 2, 3, 4]); // 4x⁴ + 3x³ + 2x² + 1x¹ + 0
    /// let g = f.mod_x_to_the_n(2);                      // 1x¹ + 0
    /// assert_eq!(Polynomial::new(bfe_vec![0, 1]), g);
    /// ```
    pub fn mod_x_to_the_n(&self, n: usize) -> Self {
        let num_coefficients_to_retain = n.min(self.coefficients.len());
        Self::new(self.coefficients[..num_coefficients_to_retain].into())
    }

    /// Preprocessing data for
    /// [fast modular coset interpolation](Self::fast_modular_coset_interpolate).
    /// Marked `pub` for benchmarking. Not considered part of the public API.
    #[doc(hidden)]
    pub fn fast_modular_coset_interpolate_preprocess(
        n: usize,
        offset: BFieldElement,
        modulus: &Polynomial<FF>,
    ) -> ModularInterpolationPreprocessingData<'static, FF> {
        let omega = BFieldElement::primitive_root_of_unity(n as u64).unwrap();
        // a list of polynomials whose ith element is X^(2^i) mod m(X)
        let modular_squares = (0..n.ilog2())
            .scan(Polynomial::<FF>::x_to_the(1), |acc, _| {
                let yld = acc.clone();
                *acc = acc.multiply(acc).reduce(modulus);
                Some(yld)
            })
            .collect_vec();
        let even_zerofiers = (0..n.ilog2())
            .map(|i| offset.inverse().mod_pow(1u64 << i))
            .zip(modular_squares.iter())
            .map(|(lc, sq)| sq.scalar_mul(FF::from(lc.value())) - Polynomial::one())
            .collect_vec();
        let odd_zerofiers = (0..n.ilog2())
            .map(|i| (offset * omega).inverse().mod_pow(1u64 << i))
            .zip(modular_squares.iter())
            .map(|(lc, sq)| sq.scalar_mul(FF::from(lc.value())) - Polynomial::one())
            .collect_vec();

        // precompute NTT-friendly multiple of the modulus
        let (shift_coefficients, tail_length) = modulus.shift_factor_ntt_with_tail_length();

        ModularInterpolationPreprocessingData {
            even_zerofiers,
            odd_zerofiers,
            shift_coefficients,
            tail_length,
        }
    }

    /// Compute f(X) mod m(X) where m(X) is a given modulus and f(X) is the
    /// interpolant of a list of n values on a domain which is a coset of the
    /// size-n subgroup that is identified by some offset.
    fn fast_modular_coset_interpolate(
        values: &[FF],
        offset: BFieldElement,
        modulus: &Polynomial<FF>,
    ) -> Self {
        let preprocessing_data =
            Self::fast_modular_coset_interpolate_preprocess(values.len(), offset, modulus);
        Self::fast_modular_coset_interpolate_with_zerofiers_and_ntt_friendly_multiple(
            values,
            offset,
            modulus,
            &preprocessing_data,
        )
    }

    /// Only marked `pub` for benchmarking purposes. Not considered part of the
    /// interface.
    #[doc(hidden)]
    pub fn fast_modular_coset_interpolate_with_zerofiers_and_ntt_friendly_multiple(
        values: &[FF],
        offset: BFieldElement,
        modulus: &Polynomial<FF>,
        preprocessed: &ModularInterpolationPreprocessingData<FF>,
    ) -> Self {
        if modulus.degree() < 0 {
            panic!("cannot reduce modulo zero")
        };
        let n = values.len();
        let omega = BFieldElement::primitive_root_of_unity(n as u64).unwrap();

        if n < Self::FAST_MODULAR_COSET_INTERPOLATE_CUTOFF_THRESHOLD_PREFER_LAGRANGE {
            let domain = (0..n)
                .scan(FF::from(offset.value()), |acc: &mut FF, _| {
                    let yld = *acc;
                    *acc *= omega;
                    Some(yld)
                })
                .collect::<Vec<FF>>();
            let interpolant = Self::lagrange_interpolate(&domain, values);
            return interpolant.reduce(modulus);
        } else if n <= Self::FAST_MODULAR_COSET_INTERPOLATE_CUTOFF_THRESHOLD_PREFER_INTT {
            let mut coefficients = values.to_vec();
            intt(&mut coefficients);
            let interpolant = Polynomial::new(coefficients);

            return interpolant
                .scale(FF::from(offset.inverse().value()))
                .reduce_by_ntt_friendly_modulus(
                    &preprocessed.shift_coefficients,
                    preprocessed.tail_length,
                )
                .reduce(modulus);
        }

        // Use even-odd domain split.
        // Even: {offset * omega^{2*i} | i in 0..n/2}
        // Odd: {offset * omega^{2*i+1} | i in 0..n/2}
        //      = {(offset * omega) * omega^{2*i} | i in 0..n/2}
        // But we don't actually need to represent the domains explicitly.

        // 1. Get zerofiers.
        // The zerofiers are sparse because the domain is structured.
        // Even: (offset^-1 * X)^(n/2) - 1 = offset^{-n/2} * X^{n/2} - 1
        // Odd: ((offset * omega)^-1 * X)^(n/2) - 1
        //      = offset^{-n/2} * omega^{n/2} * X^{n/2} - 1
        // Note that we are getting the (modularly reduced) zerofiers as
        // function arguments.

        // 2. Evaluate zerofiers on opposite domains.
        // Actually, the values are compressible because the zerofiers are
        // sparse and the domains are structured (compatibly).
        // Even zerofier on odd domain:
        // (offset^-1 * X)^(n/2) - 1 on
        // {(offset * omega) * omega^{2*i} | i in 0..n/2}
        // = {omega^{n/2}-1 | i in 0..n/2} = {-2, -2, -2, ...}
        // Odd zerofier on even domain: {omega^-i | i in 0..n/2}
        // ((offset * omega)^-1 * X)^(n/2) - 1
        // on {offset * omega^{2*i} | i in 0..n/2}
        // = {omega^{-n/2} - 1 | i in 0..n/2} = {-2, -2, -2, ...}
        // Since these values are always the same, there's no point generating
        // them at runtime. Moreover, we need their batch-inverses in the next
        // step.

        // 3. Batch-invert zerofiers on opposite domains.
        // The batch-inversion is actually not performed because we already know
        // the result: {(-2)^-1, (-2)^-1, (-2)^-1, ...}.
        const MINUS_TWO_INVERSE: BFieldElement = BFieldElement::MINUS_TWO_INVERSE;
        let even_zerofier_on_odd_domain_inverted = vec![FF::from(MINUS_TWO_INVERSE.value()); n / 2];
        let odd_zerofier_on_even_domain_inverted = vec![FF::from(MINUS_TWO_INVERSE.value()); n / 2];

        // 4. Construct interpolation values through Hadamard products.
        let mut odd_domain_targets = even_zerofier_on_odd_domain_inverted;
        let mut even_domain_targets = odd_zerofier_on_even_domain_inverted;
        for i in 0..(n / 2) {
            even_domain_targets[i] *= values[2 * i];
            odd_domain_targets[i] *= values[2 * i + 1];
        }

        // 5. Interpolate using recursion
        let even_interpolant =
            Self::fast_modular_coset_interpolate(&even_domain_targets, offset, modulus);
        let odd_interpolant =
            Self::fast_modular_coset_interpolate(&odd_domain_targets, offset * omega, modulus);

        // 6. Multiply with zerofiers and add.
        let interpolant = even_interpolant
            .multiply(&preprocessed.odd_zerofiers[(n / 2).ilog2() as usize])
            + odd_interpolant.multiply(&preprocessed.even_zerofiers[(n / 2).ilog2() as usize]);

        // 7. Reduce by modulus and return.
        interpolant.reduce(modulus)
    }

    /// Extrapolate a Reed-Solomon codeword, defined relative to a coset of the
    /// subgroup of order n (codeword length), in new points.
    pub fn coset_extrapolate(
        domain_offset: BFieldElement,
        codeword: &[FF],
        points: &[FF],
    ) -> Vec<FF> {
        if points.len() < Self::FAST_COSET_EXTRAPOLATE_THRESHOLD {
            Self::fast_coset_extrapolate(domain_offset, codeword, points)
        } else {
            Self::naive_coset_extrapolate(domain_offset, codeword, points)
        }
    }

    fn naive_coset_extrapolate_preprocessing(
        points: &[FF],
    ) -> (ZerofierTree<'_, FF>, Vec<FF>, usize) {
        let zerofier_tree = ZerofierTree::new_from_domain(points);
        let (shift_coefficients, tail_length) =
            Self::shift_factor_ntt_with_tail_length(&zerofier_tree.zerofier());
        (zerofier_tree, shift_coefficients, tail_length)
    }

    fn naive_coset_extrapolate(
        domain_offset: BFieldElement,
        codeword: &[FF],
        points: &[FF],
    ) -> Vec<FF> {
        let mut coefficients = codeword.to_vec();
        intt(&mut coefficients);
        let interpolant =
            Polynomial::new(coefficients).scale(FF::from(domain_offset.inverse().value()));
        interpolant.batch_evaluate(points)
    }

    fn fast_coset_extrapolate(
        domain_offset: BFieldElement,
        codeword: &[FF],
        points: &[FF],
    ) -> Vec<FF> {
        let zerofier_tree = ZerofierTree::new_from_domain(points);
        let minimal_interpolant = Self::fast_modular_coset_interpolate(
            codeword,
            domain_offset,
            &zerofier_tree.zerofier(),
        );
        minimal_interpolant.divide_and_conquer_batch_evaluate(&zerofier_tree)
    }

    /// Extrapolate many Reed-Solomon codewords, defined relative to the same
    /// coset of the subgroup of order `codeword_length`, in the same set of
    /// new points.
    ///
    /// # Example
    /// ```
    /// # use twenty_first::prelude::*;
    /// let n = 1 << 5;
    /// let domain_offset = bfe!(7);
    /// let codewords = [bfe_vec![3; n], bfe_vec![2; n]].concat();
    /// let points = bfe_vec![0, 1];
    /// assert_eq!(
    ///     bfe_vec![3, 3, 2, 2],
    ///     Polynomial::<BFieldElement>::batch_coset_extrapolate(
    ///         domain_offset,
    ///         n,
    ///         &codewords,
    ///         &points
    ///     )
    /// );
    /// ```
    ///
    /// # Panics
    /// Panics if the `codeword_length` is not a power of two.
    pub fn batch_coset_extrapolate(
        domain_offset: BFieldElement,
        codeword_length: usize,
        codewords: &[FF],
        points: &[FF],
    ) -> Vec<FF> {
        if points.len() < Self::FAST_COSET_EXTRAPOLATE_THRESHOLD {
            Self::batch_fast_coset_extrapolate(domain_offset, codeword_length, codewords, points)
        } else {
            Self::batch_naive_coset_extrapolate(domain_offset, codeword_length, codewords, points)
        }
    }

    fn batch_fast_coset_extrapolate(
        domain_offset: BFieldElement,
        codeword_length: usize,
        codewords: &[FF],
        points: &[FF],
    ) -> Vec<FF> {
        let n = codeword_length;

        let zerofier_tree = ZerofierTree::new_from_domain(points);
        let modulus = zerofier_tree.zerofier();
        let preprocessing_data = Self::fast_modular_coset_interpolate_preprocess(
            codeword_length,
            domain_offset,
            &modulus,
        );

        (0..codewords.len() / n)
            .flat_map(|i| {
                let codeword = &codewords[i * n..(i + 1) * n];
                let minimal_interpolant =
                    Self::fast_modular_coset_interpolate_with_zerofiers_and_ntt_friendly_multiple(
                        codeword,
                        domain_offset,
                        &modulus,
                        &preprocessing_data,
                    );
                minimal_interpolant.divide_and_conquer_batch_evaluate(&zerofier_tree)
            })
            .collect()
    }

    fn batch_naive_coset_extrapolate(
        domain_offset: BFieldElement,
        codeword_length: usize,
        codewords: &[FF],
        points: &[FF],
    ) -> Vec<FF> {
        let (zerofier_tree, shift_coefficients, tail_length) =
            Self::naive_coset_extrapolate_preprocessing(points);
        let n = codeword_length;

        (0..codewords.len() / n)
            .flat_map(|i| {
                let mut coefficients = codewords[i * n..(i + 1) * n].to_vec();
                intt(&mut coefficients);
                Polynomial::new(coefficients)
                    .scale(FF::from(domain_offset.inverse().value()))
                    .reduce_by_ntt_friendly_modulus(&shift_coefficients, tail_length)
                    .divide_and_conquer_batch_evaluate(&zerofier_tree)
            })
            .collect()
    }

    /// Parallel version of [`batch_coset_extrapolate`](Self::batch_coset_extrapolate).
    pub fn par_batch_coset_extrapolate(
        domain_offset: BFieldElement,
        codeword_length: usize,
        codewords: &[FF],
        points: &[FF],
    ) -> Vec<FF> {
        if points.len() < Self::FAST_COSET_EXTRAPOLATE_THRESHOLD {
            Self::par_batch_fast_coset_extrapolate(
                domain_offset,
                codeword_length,
                codewords,
                points,
            )
        } else {
            Self::par_batch_naive_coset_extrapolate(
                domain_offset,
                codeword_length,
                codewords,
                points,
            )
        }
    }

    fn par_batch_fast_coset_extrapolate(
        domain_offset: BFieldElement,
        codeword_length: usize,
        codewords: &[FF],
        points: &[FF],
    ) -> Vec<FF> {
        let n = codeword_length;

        let zerofier_tree = ZerofierTree::new_from_domain(points);
        let modulus = zerofier_tree.zerofier();
        let preprocessing_data = Self::fast_modular_coset_interpolate_preprocess(
            codeword_length,
            domain_offset,
            &modulus,
        );

        (0..codewords.len() / n)
            .into_par_iter()
            .flat_map(|i| {
                let codeword = &codewords[i * n..(i + 1) * n];
                let minimal_interpolant =
                    Self::fast_modular_coset_interpolate_with_zerofiers_and_ntt_friendly_multiple(
                        codeword,
                        domain_offset,
                        &modulus,
                        &preprocessing_data,
                    );
                minimal_interpolant.divide_and_conquer_batch_evaluate(&zerofier_tree)
            })
            .collect()
    }

    fn par_batch_naive_coset_extrapolate(
        domain_offset: BFieldElement,
        codeword_length: usize,
        codewords: &[FF],
        points: &[FF],
    ) -> Vec<FF> {
        let (zerofier_tree, shift_coefficients, tail_length) =
            Self::naive_coset_extrapolate_preprocessing(points);
        let n = codeword_length;

        (0..codewords.len() / n)
            .into_par_iter()
            .flat_map(|i| {
                let mut coefficients = codewords[i * n..(i + 1) * n].to_vec();
                intt(&mut coefficients);
                Polynomial::new(coefficients)
                    .scale(FF::from(domain_offset.inverse().value()))
                    .reduce_by_ntt_friendly_modulus(&shift_coefficients, tail_length)
                    .divide_and_conquer_batch_evaluate(&zerofier_tree)
            })
            .collect()
    }
}

impl Polynomial<'_, BFieldElement> {
    /// [Clean division](Self::clean_divide) is slower than [naïve
    /// division](Self::naive_divide) for polynomials of degree less than this
    /// threshold.
    ///
    /// Extracted from `cargo bench --bench poly_clean_div` on mjolnir.
    const CLEAN_DIVIDE_CUTOFF_THRESHOLD: isize = { if cfg!(test) { 0 } else { 1 << 9 } };

    /// A fast way of dividing two polynomials. Only works if division is clean,
    /// _i.e._, if the remainder of polynomial long division is [zero]. This
    /// **must** be known ahead of time. If division is unclean, this method
    /// might panic or produce a wrong result. Use [`Polynomial::divide`] for
    /// more generality.
    ///
    /// # Panics
    ///
    /// Panics if
    /// - the divisor is [zero], or
    /// - division is not clean, _i.e._, if polynomial long division leaves some
    ///   non-zero remainder.
    ///
    /// [zero]: Polynomial::is_zero
    #[must_use]
    #[expect(clippy::shadow_unrelated)]
    pub fn clean_divide(self, divisor: Self) -> Polynomial<'static, BFieldElement> {
        let dividend = self;
        if divisor.degree() < Self::CLEAN_DIVIDE_CUTOFF_THRESHOLD {
            let (quotient, remainder) = dividend.divide(&divisor);
            debug_assert!(remainder.is_zero());
            return quotient;
        }

        // Incompleteness workaround: Manually check whether 0 is a root of the
        // divisor.
        // f(0) == 0 <=> f's constant term is 0
        let mut dividend_coefficients = dividend.coefficients.into_owned();
        let mut divisor_coefficients = divisor.coefficients.into_owned();
        if divisor_coefficients.first().is_some_and(Zero::is_zero) {
            // Clean division implies the dividend also has 0 as a root.
            assert!(dividend_coefficients[0].is_zero());
            dividend_coefficients.remove(0);
            divisor_coefficients.remove(0);
        }
        let dividend = Polynomial::new(dividend_coefficients);
        let divisor = Polynomial::new(divisor_coefficients);

        // Incompleteness workaround: Move both dividend and divisor to an
        // extension field.
        let offset = XFieldElement::from([0, 1, 0]);
        let mut dividend_coefficients = dividend.scale(offset).coefficients.into_owned();
        let mut divisor_coefficients = divisor.scale(offset).coefficients.into_owned();

        // See the comment in `fast_coset_evaluate` why this bound is necessary.
        let dividend_deg_plus_1 = usize::try_from(dividend.degree() + 1).unwrap();
        let order = dividend_deg_plus_1.next_power_of_two();

        dividend_coefficients.resize(order, XFieldElement::ZERO);
        divisor_coefficients.resize(order, XFieldElement::ZERO);

        ntt(&mut dividend_coefficients);
        ntt(&mut divisor_coefficients);

        let divisor_inverses = XFieldElement::batch_inversion(divisor_coefficients);
        let mut quotient_codeword = dividend_coefficients
            .into_iter()
            .zip(divisor_inverses)
            .map(|(l, r)| l * r)
            .collect_vec();

        intt(&mut quotient_codeword);
        let quotient = Polynomial::new(quotient_codeword);

        // If the division was clean, “unscaling” brings all coefficients back
        // to the base field.
        let Cow::Owned(coeffs) = quotient.scale(offset.inverse()).coefficients else {
            unreachable!();
        };

        Polynomial::new(coeffs.into_iter().map(|c| c.unlift().unwrap()).collect())
    }

    /// Parallel version of [`clean_divide`](Self::clean_divide).
    pub fn par_clean_divide(self, divisor: Self) -> Polynomial<'static, BFieldElement> {
        let dividend = self;
        if divisor.degree() < Self::CLEAN_DIVIDE_CUTOFF_THRESHOLD {
            let (quotient, remainder) = dividend.divide(&divisor);
            debug_assert!(remainder.is_zero());
            return quotient;
        }

        // See `clean_divide` for the reasoning behind the following two
        // workarounds.
        let mut dividend_coefficients = dividend.coefficients.into_owned();
        let mut divisor_coefficients = divisor.coefficients.into_owned();
        if divisor_coefficients.first().is_some_and(Zero::is_zero) {
            assert!(dividend_coefficients[0].is_zero());
            dividend_coefficients.remove(0);
            divisor_coefficients.remove(0);
        }
        let reduced_dividend = Polynomial::new(dividend_coefficients);
        let reduced_divisor = Polynomial::new(divisor_coefficients);

        let offset = XFieldElement::from([0, 1, 0]);
        let dividend_deg_plus_1 = usize::try_from(reduced_dividend.degree() + 1).unwrap();
        let order = dividend_deg_plus_1.next_power_of_two();

        let mut scaled_dividend = reduced_dividend.par_scaled_coefficients(offset, order);
        let mut scaled_divisor = reduced_divisor.par_scaled_coefficients(offset, order);
        scaled_dividend.resize(order, XFieldElement::ZERO);
        scaled_divisor.resize(order, XFieldElement::ZERO);

        rayon::join(
            || par_ntt(&mut scaled_dividend),
            || par_ntt(&mut scaled_divisor),
        );

        let divisor_inverses = XFieldElement::par_batch_inversion(scaled_divisor);
        let mut quotient_codeword = scaled_dividend
            .into_par_iter()
            .zip(divisor_inverses)
            .map(|(l, r)| l * r)
            .collect::<Vec<_>>();

        par_intt(&mut quotient_codeword);
        let quotient = Polynomial::new(quotient_codeword);

        let Cow::Owned(coeffs) = quotient.par_scale(offset.inverse()).coefficients else {
            unreachable!();
        };

        Polynomial::new(
            coeffs
                .into_par_iter()
                .map(|c| c.unlift().unwrap())
                .collect(),
        )
    }
}

impl<const N: usize, FF, E> From<[E; N]> for Polynomial<'static, FF>
where
    FF: FiniteField,
    E: Into<FF>,
{
    fn from(coefficients: [E; N]) -> Self {
        Self::new(coefficients.into_iter().map(|x| x.into()).collect())
    }
}

impl<'c, FF> From<&'c [FF]> for Polynomial<'c, FF>
where
    FF: FiniteField,
{
    fn from(coefficients: &'c [FF]) -> Self {
        Self::new_borrowed(coefficients)
    }
}

impl<FF, E> From<Vec<E>> for Polynomial<'static, FF>
where
    FF: FiniteField,
    E: Into<FF>,
{
    fn from(coefficients: Vec<E>) -> Self {
        Self::new(coefficients.into_iter().map(|c| c.into()).collect())
    }
}

impl From<XFieldElement> for Polynomial<'static, BFieldElement> {
    fn from(xfe: XFieldElement) -> Self {
        Self::new(xfe.coefficients.to_vec())
    }
}

impl<FF> Polynomial<'static, FF>
where
    FF: FiniteField,
{
    /// Create a new polynomial with the given coefficients. The first
    /// coefficient is the constant term, the last coefficient has the highest
    /// degree.
    ///
    /// See also [`Self::new_borrowed`].
    pub fn new(coefficients: Vec<FF>) -> Self {
        let coefficients = Cow::Owned(coefficients);
        Self { coefficients }
    }

    /// Create a new polynomial that corresponds to the single monomial `x^n`
    /// with coefficient 1, where `n` is the provided argument.
    pub fn x_to_the(n: usize) -> Self {
        let mut coefficients = vec![FF::ZERO; n + 1];
        coefficients[n] = FF::ONE;
        Self::new(coefficients)
    }

    /// Create a new polynomial that corresponds to the provided constant.
    ///
    /// In particular, the degree of the new polynomial is 0.
    pub fn from_constant(constant: FF) -> Self {
        Self::new(vec![constant])
    }

    /// Only `pub` to allow benchmarking; not considered part of the public API.
    #[doc(hidden)]
    pub fn naive_zerofier(domain: &[FF]) -> Self {
        domain
            .iter()
            .map(|&r| Self::new(vec![-r, FF::ONE]))
            .reduce(|accumulator, linear_poly| accumulator * linear_poly)
            .unwrap_or_else(Self::one)
    }
}

impl<'coeffs, FF> Polynomial<'coeffs, FF>
where
    FF: FiniteField,
{
    /// Like [`Self::new`], but without owning the coefficients.
    pub fn new_borrowed(coefficients: &'coeffs [FF]) -> Self {
        let coefficients = Cow::Borrowed(coefficients);
        Self { coefficients }
    }
}

impl<FF> Div<Polynomial<'_, FF>> for Polynomial<'_, FF>
where
    FF: FiniteField + 'static,
{
    type Output = Polynomial<'static, FF>;

    fn div(self, other: Polynomial<'_, FF>) -> Self::Output {
        let (quotient, _) = self.naive_divide(&other);
        quotient
    }
}

impl<FF> Rem<Polynomial<'_, FF>> for Polynomial<'_, FF>
where
    FF: FiniteField + 'static,
{
    type Output = Polynomial<'static, FF>;

    fn rem(self, other: Polynomial<'_, FF>) -> Self::Output {
        let (_, remainder) = self.naive_divide(&other);
        remainder
    }
}

impl<FF> Add<Polynomial<'_, FF>> for Polynomial<'_, FF>
where
    FF: FiniteField + 'static,
{
    type Output = Polynomial<'static, FF>;

    fn add(self, other: Polynomial<'_, FF>) -> Self::Output {
        let summed = self
            .coefficients
            .iter()
            .zip_longest(other.coefficients.iter())
            .map(|a| match a {
                EitherOrBoth::Both(&l, &r) => l + r,
                EitherOrBoth::Left(&c) | EitherOrBoth::Right(&c) => c,
            })
            .collect();

        Polynomial::new(summed)
    }
}

impl<FF: FiniteField> AddAssign<Polynomial<'_, FF>> for Polynomial<'_, FF> {
    fn add_assign(&mut self, rhs: Polynomial<'_, FF>) {
        let rhs_len = rhs.coefficients.len();
        let self_len = self.coefficients.len();
        let mut self_coefficients = std::mem::take(&mut self.coefficients).into_owned();

        for (l, &r) in self_coefficients.iter_mut().zip(rhs.coefficients.iter()) {
            *l += r;
        }

        if rhs_len > self_len {
            self_coefficients.extend(&rhs.coefficients[self_len..]);
        }

        self.coefficients = Cow::Owned(self_coefficients);
    }
}

impl<FF> Sub<Polynomial<'_, FF>> for Polynomial<'_, FF>
where
    FF: FiniteField + 'static,
{
    type Output = Polynomial<'static, FF>;

    fn sub(self, other: Polynomial<'_, FF>) -> Self::Output {
        let coefficients = self
            .coefficients
            .iter()
            .zip_longest(other.coefficients.iter())
            .map(|a| match a {
                EitherOrBoth::Both(&l, &r) => l - r,
                EitherOrBoth::Left(&l) => l,
                EitherOrBoth::Right(&r) => FF::ZERO - r,
            })
            .collect();

        Polynomial::new(coefficients)
    }
}

/// Use the barycentric Lagrange evaluation formula to evaluate a polynomial in
/// “value form”, also known as a codeword. This is generally more efficient
/// than first [interpolating](Polynomial::interpolate), then
/// [evaluating](Polynomial::evaluate).
///
/// [Credit] for (re)discovering this formula goes to Al-Kindi.
///
/// # Panics
///
/// Panics if the codeword is some length that is
/// - not a power of 2, or
/// - greater than (1 << 32).
///
/// [Credit]: https://github.com/0xPolygonMiden/miden-vm/issues/568
//
// The trait bounds of the form `A: Mul<B, Output = C>` allow using both
// base & extension field elements for both `A` and `B`, giving the greatest
// generality in using the function.
//
// It is possible to remove one of the generics by returning type
// `<<Coeff as Mul<Ind>>::Output as Mul<Ind>>::Output`
// (and changing a few trait bounds) but who would want to read that?
pub fn barycentric_evaluate<Ind, Coeff, Eval>(
    codeword: &[Coeff],
    indeterminate: Ind,
) -> <Eval as Mul<Ind>>::Output
where
    Ind: FiniteField + Mul<BFieldElement, Output = Ind> + Sub<BFieldElement, Output = Ind>,
    Coeff: FiniteField + Mul<Ind, Output = Eval>,
    Eval: FiniteField + Mul<Ind>,
{
    let root_order = codeword.len().try_into().unwrap();
    let generator = BFieldElement::primitive_root_of_unity(root_order).unwrap();
    let domain_iter = (0..root_order).scan(BFieldElement::ONE, |acc, _| {
        let to_yield = Some(*acc);
        *acc *= generator;
        to_yield
    });

    let domain_shift = domain_iter.clone().map(|d| indeterminate - d).collect();
    let domain_shift_inverses = Ind::batch_inversion(domain_shift);
    let domain_over_domain_shift = domain_iter
        .zip(domain_shift_inverses)
        .map(|(d, inv)| inv * d);
    let denominator = domain_over_domain_shift.clone().fold(Ind::ZERO, Ind::add);
    let numerator = domain_over_domain_shift
        .zip(codeword)
        .map(|(dsi, &abscis)| abscis * dsi)
        .fold(Eval::ZERO, Eval::add);

    numerator * denominator.inverse()
}

// It is impossible to
// `impl<FF: FiniteField> Mul<Polynomial<FF>> for FF`
// because of Rust's orphan rules [E0210]. Citing RFC 2451:
//
// > Rust’s orphan rule always permits an impl if either the trait or the type
// > being implemented are local to the current crate. Therefore, we can’t allow
// > `impl<T> ForeignTrait<LocalTypeCrateA> for T`, because it might conflict
// > with another crate writing `impl<T> ForeignTrait<T> for LocalTypeCrateB`,
// > which we will always permit.

impl<FF, FF2> Mul<Polynomial<'_, FF>> for BFieldElement
where
    FF: FiniteField + Mul<BFieldElement, Output = FF2>,
    FF2: 'static + FiniteField,
{
    type Output = Polynomial<'static, FF2>;

    fn mul(self, other: Polynomial<FF>) -> Self::Output {
        other.scalar_mul(self)
    }
}

impl<FF, FF2> Mul<Polynomial<'_, FF>> for XFieldElement
where
    FF: FiniteField + Mul<XFieldElement, Output = FF2>,
    FF2: 'static + FiniteField,
{
    type Output = Polynomial<'static, FF2>;

    fn mul(self, other: Polynomial<FF>) -> Self::Output {
        other.scalar_mul(self)
    }
}

impl<S, FF, FF2> Mul<S> for Polynomial<'_, FF>
where
    S: FiniteField,
    FF: FiniteField + Mul<S, Output = FF2>,
    FF2: 'static + FiniteField,
{
    type Output = Polynomial<'static, FF2>;

    fn mul(self, other: S) -> Self::Output {
        self.scalar_mul(other)
    }
}

impl<FF, FF2> Mul<Polynomial<'_, FF2>> for Polynomial<'_, FF>
where
    FF: FiniteField + Mul<FF2>,
    FF2: FiniteField,
    <FF as Mul<FF2>>::Output: 'static + FiniteField,
{
    type Output = Polynomial<'static, <FF as Mul<FF2>>::Output>;

    fn mul(self, other: Polynomial<'_, FF2>) -> Polynomial<'static, <FF as Mul<FF2>>::Output> {
        self.naive_multiply(&other)
    }
}

impl<FF> Neg for Polynomial<'_, FF>
where
    FF: FiniteField + 'static,
{
    type Output = Polynomial<'static, FF>;

    fn neg(mut self) -> Self::Output {
        self.scalar_mul_mut(-FF::ONE);

        // communicate the cloning that has already happened in scalar_mul_mut()
        self.into_owned()
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use num_traits::ConstZero;
    use proptest::collection::size_range;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_adapter::arb;

    use super::*;
    use crate::math::other::random_elements;
    use crate::prelude::*;
    use crate::tests::proptest;
    use crate::tests::test;

    /// A type alias exclusive to the test module.
    type BfePoly = Polynomial<'static, BFieldElement>;

    /// A type alias exclusive to the test module.
    type XfePoly = Polynomial<'static, XFieldElement>;

    impl proptest::arbitrary::Arbitrary for BfePoly {
        type Parameters = ();

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            arb().boxed()
        }

        type Strategy = BoxedStrategy<Self>;
    }

    impl proptest::arbitrary::Arbitrary for XfePoly {
        type Parameters = ();

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            arb().boxed()
        }

        type Strategy = BoxedStrategy<Self>;
    }

    #[macro_rules_attr::apply(test)]
    fn polynomial_can_be_debug_printed() {
        let polynomial = Polynomial::new(bfe_vec![1, 2, 3]);
        println!("{polynomial:?}");
    }

    #[macro_rules_attr::apply(proptest)]
    fn unequal_hash_implies_unequal_polynomials(poly_0: BfePoly, poly_1: BfePoly) {
        let hash = |poly: &Polynomial<_>| {
            let mut hasher = std::hash::DefaultHasher::new();
            poly.hash(&mut hasher);
            std::hash::Hasher::finish(&hasher)
        };

        // The `Hash` trait requires:
        // poly_0 == poly_1 => hash(poly_0) == hash(poly_1)
        //
        // By De-Morgan's law, this is equivalent to the more meaningful test:
        // hash(poly_0) != hash(poly_1) => poly_0 != poly_1
        if hash(&poly_0) != hash(&poly_1) {
            prop_assert_ne!(poly_0, poly_1);
        }
    }

    #[macro_rules_attr::apply(test)]
    fn polynomial_display_test() {
        fn polynomial<const N: usize>(coeffs: [u64; N]) -> BfePoly {
            Polynomial::new(coeffs.map(BFieldElement::new).to_vec())
        }

        assert_eq!("0", polynomial([]).to_string());
        assert_eq!("0", polynomial([0]).to_string());
        assert_eq!("0", polynomial([0, 0]).to_string());

        assert_eq!("1", polynomial([1]).to_string());
        assert_eq!("2", polynomial([2, 0]).to_string());
        assert_eq!("3", polynomial([3, 0, 0]).to_string());

        assert_eq!("x", polynomial([0, 1]).to_string());
        assert_eq!("2x", polynomial([0, 2]).to_string());
        assert_eq!("3x", polynomial([0, 3]).to_string());

        assert_eq!("5x + 2", polynomial([2, 5]).to_string());
        assert_eq!("9x + 7", polynomial([7, 9, 0, 0, 0]).to_string());

        assert_eq!("4x^4 + 3x^3", polynomial([0, 0, 0, 3, 4]).to_string());
        assert_eq!("2x^4 + 1", polynomial([1, 0, 0, 0, 2]).to_string());
    }

    #[macro_rules_attr::apply(proptest)]
    fn leading_coefficient_of_zero_polynomial_is_none(#[strategy(0usize..30)] num_zeros: usize) {
        let coefficients = vec![BFieldElement::ZERO; num_zeros];
        let polynomial = Polynomial::new(coefficients);
        prop_assert!(polynomial.leading_coefficient().is_none());
    }

    #[macro_rules_attr::apply(proptest)]
    fn leading_coefficient_of_non_zero_polynomial_is_some(
        polynomial: BfePoly,
        leading_coefficient: BFieldElement,
        #[strategy(0usize..30)] num_leading_zeros: usize,
    ) {
        let mut coefficients = polynomial.coefficients.into_owned();
        coefficients.push(leading_coefficient);
        coefficients.extend(vec![BFieldElement::ZERO; num_leading_zeros]);
        let polynomial_with_leading_zeros = Polynomial::new(coefficients);
        prop_assert_eq!(
            leading_coefficient,
            polynomial_with_leading_zeros.leading_coefficient().unwrap()
        );
    }

    #[macro_rules_attr::apply(test)]
    fn normalizing_canonical_zero_polynomial_has_no_effect() {
        let mut zero_polynomial = Polynomial::<BFieldElement>::zero();
        zero_polynomial.normalize();
        assert_eq!(Polynomial::zero(), zero_polynomial);
    }

    #[macro_rules_attr::apply(proptest)]
    fn spurious_leading_zeros_dont_affect_equality(
        polynomial: BfePoly,
        #[strategy(0usize..30)] num_leading_zeros: usize,
    ) {
        let mut coefficients = polynomial.clone().coefficients.into_owned();
        coefficients.extend(vec![BFieldElement::ZERO; num_leading_zeros]);
        let polynomial_with_leading_zeros = Polynomial::new(coefficients);

        prop_assert_eq!(polynomial, polynomial_with_leading_zeros);
    }

    #[macro_rules_attr::apply(proptest)]
    fn normalizing_removes_spurious_leading_zeros(
        polynomial: BfePoly,
        #[filter(!#leading_coefficient.is_zero())] leading_coefficient: BFieldElement,
        #[strategy(0usize..30)] num_leading_zeros: usize,
    ) {
        let mut coefficients = polynomial.clone().coefficients.into_owned();
        coefficients.push(leading_coefficient);
        coefficients.extend(vec![BFieldElement::ZERO; num_leading_zeros]);
        let mut polynomial_with_leading_zeros = Polynomial::new(coefficients);
        polynomial_with_leading_zeros.normalize();

        let num_inserted_coefficients = 1;
        let expected_num_coefficients = polynomial.coefficients.len() + num_inserted_coefficients;
        let num_coefficients = polynomial_with_leading_zeros.coefficients.len();

        prop_assert_eq!(expected_num_coefficients, num_coefficients);
    }

    #[macro_rules_attr::apply(test)]
    fn accessing_coefficients_of_empty_polynomial_gives_empty_slice() {
        let poly = BfePoly::new(vec![]);
        assert!(poly.coefficients().is_empty());
        assert!(poly.into_coefficients().is_empty());
    }

    #[macro_rules_attr::apply(proptest)]
    fn accessing_coefficients_of_polynomial_with_only_zero_coefficients_gives_empty_slice(
        #[strategy(0_usize..30)] num_zeros: usize,
    ) {
        let poly = Polynomial::new(vec![BFieldElement::ZERO; num_zeros]);
        prop_assert!(poly.coefficients().is_empty());
        prop_assert!(poly.into_coefficients().is_empty());
    }

    #[macro_rules_attr::apply(proptest)]
    fn accessing_the_coefficients_is_equivalent_to_normalizing_then_raw_access(
        mut coefficients: Vec<BFieldElement>,
        #[strategy(0_usize..30)] num_leading_zeros: usize,
    ) {
        coefficients.extend(vec![BFieldElement::ZERO; num_leading_zeros]);
        let mut polynomial = Polynomial::new(coefficients);

        let accessed_coefficients_borrow = polynomial.coefficients().to_vec();
        let accessed_coefficients_owned = polynomial.clone().into_coefficients();

        polynomial.normalize();
        let raw_coefficients = polynomial.coefficients.into_owned();

        prop_assert_eq!(&raw_coefficients, &accessed_coefficients_borrow);
        prop_assert_eq!(&raw_coefficients, &accessed_coefficients_owned);
    }

    #[macro_rules_attr::apply(test)]
    fn x_to_the_0_is_constant_1() {
        assert!(Polynomial::<BFieldElement>::x_to_the(0).is_one());
        assert!(Polynomial::<XFieldElement>::x_to_the(0).is_one());
    }

    #[macro_rules_attr::apply(test)]
    fn x_to_the_1_is_x() {
        assert!(Polynomial::<BFieldElement>::x_to_the(1).is_x());
        assert!(Polynomial::<XFieldElement>::x_to_the(1).is_x());
    }

    #[macro_rules_attr::apply(proptest)]
    fn x_to_the_n_to_the_m_is_homomorphic(
        #[strategy(0_usize..50)] n: usize,
        #[strategy(0_usize..50)] m: usize,
    ) {
        let to_the_n_times_m = Polynomial::<BFieldElement>::x_to_the(n * m);
        let to_the_n_then_to_the_m = Polynomial::x_to_the(n).pow(m as u32);
        prop_assert_eq!(to_the_n_times_m, to_the_n_then_to_the_m);
    }

    #[macro_rules_attr::apply(test)]
    fn scaling_a_polynomial_works_with_different_fields_as_the_offset() {
        let bfe_poly = Polynomial::new(bfe_vec![0, 1, 2]);
        let _ = bfe_poly.scale(bfe!(42));
        let _ = bfe_poly.scale(xfe!(42));

        let xfe_poly = Polynomial::new(xfe_vec![0, 1, 2]);
        let _ = xfe_poly.scale(bfe!(42));
        let _ = xfe_poly.scale(xfe!(42));
    }

    #[macro_rules_attr::apply(proptest)]
    fn polynomial_scaling_is_equivalent_in_extension_field(
        bfe_polynomial: BfePoly,
        alpha: BFieldElement,
    ) {
        let bfe_coefficients = bfe_polynomial.coefficients.iter();
        let xfe_coefficients = bfe_coefficients.map(|bfe| bfe.lift()).collect();
        let xfe_polynomial = Polynomial::<XFieldElement>::new(xfe_coefficients);

        let xfe_poly_bfe_scalar = xfe_polynomial.scale(alpha);
        let bfe_poly_xfe_scalar = bfe_polynomial.scale(alpha.lift());
        prop_assert_eq!(xfe_poly_bfe_scalar, bfe_poly_xfe_scalar);
    }

    #[macro_rules_attr::apply(proptest)]
    fn evaluating_scaled_polynomial_is_equivalent_to_evaluating_original_in_offset_point(
        polynomial: BfePoly,
        alpha: BFieldElement,
        x: BFieldElement,
    ) {
        let scaled_polynomial = polynomial.scale(alpha);
        prop_assert_eq!(
            polynomial.evaluate_in_same_field(alpha * x),
            scaled_polynomial.evaluate_in_same_field(x)
        );
    }

    #[macro_rules_attr::apply(proptest)]
    fn polynomial_multiplication_with_scalar_is_equivalent_for_the_two_methods(
        mut polynomial: BfePoly,
        scalar: BFieldElement,
    ) {
        let new_polynomial = polynomial.scalar_mul(scalar);
        polynomial.scalar_mul_mut(scalar);
        prop_assert_eq!(polynomial, new_polynomial);
    }

    #[macro_rules_attr::apply(proptest)]
    fn polynomial_multiplication_with_scalar_is_equivalent_for_all_mul_traits(
        polynomial: BfePoly,
        scalar: BFieldElement,
    ) {
        let bfe_rhs = polynomial.clone() * scalar;
        let xfe_rhs = polynomial.clone() * scalar.lift();
        let bfe_lhs = scalar * polynomial.clone();
        let xfe_lhs = scalar.lift() * polynomial;

        prop_assert_eq!(bfe_lhs.clone(), bfe_rhs);
        prop_assert_eq!(xfe_lhs.clone(), xfe_rhs);

        prop_assert_eq!(bfe_lhs * XFieldElement::ONE, xfe_lhs);
    }

    #[macro_rules_attr::apply(test)]
    fn polynomial_multiplication_with_scalar_works_for_various_types() {
        let bfe_poly = Polynomial::new(bfe_vec![0, 1, 2]);
        let _: Polynomial<BFieldElement> = bfe_poly.scalar_mul(bfe!(42));
        let _: Polynomial<XFieldElement> = bfe_poly.scalar_mul(xfe!(42));

        let xfe_poly = Polynomial::new(xfe_vec![0, 1, 2]);
        let _: Polynomial<XFieldElement> = xfe_poly.scalar_mul(bfe!(42));
        let _: Polynomial<XFieldElement> = xfe_poly.scalar_mul(xfe!(42));

        let mut bfe_poly = bfe_poly;
        bfe_poly.scalar_mul_mut(bfe!(42));

        let mut xfe_poly = xfe_poly;
        xfe_poly.scalar_mul_mut(bfe!(42));
        xfe_poly.scalar_mul_mut(xfe!(42));
    }

    #[macro_rules_attr::apply(proptest)]
    fn slow_lagrange_interpolation(
        polynomial: BfePoly,
        #[strategy(Just(#polynomial.coefficients.len().max(1)))] _min_points: usize,
        #[any(size_range(#_min_points..8 * #_min_points).lift())] points: Vec<BFieldElement>,
    ) {
        let evaluations = points
            .into_iter()
            .map(|x| (x, polynomial.evaluate(x)))
            .collect_vec();
        let interpolation_polynomial = Polynomial::lagrange_interpolate_zipped(&evaluations);
        prop_assert_eq!(polynomial, interpolation_polynomial);
    }

    #[macro_rules_attr::apply(proptest)]
    fn three_colinear_points_are_colinear(
        p0: (BFieldElement, BFieldElement),
        #[filter(#p0.0 != #p1.0)] p1: (BFieldElement, BFieldElement),
        #[filter(#p0.0 != #p2_x && #p1.0 != #p2_x)] p2_x: BFieldElement,
    ) {
        let line = Polynomial::lagrange_interpolate_zipped(&[p0, p1]);
        let p2 = (p2_x, line.evaluate(p2_x));
        prop_assert!(Polynomial::are_colinear(&[p0, p1, p2]));
    }

    #[macro_rules_attr::apply(proptest)]
    fn three_non_colinear_points_are_not_colinear(
        p0: (BFieldElement, BFieldElement),
        #[filter(#p0.0 != #p1.0)] p1: (BFieldElement, BFieldElement),
        #[filter(#p0.0 != #p2_x && #p1.0 != #p2_x)] p2_x: BFieldElement,
        #[filter(!#disturbance.is_zero())] disturbance: BFieldElement,
    ) {
        let line = Polynomial::lagrange_interpolate_zipped(&[p0, p1]);
        let p2 = (p2_x, line.evaluate_in_same_field(p2_x) + disturbance);
        prop_assert!(!Polynomial::are_colinear(&[p0, p1, p2]));
    }

    #[macro_rules_attr::apply(proptest)]
    fn colinearity_check_needs_at_least_three_points(
        p0: (BFieldElement, BFieldElement),
        #[filter(#p0.0 != #p1.0)] p1: (BFieldElement, BFieldElement),
    ) {
        prop_assert!(!Polynomial::<BFieldElement>::are_colinear(&[]));
        prop_assert!(!Polynomial::are_colinear(&[p0]));
        prop_assert!(!Polynomial::are_colinear(&[p0, p1]));
    }

    #[macro_rules_attr::apply(proptest)]
    fn colinearity_check_with_repeated_points_fails(
        p0: (BFieldElement, BFieldElement),
        #[filter(#p0.0 != #p1.0)] p1: (BFieldElement, BFieldElement),
    ) {
        prop_assert!(!Polynomial::are_colinear(&[p0, p1, p1]));
    }

    #[macro_rules_attr::apply(proptest)]
    fn colinear_points_are_colinear(
        p0: (BFieldElement, BFieldElement),
        #[filter(#p0.0 != #p1.0)] p1: (BFieldElement, BFieldElement),
        #[filter(!#additional_points_xs.contains(&#p0.0))]
        #[filter(!#additional_points_xs.contains(&#p1.0))]
        #[filter(#additional_points_xs.iter().all_unique())]
        #[any(size_range(1..100).lift())]
        additional_points_xs: Vec<BFieldElement>,
    ) {
        let line = Polynomial::lagrange_interpolate_zipped(&[p0, p1]);
        let additional_points = additional_points_xs
            .into_iter()
            .map(|x| (x, line.evaluate(x)))
            .collect_vec();
        let all_points = [p0, p1].into_iter().chain(additional_points).collect_vec();
        prop_assert!(Polynomial::are_colinear(&all_points));
    }

    #[macro_rules_attr::apply(test)]
    #[should_panic(expected = "Line must not be parallel to y-axis")]
    fn getting_point_on_invalid_line_fails() {
        let one = BFieldElement::ONE;
        let two = one + one;
        let three = two + one;
        Polynomial::<BFieldElement>::get_colinear_y((one, one), (one, three), two);
    }

    #[macro_rules_attr::apply(proptest)]
    fn point_on_line_and_colinear_point_are_identical(
        p0: (BFieldElement, BFieldElement),
        #[filter(#p0.0 != #p1.0)] p1: (BFieldElement, BFieldElement),
        x: BFieldElement,
    ) {
        let line = Polynomial::lagrange_interpolate_zipped(&[p0, p1]);
        let y = line.evaluate_in_same_field(x);
        let y_from_get_point_on_line = Polynomial::get_colinear_y(p0, p1, x);
        prop_assert_eq!(y, y_from_get_point_on_line);
    }

    #[macro_rules_attr::apply(proptest)]
    fn point_on_line_and_colinear_point_are_identical_in_extension_field(
        p0: (XFieldElement, XFieldElement),
        #[filter(#p0.0 != #p1.0)] p1: (XFieldElement, XFieldElement),
        x: XFieldElement,
    ) {
        let line = Polynomial::lagrange_interpolate_zipped(&[p0, p1]);
        let y = line.evaluate_in_same_field(x);
        let y_from_get_point_on_line = Polynomial::get_colinear_y(p0, p1, x);
        prop_assert_eq!(y, y_from_get_point_on_line);
    }

    #[macro_rules_attr::apply(proptest)]
    fn shifting_polynomial_coefficients_by_zero_is_the_same_as_not_shifting_it(poly: BfePoly) {
        prop_assert_eq!(poly.clone(), poly.shift_coefficients(0));
    }

    #[macro_rules_attr::apply(proptest)]
    fn shifting_polynomial_one_is_equivalent_to_raising_polynomial_x_to_the_power_of_the_shift(
        #[strategy(0usize..30)] shift: usize,
    ) {
        let shifted_one = Polynomial::one().shift_coefficients(shift);
        let x_to_the_shift = Polynomial::<BFieldElement>::from([0, 1]).pow(shift as u32);
        prop_assert_eq!(shifted_one, x_to_the_shift);
    }

    #[macro_rules_attr::apply(test)]
    fn polynomial_shift_test() {
        fn polynomial<const N: usize>(coeffs: [u64; N]) -> BfePoly {
            Polynomial::new(coeffs.map(BFieldElement::new).to_vec())
        }

        assert_eq!(
            polynomial([17, 14]),
            polynomial([17, 14]).shift_coefficients(0)
        );
        assert_eq!(
            polynomial([0, 17, 14]),
            polynomial([17, 14]).shift_coefficients(1)
        );
        assert_eq!(
            polynomial([0, 0, 0, 0, 17, 14]),
            polynomial([17, 14]).shift_coefficients(4)
        );

        let poly = polynomial([17, 14]);
        let poly_shift_0 = poly.clone().shift_coefficients(0);
        assert_eq!(polynomial([17, 14]), poly_shift_0);

        let poly_shift_1 = poly.clone().shift_coefficients(1);
        assert_eq!(polynomial([0, 17, 14]), poly_shift_1);

        let poly_shift_4 = poly.clone().shift_coefficients(4);
        assert_eq!(polynomial([0, 0, 0, 0, 17, 14]), poly_shift_4);
    }

    #[macro_rules_attr::apply(proptest)]
    fn shifting_a_polynomial_means_prepending_zeros_to_its_coefficients(
        poly: BfePoly,
        #[strategy(0usize..30)] shift: usize,
    ) {
        let shifted_poly = poly.clone().shift_coefficients(shift);
        let mut expected_coefficients = vec![BFieldElement::ZERO; shift];
        expected_coefficients.extend(poly.coefficients.to_vec());
        prop_assert_eq!(expected_coefficients, shifted_poly.coefficients.to_vec());
    }

    #[macro_rules_attr::apply(proptest)]
    fn any_polynomial_to_the_power_of_zero_is_one(poly: BfePoly) {
        let poly_to_the_zero = poly.pow(0);
        prop_assert_eq!(Polynomial::one(), poly_to_the_zero);
    }

    #[macro_rules_attr::apply(proptest)]
    fn any_polynomial_to_the_power_one_is_itself(poly: BfePoly) {
        let poly_to_the_one = poly.pow(1);
        prop_assert_eq!(poly, poly_to_the_one);
    }

    #[macro_rules_attr::apply(proptest)]
    fn polynomial_one_to_any_power_is_one(#[strategy(0u32..30)] exponent: u32) {
        let one_to_the_exponent = Polynomial::<BFieldElement>::one().pow(exponent);
        prop_assert_eq!(Polynomial::one(), one_to_the_exponent);
    }

    #[macro_rules_attr::apply(test)]
    fn pow_test() {
        fn polynomial<const N: usize>(coeffs: [u64; N]) -> BfePoly {
            Polynomial::new(coeffs.map(BFieldElement::new).to_vec())
        }

        let pol = polynomial([0, 14, 0, 4, 0, 8, 0, 3]);
        let pol_squared = polynomial([0, 0, 196, 0, 112, 0, 240, 0, 148, 0, 88, 0, 48, 0, 9]);
        let pol_cubed = polynomial([
            0, 0, 0, 2744, 0, 2352, 0, 5376, 0, 4516, 0, 4080, 0, 2928, 0, 1466, 0, 684, 0, 216, 0,
            27,
        ]);

        assert_eq!(pol_squared, pol.pow(2));
        assert_eq!(pol_cubed, pol.pow(3));

        let parabola = polynomial([5, 41, 19]);
        let parabola_squared = polynomial([25, 410, 1871, 1558, 361]);
        assert_eq!(parabola_squared, parabola.pow(2));
    }

    #[macro_rules_attr::apply(proptest)]
    fn pow_arbitrary_test(poly: BfePoly, #[strategy(0u32..15)] exponent: u32) {
        let actual = poly.pow(exponent);
        let fast_actual = poly.fast_pow(exponent);
        let mut expected = Polynomial::one();
        for _ in 0..exponent {
            expected = expected.clone() * poly.clone();
        }

        prop_assert_eq!(expected.clone(), actual);
        prop_assert_eq!(expected, fast_actual);
    }

    #[macro_rules_attr::apply(proptest)]
    fn polynomial_zero_is_neutral_element_for_addition(a: BfePoly) {
        prop_assert_eq!(a.clone() + Polynomial::zero(), a.clone());
        prop_assert_eq!(Polynomial::zero() + a.clone(), a);
    }

    #[macro_rules_attr::apply(proptest)]
    fn polynomial_one_is_neutral_element_for_multiplication(a: BfePoly) {
        prop_assert_eq!(a.clone() * Polynomial::<BFieldElement>::one(), a.clone());
        prop_assert_eq!(Polynomial::<BFieldElement>::one() * a.clone(), a);
    }

    #[macro_rules_attr::apply(proptest)]
    fn multiplication_by_zero_is_zero(a: BfePoly) {
        let zero = Polynomial::<BFieldElement>::zero();

        prop_assert_eq!(Polynomial::zero(), a.clone() * zero.clone());
        prop_assert_eq!(Polynomial::zero(), zero * a);
    }

    #[macro_rules_attr::apply(proptest)]
    fn polynomial_addition_is_commutative(a: BfePoly, b: BfePoly) {
        prop_assert_eq!(a.clone() + b.clone(), b + a);
    }

    #[macro_rules_attr::apply(proptest)]
    fn polynomial_multiplication_is_commutative(a: BfePoly, b: BfePoly) {
        prop_assert_eq!(a.clone() * b.clone(), b * a);
    }

    #[macro_rules_attr::apply(proptest)]
    fn polynomial_addition_is_associative(a: BfePoly, b: BfePoly, c: BfePoly) {
        prop_assert_eq!((a.clone() + b.clone()) + c.clone(), a + (b + c));
    }

    #[macro_rules_attr::apply(proptest)]
    fn polynomial_multiplication_is_associative(a: BfePoly, b: BfePoly, c: BfePoly) {
        prop_assert_eq!((a.clone() * b.clone()) * c.clone(), a * (b * c));
    }

    #[macro_rules_attr::apply(proptest)]
    fn polynomial_multiplication_is_distributive(a: BfePoly, b: BfePoly, c: BfePoly) {
        prop_assert_eq!(
            (a.clone() + b.clone()) * c.clone(),
            (a * c.clone()) + (b * c)
        );
    }

    #[macro_rules_attr::apply(proptest)]
    fn polynomial_subtraction_of_self_is_zero(a: BfePoly) {
        prop_assert_eq!(Polynomial::zero(), a.clone() - a);
    }

    #[macro_rules_attr::apply(proptest)]
    fn polynomial_division_by_self_is_one(#[filter(!#a.is_zero())] a: BfePoly) {
        prop_assert_eq!(Polynomial::one(), a.clone() / a);
    }

    #[macro_rules_attr::apply(proptest)]
    fn polynomial_division_removes_common_factors(a: BfePoly, #[filter(!#b.is_zero())] b: BfePoly) {
        prop_assert_eq!(a.clone(), a * b.clone() / b);
    }

    #[macro_rules_attr::apply(proptest)]
    fn polynomial_multiplication_raises_degree_at_maximum_to_sum_of_degrees(
        a: BfePoly,
        b: BfePoly,
    ) {
        let sum_of_degrees = (a.degree() + b.degree()).max(-1);
        prop_assert!((a * b).degree() <= sum_of_degrees);
    }

    #[macro_rules_attr::apply(test)]
    fn leading_zeros_dont_affect_polynomial_division() {
        // This test was used to catch a bug where the polynomial division was
        // wrong when the divisor has a leading zero coefficient, i.e. when it
        // was not normalized

        fn polynomial<const N: usize>(coeffs: [u64; N]) -> BfePoly {
            Polynomial::new(coeffs.map(BFieldElement::new).to_vec())
        }

        // x^3 - x + 1 / y = x
        let numerator = polynomial([1, BFieldElement::P - 1, 0, 1]);
        let numerator_with_leading_zero = polynomial([1, BFieldElement::P - 1, 0, 1, 0]);

        let divisor_normalized = polynomial([0, 1]);
        let divisor_not_normalized = polynomial([0, 1, 0]);
        let divisor_more_leading_zeros = polynomial([0, 1, 0, 0, 0, 0, 0, 0, 0]);

        let expected = polynomial([BFieldElement::P - 1, 0, 1]);

        // Verify that the divisor need not be normalized
        assert_eq!(expected, numerator.clone() / divisor_normalized.clone());
        assert_eq!(expected, numerator.clone() / divisor_not_normalized.clone());
        assert_eq!(expected, numerator / divisor_more_leading_zeros.clone());

        // Verify that numerator need not be normalized
        let res_numerator_not_normalized_0 =
            numerator_with_leading_zero.clone() / divisor_normalized;
        let res_numerator_not_normalized_1 =
            numerator_with_leading_zero.clone() / divisor_not_normalized;
        let res_numerator_not_normalized_2 =
            numerator_with_leading_zero / divisor_more_leading_zeros;
        assert_eq!(expected, res_numerator_not_normalized_0);
        assert_eq!(expected, res_numerator_not_normalized_1);
        assert_eq!(expected, res_numerator_not_normalized_2);
    }

    #[macro_rules_attr::apply(proptest)]
    fn leading_coefficient_of_truncated_polynomial_is_same_as_original_leading_coefficient(
        poly: BfePoly,
        #[strategy(..50_usize)] truncation_point: usize,
    ) {
        let Some(lc) = poly.leading_coefficient() else {
            let reason = "test is only sensible if polynomial has a leading coefficient";
            return Err(TestCaseError::Reject(reason.into()));
        };
        let truncated_poly = poly.truncate(truncation_point);
        let Some(trunc_lc) = truncated_poly.leading_coefficient() else {
            let reason = "test is only sensible if truncated polynomial has a leading coefficient";
            return Err(TestCaseError::Reject(reason.into()));
        };
        prop_assert_eq!(lc, trunc_lc);
    }

    #[macro_rules_attr::apply(proptest)]
    fn truncated_polynomial_is_of_degree_min_of_truncation_point_and_poly_degree(
        poly: BfePoly,
        #[strategy(..50_usize)] truncation_point: usize,
    ) {
        let expected_degree = poly.degree().min(truncation_point.try_into().unwrap());
        prop_assert_eq!(expected_degree, poly.truncate(truncation_point).degree());
    }

    #[macro_rules_attr::apply(proptest)]
    fn truncating_zero_polynomial_gives_zero_polynomial(
        #[strategy(..50_usize)] truncation_point: usize,
    ) {
        let poly = Polynomial::<BFieldElement>::zero().truncate(truncation_point);
        prop_assert!(poly.is_zero());
    }

    #[macro_rules_attr::apply(proptest)]
    fn truncation_negates_degree_shifting(
        #[strategy(0_usize..30)] shift: usize,
        #[strategy(..50_usize)] truncation_point: usize,
        #[filter(#poly.degree() >= #truncation_point as isize)] poly: BfePoly,
    ) {
        prop_assert_eq!(
            poly.truncate(truncation_point),
            poly.shift_coefficients(shift).truncate(truncation_point)
        );
    }

    #[macro_rules_attr::apply(proptest)]
    fn zero_polynomial_mod_any_power_of_x_is_zero_polynomial(power: usize) {
        let must_be_zero = Polynomial::<BFieldElement>::zero().mod_x_to_the_n(power);
        prop_assert!(must_be_zero.is_zero());
    }

    #[macro_rules_attr::apply(proptest)]
    fn polynomial_mod_some_power_of_x_results_in_polynomial_of_degree_one_less_than_power(
        #[filter(!#poly.is_zero())] poly: BfePoly,
        #[strategy(..=usize::try_from(#poly.degree()).unwrap())] power: usize,
    ) {
        let remainder = poly.mod_x_to_the_n(power);
        prop_assert_eq!(isize::try_from(power).unwrap() - 1, remainder.degree());
    }

    #[macro_rules_attr::apply(proptest)]
    fn polynomial_mod_some_power_of_x_shares_low_degree_terms_coefficients_with_original_polynomial(
        #[filter(!#poly.is_zero())] poly: BfePoly,
        power: usize,
    ) {
        let remainder = poly.mod_x_to_the_n(power);
        let min_num_coefficients = poly.coefficients.len().min(remainder.coefficients.len());
        prop_assert_eq!(
            &poly.coefficients[..min_num_coefficients],
            &remainder.coefficients[..min_num_coefficients]
        );
    }

    #[macro_rules_attr::apply(proptest)]
    fn fast_multiplication_by_zero_gives_zero(poly: BfePoly) {
        let product = poly.fast_multiply(&Polynomial::<BFieldElement>::zero());
        prop_assert_eq!(Polynomial::zero(), product);
    }

    #[macro_rules_attr::apply(proptest)]
    fn fast_multiplication_by_one_gives_self(poly: BfePoly) {
        let product = poly.fast_multiply(&Polynomial::<BFieldElement>::one());
        prop_assert_eq!(poly, product);
    }

    #[macro_rules_attr::apply(proptest)]
    fn fast_multiplication_is_commutative(a: BfePoly, b: BfePoly) {
        prop_assert_eq!(a.fast_multiply(&b), b.fast_multiply(&a));
    }

    #[macro_rules_attr::apply(proptest)]
    fn fast_multiplication_and_normal_multiplication_are_equivalent(a: BfePoly, b: BfePoly) {
        let product = a.fast_multiply(&b);
        prop_assert_eq!(a * b, product);
    }

    #[macro_rules_attr::apply(proptest)]
    fn batch_multiply_agrees_with_iterative_multiply(a: Vec<BfePoly>) {
        let mut acc = Polynomial::one();
        for factor in &a {
            acc = acc.multiply(factor);
        }
        prop_assert_eq!(acc, Polynomial::batch_multiply(&a));
    }

    #[macro_rules_attr::apply(proptest)]
    fn par_batch_multiply_agrees_with_batch_multiply(a: Vec<BfePoly>) {
        prop_assert_eq!(
            Polynomial::batch_multiply(&a),
            Polynomial::par_batch_multiply(&a)
        );
    }

    #[macro_rules_attr::apply(proptest(cases = 50))]
    fn naive_zerofier_and_fast_zerofier_are_identical(
        #[any(size_range(..Polynomial::<BFieldElement>::FAST_ZEROFIER_CUTOFF_THRESHOLD * 2).lift())]
        roots: Vec<BFieldElement>,
    ) {
        let naive_zerofier = Polynomial::naive_zerofier(&roots);
        let fast_zerofier = Polynomial::fast_zerofier(&roots);
        prop_assert_eq!(naive_zerofier, fast_zerofier);
    }

    #[macro_rules_attr::apply(proptest(cases = 50))]
    fn smart_zerofier_and_fast_zerofier_are_identical(
        #[any(size_range(..Polynomial::<BFieldElement>::FAST_ZEROFIER_CUTOFF_THRESHOLD * 2).lift())]
        roots: Vec<BFieldElement>,
    ) {
        let smart_zerofier = Polynomial::smart_zerofier(&roots);
        let fast_zerofier = Polynomial::fast_zerofier(&roots);
        prop_assert_eq!(smart_zerofier, fast_zerofier);
    }

    #[macro_rules_attr::apply(proptest(cases = 50))]
    fn zerofier_and_naive_zerofier_are_identical(
        #[any(size_range(..Polynomial::<BFieldElement>::FAST_ZEROFIER_CUTOFF_THRESHOLD * 2).lift())]
        roots: Vec<BFieldElement>,
    ) {
        let zerofier = Polynomial::zerofier(&roots);
        let naive_zerofier = Polynomial::naive_zerofier(&roots);
        prop_assert_eq!(zerofier, naive_zerofier);
    }

    #[macro_rules_attr::apply(proptest(cases = 50))]
    fn zerofier_is_zero_only_on_domain(
        #[any(size_range(..1024).lift())] domain: Vec<BFieldElement>,
        #[filter(#out_of_domain_points.iter().all(|p| !#domain.contains(p)))]
        out_of_domain_points: Vec<BFieldElement>,
    ) {
        let zerofier = Polynomial::zerofier(&domain);
        for point in domain {
            prop_assert_eq!(BFieldElement::ZERO, zerofier.evaluate(point));
        }
        for point in out_of_domain_points {
            prop_assert_ne!(BFieldElement::ZERO, zerofier.evaluate(point));
        }
    }

    #[macro_rules_attr::apply(proptest)]
    fn zerofier_has_leading_coefficient_one(domain: Vec<BFieldElement>) {
        let zerofier = Polynomial::zerofier(&domain);
        prop_assert_eq!(BFieldElement::ONE, zerofier.leading_coefficient().unwrap());
    }
    #[macro_rules_attr::apply(proptest)]
    fn par_zerofier_agrees_with_zerofier(domain: Vec<BFieldElement>) {
        prop_assert_eq!(
            Polynomial::zerofier(&domain),
            Polynomial::par_zerofier(&domain)
        );
    }

    #[macro_rules_attr::apply(test)]
    fn fast_evaluate_on_hardcoded_domain_and_polynomial() {
        let domain = bfe_array![6, 12];
        let x_to_the_5_plus_x_to_the_3 = Polynomial::new(bfe_vec![0, 0, 0, 1, 0, 1]);

        let manual_evaluations = domain.map(|x| x.mod_pow(5) + x.mod_pow(3)).to_vec();
        let fast_evaluations = x_to_the_5_plus_x_to_the_3.batch_evaluate(&domain);
        assert_eq!(manual_evaluations, fast_evaluations);
    }

    #[macro_rules_attr::apply(proptest)]
    fn slow_and_fast_polynomial_evaluation_are_equivalent(
        poly: BfePoly,
        #[any(size_range(..1024).lift())] domain: Vec<BFieldElement>,
    ) {
        let evaluations = domain
            .iter()
            .map(|&x| poly.evaluate_in_same_field(x))
            .collect_vec();
        let fast_evaluations = poly.batch_evaluate(&domain);
        prop_assert_eq!(evaluations, fast_evaluations);
    }

    #[macro_rules_attr::apply(test)]
    #[should_panic(expected = "zero points")]
    fn interpolation_through_no_points_is_impossible() {
        let _ = Polynomial::<BFieldElement>::interpolate(&[], &[]);
    }

    #[macro_rules_attr::apply(test)]
    #[should_panic(expected = "zero points")]
    fn lagrange_interpolation_through_no_points_is_impossible() {
        let _ = Polynomial::<BFieldElement>::lagrange_interpolate(&[], &[]);
    }

    #[macro_rules_attr::apply(test)]
    #[should_panic(expected = "zero points")]
    fn zipped_lagrange_interpolation_through_no_points_is_impossible() {
        let _ = Polynomial::<BFieldElement>::lagrange_interpolate_zipped(&[]);
    }

    #[macro_rules_attr::apply(test)]
    #[should_panic(expected = "zero points")]
    fn fast_interpolation_through_no_points_is_impossible() {
        let _ = Polynomial::<BFieldElement>::fast_interpolate(&[], &[]);
    }

    #[macro_rules_attr::apply(test)]
    #[should_panic(expected = "equal length")]
    fn interpolation_with_domain_size_different_from_number_of_points_is_impossible() {
        let domain = bfe_array![1, 2, 3];
        let points = bfe_array![1, 2];
        let _ = Polynomial::interpolate(&domain, &points);
    }

    #[macro_rules_attr::apply(test)]
    #[should_panic(expected = "Repeated")]
    fn zipped_lagrange_interpolate_using_repeated_domain_points_is_impossible() {
        let domain = bfe_array![1, 1, 2];
        let points = bfe_array![1, 2, 3];
        let zipped = domain.into_iter().zip(points).collect_vec();
        let _ = Polynomial::lagrange_interpolate_zipped(&zipped);
    }

    #[macro_rules_attr::apply(proptest)]
    fn interpolating_through_one_point_gives_constant_polynomial(
        x: BFieldElement,
        y: BFieldElement,
    ) {
        let interpolant = Polynomial::lagrange_interpolate(&[x], &[y]);
        let polynomial = Polynomial::from_constant(y);
        prop_assert_eq!(polynomial, interpolant);
    }

    #[macro_rules_attr::apply(proptest(cases = 10))]
    fn lagrange_and_fast_interpolation_are_identical(
        #[any(size_range(1..2048).lift())]
        #[filter(#domain.iter().all_unique())]
        domain: Vec<BFieldElement>,
        #[strategy(vec(arb(), #domain.len()))] values: Vec<BFieldElement>,
    ) {
        let lagrange_interpolant = Polynomial::lagrange_interpolate(&domain, &values);
        let fast_interpolant = Polynomial::fast_interpolate(&domain, &values);
        prop_assert_eq!(lagrange_interpolant, fast_interpolant);
    }

    #[macro_rules_attr::apply(proptest(cases = 10))]
    fn par_fast_interpolate_and_fast_interpolation_are_identical(
        #[any(size_range(1..2048).lift())]
        #[filter(#domain.iter().all_unique())]
        domain: Vec<BFieldElement>,
        #[strategy(vec(arb(), #domain.len()))] values: Vec<BFieldElement>,
    ) {
        let par_fast_interpolant = Polynomial::par_fast_interpolate(&domain, &values);
        let fast_interpolant = Polynomial::fast_interpolate(&domain, &values);
        prop_assert_eq!(par_fast_interpolant, fast_interpolant);
    }

    #[macro_rules_attr::apply(proptest(cases = 10))]
    fn interpolation_with_zerofier_tree_and_weights_agrees_with_fast_interpolation(
        #[any(size_range(1..2048).lift())]
        #[filter(#domain.iter().all_unique())]
        domain: Vec<BFieldElement>,
        #[strategy(vec(arb(), #domain.len()))] values: Vec<BFieldElement>,
    ) {
        let zerofier_tree = ZerofierTree::par_new_from_domain(&domain);
        let derivative = zerofier_tree.zerofier().formal_derivative();
        let weights = domain
            .iter()
            .zip(&values)
            .map(|(&x, &y)| y / derivative.evaluate::<_, BFieldElement>(x))
            .collect_vec();

        let weighted_interpolant =
            Polynomial::par_interpolate_with_zerofier_tree_and_weights(&zerofier_tree, &weights);
        let fast_interpolant = Polynomial::fast_interpolate(&domain, &values);
        prop_assert_eq!(fast_interpolant, weighted_interpolant);
    }

    #[macro_rules_attr::apply(proptest(cases = 20))]
    fn par_evaluate_agrees_with_evaluate(
        #[strategy(vec(arb(), 0..(1 << 13)))] coefficients: Vec<BFieldElement>,
        #[strategy(arb())] x: XFieldElement,
    ) {
        let polynomial = Polynomial::new(coefficients);
        let sequential = polynomial.evaluate::<_, XFieldElement>(x);
        let parallel = polynomial.par_evaluate::<_, XFieldElement>(x);
        prop_assert_eq!(sequential, parallel);
    }

    #[macro_rules_attr::apply(test)]
    fn fast_interpolation_through_a_single_point_succeeds() {
        let zero_arr = bfe_array![0];
        let _ = Polynomial::fast_interpolate(&zero_arr, &zero_arr);
    }

    #[macro_rules_attr::apply(proptest(cases = 20))]
    fn interpolation_then_evaluation_is_identity(
        #[any(size_range(1..2048).lift())]
        #[filter(#domain.iter().all_unique())]
        domain: Vec<BFieldElement>,
        #[strategy(vec(arb(), #domain.len()))] values: Vec<BFieldElement>,
    ) {
        let interpolant = Polynomial::fast_interpolate(&domain, &values);
        let evaluations = interpolant.batch_evaluate(&domain);
        prop_assert_eq!(values, evaluations);
    }

    #[macro_rules_attr::apply(proptest(cases = 1))]
    fn fast_batch_interpolation_is_equivalent_to_fast_interpolation(
        #[any(size_range(1..2048).lift())]
        #[filter(#domain.iter().all_unique())]
        domain: Vec<BFieldElement>,
        #[strategy(vec(vec(arb(), #domain.len()), 0..10))] value_vecs: Vec<Vec<BFieldElement>>,
    ) {
        let root_order = domain.len().next_power_of_two();
        let root_of_unity = BFieldElement::primitive_root_of_unity(root_order as u64).unwrap();

        let interpolants = value_vecs
            .iter()
            .map(|values| Polynomial::fast_interpolate(&domain, values))
            .collect_vec();

        let batched_interpolants =
            Polynomial::batch_fast_interpolate(&domain, &value_vecs, root_of_unity, root_order);
        prop_assert_eq!(interpolants, batched_interpolants);
    }

    fn coset_domain_of_size_from_generator_with_offset(
        size: usize,
        generator: BFieldElement,
        offset: BFieldElement,
    ) -> Vec<BFieldElement> {
        let mut domain = vec![offset];
        for _ in 1..size {
            domain.push(domain.last().copied().unwrap() * generator);
        }
        domain
    }

    #[macro_rules_attr::apply(proptest)]
    fn fast_coset_evaluation_and_fast_evaluation_on_coset_are_identical(
        polynomial: BfePoly,
        offset: BFieldElement,
        #[strategy(0..8usize)]
        #[map(|x: usize| 1 << x)]
        // due to current limitation in `Polynomial::fast_coset_evaluate`
        #[filter((#root_order as isize) > #polynomial.degree())]
        root_order: usize,
    ) {
        let root_of_unity = BFieldElement::primitive_root_of_unity(root_order as u64).unwrap();
        let domain =
            coset_domain_of_size_from_generator_with_offset(root_order, root_of_unity, offset);

        let fast_values = polynomial.batch_evaluate(&domain);
        let fast_coset_values = polynomial.fast_coset_evaluate(offset, root_order);
        prop_assert_eq!(fast_values, fast_coset_values);
    }

    #[macro_rules_attr::apply(proptest)]
    fn fast_coset_interpolation_and_and_fast_interpolation_on_coset_are_identical(
        #[filter(!#offset.is_zero())] offset: BFieldElement,
        #[strategy(1..8usize)]
        #[map(|x: usize| 1 << x)]
        root_order: usize,
        #[strategy(vec(arb(), #root_order))] values: Vec<BFieldElement>,
    ) {
        let root_of_unity = BFieldElement::primitive_root_of_unity(root_order as u64).unwrap();
        let domain =
            coset_domain_of_size_from_generator_with_offset(root_order, root_of_unity, offset);

        let fast_interpolant = Polynomial::fast_interpolate(&domain, &values);
        let fast_coset_interpolant = Polynomial::fast_coset_interpolate(offset, &values);
        prop_assert_eq!(fast_interpolant, fast_coset_interpolant);
    }

    #[macro_rules_attr::apply(proptest)]
    fn naive_division_gives_quotient_and_remainder_with_expected_properties(
        a: BfePoly,
        #[filter(!#b.is_zero())] b: BfePoly,
    ) {
        let (quot, rem) = a.naive_divide(&b);
        prop_assert!(rem.degree() < b.degree());
        prop_assert_eq!(a, quot * b + rem);
    }

    #[macro_rules_attr::apply(proptest)]
    fn clean_naive_division_gives_quotient_and_remainder_with_expected_properties(
        #[filter(!#a_roots.is_empty())] a_roots: Vec<BFieldElement>,
        #[strategy(vec(0..#a_roots.len(), 1..=#a_roots.len()))]
        #[filter(#b_root_indices.iter().all_unique())]
        b_root_indices: Vec<usize>,
    ) {
        let b_roots = b_root_indices.into_iter().map(|i| a_roots[i]).collect_vec();
        let a = Polynomial::zerofier(&a_roots);
        let b = Polynomial::zerofier(&b_roots);
        let (quot, rem) = a.naive_divide(&b);
        prop_assert!(rem.is_zero());
        prop_assert_eq!(a, quot * b);
    }

    #[macro_rules_attr::apply(proptest)]
    fn clean_division_agrees_with_divide_on_clean_division(
        #[strategy(arb())] a: BfePoly,
        #[strategy(arb())]
        #[filter(!#b.is_zero())]
        b: BfePoly,
    ) {
        let product = a.clone() * b.clone();
        let (naive_quotient, naive_remainder) = product.naive_divide(&b);
        let clean_quotient = product.clone().clean_divide(b.clone());
        let err = format!("{product} / {b} == {naive_quotient} != {clean_quotient}");
        prop_assert_eq!(naive_quotient, clean_quotient, "{}", err);
        prop_assert_eq!(Polynomial::<BFieldElement>::zero(), naive_remainder);
    }

    #[macro_rules_attr::apply(proptest)]
    fn clean_division_agrees_with_division_if_divisor_has_only_0_as_root(
        #[strategy(arb())] mut dividend_roots: Vec<BFieldElement>,
    ) {
        dividend_roots.push(bfe!(0));
        let dividend = Polynomial::zerofier(&dividend_roots);
        let divisor = Polynomial::zerofier(&[bfe!(0)]);

        let (naive_quotient, remainder) = dividend.naive_divide(&divisor);
        let clean_quotient = dividend.clean_divide(divisor);
        prop_assert_eq!(naive_quotient, clean_quotient);
        prop_assert_eq!(Polynomial::<BFieldElement>::zero(), remainder);
    }

    #[macro_rules_attr::apply(proptest)]
    fn clean_division_agrees_with_division_if_divisor_has_only_0_as_multiple_root(
        #[strategy(arb())] mut dividend_roots: Vec<BFieldElement>,
        #[strategy(0_usize..300)] num_roots: usize,
    ) {
        let multiple_roots = bfe_vec![0; num_roots];
        let divisor = Polynomial::zerofier(&multiple_roots);
        dividend_roots.extend(multiple_roots);
        let dividend = Polynomial::zerofier(&dividend_roots);

        let (naive_quotient, remainder) = dividend.naive_divide(&divisor);
        let clean_quotient = dividend.clean_divide(divisor);
        prop_assert_eq!(naive_quotient, clean_quotient);
        prop_assert_eq!(Polynomial::<BFieldElement>::zero(), remainder);
    }

    #[macro_rules_attr::apply(proptest)]
    fn clean_division_agrees_with_division_if_divisor_has_0_as_root(
        #[strategy(arb())] mut dividend_roots: Vec<BFieldElement>,
        #[strategy(vec(0..#dividend_roots.len(), 0..=#dividend_roots.len()))]
        #[filter(#divisor_root_indices.iter().all_unique())]
        divisor_root_indices: Vec<usize>,
    ) {
        // ensure clean division: make divisor's roots a subset of dividend's
        let mut divisor_roots = divisor_root_indices
            .into_iter()
            .map(|i| dividend_roots[i])
            .collect_vec();

        // ensure clean division: make 0 a root of both dividend and divisor
        dividend_roots.push(bfe!(0));
        divisor_roots.push(bfe!(0));

        let dividend = Polynomial::zerofier(&dividend_roots);
        let divisor = Polynomial::zerofier(&divisor_roots);
        let quotient = dividend.clone().clean_divide(divisor.clone());
        prop_assert_eq!(dividend / divisor, quotient);
    }

    #[macro_rules_attr::apply(proptest)]
    fn clean_division_agrees_with_division_if_divisor_has_0_through_9_as_roots(
        #[strategy(arb())] additional_dividend_roots: Vec<BFieldElement>,
    ) {
        let divisor_roots = (0..10).map(BFieldElement::new).collect_vec();
        let divisor = Polynomial::zerofier(&divisor_roots);
        let dividend_roots = [additional_dividend_roots, divisor_roots].concat();
        let dividend = Polynomial::zerofier(&dividend_roots);
        dbg!(dividend.to_string());
        dbg!(divisor.to_string());
        let quotient = dividend.clone().clean_divide(divisor.clone());
        prop_assert_eq!(dividend / divisor, quotient);
    }

    #[macro_rules_attr::apply(proptest)]
    fn clean_division_gives_quotient_and_remainder_with_expected_properties(
        #[filter(!#a_roots.is_empty())] a_roots: Vec<BFieldElement>,
        #[strategy(vec(0..#a_roots.len(), 1..=#a_roots.len()))]
        #[filter(#b_root_indices.iter().all_unique())]
        b_root_indices: Vec<usize>,
    ) {
        let b_roots = b_root_indices.into_iter().map(|i| a_roots[i]).collect_vec();
        let a = Polynomial::zerofier(&a_roots);
        let b = Polynomial::zerofier(&b_roots);
        let quotient = a.clone().clean_divide(b.clone());
        prop_assert_eq!(a.clone(), quotient * b.clone());

        let par_quotient = a.clone().par_clean_divide(b.clone());
        prop_assert_eq!(a, par_quotient * b);
    }

    #[macro_rules_attr::apply(proptest(cases = 8))]
    fn parallel_and_sequential_clean_division_agree_on_large_input(
        #[strategy(vec(arb(), 600..1200))] a_roots: Vec<BFieldElement>,
        #[strategy(1_usize..600)] num_b_roots: usize,
    ) {
        let a = Polynomial::zerofier(&a_roots);
        let b = Polynomial::zerofier(&a_roots[..num_b_roots]);
        prop_assert_eq!(a.clone().clean_divide(b.clone()), a.par_clean_divide(b));
    }

    #[macro_rules_attr::apply(proptest)]
    fn dividing_constant_polynomials_is_equivalent_to_dividing_constants(
        a: BFieldElement,
        #[filter(!#b.is_zero())] b: BFieldElement,
    ) {
        let a_poly = Polynomial::from_constant(a);
        let b_poly = Polynomial::from_constant(b);
        let expected_quotient = Polynomial::from_constant(a / b);
        prop_assert_eq!(expected_quotient, a_poly / b_poly);
    }

    #[macro_rules_attr::apply(proptest)]
    fn dividing_any_polynomial_by_a_constant_polynomial_results_in_remainder_zero(
        a: BfePoly,
        #[filter(!#b.is_zero())] b: BFieldElement,
    ) {
        let b_poly = Polynomial::from_constant(b);
        let (_, remainder) = a.naive_divide(&b_poly);
        prop_assert_eq!(Polynomial::zero(), remainder);
    }

    #[macro_rules_attr::apply(test)]
    fn polynomial_division_by_and_with_shah_polynomial() {
        fn polynomial<const N: usize>(coeffs: [u64; N]) -> BfePoly {
            Polynomial::new(coeffs.map(BFieldElement::new).to_vec())
        }

        let shah = XFieldElement::shah_polynomial();
        let x_to_the_3 = polynomial([1]).shift_coefficients(3);
        let (shah_div_x_to_the_3, shah_mod_x_to_the_3) = shah.naive_divide(&x_to_the_3);
        assert_eq!(polynomial([1]), shah_div_x_to_the_3);
        assert_eq!(polynomial([1, BFieldElement::P - 1]), shah_mod_x_to_the_3);

        let x_to_the_6 = polynomial([1]).shift_coefficients(6);
        let (x_to_the_6_div_shah, x_to_the_6_mod_shah) = x_to_the_6.naive_divide(&shah);

        // x^3 + x - 1
        let expected_quot = polynomial([BFieldElement::P - 1, 1, 0, 1]);
        assert_eq!(expected_quot, x_to_the_6_div_shah);

        // x^2 - 2x + 1
        let expected_rem = polynomial([1, BFieldElement::P - 2, 1]);
        assert_eq!(expected_rem, x_to_the_6_mod_shah);
    }

    #[macro_rules_attr::apply(test)]
    fn xgcd_does_not_panic_on_input_zero() {
        let zero = Polynomial::<BFieldElement>::zero;
        let (gcd, a, b) = Polynomial::xgcd(zero(), zero());
        assert_eq!(zero(), gcd);
        println!("a = {a}");
        println!("b = {b}");
    }

    #[macro_rules_attr::apply(proptest)]
    fn xgcd_b_field_pol_test(x: BfePoly, y: BfePoly) {
        let (gcd, a, b) = Polynomial::xgcd(x.clone(), y.clone());
        // Bezout relation
        prop_assert_eq!(gcd, a * x + b * y);
    }

    #[macro_rules_attr::apply(proptest)]
    fn xgcd_x_field_pol_test(x: XfePoly, y: XfePoly) {
        let (gcd, a, b) = Polynomial::xgcd(x.clone(), y.clone());
        // Bezout relation
        prop_assert_eq!(gcd, a * x + b * y);
    }

    #[macro_rules_attr::apply(proptest)]
    fn add_assign_is_equivalent_to_adding_and_assigning(a: BfePoly, b: BfePoly) {
        let mut c = a.clone();
        c += b.clone();
        prop_assert_eq!(a + b, c);
    }

    #[macro_rules_attr::apply(test)]
    fn only_monic_polynomial_of_degree_1_is_x() {
        fn polynomial<const N: usize>(coeffs: [u64; N]) -> BfePoly {
            Polynomial::new(coeffs.map(BFieldElement::new).to_vec())
        }

        assert!(polynomial([0, 1]).is_x());
        assert!(polynomial([0, 1, 0]).is_x());
        assert!(polynomial([0, 1, 0, 0]).is_x());

        assert!(!polynomial([]).is_x());
        assert!(!polynomial([0]).is_x());
        assert!(!polynomial([1]).is_x());
        assert!(!polynomial([1, 0]).is_x());
        assert!(!polynomial([0, 2]).is_x());
        assert!(!polynomial([0, 0, 1]).is_x());
    }

    #[macro_rules_attr::apply(test)]
    fn hardcoded_polynomial_squaring() {
        fn polynomial<const N: usize>(coeffs: [u64; N]) -> BfePoly {
            Polynomial::new(coeffs.map(BFieldElement::new).to_vec())
        }

        assert_eq!(Polynomial::zero(), polynomial([]).square());

        let x_plus_1 = polynomial([1, 1]);
        assert_eq!(polynomial([1, 2, 1]), x_plus_1.square());

        let x_to_the_15 = polynomial([1]).shift_coefficients(15);
        let x_to_the_30 = polynomial([1]).shift_coefficients(30);
        assert_eq!(x_to_the_30, x_to_the_15.square());

        let some_poly = polynomial([14, 1, 3, 4]);
        assert_eq!(
            polynomial([196, 28, 85, 118, 17, 24, 16]),
            some_poly.square()
        );
    }

    #[macro_rules_attr::apply(proptest)]
    fn polynomial_squaring_is_equivalent_to_multiplication_with_self(poly: BfePoly) {
        prop_assert_eq!(poly.clone() * poly.clone(), poly.square());
    }

    #[macro_rules_attr::apply(proptest)]
    fn slow_and_normal_squaring_are_equivalent(poly: BfePoly) {
        prop_assert_eq!(poly.slow_square(), poly.square());
    }

    #[macro_rules_attr::apply(proptest)]
    fn normal_and_fast_squaring_are_equivalent(poly: BfePoly) {
        prop_assert_eq!(poly.square(), poly.fast_square());
    }

    #[macro_rules_attr::apply(test)]
    fn constant_zero_eq_constant_zero() {
        let zero_polynomial1 = Polynomial::<BFieldElement>::zero();
        let zero_polynomial2 = Polynomial::<BFieldElement>::zero();

        assert_eq!(zero_polynomial1, zero_polynomial2)
    }

    #[macro_rules_attr::apply(test)]
    fn zero_polynomial_is_zero() {
        assert!(Polynomial::<BFieldElement>::zero().is_zero());
        assert!(Polynomial::<XFieldElement>::zero().is_zero());
    }

    #[macro_rules_attr::apply(proptest)]
    fn zero_polynomial_is_zero_independent_of_spurious_leading_zeros(
        #[strategy(..500usize)] num_zeros: usize,
    ) {
        let coefficients = vec![0; num_zeros];
        prop_assert_eq!(
            Polynomial::zero(),
            Polynomial::<BFieldElement>::from(coefficients)
        );
    }

    #[macro_rules_attr::apply(proptest)]
    fn no_constant_polynomial_with_non_zero_coefficient_is_zero(
        #[filter(!#constant.is_zero())] constant: BFieldElement,
    ) {
        let constant_polynomial = Polynomial::from_constant(constant);
        prop_assert!(!constant_polynomial.is_zero());
    }

    #[macro_rules_attr::apply(test)]
    fn constant_one_eq_constant_one() {
        let one_polynomial1 = Polynomial::<BFieldElement>::one();
        let one_polynomial2 = Polynomial::<BFieldElement>::one();

        assert_eq!(one_polynomial1, one_polynomial2)
    }

    #[macro_rules_attr::apply(test)]
    fn one_polynomial_is_one() {
        assert!(Polynomial::<BFieldElement>::one().is_one());
        assert!(Polynomial::<XFieldElement>::one().is_one());
    }

    #[macro_rules_attr::apply(proptest)]
    fn one_polynomial_is_one_independent_of_spurious_leading_zeros(
        #[strategy(..500usize)] num_leading_zeros: usize,
    ) {
        let spurious_leading_zeros = vec![0; num_leading_zeros];
        let mut coefficients = vec![1];
        coefficients.extend(spurious_leading_zeros);
        prop_assert_eq!(
            Polynomial::one(),
            Polynomial::<BFieldElement>::from(coefficients)
        );
    }

    #[macro_rules_attr::apply(proptest)]
    fn no_constant_polynomial_with_non_one_coefficient_is_one(
        #[filter(!#constant.is_one())] constant: BFieldElement,
    ) {
        let constant_polynomial = Polynomial::from_constant(constant);
        prop_assert!(!constant_polynomial.is_one());
    }

    #[macro_rules_attr::apply(test)]
    fn formal_derivative_of_zero_is_zero() {
        let bfe_0_poly = Polynomial::<BFieldElement>::zero();
        assert!(bfe_0_poly.formal_derivative().is_zero());

        let xfe_0_poly = Polynomial::<XFieldElement>::zero();
        assert!(xfe_0_poly.formal_derivative().is_zero());
    }

    #[macro_rules_attr::apply(proptest)]
    fn formal_derivative_of_constant_polynomial_is_zero(constant: BFieldElement) {
        let formal_derivative = Polynomial::from_constant(constant).formal_derivative();
        prop_assert!(formal_derivative.is_zero());
    }

    #[macro_rules_attr::apply(proptest)]
    fn formal_derivative_of_non_zero_polynomial_is_of_degree_one_less_than_the_polynomial(
        #[filter(!#poly.is_zero())] poly: BfePoly,
    ) {
        prop_assert_eq!(poly.degree() - 1, poly.formal_derivative().degree());
    }

    #[macro_rules_attr::apply(proptest)]
    fn formal_derivative_of_product_adheres_to_the_leibniz_product_rule(a: BfePoly, b: BfePoly) {
        let product_formal_derivative = (a.clone() * b.clone()).formal_derivative();
        let product_rule = a.formal_derivative() * b.clone() + a * b.formal_derivative();
        prop_assert_eq!(product_rule, product_formal_derivative);
    }

    #[macro_rules_attr::apply(test)]
    fn zero_is_zero() {
        let f = Polynomial::new(vec![BFieldElement::new(0)]);
        assert!(f.is_zero());
    }

    #[macro_rules_attr::apply(proptest)]
    fn formal_power_series_inverse_newton(
        #[strategy(2usize..20)] precision: usize,
        #[filter(!#f.coefficients.is_empty())]
        #[filter(!#f.coefficients[0].is_zero())]
        #[filter(#precision > 1 + #f.degree() as usize)]
        f: BfePoly,
    ) {
        let g = f.clone().formal_power_series_inverse_newton(precision);
        let mut coefficients = bfe_vec![0; precision + 1];
        coefficients[precision] = BFieldElement::ONE;
        let xn = Polynomial::new(coefficients);
        let (_quotient, remainder) = g.multiply(&f).divide(&xn);
        prop_assert!(remainder.is_one());
    }

    #[macro_rules_attr::apply(test)]
    fn formal_power_series_inverse_newton_concrete() {
        let f = Polynomial::new(vec![
            BFieldElement::new(3618372803227210457),
            BFieldElement::new(14620511201754172786),
            BFieldElement::new(2577803283145951105),
            BFieldElement::new(1723541458268087404),
            BFieldElement::new(4119508755381840018),
            BFieldElement::new(8592072587377832596),
            BFieldElement::new(236223201225),
        ]);
        let precision = 8;

        let g = f.clone().formal_power_series_inverse_newton(precision);
        let mut coefficients = vec![BFieldElement::ZERO; precision + 1];
        coefficients[precision] = BFieldElement::ONE;
        let xn = Polynomial::new(coefficients);
        let (_quotient, remainder) = g.multiply(&f).divide(&xn);
        assert!(remainder.is_one());
    }

    #[macro_rules_attr::apply(proptest)]
    fn structured_multiple_of_degree_is_multiple(
        #[strategy(2usize..100)] n: usize,
        #[filter(#coefficients.iter().any(|c|!c.is_zero()))]
        #[strategy(vec(arb(), 1..usize::min(30, #n)))]
        coefficients: Vec<BFieldElement>,
    ) {
        let polynomial = Polynomial::new(coefficients);
        let multiple = polynomial.structured_multiple_of_degree(n);
        let remainder = multiple.reduce_long_division(&polynomial);
        prop_assert!(remainder.is_zero());
    }

    #[macro_rules_attr::apply(proptest)]
    fn structured_multiple_of_degree_generates_structure(
        #[strategy(4usize..100)] n: usize,
        #[strategy(vec(arb(), 3..usize::min(30, #n)))] mut coefficients: Vec<BFieldElement>,
    ) {
        *coefficients.last_mut().unwrap() = BFieldElement::ONE;
        let polynomial = Polynomial::new(coefficients);
        let structured_multiple = polynomial.structured_multiple_of_degree(n);

        let xn =
            Polynomial::new([vec![BFieldElement::ZERO; n], vec![BFieldElement::ONE; 1]].concat());
        let remainder = structured_multiple.reduce_long_division(&xn);
        assert_eq!(
            (structured_multiple.clone() - remainder.clone())
                .reverse()
                .degree() as usize,
            0
        );
        assert_eq!(
            BFieldElement::ONE,
            *(structured_multiple.clone() - remainder)
                .coefficients
                .last()
                .unwrap()
        );
    }

    #[macro_rules_attr::apply(proptest)]
    fn structured_multiple_of_degree_has_given_degree(
        #[strategy(2usize..100)] n: usize,
        #[filter(#coefficients.iter().any(|c|!c.is_zero()))]
        #[strategy(vec(arb(), 1..usize::min(30, #n)))]
        coefficients: Vec<BFieldElement>,
    ) {
        let polynomial = Polynomial::new(coefficients);
        let multiple = polynomial.structured_multiple_of_degree(n);
        prop_assert_eq!(
            multiple.degree() as usize,
            n,
            "polynomial: {} whereas multiple {}",
            polynomial,
            multiple
        );
    }

    #[macro_rules_attr::apply(proptest)]
    fn reverse_polynomial_with_nonzero_constant_term_twice_gives_original_back(f: BfePoly) {
        let fx_plus_1 = f.shift_coefficients(1) + Polynomial::from_constant(bfe!(1));
        prop_assert_eq!(fx_plus_1.clone(), fx_plus_1.reverse().reverse());
    }

    #[macro_rules_attr::apply(proptest)]
    fn reverse_polynomial_with_zero_constant_term_twice_gives_shift_back(
        #[filter(!#f.is_zero())] f: BfePoly,
    ) {
        let fx_plus_1 = f.shift_coefficients(1);
        prop_assert_ne!(fx_plus_1.clone(), fx_plus_1.reverse().reverse());
        prop_assert_eq!(
            fx_plus_1.clone(),
            fx_plus_1.reverse().reverse().shift_coefficients(1)
        );
    }

    #[macro_rules_attr::apply(proptest)]
    fn reduce_by_ntt_friendly_modulus_and_reduce_long_division_agree(
        #[strategy(1usize..10)] m: usize,
        #[strategy(vec(arb(), #m))] b_coefficients: Vec<BFieldElement>,
        #[strategy(1usize..100)] _deg_a: usize,
        #[strategy(vec(arb(), #_deg_a + 1))] _a_coefficients: Vec<BFieldElement>,
        #[strategy(Just(Polynomial::new(#_a_coefficients)))] a: BfePoly,
    ) {
        let b = Polynomial::new(b_coefficients.clone());
        if b.is_zero() {
            return Err(TestCaseError::Reject("some reason".into()));
        }
        let n = (b_coefficients.len() + 1).next_power_of_two();
        let mut full_modulus_coefficients = b_coefficients.clone();
        full_modulus_coefficients.resize(n + 1, BFieldElement::from(0));
        *full_modulus_coefficients.last_mut().unwrap() = BFieldElement::from(1);
        let full_modulus = Polynomial::new(full_modulus_coefficients);

        let long_remainder = a.reduce_long_division(&full_modulus);

        let mut shift_ntt = b_coefficients.clone();
        shift_ntt.resize(n, BFieldElement::from(0));
        ntt(&mut shift_ntt);
        let structured_remainder = a.reduce_by_ntt_friendly_modulus(&shift_ntt, m);

        prop_assert_eq!(long_remainder, structured_remainder);
    }

    #[macro_rules_attr::apply(test)]
    fn reduce_by_ntt_friendly_modulus_and_reduce_agree_concrete() {
        let m = 1;
        let a_coefficients = bfe_vec![0, 0, 75944580];
        let a = Polynomial::new(a_coefficients);
        let b_coefficients = vec![BFieldElement::new(944892804900)];
        let n = (b_coefficients.len() + 1).next_power_of_two();
        let mut full_modulus_coefficients = b_coefficients.clone();
        full_modulus_coefficients.resize(n + 1, BFieldElement::from(0));
        *full_modulus_coefficients.last_mut().unwrap() = BFieldElement::from(1);
        let full_modulus = Polynomial::new(full_modulus_coefficients);

        let long_remainder = a.reduce_long_division(&full_modulus);

        let mut shift_ntt = b_coefficients.clone();
        shift_ntt.resize(n, BFieldElement::from(0));
        ntt(&mut shift_ntt);
        let structured_remainder = a.reduce_by_ntt_friendly_modulus(&shift_ntt, m);

        assert_eq!(
            long_remainder, structured_remainder,
            "full modulus: {full_modulus}",
        );
    }

    #[macro_rules_attr::apply(proptest)]
    fn reduce_fast_and_reduce_long_division_agree(
        numerator: BfePoly,
        #[filter(!#modulus.is_zero())] modulus: BfePoly,
    ) {
        prop_assert_eq!(
            numerator.fast_reduce(&modulus),
            numerator.reduce_long_division(&modulus)
        );
    }

    #[macro_rules_attr::apply(test)]
    fn reduce_and_fast_reduce_long_division_agree_on_fixed_input() {
        // The bug exhibited by this minimal failing test case has since been
        // fixed. The comments are kept as-is for historical accuracy and
        // didactics, and do not reflect an on-going bug hunt anymore.
        let mut failures = vec![];
        for i in 1..100 {
            // Is this setup convoluted? Maybe. It's the only way I've managed
            // to trigger the discrepancy so far. The historic context of
            // finding Bezout coefficients shimmers through. :)
            let roots = (0..i).map(BFieldElement::new).collect_vec();
            let dividend = Polynomial::zerofier(&roots).formal_derivative();

            // Fractions of 1/4th, 1/5th, 1/6th, and so on trigger the failure.
            // Fraction 1/5th seems to trigger both a failure for the smallest
            // `i` (10) and the most failures (90 out of 100). Fractions 1/2 or
            // 1/3rd don't trigger the failure.
            let divisor_roots = &roots[..roots.len() / 5];
            let divisor = Polynomial::zerofier(divisor_roots);

            let long_div_remainder = dividend.reduce_long_division(&divisor);
            let preprocessed_remainder = dividend.fast_reduce(&divisor);

            if long_div_remainder != preprocessed_remainder {
                failures.push(i);
            }
        }

        assert_eq!(0, failures.len(), "failures at indices: {failures:?}");
    }

    #[macro_rules_attr::apply(test)]
    fn reduce_long_division_and_fast_reduce_agree_simple_fixed() {
        let roots = (0..10).map(BFieldElement::new).collect_vec();
        let numerator = Polynomial::zerofier(&roots).formal_derivative();

        let divisor_roots = &roots[..roots.len() / 5];
        let denominator = Polynomial::zerofier(divisor_roots);

        let (quotient, remainder) = numerator.divide(&denominator);
        assert_eq!(
            numerator,
            denominator.clone() * quotient + remainder.clone()
        );

        let long_div_remainder = numerator.reduce_long_division(&denominator);
        assert_eq!(remainder, long_div_remainder);

        let preprocessed_remainder = numerator.fast_reduce(&denominator);

        assert_eq!(long_div_remainder, preprocessed_remainder);
    }

    #[macro_rules_attr::apply(proptest(cases = 100))]
    fn batch_evaluate_methods_are_equivalent(
        #[strategy(vec(arb(), (1<<10)..(1<<11)))] coefficients: Vec<BFieldElement>,
        #[strategy(vec(arb(), (1<<5)..(1<<7)))] domain: Vec<BFieldElement>,
    ) {
        let polynomial = Polynomial::new(coefficients);
        let evaluations_iterative = polynomial.iterative_batch_evaluate(&domain);
        let zerofier_tree = ZerofierTree::new_from_domain(&domain);
        let evaluations_fast = polynomial.divide_and_conquer_batch_evaluate(&zerofier_tree);
        let evaluations_reduce_then = polynomial.reduce_then_batch_evaluate(&domain);

        prop_assert_eq!(evaluations_iterative.clone(), evaluations_fast);
        prop_assert_eq!(evaluations_iterative, evaluations_reduce_then);
    }

    #[macro_rules_attr::apply(proptest)]
    fn reduce_agrees_with_division(a: BfePoly, #[filter(!#b.is_zero())] b: BfePoly) {
        prop_assert_eq!(a.divide(&b).1, a.reduce(&b));
    }

    #[macro_rules_attr::apply(proptest)]
    fn structured_multiple_of_monomial_term_is_actually_multiple_and_of_right_degree(
        #[strategy(1usize..1000)] degree: usize,
        #[filter(!#leading_coefficient.is_zero())] leading_coefficient: BFieldElement,
        #[strategy(#degree+1..#degree+200)] target_degree: usize,
    ) {
        let coefficients = [bfe_vec![0; degree], vec![leading_coefficient]].concat();
        let polynomial = Polynomial::new(coefficients);
        let multiple = polynomial.structured_multiple_of_degree(target_degree);
        prop_assert_eq!(Polynomial::zero(), multiple.reduce(&polynomial));
        prop_assert_eq!(multiple.degree() as usize, target_degree);
    }

    #[macro_rules_attr::apply(proptest)]
    fn monomial_term_divided_by_smaller_monomial_term_gives_clean_division(
        #[strategy(100usize..102)] high_degree: usize,
        #[filter(!#high_lc.is_zero())] high_lc: BFieldElement,
        #[strategy(83..#high_degree)] low_degree: usize,
        #[filter(!#low_lc.is_zero())] low_lc: BFieldElement,
    ) {
        let numerator = Polynomial::new([bfe_vec![0; high_degree], vec![high_lc]].concat());
        let denominator = Polynomial::new([bfe_vec![0; low_degree], vec![low_lc]].concat());
        let (quotient, remainder) = numerator.divide(&denominator);
        prop_assert_eq!(
            quotient
                .coefficients
                .iter()
                .filter(|c| !c.is_zero())
                .count(),
            1
        );
        prop_assert_eq!(Polynomial::zero(), remainder);
    }

    #[macro_rules_attr::apply(proptest)]
    fn fast_modular_coset_interpolate_agrees_with_interpolate_then_reduce_property(
        #[filter(!#modulus.is_zero())] modulus: BfePoly,
        #[strategy(0usize..10)] _logn: usize,
        #[strategy(Just(1 << #_logn))] n: usize,
        #[strategy(vec(arb(), #n))] values: Vec<BFieldElement>,
        #[strategy(arb())] offset: BFieldElement,
    ) {
        let omega = BFieldElement::primitive_root_of_unity(n as u64).unwrap();
        let domain = (0..n)
            .scan(offset, |acc: &mut BFieldElement, _| {
                let yld = *acc;
                *acc *= omega;
                Some(yld)
            })
            .collect_vec();
        prop_assert_eq!(
            Polynomial::fast_modular_coset_interpolate(&values, offset, &modulus),
            Polynomial::interpolate(&domain, &values).reduce(&modulus)
        )
    }

    #[macro_rules_attr::apply(test)]
    fn fast_modular_coset_interpolate_agrees_with_interpolate_then_reduce_concrete() {
        let logn = 8;
        let n = 1u64 << logn;
        let modulus = Polynomial::new(bfe_vec![2, 3, 1]);
        let values = (0..n).map(|i| BFieldElement::new(i / 5)).collect_vec();
        let offset = BFieldElement::new(7);

        let omega = BFieldElement::primitive_root_of_unity(n).unwrap();
        let mut domain = bfe_vec![0; n as usize];
        domain[0] = offset;
        for i in 1..n as usize {
            domain[i] = domain[i - 1] * omega;
        }
        assert_eq!(
            Polynomial::interpolate(&domain, &values).reduce(&modulus),
            Polynomial::fast_modular_coset_interpolate(&values, offset, &modulus),
        )
    }

    #[macro_rules_attr::apply(proptest(cases = 100))]
    fn coset_extrapolation_methods_agree_with_interpolate_then_evaluate(
        #[strategy(0usize..10)] _logn: usize,
        #[strategy(Just(1 << #_logn))] n: usize,
        #[strategy(vec(arb(), #n))] values: Vec<BFieldElement>,
        #[strategy(arb())] offset: BFieldElement,
        #[strategy(vec(arb(), 1..1000))] points: Vec<BFieldElement>,
    ) {
        let omega = BFieldElement::primitive_root_of_unity(n as u64).unwrap();
        let domain = (0..n)
            .scan(offset, |acc: &mut BFieldElement, _| {
                let yld = *acc;
                *acc *= omega;
                Some(yld)
            })
            .collect_vec();
        let fast_coset_extrapolation = Polynomial::fast_coset_extrapolate(offset, &values, &points);
        let naive_coset_extrapolation =
            Polynomial::naive_coset_extrapolate(offset, &values, &points);
        let interpolation_then_evaluation =
            Polynomial::interpolate(&domain, &values).batch_evaluate(&points);
        prop_assert_eq!(fast_coset_extrapolation.clone(), naive_coset_extrapolation);
        prop_assert_eq!(fast_coset_extrapolation, interpolation_then_evaluation);
    }

    #[macro_rules_attr::apply(proptest)]
    fn coset_extrapolate_and_batch_coset_extrapolate_agree(
        #[strategy(1usize..10)] _logn: usize,
        #[strategy(Just(1<<#_logn))] n: usize,
        #[strategy(0usize..5)] _m: usize,
        #[strategy(vec(arb(), #_m*#n))] codewords: Vec<BFieldElement>,
        #[strategy(vec(arb(), 0..20))] points: Vec<BFieldElement>,
    ) {
        let offset = BFieldElement::new(7);

        let one_by_one_dispatch = codewords
            .chunks(n)
            .flat_map(|chunk| Polynomial::coset_extrapolate(offset, chunk, &points))
            .collect_vec();
        let batched_dispatch = Polynomial::batch_coset_extrapolate(offset, n, &codewords, &points);
        let par_batched_dispatch =
            Polynomial::par_batch_coset_extrapolate(offset, n, &codewords, &points);
        prop_assert_eq!(one_by_one_dispatch.clone(), batched_dispatch);
        prop_assert_eq!(one_by_one_dispatch, par_batched_dispatch);

        let one_by_one_fast = codewords
            .chunks(n)
            .flat_map(|chunk| Polynomial::fast_coset_extrapolate(offset, chunk, &points))
            .collect_vec();
        let batched_fast = Polynomial::batch_fast_coset_extrapolate(offset, n, &codewords, &points);
        let par_batched_fast =
            Polynomial::par_batch_fast_coset_extrapolate(offset, n, &codewords, &points);
        prop_assert_eq!(one_by_one_fast.clone(), batched_fast);
        prop_assert_eq!(one_by_one_fast, par_batched_fast);

        let one_by_one_naive = codewords
            .chunks(n)
            .flat_map(|chunk| Polynomial::naive_coset_extrapolate(offset, chunk, &points))
            .collect_vec();
        let batched_naive =
            Polynomial::batch_naive_coset_extrapolate(offset, n, &codewords, &points);
        let par_batched_naive =
            Polynomial::par_batch_naive_coset_extrapolate(offset, n, &codewords, &points);
        prop_assert_eq!(one_by_one_naive.clone(), batched_naive);
        prop_assert_eq!(one_by_one_naive, par_batched_naive);
    }

    #[macro_rules_attr::apply(test)]
    fn fast_modular_coset_interpolate_thresholds_relate_properly() {
        type BfePoly = Polynomial<'static, BFieldElement>;

        let intt = BfePoly::FAST_MODULAR_COSET_INTERPOLATE_CUTOFF_THRESHOLD_PREFER_INTT;
        let lagrange = BfePoly::FAST_MODULAR_COSET_INTERPOLATE_CUTOFF_THRESHOLD_PREFER_LAGRANGE;
        assert!(intt > lagrange);
    }

    #[macro_rules_attr::apply(proptest)]
    fn interpolate_and_par_interpolate_agree(
        #[filter(!#points.is_empty())] points: Vec<BFieldElement>,
        #[strategy(vec(arb(), #points.len()))] domain: Vec<BFieldElement>,
    ) {
        let expected_interpolant = Polynomial::interpolate(&domain, &points);
        let observed_interpolant = Polynomial::par_interpolate(&domain, &points);
        prop_assert_eq!(expected_interpolant, observed_interpolant);
    }

    #[macro_rules_attr::apply(proptest)]
    fn batch_evaluate_agrees_with_par_batch_evalaute(
        polynomial: BfePoly,
        points: Vec<BFieldElement>,
    ) {
        prop_assert_eq!(
            polynomial.batch_evaluate(&points),
            polynomial.par_batch_evaluate(&points)
        );
    }

    #[macro_rules_attr::apply(proptest(cases = 32))]
    fn par_batch_evaluate_agrees_with_iterative_evaluation_on_many_points(
        #[strategy(vec(arb(), 0..(1 << 10)))] coefficients: Vec<BFieldElement>,
        #[strategy(vec(arb(), 1..(1 << 8)))] points: Vec<BFieldElement>,
    ) {
        let polynomial = Polynomial::new(coefficients);
        prop_assert_eq!(
            polynomial.iterative_batch_evaluate(&points),
            polynomial.par_batch_evaluate(&points)
        );
    }

    #[macro_rules_attr::apply(proptest)]
    fn formal_power_series_inverse_minimal(
        #[strategy(2usize..20)] precision: usize,
        #[filter(!#f.coefficients.is_empty())]
        #[filter(!#f.coefficients[0].is_zero())]
        #[filter(#precision > 1 + #f.degree() as usize)]
        f: BfePoly,
    ) {
        let g = f.formal_power_series_inverse_minimal(precision);
        let mut coefficients = vec![BFieldElement::ZERO; precision + 1];
        coefficients[precision] = BFieldElement::ONE;
        let xn = Polynomial::new(coefficients);
        let (_quotient, remainder) = g.multiply(&f).divide(&xn);

        // inverse in formal power series ring
        prop_assert!(remainder.is_one());

        // minimal?
        prop_assert!(g.degree() <= precision as isize);
    }

    #[macro_rules_attr::apply(proptest)]
    fn power_series_inverse_is_inverse_modulo_x_to_the_precision(
        #[filter(!#f.coefficients.is_empty())]
        #[filter(!#f.coefficients[0].is_zero())]
        f: BfePoly,
        #[strategy(1_usize..2000)] precision: usize,
    ) {
        let g = f.power_series_inverse(precision);
        prop_assert!(g.degree() < precision as isize);
        let product = f.multiply(&g).mod_x_to_the_n(precision);
        prop_assert!(product.is_one());
    }

    #[macro_rules_attr::apply(proptest)]
    fn reduction_with_reversed_inverse_agrees_with_reduction(
        #[filter(#a.degree() >= 0)] a: BfePoly,
        #[filter(#m.degree() >= 1)] m: BfePoly,
    ) {
        let quotient_degree = (a.degree() - m.degree()).max(0) as usize;
        let reversed_inverse = m.reverse().power_series_inverse(quotient_degree + 1);
        prop_assert_eq!(
            a.reduce(&m),
            a.reduce_with_reversed_inverse(&m, &reversed_inverse)
        );
    }

    #[macro_rules_attr::apply(proptest(cases = 16))]
    fn par_fast_interpolation_recovers_polynomial_through_many_points(
        #[strategy(1_usize..10)] _log_num_points: usize,
        #[strategy(vec(arb(), 1 << #_log_num_points))] points: Vec<BFieldElement>,
        #[strategy(vec(arb(), #points.len()))] values: Vec<BFieldElement>,
    ) {
        let points = points.into_iter().unique().collect_vec();
        let values = values[..points.len()].to_vec();
        let interpolant = Polynomial::par_fast_interpolate(&points, &values);
        prop_assert!(interpolant.degree() < points.len() as isize);
        prop_assert_eq!(values, interpolant.iterative_batch_evaluate(&points));
    }

    #[macro_rules_attr::apply(proptest(cases = 20))]
    fn polynomial_evaluation_and_barycentric_evaluation_are_equivalent(
        #[strategy(1_usize..8)] _log_num_coefficients: usize,
        #[strategy(1_usize..6)] log_expansion_factor: usize,
        #[strategy(vec(arb(), 1 << #_log_num_coefficients))] coefficients: Vec<XFieldElement>,
        #[strategy(arb())] indeterminate: XFieldElement,
    ) {
        let domain_len = coefficients.len() * (1 << log_expansion_factor);
        let domain_gen = BFieldElement::primitive_root_of_unity(domain_len.try_into()?).unwrap();
        let domain = (0..domain_len)
            .scan(XFieldElement::ONE, |acc, _| {
                let current = *acc;
                *acc *= domain_gen;
                Some(current)
            })
            .collect_vec();

        let polynomial = Polynomial::new(coefficients);
        let codeword = polynomial.batch_evaluate(&domain);
        prop_assert_eq!(
            polynomial.evaluate_in_same_field(indeterminate),
            barycentric_evaluate(&codeword, indeterminate)
        );
    }

    #[macro_rules_attr::apply(test)]
    fn regular_evaluation_works_with_various_types() {
        let bfe_poly = Polynomial::new(bfe_vec![1]);
        let _: BFieldElement = bfe_poly.evaluate(bfe!(0));
        let _: XFieldElement = bfe_poly.evaluate(bfe!(0));
        let _: XFieldElement = bfe_poly.evaluate(xfe!(0));

        let xfe_poly = Polynomial::new(xfe_vec![1]);
        let _: XFieldElement = xfe_poly.evaluate(bfe!(0));
        let _: XFieldElement = xfe_poly.evaluate(xfe!(0));
    }

    #[macro_rules_attr::apply(test)]
    fn barycentric_evaluation_works_with_many_types() {
        let bfe_codeword = bfe_array![1];
        let _ = barycentric_evaluate(&bfe_codeword, bfe!(0));
        let _ = barycentric_evaluate(&bfe_codeword, xfe!(0));

        let xfe_codeword = xfe_array![[1; 3]];
        let _ = barycentric_evaluate(&xfe_codeword, bfe!(0));
        let _ = barycentric_evaluate(&xfe_codeword, xfe!(0));
    }

    #[macro_rules_attr::apply(test)]
    fn various_multiplications_work_with_various_types() {
        let b = Polynomial::<BFieldElement>::zero;
        let x = Polynomial::<XFieldElement>::zero;

        let _ = b() * b();
        let _ = b() * x();
        let _ = x() * b();
        let _ = x() * x();

        let _ = b().multiply(&b());
        let _ = b().multiply(&x());
        let _ = x().multiply(&b());
        let _ = x().multiply(&x());

        let _ = b().naive_multiply(&b());
        let _ = b().naive_multiply(&x());
        let _ = x().naive_multiply(&b());
        let _ = x().naive_multiply(&x());

        let _ = b().fast_multiply(&b());
        let _ = b().fast_multiply(&x());
        let _ = x().fast_multiply(&b());
        let _ = x().fast_multiply(&x());
    }

    #[macro_rules_attr::apply(test)]
    fn evaluating_polynomial_with_borrowed_coefficients_leaves_coefficients_borrowed() {
        let coefficients = bfe_vec![4, 5, 6];
        let poly = Polynomial::new_borrowed(&coefficients);
        let _ = poly.evaluate_in_same_field(bfe!(0));
        let _ = poly.evaluate::<_, XFieldElement>(bfe!(0));
        let _ = poly.fast_coset_evaluate(bfe!(3), 128);

        let Cow::Borrowed(_) = poly.coefficients else {
            panic!("evaluating must not clone the coefficient vector")
        };

        // make sure the coefficients are still owned by this scope
        drop(coefficients);
    }
    #[macro_rules_attr::apply(test)]
    fn par_scale_agrees_with_scale() {
        // lengths around the chunk boundaries of `par_scale`
        for len in [
            0,
            1,
            7,
            (1 << 12) - 1,
            1 << 12,
            (1 << 12) + 1,
            (1 << 14) + 3,
        ] {
            let poly = Polynomial::<XFieldElement>::new(random_elements(len));
            let bfe_scalar: BFieldElement = random_elements(1)[0];
            let xfe_scalar: XFieldElement = random_elements(1)[0];
            assert_eq!(poly.scale(bfe_scalar), poly.par_scale(bfe_scalar), "{len}");
            assert_eq!(poly.scale(xfe_scalar), poly.par_scale(xfe_scalar), "{len}");

            let bfe_poly = Polynomial::<BFieldElement>::new(random_elements(len));
            assert_eq!(
                bfe_poly.scale(bfe_scalar),
                bfe_poly.par_scale(bfe_scalar),
                "{len}"
            );
            assert_eq!(
                bfe_poly.scale(xfe_scalar),
                bfe_poly.par_scale(xfe_scalar),
                "{len}"
            );
        }
    }

    #[macro_rules_attr::apply(test)]
    fn parallel_and_serial_coset_evaluation_and_interpolation_agree() {
        let offset = BFieldElement::generator();
        for log_order in [1, 5, 10, 17] {
            let order = 1_usize << log_order;
            let poly = Polynomial::<XFieldElement>::new(random_elements(order - 1));

            let serial = poly.fast_coset_evaluate(offset, order);
            let parallel = poly.par_fast_coset_evaluate(offset, order);
            assert_eq!(serial, parallel, "log_order: {log_order}");

            let serial_interpolant = Polynomial::fast_coset_interpolate(offset, &parallel);
            let parallel_interpolant = Polynomial::par_fast_coset_interpolate(offset, &parallel);
            assert_eq!(
                serial_interpolant, parallel_interpolant,
                "log_order: {log_order}"
            );
            assert_eq!(poly, parallel_interpolant, "log_order: {log_order}");
        }
    }
}
