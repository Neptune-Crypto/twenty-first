use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};

use itertools::Itertools;
use num_traits::Zero;

const CUTOFF: usize = 1;

fn schoolbook<T: Clone + Zero + Add<Output = T> + Mul<Output = T>>(
    a: &[T],
    b: &[T],
    n: usize,
) -> Vec<T> {
    let mut res = vec![T::zero(); 2 * n - 1];
    for i in 0..n {
        for j in 0..n {
            res[i + j] = res[i + j].clone() + a[i].clone() * b[j].clone();
        }
    }
    res
}

fn karatsuba<T: Clone + Zero + Add<Output = T> + Sub<Output = T> + Mul<Output = T>>(
    a: &[T],
    b: &[T],
    n: usize,
) -> Vec<T> {
    debug_assert_eq!(n & (n - 1), 0);
    if n <= CUTOFF {
        return schoolbook(a, b, n);
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

    let los = karatsuba(alo, blo, half);
    let his = karatsuba(ahi, bhi, half);
    let sus = karatsuba(&asu, &bsu, half);

    let mut c = vec![T::zero(); 2 * n - 1];
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
    T: Clone + Debug + Zero + Sub<Output = T> + Add<Output = T> + Mul<Output = T>,
>(
    a: &[T],
    b: &[T],
    n: usize,
) -> Vec<T> {
    let prod = karatsuba(a, b, n);
    let mut res = vec![T::zero(); n];
    for i in 0..n - 1 {
        res[i] = prod[i].clone() - prod[n + i].clone();
    }
    res[n - 1] = prod[n - 1].clone();
    res
}

fn quadratic_negacyclic_mul<
    T: Clone + Debug + Zero + Sub<Output = T> + Add<Output = T> + Mul<Output = T>,
>(
    a: &[T],
    b: &[T],
    n: usize,
) -> Vec<T> {
    debug_assert_eq!(a.len(), n);
    debug_assert_eq!(b.len(), n);
    if n == 1 {
        return vec![a[0].clone() * b[0].clone()];
    }
    let mut res = vec![T::zero(); n];
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

fn quadratic_cyclic_mul<
    T: Clone + Debug + Zero + Sub<Output = T> + Add<Output = T> + Mul<Output = T>,
>(
    a: &[T],
    b: &[T],
    n: usize,
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
    let mut res = vec![T::zero(); n];
    for i in 0..n {
        for j in 0..n {
            let product = b[j].clone() * a_row[(n - i + j) % n].clone();
            res[i] = res[i].clone() + product;
        }
    }
    res
}

fn karatsuba_cyclic_mul<T: Clone + Zero + Sub<Output = T> + Add<Output = T> + Mul<Output = T>>(
    a: &[T],
    b: &[T],
    n: usize,
) -> Vec<T> {
    let prod = karatsuba(a, b, n);
    let mut res = vec![T::zero(); n];
    for i in 0..n - 1 {
        res[i] = prod[i].clone() + prod[n + i].clone();
    }
    res[n - 1] = prod[n - 1].clone();
    res
}

fn recursive_cyclic_mul<
    T: Clone + Zero + Debug + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
>(
    a: &[T],
    b: &[T],
    n: usize,
    two: T,
) -> Vec<T> {
    debug_assert_eq!(n & (n - 1), 0);
    if n <= CUTOFF {
        let mut res = quadratic_cyclic_mul(a, b, n);
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

    let sums = recursive_cyclic_mul(&asum, &bsum, half, two.clone());
    println!("asum: {asum:?}");
    println!("bsum: {bsum:?}");
    println!("sums: {sums:?}");
    let mut expc = quadratic_cyclic_mul(&asum, &bsum, half);
    let mut nn = half;
    while nn > 1 {
        expc = expc.into_iter().map(|i| i * two.clone()).collect_vec();
        nn >>= 1;
    }
    println!("expc: {:?}", expc);

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

    let mut diffs = karatsuba_negacyclic_mul(&adiff, &bdiff, half);
    let mut expc_ = quadratic_negacyclic_mul(&adiff, &bdiff, half);
    let mut nnn = half;
    while nnn > 1 {
        diffs = diffs.into_iter().map(|i| i * two.clone()).collect_vec();
        expc_ = expc_.into_iter().map(|i| i * two.clone()).collect_vec();
        nnn >>= 1;
    }
    println!("adiff: {adiff:?}");
    println!("bdiff: {bdiff:?}");
    println!("diffs: {diffs:?}");
    println!("expc: {:?}", expc_);

    let mut res = vec![T::zero(); n];
    for i in 0..half {
        res[i] = (sums[i].clone() + diffs[i].clone());
        res[i + half] = (sums[i].clone() - diffs[i].clone());
    }

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
            .map(|_| (rng.next_u32() % (1 << 20)) as i64)
            .collect_vec();
        let b = (0..n)
            .map(|_| (rng.next_u32() % (1 << 20)) as i64)
            .collect_vec();

        let c_schoolbook = schoolbook(&a, &b, n);
        let c_karatsuba = karatsuba(&a, &b, n);

        assert_eq!(c_schoolbook, c_karatsuba);

        println!("{}", c_schoolbook.iter().map(|x| x.to_string()).join(","));
        println!("{}", c_karatsuba.iter().map(|x| x.to_string()).join(","));
    }

    #[test]
    fn test_negacyclic_mul() {
        let mut rng = thread_rng();
        let n = 8;
        let a = (0..n)
            .map(|_| (rng.next_u32() % (1 << 20)) as i64)
            .collect_vec();
        let b = (0..n)
            .map(|_| (rng.next_u32() % (1 << 20)) as i64)
            .collect_vec();

        println!("a = {a:?}");
        println!("b = {b:?}");

        let c_quadratic = quadratic_negacyclic_mul(&a, &b, n);
        let c_karatsuba = karatsuba_negacyclic_mul(&a, &b, n);

        assert_eq!(c_quadratic, c_karatsuba);

        println!("{}", c_quadratic.iter().map(|x| x.to_string()).join(","));
        println!("{}", c_karatsuba.iter().map(|x| x.to_string()).join(","));
    }

    #[test]
    fn test_recursive_cyclic_mul() {
        let mut rng = thread_rng();
        let n = 16;
        let a = (0..n)
            .map(|_| (rng.next_u32() % (1 << 20)) as i64)
            .collect_vec();
        let b = (0..n)
            .map(|_| (rng.next_u32() % (1 << 20)) as i64)
            .collect_vec();

        println!("a = {a:?}");
        println!("b = {b:?}");

        let c_quadratic = quadratic_cyclic_mul(&a, &b, n);
        let c_karatsuba = karatsuba_cyclic_mul(&a, &b, n);
        let c_recursive = recursive_cyclic_mul(&a, &b, n, 2)
            .iter()
            .map(|i| i / n as i64)
            .collect_vec();

        assert_eq!(c_quadratic, c_karatsuba);
        assert_eq!(c_karatsuba, c_recursive);
        assert_eq!(c_quadratic, c_recursive);

        println!("{}", c_quadratic.iter().map(|x| x.to_string()).join(","));
        println!("{}", c_karatsuba.iter().map(|x| x.to_string()).join(","));
        println!("{}", c_recursive.iter().map(|x| x.to_string()).join(","));
    }
}
