use std::collections::VecDeque;
use std::ops::MulAssign;

use num_traits::One;
use rayon::prelude::*;

use super::b_field_element::BFieldElement;
use super::polynomial::Polynomial;
use super::traits::FiniteField;

#[derive(Debug, Clone, PartialEq)]
pub struct Leaf<'c, FF: FiniteField + MulAssign<BFieldElement>> {
    pub(crate) points: Vec<FF>,
    zerofier: Polynomial<'c, FF>,
}

impl<FF> Leaf<'static, FF>
where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    pub fn new(points: Vec<FF>) -> Leaf<'static, FF> {
        let zerofier = Polynomial::zerofier(&points);
        Self { points, zerofier }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Branch<'c, FF: FiniteField + MulAssign<BFieldElement>> {
    zerofier: Polynomial<'c, FF>,
    pub(crate) left: ZerofierTree<'c, FF>,
    pub(crate) right: ZerofierTree<'c, FF>,

    /// The number of points in this subtree.
    pub(crate) num_points: usize,
}

impl<'c, FF> Branch<'c, FF>
where
    FF: FiniteField + MulAssign<BFieldElement> + 'static,
{
    pub fn new(left: ZerofierTree<'c, FF>, right: ZerofierTree<'c, FF>) -> Self {
        let zerofier = left.zerofier_view().multiply_monic(&right.zerofier_view());
        let num_points = left.num_points() + right.num_points();

        Self {
            zerofier,
            left,
            right,
            num_points,
        }
    }
}

/// A zerofier tree is a balanced binary tree of vanishing polynomials.
/// Conceptually, every leaf corresponds to a single point, and the value of
/// that leaf is the monic linear polynomial that evaluates to zero there and
/// no-where else. Every non-leaf node is the product of its two children.
/// In practice, it makes sense to truncate the tree depth, in which case every
/// leaf contains a chunk of points whose size is upper-bounded and more or less
/// equal to some constant threshold.
#[derive(Debug, Clone, PartialEq)]
pub enum ZerofierTree<'c, FF: FiniteField + MulAssign<BFieldElement>> {
    Leaf(Leaf<'c, FF>),
    Branch(Box<Branch<'c, FF>>),
    Padding,
}

impl<FF: FiniteField + MulAssign<BFieldElement>> ZerofierTree<'static, FF> {
    /// Regulates the depth at which the tree is truncated. Phrased differently,
    /// regulates the number of points contained by each leaf.
    const RECURSION_CUTOFF_THRESHOLD: usize = 16;

    /// Parallel version of [`new_from_domain`](Self::new_from_domain).
    pub fn par_new_from_domain(domain: &[FF]) -> Self {
        let mut nodes = domain
            .par_chunks(Self::RECURSION_CUTOFF_THRESHOLD)
            .map(|chunk| ZerofierTree::Leaf(Leaf::new(chunk.to_vec())))
            .collect::<Vec<_>>();
        nodes.resize(nodes.len().next_power_of_two(), ZerofierTree::Padding);
        while nodes.len() > 1 {
            nodes = nodes
                .into_par_iter()
                .chunks(2)
                .map(|pair| {
                    let [left, right] = <[_; 2]>::try_from(pair).unwrap();
                    if left == ZerofierTree::Padding {
                        ZerofierTree::Padding
                    } else {
                        ZerofierTree::Branch(Box::new(Branch::new(left, right)))
                    }
                })
                .collect();
        }
        nodes.pop().unwrap()
    }

    pub fn new_from_domain(domain: &[FF]) -> Self {
        let mut nodes = domain
            .chunks(Self::RECURSION_CUTOFF_THRESHOLD)
            .map(|chunk| {
                let leaf = Leaf::new(chunk.to_vec());
                ZerofierTree::Leaf(leaf)
            })
            .collect::<VecDeque<_>>();
        nodes.resize(nodes.len().next_power_of_two(), ZerofierTree::Padding);
        while nodes.len() > 1 {
            let right = nodes.pop_back().unwrap();
            let left = nodes.pop_back().unwrap();
            if left == ZerofierTree::Padding {
                nodes.push_front(ZerofierTree::Padding);
            } else {
                let new_node = Branch::new(left, right);
                nodes.push_front(ZerofierTree::Branch(Box::new(new_node)));
            }
        }
        nodes.pop_front().unwrap()
    }
}

impl<'c, FF> ZerofierTree<'c, FF>
where
    FF: FiniteField + MulAssign<BFieldElement> + 'static,
{
    pub fn zerofier(&self) -> Polynomial<'c, FF> {
        match self {
            ZerofierTree::Leaf(leaf) => leaf.zerofier.clone(),
            ZerofierTree::Branch(branch) => branch.zerofier.clone(),
            ZerofierTree::Padding => Polynomial::one(),
        }
    }

    /// Like [`zerofier`](Self::zerofier), but borrowing the coefficients
    /// where possible.
    pub(crate) fn zerofier_view(&self) -> Polynomial<'_, FF> {
        match self {
            ZerofierTree::Leaf(leaf) => Polynomial::new_borrowed(leaf.zerofier.coefficients()),
            ZerofierTree::Branch(branch) => {
                Polynomial::new_borrowed(branch.zerofier.coefficients())
            }
            ZerofierTree::Padding => Polynomial::one(),
        }
    }

    /// The number of points this (sub)tree was built from.
    pub fn num_points(&self) -> usize {
        match self {
            ZerofierTree::Leaf(leaf) => leaf.points.len(),
            ZerofierTree::Branch(branch) => branch.num_points,
            ZerofierTree::Padding => 0,
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use num_traits::ConstZero;
    use num_traits::Zero;
    use proptest::collection::vec;
    use proptest::prop_assert_eq;
    use proptest_arbitrary_adapter::arb;

    use crate::math::zerofier_tree::ZerofierTree;
    use crate::prelude::BFieldElement;
    use crate::prelude::Polynomial;
    use crate::tests::proptest;
    use crate::tests::test;

    #[macro_rules_attr::apply(test)]
    fn zerofier_tree_can_be_empty() {
        ZerofierTree::<BFieldElement>::new_from_domain(&[]);
    }

    #[macro_rules_attr::apply(proptest)]
    fn zerofier_tree_root_is_multiple_of_children(
        #[strategy(vec(arb(), 2*ZerofierTree::<BFieldElement>::RECURSION_CUTOFF_THRESHOLD))]
        points: Vec<BFieldElement>,
    ) {
        let zerofier_tree = ZerofierTree::new_from_domain(&points);
        let ZerofierTree::Branch(ref branch) = zerofier_tree else {
            panic!("not enough leafs");
        };
        prop_assert_eq!(
            Polynomial::zero(),
            zerofier_tree.zerofier().reduce(&branch.left.zerofier())
        );
        prop_assert_eq!(
            Polynomial::zero(),
            zerofier_tree.zerofier().reduce(&branch.right.zerofier())
        );
    }

    #[macro_rules_attr::apply(proptest)]
    fn zerofier_tree_root_has_right_degree(
        #[strategy(vec(arb(), 1..(1<<10)))] points: Vec<BFieldElement>,
    ) {
        let zerofier_tree = ZerofierTree::new_from_domain(&points);
        prop_assert_eq!(points.len(), zerofier_tree.zerofier().degree() as usize);
    }

    #[macro_rules_attr::apply(proptest)]
    fn zerofier_tree_root_zerofies(
        #[strategy(vec(arb(), 1..(1<<10)))] points: Vec<BFieldElement>,
        #[strategy(0usize..#points.len())] index: usize,
    ) {
        let zerofier_tree = ZerofierTree::new_from_domain(&points);
        prop_assert_eq!(
            BFieldElement::ZERO,
            zerofier_tree.zerofier().evaluate(points[index])
        );
    }

    #[macro_rules_attr::apply(proptest)]
    fn zerofier_tree_and_polynomial_agree_on_zerofiers(
        #[strategy(vec(arb(), 1..(1<<10)))] points: Vec<BFieldElement>,
    ) {
        let zerofier_tree = ZerofierTree::new_from_domain(&points);
        let polynomial_zerofier = Polynomial::zerofier(&points);
        prop_assert_eq!(polynomial_zerofier, zerofier_tree.zerofier());
    }

    #[macro_rules_attr::apply(proptest)]
    fn parallel_and_sequential_construction_agree(
        #[strategy(vec(arb(), 0..(1 << 8)))] points: Vec<BFieldElement>,
    ) {
        let sequential = ZerofierTree::new_from_domain(&points);
        let parallel = ZerofierTree::par_new_from_domain(&points);
        prop_assert_eq!(sequential, parallel);
    }

    #[macro_rules_attr::apply(proptest)]
    fn num_points_is_number_of_points(
        #[strategy(vec(arb(), 0..(1 << 8)))] points: Vec<BFieldElement>,
    ) {
        let zerofier_tree = ZerofierTree::new_from_domain(&points);
        prop_assert_eq!(points.len(), zerofier_tree.num_points());
    }
}
