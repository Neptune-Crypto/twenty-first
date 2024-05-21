use std::collections::VecDeque;
use std::ops::MulAssign;

use num_traits::One;

use super::b_field_element::BFieldElement;
use super::polynomial::Polynomial;
use super::traits::FiniteField;

#[derive(Debug, Clone, PartialEq)]
pub struct Leaf<FF: FiniteField + MulAssign<BFieldElement>> {
    pub points: Vec<FF>,
    zerofier: Polynomial<FF>,
}
impl<FF> Leaf<FF>
where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    pub fn new(points: Vec<FF>) -> Self {
        let zerofier = Polynomial::zerofier(&points);
        Self { points, zerofier }
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct Branch<FF: FiniteField + MulAssign<BFieldElement>> {
    zerofier: Polynomial<FF>,
    pub(crate) left: ZerofierTree<FF>,
    pub(crate) right: ZerofierTree<FF>,
}
impl<FF> Branch<FF>
where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    pub fn new(left: ZerofierTree<FF>, right: ZerofierTree<FF>) -> Self {
        let zerofier = left.zerofier().multiply(&right.zerofier());
        Self {
            zerofier,
            left,
            right,
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
pub enum ZerofierTree<FF: FiniteField + MulAssign<BFieldElement>> {
    Leaf(Leaf<FF>),
    Branch(Box<Branch<FF>>),
    Padding,
}

impl<FF: FiniteField + MulAssign<BFieldElement>> ZerofierTree<FF> {
    /// Regulates the depth at which the tree is truncated. Phrased differently,
    /// regulates the number of points contained by each leaf.
    const ZEROFIER_TREE_RECURSION_CUTOFF_THRESHOLD: usize = 16;

    pub fn new_from_domain(domain: &[FF]) -> Self {
        let mut nodes = domain
            .chunks(Self::ZEROFIER_TREE_RECURSION_CUTOFF_THRESHOLD)
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
        nodes.front().unwrap().clone()
    }

    pub fn zerofier(&self) -> Polynomial<FF> {
        match self {
            ZerofierTree::Leaf(leaf) => leaf.zerofier.clone(),
            ZerofierTree::Branch(branch) => branch.zerofier.clone(),
            ZerofierTree::Padding => Polynomial::<FF>::one(),
        }
    }
}

#[cfg(test)]
mod test {
    use num_traits::Zero;
    use proptest::collection::vec;
    use proptest::prop_assert_eq;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use crate::math::zerofier_tree::ZerofierTree;
    use crate::prelude::BFieldElement;
    use crate::prelude::Polynomial;

    #[proptest]
    fn zerofier_tree_root_is_multiple_of_children(
        #[strategy(vec(arb(), 2usize*ZerofierTree::<BFieldElement>::ZEROFIER_TREE_RECURSION_CUTOFF_THRESHOLD))]
        points: Vec<BFieldElement>,
    ) {
        let zerofier_tree = ZerofierTree::new_from_domain(&points);
        let (left, right) = match &zerofier_tree {
            ZerofierTree::Branch(branch) => (&branch.left, &branch.right),
            _ => panic!("not enough leafs"),
        };
        prop_assert_eq!(
            Polynomial::zero(),
            zerofier_tree.zerofier().reduce(&left.zerofier())
        );
        prop_assert_eq!(
            Polynomial::zero(),
            zerofier_tree.zerofier().reduce(&right.zerofier())
        );
    }

    #[proptest]
    fn zerofier_tree_root_has_right_degree(
        #[strategy(vec(arb(), 1usize..(1<<10)))] points: Vec<BFieldElement>,
    ) {
        let zerofier_tree = ZerofierTree::new_from_domain(&points);
        prop_assert_eq!(points.len(), zerofier_tree.zerofier().degree() as usize);
    }

    #[proptest]
    fn zerofier_tree_root_zerofies(
        #[strategy(vec(arb(), 1usize..(1<<10)))] points: Vec<BFieldElement>,
        #[strategy(0usize..#points.len())] index: usize,
    ) {
        let zerofier_tree = ZerofierTree::new_from_domain(&points);
        prop_assert_eq!(
            BFieldElement::zero(),
            zerofier_tree.zerofier().evaluate(points[index])
        );
    }
}
