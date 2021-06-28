pub trait IdentityValues {
    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
    fn zero(&self) -> Self;
    fn one(&self) -> Self;
}
