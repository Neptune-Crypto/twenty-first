pub trait IdentityValues {
    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
    fn ring_zero(&self) -> Self;
    fn ring_one(&self) -> Self;
}
