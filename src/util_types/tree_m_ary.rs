use std::{cell::RefCell, rc::Rc};

#[derive(Debug, Clone)]
pub struct Node<T: Sized> {
    pub children: Vec<Rc<RefCell<Node<T>>>>,

    // Must have same length as number of variables in associated mpol
    pub data: T,
}
