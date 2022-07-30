use std::{cell::RefCell, rc::Rc};

#[derive(Debug, Clone)]
pub struct Node<T: Sized> {
    pub children: Vec<Rc<RefCell<Node<T>>>>,

    pub data: T,
}
