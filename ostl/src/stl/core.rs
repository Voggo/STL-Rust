use std::time::Duration;
use crate::ring_buffer::Step;
use dyn_clone::{DynClone, clone_trait_object};

// Time interval type
#[derive(Debug, Copy, Clone)]
pub struct TimeInterval {
    pub start: Duration,
    pub end: Duration,
}

// stloperator trait
pub trait StlOperatorTrait<T: Clone>: DynClone { // added DynClone for cloning trait objects
    fn robustness(&mut self, step: &Step<T>) -> Option<f64>;
    fn to_string(&self) -> String;
} 

clone_trait_object!(<T: Clone> StlOperatorTrait<T>);