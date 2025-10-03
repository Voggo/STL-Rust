use crate::ring_buffer::Step;
use dyn_clone::{DynClone, clone_trait_object};
use std::fmt::Display;
use std::time::Duration;

// Time interval type
#[derive(Debug, Copy, Clone)]
pub struct TimeInterval {
    pub start: Duration,
    pub end: Duration,
}

// stloperator trait
pub trait StlOperatorTrait<T: Clone, Y>: DynClone + Display {
    // added DynClone for cloning trait objects
    fn robustness(&mut self, step: &Step<T>) -> Option<Y>;
    // fn to_string(&self) -> String;
}

clone_trait_object!(<T: Clone, Y> StlOperatorTrait<T, Y>);

pub trait LogicalOperatorTrait<T: Clone, Y> {
    fn left(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Y>>;
    fn right(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Y>>;

    fn robustness_with<F>(&mut self, step: &Step<T>, f: F) -> Option<Y>
    where
        F: FnOnce(Y, Y) -> Y,
        Y: Clone,
        T: Clone,
        Self: StlOperatorTrait<T, Y>,
    {
        self.left()
            .as_mut()
            .robustness(step)
            .zip(self.right().as_mut().robustness(step))
            .map(|(l, r)| f(l, r))
    }
}

pub trait TemporalOperatorTrait<T: Clone, Y>: {}
