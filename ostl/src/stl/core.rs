use crate::ring_buffer::Step;
use dyn_clone::{DynClone, clone_trait_object};
use std::collections::HashSet;
use std::fmt::Display;
use std::time::Duration;

// Time interval type
#[derive(Debug, Copy, Clone)]
pub struct TimeInterval {
    pub start: Duration,
    pub end: Duration,
}

// stloperator trait
// DynClone for cloning trait objects
pub trait StlOperatorTrait<T: Clone>: DynClone + Display {
    type Output;

    fn robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>>;
    fn get_max_lookahead(&self) -> Duration;
}

clone_trait_object!(<T: Clone, Y> StlOperatorTrait<T, Output = Y>);

pub trait SignalIdentifier {
    fn get_signal_identifiers(&mut self) -> HashSet<&'static str>;
}

pub trait StlOperatorAndSignalIdentifier<T: Clone, Y>:
    StlOperatorTrait<T, Output = Y> + SignalIdentifier
{
}

impl<C, Y, U> StlOperatorAndSignalIdentifier<C, Y> for U
where
    C: Clone,
    U: StlOperatorTrait<C, Output = Y> + SignalIdentifier,
{
}

clone_trait_object!(<T: Clone, Y> StlOperatorAndSignalIdentifier<T, Y>);

// should maybe just use refs for the operations
pub trait RobustnessSemantics: Clone + PartialEq {
    fn and(l: Self, r: Self) -> Self;
    fn or(l: Self, r: Self) -> Self;
    fn not(val: Self) -> Self;
    fn implies(antecedent: Self, consequent: Self) -> Self;
    fn eventually_identity() -> Self;
    fn globally_identity() -> Self;
    fn atomic_true() -> Self;
    fn atomic_false() -> Self;
    fn atomic_greater_than(value: f64, c: f64) -> Self;
    fn atomic_less_than(value: f64, c: f64) -> Self;
}
impl RobustnessSemantics for f64 {
    fn and(l: f64, r: f64) -> f64 {
        let result = l.min(r);
        if result == -0.0 { 0.0 } else { result }
    }
    fn or(l: f64, r: f64) -> f64 {
        let result = l.max(r);
        if result == -0.0 { 0.0 } else { result }
    }
    fn not(val: f64) -> f64 {
        let result = -val;
        if result == -0.0 { 0.0 } else { result }
    }
    fn implies(antecedent: f64, consequent: f64) -> f64 {
        let result = (-antecedent).max(consequent);
        if result == -0.0 { 0.0 } else { result }
    }
    fn eventually_identity() -> Self {
        f64::NEG_INFINITY
    }
    fn globally_identity() -> Self {
        f64::INFINITY
    }
    fn atomic_true() -> Self {
        f64::INFINITY
    }
    fn atomic_false() -> Self {
        f64::NEG_INFINITY
    }
    fn atomic_greater_than(value: f64, c: f64) -> Self {
        let result = value - c;
        if result == -0.0 { 0.0 } else { result }
    }
    fn atomic_less_than(value: f64, c: f64) -> Self {
        let result = c - value;
        if result == -0.0 { 0.0 } else { result }
    }
}

// Implement the semantics for bool
impl RobustnessSemantics for bool {
    fn and(l: bool, r: bool) -> bool {
        l && r
    }
    fn or(l: bool, r: bool) -> bool {
        l || r
    }
    fn not(val: bool) -> bool {
        !val
    }
    fn implies(antecedent: bool, consequent: bool) -> bool {
        !antecedent || consequent
    }
    fn eventually_identity() -> Self {
        false
    }
    fn globally_identity() -> Self {
        true
    }
    fn atomic_true() -> Self {
        true
    }
    fn atomic_false() -> Self {
        false
    }
    fn atomic_greater_than(value: f64, c: f64) -> Self {
        value > c
    }
    fn atomic_less_than(value: f64, c: f64) -> Self {
        value < c
    }
}
