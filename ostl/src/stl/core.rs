use crate::ring_buffer::Step;
use core::f64;
use dyn_clone::{DynClone, clone_trait_object};
use std::collections::HashSet;
use std::fmt::Display;
use std::ops::Add;
use std::ops::Neg;
use std::ops::Sub;
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

    fn update(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>>;
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

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct RobustnessInterval(pub f64, pub f64);

impl Add<Self> for RobustnessInterval {
    type Output = RobustnessInterval;

    fn add(self, other: Self) -> RobustnessInterval {
        RobustnessInterval(self.0 + other.0, self.1 + other.1)
    }
}

impl Add<f64> for RobustnessInterval {
    type Output = RobustnessInterval;

    fn add(self, other: f64) -> RobustnessInterval {
        RobustnessInterval(self.0 + other, self.1 + other)
    }
}

impl Add<RobustnessInterval> for f64 {
    type Output = RobustnessInterval;

    fn add(self, other: RobustnessInterval) -> RobustnessInterval {
        // You can just reuse the logic you already wrote!
        other + self
        // Or be explicit:
        // Interval(self + other.0, self + other.1)
    }
}

impl Sub<Self> for RobustnessInterval {
    type Output = RobustnessInterval;

    fn sub(self, other: Self) -> RobustnessInterval {
        RobustnessInterval(self.0 - other.1, self.1 - other.0)
    }
}

impl Sub<f64> for RobustnessInterval {
    type Output = RobustnessInterval;

    fn sub(self, other: f64) -> RobustnessInterval {
        RobustnessInterval(self.0 - other, self.1 - other)
    }
}

impl Sub<RobustnessInterval> for f64 {
    type Output = RobustnessInterval;

    fn sub(self, other: RobustnessInterval) -> RobustnessInterval {
        RobustnessInterval(self - other.1, self - other.0)
    }
}

impl Neg for RobustnessInterval {
    type Output = RobustnessInterval;

    fn neg(self) -> RobustnessInterval {
        RobustnessInterval(-self.1, -self.0)
    }
}

// Min, Max, Intersection traits
pub trait Min {
    fn min(self, other: Self) -> Self;
}
pub trait Max {
    fn max(self, other: Self) -> Self;
}
pub trait Intersection {
    fn intersection(self, other: Self) -> Self;
}

impl Min for RobustnessInterval {
    fn min(self, other: Self) -> Self {
        RobustnessInterval(self.0.min(other.0), self.1.min(other.1))
    }
}
impl Max for RobustnessInterval {
    fn max(self, other: Self) -> Self {
        RobustnessInterval(self.0.max(other.0), self.1.max(other.1))
    }
}

impl Intersection for RobustnessInterval {
    /// Calculates the intersection of two intervals.
    /// Returns None if there is no overlap (the empty set).
    fn intersection(self, other: Self) -> Self {
        let new_start = self.0.max(other.0);
        let new_end = self.1.min(other.1);
        if new_end < new_start {
            // The intervals do not overlap.
            RobustnessInterval(f64::NEG_INFINITY, f64::INFINITY)
        } else {
            // The intervals overlap. Return the new intersected interval.
            RobustnessInterval(new_start, new_end)
        }
    }
}

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
        l.min(r)
    }
    fn or(l: f64, r: f64) -> f64 {
        l.max(r)
    }
    fn not(val: f64) -> f64 {
        -val
    }
    fn implies(antecedent: f64, consequent: f64) -> f64 {
        (-antecedent).max(consequent)
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
        value - c
    }
    fn atomic_less_than(value: f64, c: f64) -> Self {
        c - value
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

impl RobustnessSemantics for RobustnessInterval {
    fn and(l: Self, r: Self) -> Self {
        // conjunction => pointwise min over interval bounds
        l.min(r)
    }

    fn or(l: Self, r: Self) -> Self {
        // disjunction => pointwise max over interval bounds
        l.max(r)
    }

    fn not(val: Self) -> Self {
        // negation => interval negation (flip and negate bounds)
        -val
    }

    fn implies(antecedent: Self, consequent: Self) -> Self {
        // implication: max(-antecedent, consequent)
        (-antecedent).max(consequent)
    }

    fn eventually_identity() -> Self {
        // identity for sup / eventuality is negative infinity
        RobustnessInterval(f64::NEG_INFINITY, f64::INFINITY)
    }

    fn globally_identity() -> Self {
        // identity for inf / globally is positive infinity
        RobustnessInterval(f64::INFINITY, f64::NEG_INFINITY)
    }

    fn atomic_true() -> Self {
        // atomic true represents the maximal robustness
        RobustnessInterval(f64::INFINITY, f64::INFINITY)
    }

    fn atomic_false() -> Self {
        // atomic false represents the minimal robustness
        RobustnessInterval(f64::NEG_INFINITY, f64::NEG_INFINITY)
    }

    fn atomic_greater_than(value: f64, c: f64) -> Self {
        RobustnessInterval(value - c, value - c)
    }

    fn atomic_less_than(value: f64, c: f64) -> Self {
        RobustnessInterval(c - value, c - value)
    }
}
