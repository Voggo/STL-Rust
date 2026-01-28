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
        other + self
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
    fn unknown() -> Self;

    /// Returns true if 'old' is strictly dominated by 'new' such that 'old'
    /// can be safely discarded from a Lemire sliding window.
    ///
    /// For RoSI, this requires strict separation of intervals.
    ///
    /// # Arguments
    /// * `old` - The value to check for domination
    /// * `new` - The potentially dominating value
    /// * `is_max` - If true, checks domination for max operations; if false, for min operations
    fn prune_dominated(old: Self, new: Self, is_max: bool) -> bool;
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
    fn unknown() -> Self {
        f64::NAN
    }
    fn prune_dominated(old: Self, new: Self, is_max: bool) -> bool {
        if is_max { old <= new } else { old >= new }
    }
}

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
    fn unknown() -> Self {
        // In the boolean case, we can represent "unknown" as false
        false
    }
    fn prune_dominated(old: Self, new: Self, is_max: bool) -> bool {
        // false <= true
        if is_max {
            !old || new // old <= new
        } else {
            old || !new // old >= new
        }
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
        // (-antecedent).max(consequent)
        Self::or(-antecedent, consequent)
    }

    fn eventually_identity() -> Self {
        // identity for sup / eventuality is negative infinity
        RobustnessInterval(f64::NEG_INFINITY, f64::NEG_INFINITY)
    }

    fn globally_identity() -> Self {
        // identity for inf / globally is positive infinity
        RobustnessInterval(f64::INFINITY, f64::INFINITY)
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

    fn unknown() -> Self {
        RobustnessInterval(f64::NEG_INFINITY, f64::INFINITY)
    }
    fn prune_dominated(old: Self, new: Self, is_max: bool) -> bool {
        // example: F[a,b] x>0
        // x0 = -2, x1 = 2
        // old = (-2, -2), new = (2, 2), is_max = true
        // returns true: old can be discarded since it can never exceed new and we want the best only
        // example: G[a,b] x>0
        // x0 = 2, x1 = -2
        // old = (2, 2), new = (-2, -2), is_max = false
        // returns true: old can be discarded since it can never be smaller than new and we want the worst only

        if is_max {
            // Max/Eventually: Discard old if it can never exceed new.
            // We need new's lower bound to be >= old's upper bound.
            old.1 <= new.0
        } else {
            // Min/Globally: Discard old if it can never be smaller than new.
            // We need new's upper bound to be <= old's lower bound.
            old.0 >= new.1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ri_add_sub_neg() {
        let a = RobustnessInterval(1.0, 2.0);
        let b = RobustnessInterval(0.5, 1.5);

        assert_eq!(a + b, RobustnessInterval(1.5, 3.5));
        assert_eq!(a + 1.0f64, RobustnessInterval(2.0, 3.0));
        assert_eq!(1.0f64 + a, RobustnessInterval(2.0, 3.0));

        let sub = a - b;
        // subtraction defined as (a.lo - b.hi, a.hi - b.lo)
        assert_eq!(sub, RobustnessInterval(1.0 - 1.5, 2.0 - 0.5));

        let subf = a - 1.0f64;
        assert_eq!(subf, RobustnessInterval(0.0, 1.0));

        let neg = -a;
        assert_eq!(neg, RobustnessInterval(-2.0, -1.0));

        // f64 - RobustnessInterval is implemented as (self - other.hi, self - other.lo)
        let left_f64 = 5.0f64;
        let res = left_f64 - a;
        assert_eq!(res, RobustnessInterval(5.0 - 2.0, 5.0 - 1.0));
    }

    #[test]
    fn ri_min_max_intersection() {
        let a = RobustnessInterval(1.0, 4.0);
        let b = RobustnessInterval(2.0, 3.0);

        assert_eq!(a.min(b), RobustnessInterval(1.0, 3.0));
        assert_eq!(a.max(b), RobustnessInterval(2.0, 4.0));

        let inter = a.intersection(b);
        assert_eq!(inter, RobustnessInterval(2.0, 3.0));

        // non-overlapping intervals -> returns (NEG_INFINITY, INFINITY) per implementation
        let c = RobustnessInterval(1.0, 2.0);
        let d = RobustnessInterval(3.0, 4.0);
        let non = c.intersection(d);
        assert!(non.0.is_infinite() && non.0.is_sign_negative());
        assert!(non.1.is_infinite() && non.1.is_sign_positive());
    }

    #[test]
    fn f64_semantics_basic() {
        let a = 1.5f64;
        let b = 2.0f64;

        assert_eq!(<f64 as RobustnessSemantics>::and(a, b), a.min(b));
        assert_eq!(<f64 as RobustnessSemantics>::or(a, b), a.max(b));
        assert_eq!(<f64 as RobustnessSemantics>::not(a), -a);

        // implication: max(-antecedent, consequent)
        assert_eq!(<f64 as RobustnessSemantics>::implies(a, b), (-a).max(b));

        assert!(<f64 as RobustnessSemantics>::eventually_identity().is_infinite());
        assert!(<f64 as RobustnessSemantics>::globally_identity().is_infinite());

        assert_eq!(<f64 as RobustnessSemantics>::atomic_true(), f64::INFINITY);
        assert_eq!(
            <f64 as RobustnessSemantics>::atomic_false(),
            f64::NEG_INFINITY
        );

        assert_eq!(
            <f64 as RobustnessSemantics>::atomic_greater_than(5.0, 3.0),
            2.0
        );
        assert_eq!(
            <f64 as RobustnessSemantics>::atomic_less_than(2.0, 5.0),
            3.0
        );

        let unk = <f64 as RobustnessSemantics>::unknown();
        assert!(unk.is_nan());
    }

    #[test]
    fn bool_semantics_basic() {
        assert_eq!(<bool as RobustnessSemantics>::and(true, false), false);
        assert_eq!(<bool as RobustnessSemantics>::or(true, false), true);
        assert_eq!(<bool as RobustnessSemantics>::not(true), false);
        assert_eq!(<bool as RobustnessSemantics>::implies(true, false), false);

        assert_eq!(<bool as RobustnessSemantics>::eventually_identity(), false);
        assert_eq!(<bool as RobustnessSemantics>::globally_identity(), true);

        assert_eq!(<bool as RobustnessSemantics>::atomic_true(), true);
        assert_eq!(<bool as RobustnessSemantics>::atomic_false(), false);

        assert_eq!(
            <bool as RobustnessSemantics>::atomic_greater_than(5.0, 3.0),
            true
        );
        assert_eq!(
            <bool as RobustnessSemantics>::atomic_less_than(2.0, 1.0),
            false
        );
    }

    #[test]
    fn interval_semantics_basic() {
        let a = RobustnessInterval(1.0, 2.0);
        let b = RobustnessInterval(0.5, 3.0);

        let is_dom_true = RobustnessInterval::prune_dominated(a, b, false);
        println!("is_dom_true: {}", is_dom_true);
        // and = min
        assert_eq!(
            <RobustnessInterval as RobustnessSemantics>::and(a, b),
            a.min(b)
        );
        // or = max
        assert_eq!(
            <RobustnessInterval as RobustnessSemantics>::or(a, b),
            a.max(b)
        );
        // not = negation
        assert_eq!(<RobustnessInterval as RobustnessSemantics>::not(a), -a);

        // implies = or(-antecedent, consequent)
        let imp = <RobustnessInterval as RobustnessSemantics>::implies(a, b);
        assert_eq!(imp, <RobustnessInterval as RobustnessSemantics>::or(-a, b));

        assert_eq!(
            <RobustnessInterval as RobustnessSemantics>::eventually_identity(),
            RobustnessInterval(f64::NEG_INFINITY, f64::NEG_INFINITY)
        );
        assert_eq!(
            <RobustnessInterval as RobustnessSemantics>::globally_identity(),
            RobustnessInterval(f64::INFINITY, f64::INFINITY)
        );

        assert_eq!(
            <RobustnessInterval as RobustnessSemantics>::atomic_true(),
            RobustnessInterval(f64::INFINITY, f64::INFINITY)
        );
        assert_eq!(
            <RobustnessInterval as RobustnessSemantics>::atomic_false(),
            RobustnessInterval(f64::NEG_INFINITY, f64::NEG_INFINITY)
        );

        assert_eq!(
            <RobustnessInterval as RobustnessSemantics>::atomic_greater_than(5.0, 3.0),
            RobustnessInterval(2.0, 2.0)
        );
        assert_eq!(
            <RobustnessInterval as RobustnessSemantics>::atomic_less_than(2.0, 5.0),
            RobustnessInterval(3.0, 3.0)
        );

        let unk = <RobustnessInterval as RobustnessSemantics>::unknown();
        assert_eq!(unk, RobustnessInterval(f64::NEG_INFINITY, f64::INFINITY));
    }

    #[test]
    fn time_interval_basic() {
        let ti = TimeInterval {
            start: Duration::from_secs(1),
            end: Duration::from_secs(5),
        };
        assert_eq!(ti.start, Duration::from_secs(1));
        assert_eq!(ti.end, Duration::from_secs(5));
    }
}
