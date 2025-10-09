use crate::ring_buffer::{RingBufferTrait, Step};
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
pub trait StlOperatorTrait<T: Clone>: DynClone + Display {
    type Output;

    // Added as_any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;

    // added DynClone for cloning trait objects
    fn robustness<'a>(&'a mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>>;
}

clone_trait_object!(<T: Clone, Y> StlOperatorTrait<T, Output = Y>);

pub trait RobustnessSemantics: Clone {
    fn and(l: Self, r: Self) -> Self;
    fn or(l: Self, r: Self) -> Self;
    fn not(val: Self) -> Self;
    fn implies(antecedent: Self, consequent: Self) -> Self;
    fn eventually_identity() -> Self;
    fn globally_identity() -> Self;
    fn until_identity() -> Self;
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
    fn until_identity() -> Self {
        f64::NEG_INFINITY
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
    fn until_identity() -> Self {
        false
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

pub trait TemporalOperatorBaseTrait<T, C>: StlOperatorTrait<T>
where
    T: Clone,
    C: RingBufferTrait<Value = Option<Self::Output>>,
{
    fn interval(&self) -> TimeInterval;
    fn cache(&mut self) -> &mut C;

    fn is_cache_sufficient(
        &mut self,
        lower_bound: Duration,
        upper_bound: Duration,
        t: Duration,
    ) -> bool {
        self.cache().is_empty()
            || upper_bound - lower_bound
                <= self
                    .cache()
                    .get_back()
                    .map_or(Duration::ZERO, |entry| entry.timestamp - t)
    }
}

pub trait UnaryTemporalOperatorTrait<T, C>: TemporalOperatorBaseTrait<T, C>
where
    C: RingBufferTrait<Value = Option<Self::Output>>,
    T: Clone,
{
    fn operand(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Output = Self::Output>>;

    fn robustness_unary_with<'a, F>(
        &mut self,
        step: &Step<T>,
        initial: Self::Output,
        f: F,
    ) -> &'a [Step<Option<Self::Output>>]
    where
        F: Fn(Self::Output, Self::Output) -> Self::Output,
        Self::Output: Clone,
        Self: Sized,
    {
        let sub_robustness_vec = self.operand().robustness(step).to_vec();

        sub_robustness_vec
            .into_iter()
            .for_each(|step| self.cache().add_step(step));

        let t = step.timestamp.saturating_sub(self.interval().end);
        let lower_bound = t + self.interval().start;
        let upper_bound = t + self.interval().end;

        if self.is_cache_sufficient(lower_bound, upper_bound, t) {
            let result = self
                .cache()
                .iter()
                .filter(|entry| entry.timestamp >= lower_bound && entry.timestamp <= upper_bound)
                .filter_map(|entry| entry.value.clone())
                .fold(initial, f);
        };
        &[]
    }
}

pub trait BinaryTemporalOperatorTrait<T, C>: TemporalOperatorBaseTrait<T, C>
where
    C: RingBufferTrait<Value = Option<Self::Output>>,
    T: Clone,
{
    fn left(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Output = Self::Output>>;
    fn right(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Output = Self::Output>>;
}
