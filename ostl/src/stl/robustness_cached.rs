use crate::ring_buffer::{RingBufferTrait, Step};
use crate::stl::core::{
    BinaryTemporalOperatorTrait, LogicalOperatorTrait, StlOperatorTrait, TemporalOperatorBaseTrait,
    TimeInterval, UnaryTemporalOperatorTrait,
};
use std::fmt::Display;

#[derive(Clone)]
pub struct And<T, Y> {
    pub left: Box<dyn StlOperatorTrait<T, Y>>,
    pub right: Box<dyn StlOperatorTrait<T, Y>>,
}

impl<T, Y> Display for And<T, Y> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}) ∧ ({})",
            self.left.to_string(),
            self.right.to_string()
        )
    }
}

impl<T: Clone, Y: Clone> LogicalOperatorTrait<T, Y> for And<T, Y> {
    fn left(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Y>> {
        &mut self.left
    }
    fn right(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Y>> {
        &mut self.right
    }
}

impl<T: Clone> StlOperatorTrait<T, f64> for And<T, f64> {
    fn robustness(&mut self, step: &Step<T>) -> Option<f64> {
        self.robustness_with(step, |l, r| l.min(r))
    }
}

/// Implement StlOperatorTrait for And with boolean output
impl<T: Clone> StlOperatorTrait<T, bool> for And<T, bool> {
    fn robustness(&mut self, step: &Step<T>) -> Option<bool> {
        self.robustness_with(step, |l, r| l && r)
    }
}

#[derive(Clone)]
pub struct Or<T, Y> {
    pub left: Box<dyn StlOperatorTrait<T, Y> + 'static>,
    pub right: Box<dyn StlOperatorTrait<T, Y> + 'static>,
}

impl<T: Clone, Y: Clone> LogicalOperatorTrait<T, Y> for Or<T, Y> {
    fn left(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Y>> {
        &mut self.left
    }
    fn right(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Y>> {
        &mut self.right
    }
}

impl<T: Clone> StlOperatorTrait<T, f64> for Or<T, f64> {
    fn robustness(&mut self, step: &Step<T>) -> Option<f64> {
        self.robustness_with(step, |l, r| l.max(r))
    }
}

impl<T: Clone> StlOperatorTrait<T, bool> for Or<T, bool> {
    fn robustness(&mut self, step: &Step<T>) -> Option<bool> {
        self.robustness_with(step, |l, r| l || r)
    }
}

impl<T, Y> Display for Or<T, Y> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}) v ({})",
            self.left.to_string(),
            self.right.to_string()
        )
    }
}

#[derive(Clone)]
pub struct Not<T, Y> {
    pub operand: Box<dyn StlOperatorTrait<T, Y> + 'static>,
}

impl<T: Clone, Y: Clone> LogicalOperatorTrait<T, Y> for Not<T, Y> {
    fn left(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Y>> {
        &mut self.operand
    }
    fn right(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Y>> {
        panic!("Not operator does not have a right operand");
    }
    fn robustness_with<F>(&mut self, step: &Step<T>, f: F) -> Option<Y>
    where
        F: FnOnce(Y, Y) -> Y,
        Y: Clone,
        T: Clone,
        Self: StlOperatorTrait<T, Y>,
    {
        // NOTE: currently, traiy forces f to be a binary function, but we only need a unary function here.
        // This is a workaround to reuse the robustness_with method.
        // In the future, we might want to refactor this to allow unary functions.
        self.left().robustness(step).map(|r| f(r.clone(), r))
    }
}

impl<T: Clone> StlOperatorTrait<T, f64> for Not<T, f64> {
    fn robustness(&mut self, step: &Step<T>) -> Option<f64> {
        self.robustness_with(step, |a, _| -a)
    }
}

impl<T: Clone> StlOperatorTrait<T, bool> for Not<T, bool> {
    fn robustness(&mut self, step: &Step<T>) -> Option<bool> {
        self.robustness_with(step, |a, _| !a)
    }
}

impl<T, Y> Display for Not<T, Y> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "¬({})", self.operand.to_string())
    }
}

#[derive(Clone)]
pub struct Implies<T, Y> {
    pub antecedent: Box<dyn StlOperatorTrait<T, Y> + 'static>,
    pub consequent: Box<dyn StlOperatorTrait<T, Y> + 'static>,
}

impl<T: Clone, Y: Clone> LogicalOperatorTrait<T, Y> for Implies<T, Y> {
    fn left(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Y>> {
        &mut self.antecedent
    }
    fn right(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Y>> {
        &mut self.consequent
    }
}

impl<T: Clone> StlOperatorTrait<T, f64> for Implies<T, f64> {
    fn robustness(&mut self, step: &Step<T>) -> Option<f64> {
        self.robustness_with(step, |a, c| (-a).max(c))
    }
}
impl<T, Y> Display for Implies<T, Y> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}) → ({})",
            self.antecedent.to_string(),
            self.consequent.to_string()
        )
    }
}

#[derive(Clone)]
pub struct Eventually<T, C, Y> {
    pub interval: TimeInterval,
    pub operand: Box<dyn StlOperatorTrait<T, Y> + 'static>,
    pub cache: C,
}
// Implementation of the new TemporalOperatorBaseTrait for Eventually
impl<T, Y, C> TemporalOperatorBaseTrait<T, Y, C> for Eventually<T, C, Y>
where
    T: Clone,
    Y: Clone,
    C: RingBufferTrait<Value = Option<Y>> + Clone,
{
    fn interval(&self) -> TimeInterval {
        self.interval
    }

    fn cache(&mut self) -> &mut C {
        &mut self.cache
    }
}

impl<T, Y, C> UnaryTemporalOperatorTrait<T, Y, C> for Eventually<T, C, Y>
where
    T: Clone,
    Y: Clone,
    C: RingBufferTrait<Value = Option<Y>> + Clone,
{
    fn operand(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Y>> {
        &mut self.operand
    }
}

// StlOperatorTrait for Eventually now uses the shared robustness_with method
impl<T: Clone, C> StlOperatorTrait<T, f64> for Eventually<T, C, f64>
where
    C: RingBufferTrait<Value = Option<f64>> + Clone,
{
    fn robustness(&mut self, step: &Step<T>) -> Option<f64> {
        self.robustness_unary_with(step, f64::NEG_INFINITY, f64::max)
    }
}

impl<T, C> StlOperatorTrait<T, bool> for Eventually<T, C, bool>
where
    T: Clone,
    C: RingBufferTrait<Value = Option<bool>> + Clone,
{
    // formula: exists t' in [t+start, t+end] s.t. operand is true at t'
    fn robustness(&mut self, step: &Step<T>) -> Option<bool> {
        self.robustness_unary_with(step, false, |a, b| a || b)
    }
}

impl<T, C, Y> Display for Eventually<T, C, Y> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "F[{}, {}]({})",
            self.interval.start.as_secs_f64(),
            self.interval.end.as_secs_f64(),
            self.operand.to_string()
        )
    }
}

#[derive(Clone)]
pub struct Globally<T, Y, C> {
    pub interval: TimeInterval,
    pub operand: Box<dyn StlOperatorTrait<T, Y> + 'static>,
    pub cache: C,
}

// Implementation of the new TemporalOperatorBaseTrait for Globally
impl<T, Y, C> TemporalOperatorBaseTrait<T, Y, C> for Globally<T, Y, C>
where
    T: Clone,
    Y: Clone,
    C: RingBufferTrait<Value = Option<Y>> + Clone,
{
    fn interval(&self) -> TimeInterval {
        self.interval
    }

    fn cache(&mut self) -> &mut C {
        &mut self.cache
    }
}

impl<T, Y, C> UnaryTemporalOperatorTrait<T, Y, C> for Globally<T, Y, C>
where
    T: Clone,
    Y: Clone,
    C: RingBufferTrait<Value = Option<Y>> + Clone,
{
    fn operand(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Y>> {
        &mut self.operand
    }
}

// StlOperatorTrait for Globally now uses the shared robustness_with method
impl<T: Clone, C> StlOperatorTrait<T, f64> for Globally<T, f64, C>
where
    C: RingBufferTrait<Value = Option<f64>> + Clone,
{
    fn robustness(&mut self, step: &Step<T>) -> Option<f64> {
        self.robustness_unary_with(step, f64::INFINITY, f64::min)
    }
}

impl<T, C> StlOperatorTrait<T, bool> for Globally<T, bool, C>
where
    T: Clone,
    C: RingBufferTrait<Value = Option<bool>> + Clone,
{
    // formula: for all t' in [t+start, t+end], operand is true at t'
    fn robustness(&mut self, step: &Step<T>) -> Option<bool> {
        self.robustness_unary_with(step, true, |a, b| a && b)
    }
}

impl<T, C, Y> Display for Globally<T, Y, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "G[{}, {}]({})",
            self.interval.start.as_secs_f64(),
            self.interval.end.as_secs_f64(),
            self.operand.to_string()
        )
    }
}

#[derive(Clone)]
pub struct Until<T, C, Y> {
    pub interval: TimeInterval,
    pub left: Box<dyn StlOperatorTrait<T, Y> + 'static>,
    pub right: Box<dyn StlOperatorTrait<T, Y> + 'static>,
    pub cache: C,
}

// Implementation of the new TemporalOperatorBaseTrait for Until
impl<T, Y, C> TemporalOperatorBaseTrait<T, Y, C> for Until<T, C, Y>
where
    T: Clone,
    Y: Clone,
    C: RingBufferTrait<Value = Option<Y>> + Clone,
{
    fn interval(&self) -> TimeInterval {
        self.interval
    }

    fn cache(&mut self) -> &mut C {
        &mut self.cache
    }
}

impl<T, Y, C> BinaryTemporalOperatorTrait<T, Y, C> for Until<T, C, Y>
where
    T: Clone,
    Y: Clone,
    C: RingBufferTrait<Value = Option<Y>> + Clone,
{
    fn left_operand(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Y>> {
        &mut self.left
    }
    fn right_operand(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Y>> {
        &mut self.right
    }
}

impl<T, C> StlOperatorTrait<T, f64> for Until<T, C, f64>
where
    T: Clone,
    C: RingBufferTrait<Value = Option<f64>> + Clone,
{
    fn robustness(&mut self, step: &Step<T>) -> Option<f64> {
        let right_robustness = self.right.robustness(step)?;
        self.cache
            .add_step(Some(self.left.robustness(step))?, step.timestamp);

        // The window of interest for the left operand's past robustness values
        let t = step.timestamp.saturating_sub(self.interval.end);
        let lower_bound_t_prime = t + self.interval.start;
        let upper_bound_t_prime = t + self.interval.end;

        // Ensure we have enough data to evaluate the window
        if self.is_cache_sufficient(lower_bound_t_prime, upper_bound_t_prime, t) {
            let max_robustness = self
                .cache
                .iter()
                .filter(|entry| {
                    entry.timestamp >= lower_bound_t_prime && entry.timestamp <= upper_bound_t_prime
                })
                .map(|entry| {
                    let t_prime = entry.timestamp;
                    let min_left_robustness = self
                        .cache
                        .iter()
                        .filter(|e| e.timestamp >= lower_bound_t_prime && e.timestamp <= t_prime)
                        .filter_map(|e| e.value)
                        .fold(f64::INFINITY, f64::min);

                    right_robustness.min(min_left_robustness)
                })
                .fold(f64::NEG_INFINITY, f64::max);

            Some(max_robustness)
        } else {
            None // Not enough data to evaluate
        }
    }
}

// Implement StlOperatorTrait for Until with boolean output
impl<T, C> StlOperatorTrait<T, bool> for Until<T, C, bool>
where
    T: Clone,
    C: RingBufferTrait<Value = Option<bool>> + Clone,
{
    fn robustness(&mut self, step: &Step<T>) -> Option<bool> {
        let right_robustness = self.right.robustness(step)?;
        self.cache
            .add_step(Some(self.left.robustness(step)?), step.timestamp);

        // The window of interest for the left operand's past robustness values
        let t = step.timestamp.saturating_sub(self.interval.end);
        let lower_bound_t_prime = t + self.interval.start;
        let upper_bound_t_prime = t + self.interval.end;

        // Ensure we have enough data to evaluate the window
        if self.is_cache_sufficient(lower_bound_t_prime, upper_bound_t_prime, t) {
            let max_robustness = self
                .cache
                .iter()
                .filter(|entry| {
                    entry.timestamp >= lower_bound_t_prime && entry.timestamp <= upper_bound_t_prime
                })
                .map(|entry| {
                    let t_prime = entry.timestamp;
                    let min_left_robustness = self
                        .cache
                        .iter()
                        .filter(|e| e.timestamp >= lower_bound_t_prime && e.timestamp <= t_prime)
                        .map(|e| e.value)
                        .fold(true, |a, b| a && b.unwrap_or(false));

                    right_robustness.min(min_left_robustness)
                })
                .fold(false, |a, b| a || b);

            Some(max_robustness)
        } else {
            None // Not enough data to evaluate
        }
    }
}

impl<T, C, Y> Display for Until<T, Y, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}) U[{}, {}] ({})",
            self.left.to_string(),
            self.interval.start.as_secs_f64(),
            self.interval.end.as_secs_f64(),
            self.right.to_string()
        )
    }
}

#[derive(Clone)]
pub enum Atomic {
    LessThan(f64),
    GreaterThan(f64),
    True,
    False,
}

impl<T> StlOperatorTrait<T, f64> for Atomic
where
    T: Into<f64> + Clone,
{
    fn robustness(&mut self, step: &Step<T>) -> Option<f64> {
        let value = step.value.clone().into();
        match self {
            Atomic::True => Some(f64::INFINITY),
            Atomic::False => Some(f64::NEG_INFINITY),
            Atomic::GreaterThan(c) => Some(value - *c),
            Atomic::LessThan(c) => Some(*c - value),
        }
    }
}

impl<T> StlOperatorTrait<T, bool> for Atomic
where
    T: Into<f64> + Clone,
{
    fn robustness(&mut self, step: &Step<T>) -> Option<bool> {
        let value = step.value.clone().into();
        match self {
            Atomic::True => Some(true),
            Atomic::False => Some(false),
            Atomic::GreaterThan(c) => Some(value > *c),
            Atomic::LessThan(c) => Some(value < *c),
        }
    }
}

impl Display for Atomic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Atomic::LessThan(c) => write!(f, "x < {}", c),
            Atomic::GreaterThan(c) => write!(f, "x > {}", c),
            Atomic::True => write!(f, "True"),
            Atomic::False => write!(f, "False"),
        }
    }
}
