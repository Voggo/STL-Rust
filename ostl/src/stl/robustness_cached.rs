use crate::ring_buffer::{RingBufferTrait, Step};
use crate::stl::core::{
    LogicalOperatorTrait, StlOperatorTrait, TemporalOperatorTrait, TimeInterval,
};
use std::fmt::Display;
use std::time::Duration;

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
}

impl<T: Clone> StlOperatorTrait<T, f64> for Not<T, f64> {
    fn robustness(&mut self, step: &Step<T>) -> Option<f64> {
        self.left().robustness(step).map(|r| -r)
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
pub struct Eventually<T, C, Y>
where
    C: RingBufferTrait<Value = Option<f64>> + Clone,
{
    pub interval: TimeInterval,
    pub operand: Box<dyn StlOperatorTrait<T, Y> + 'static>,
    pub cache: C,
}

impl<T, C> StlOperatorTrait<T, f64> for Eventually<T, C, f64>
where
    T: Clone,
    C: RingBufferTrait<Value = Option<f64>> + Clone,
{
    fn robustness(&mut self, step: &Step<T>) -> Option<f64> {
        self.cache
            .add_step(self.operand.robustness(step), step.timestamp);

        // The window of interest for the operand's past robustness values
        let t = step.timestamp.saturating_sub(self.interval.end);
        let lower_bound_t_prime = t + self.interval.start;
        let upper_bound_t_prime = t + self.interval.end;

        // Ensure we have enough data to evaluate the window
        if self.cache.is_empty()
            || upper_bound_t_prime - lower_bound_t_prime
                <= self
                    .cache
                    .get_back()
                    .map_or(Duration::ZERO, |entry| entry.timestamp - t)
        {
            let max_robustness = self
                .cache
                .iter()
                .filter(|entry| {
                    entry.timestamp >= lower_bound_t_prime && entry.timestamp <= upper_bound_t_prime
                })
                .filter_map(|entry| entry.value) // From Step<Option<f64>> to Option<f64>, then unwraps to f64
                .fold(f64::NEG_INFINITY, f64::max);

            Some(max_robustness)
        } else {
            None // Not enough historical data to compute robustness
        }
    }
}
impl<T, C, Y> Display for Eventually<T, C, Y>
where
    C: RingBufferTrait<Value = Option<f64>> + Clone,
{
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
pub struct Globally<T, Y, C>
where
    C: RingBufferTrait<Value = Option<f64>> + Clone,
{
    pub interval: TimeInterval,
    pub operand: Box<dyn StlOperatorTrait<T, Y> + 'static>,
    pub cache: C,
}

impl<T, C> StlOperatorTrait<T, f64> for Globally<T, f64, C>
where
    T: Clone,
    C: RingBufferTrait<Value = Option<f64>> + Clone,
{
    fn robustness(&mut self, step: &Step<T>) -> Option<f64> {
        self.cache
            .add_step(self.operand.robustness(step), step.timestamp);

        // The window of interest for the operand's past robustness values
        let t = step.timestamp.saturating_sub(self.interval.end);
        let lower_bound_t_prime = t + self.interval.start;
        let upper_bound_t_prime = t + self.interval.end;

        // We can only compute a result if we have data extending back to the start of the window.
        if self.cache.is_empty()
            || upper_bound_t_prime - lower_bound_t_prime
                <= self
                    .cache
                    .get_back()
                    .map_or(Duration::ZERO, |entry| entry.timestamp - t)
        {
            let min_robustness = self
                .cache
                .iter()
                .filter(|entry| {
                    entry.timestamp >= lower_bound_t_prime && entry.timestamp <= upper_bound_t_prime
                })
                .filter_map(|entry| entry.value) // From Step<Option<f64>> to Option<f64>, then unwraps to f64
                .fold(f64::INFINITY, f64::min);

            Some(min_robustness)
        } else {
            None // Not enough historical data to compute robustness
        }
    }
}

impl<T, C, Y> Display for Globally<T, Y, C>
where
    C: RingBufferTrait<Value = Option<f64>> + Clone,
{
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
pub struct Until<T, Y, C>
where
    C: RingBufferTrait<Value = f64> + Clone,
{
    pub interval: TimeInterval,
    pub left: Box<dyn StlOperatorTrait<T, Y> + 'static>,
    pub right: Box<dyn StlOperatorTrait<T, Y> + 'static>,
    pub cache: C,
}

impl<T, C> StlOperatorTrait<T, f64> for Until<T, f64, C>
where
    T: Clone,
    C: RingBufferTrait<Value = f64> + Clone,
{
    fn robustness(&mut self, step: &Step<T>) -> Option<f64> {
        let right_robustness = self.right.robustness(step)?;
        self.cache
            .add_step(self.left.robustness(step)?, step.timestamp);

        // The window of interest for the left operand's past robustness values
        let t = step.timestamp.saturating_sub(self.interval.end);
        let lower_bound_t_prime = t + self.interval.start;
        let upper_bound_t_prime = t + self.interval.end;

        // Ensure we have enough data to evaluate the window
        if self.cache.is_empty()
            || upper_bound_t_prime - lower_bound_t_prime
                > self
                    .cache
                    .get_back()
                    .map_or(Duration::ZERO, |entry| entry.timestamp - t)
        {
            return None; // Not enough data to evaluate
        }

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
                    .fold(f64::INFINITY, f64::min);

                right_robustness.min(min_left_robustness)
            })
            .fold(f64::NEG_INFINITY, f64::max);

        Some(max_robustness)
    }
}
impl<T, C, Y> Display for Until<T, Y, C>
where
    C: RingBufferTrait<Value = f64> + Clone,
{
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
