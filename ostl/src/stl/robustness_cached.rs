use crate::ring_buffer::{RingBufferTrait, Step};
use crate::stl::core::{StlOperatorTrait, TimeInterval};
use std::time::Duration;

#[derive(Clone)]
pub struct And<T> {
    pub left: Box<dyn StlOperatorTrait<T>>,
    pub right: Box<dyn StlOperatorTrait<T>>,
}

impl<T: Clone> StlOperatorTrait<T> for And<T> {
    fn robustness(&mut self, step: &Step<T>) -> Option<f64> {
        self.left
            .robustness(step)
            .zip(self.right.robustness(step))
            .map(|(l, r)| l.min(r))
    }
    fn to_string(&self) -> String {
        format!("({}) ∧ ({})", self.left.to_string(), self.right.to_string())
    }
}

#[derive(Clone)]
pub struct Or<T> {
    pub left: Box<dyn StlOperatorTrait<T> + 'static>,
    pub right: Box<dyn StlOperatorTrait<T> + 'static>,
}

impl<T: Clone> StlOperatorTrait<T> for Or<T> {
    fn robustness(&mut self, step: &Step<T>) -> Option<f64> {
        self.left
            .robustness(step)
            .zip(self.right.robustness(step))
            .map(|(l, r)| l.max(r))
    }
    fn to_string(&self) -> String {
        format!("({}) v ({})", self.left.to_string(), self.right.to_string())
    }
}

#[derive(Clone)]
pub struct Not<T> {
    pub operand: Box<dyn StlOperatorTrait<T> + 'static>,
}

impl<T: Clone> StlOperatorTrait<T> for Not<T> {
    fn robustness(&mut self, step: &Step<T>) -> Option<f64> {
        self.operand.robustness(step).map(|r| -r)
    }
    fn to_string(&self) -> String {
        format!("¬({})", self.operand.to_string())
    }
}

#[derive(Clone)]
pub struct Eventually<T, C>
where
    C: RingBufferTrait<Value = Option<f64>> + Clone,
{
    pub interval: TimeInterval,
    pub operand: Box<dyn StlOperatorTrait<T> + 'static>,
    pub cache: C,
}

impl<T, C: RingBufferTrait<Value = Option<f64>>> StlOperatorTrait<T> for Eventually<T, C>
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
    fn to_string(&self) -> String {
        format!(
            "F[{}, {}]({})",
            self.interval.start.as_secs_f64(),
            self.interval.end.as_secs_f64(),
            self.operand.to_string()
        )
    }
}

#[derive(Clone)]
pub struct Globally<T, C>
where
    C: RingBufferTrait<Value = Option<f64>> + Clone,
{
    pub interval: TimeInterval,
    pub operand: Box<dyn StlOperatorTrait<T> + 'static>,
    pub cache: C,
}

impl<T, C: RingBufferTrait<Value = Option<f64>>> StlOperatorTrait<T> for Globally<T, C>
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
    fn to_string(&self) -> String {
        format!(
            "G[{}, {}]({})",
            self.interval.start.as_secs_f64(),
            self.interval.end.as_secs_f64(),
            self.operand.to_string()
        )
    }
}

#[derive(Clone)]
pub struct Until<T, C>
where
    C: RingBufferTrait<Value = f64> + Clone,
{
    pub interval: TimeInterval,
    pub left: Box<dyn StlOperatorTrait<T> + 'static>,
    pub right: Box<dyn StlOperatorTrait<T> + 'static>,
    pub cache: C,
}

impl<T, C: RingBufferTrait<Value = f64>> StlOperatorTrait<T> for Until<T, C>
where
    T: Clone,
    C: RingBufferTrait<Value = f64> + Clone,
{
    fn robustness(&mut self, step: &Step<T>) -> Option<f64> {
        let left_robustness = self.left.robustness(step)?;
        let right_robustness = self.right.robustness(step)?;

        self.cache.add_step(left_robustness, step.timestamp);

        // The window of interest for the left operand's past robustness values
        let window_start = step.timestamp.saturating_sub(self.interval.end);
        let window_end = step.timestamp.saturating_sub(self.interval.start);

        // We can only compute a result if we have data extending back to the start of the window.
        if self
            .cache
            .get_back()
            .map_or(false, |h| h.timestamp <= window_start)
        {
            // Compute the minimum left robustness in the window
            let min_left_robustness = self
                .cache
                .iter()
                .filter(|entry| entry.timestamp >= window_start && entry.timestamp <= window_end)
                .map(|entry| entry.value) // From Step<f64> to f64
                .fold(f64::INFINITY, f64::min);

            Some(right_robustness.min(min_left_robustness))
        } else {
            None // Not enough historical data to compute robustness
        }
    }
    fn to_string(&self) -> String {
        format!(
            "({}) U[{}, {}] ({})",
            self.left.to_string(),
            self.interval.start.as_secs_f64(),
            self.interval.end.as_secs_f64(),
            self.right.to_string()
        )
    }
}

#[derive(Clone)]
pub struct Implies<T> {
    pub antecedent: Box<dyn StlOperatorTrait<T> + 'static>,
    pub consequent: Box<dyn StlOperatorTrait<T> + 'static>,
}

impl<T: Clone> StlOperatorTrait<T> for Implies<T> {
    fn robustness(&mut self, step: &Step<T>) -> Option<f64> {
        return self
            .antecedent
            .robustness(step)
            .zip(self.consequent.robustness(step))
            .map(|(a, c)| (-a).max(c));
    }
    fn to_string(&self) -> String {
        format!(
            "({}) -> ({})",
            self.antecedent.to_string(),
            self.consequent.to_string()
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

impl<T> StlOperatorTrait<T> for Atomic
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
    fn to_string(&self) -> String {
        match self {
            Atomic::True => "True".to_string(),
            Atomic::False => "False".to_string(),
            Atomic::GreaterThan(val) => format!("x > {}", val),
            Atomic::LessThan(val) => format!("x < {}", val),
        }
    }
}
