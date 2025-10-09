use crate::ring_buffer::{RingBufferTrait, Step};
use crate::stl::core::{
    BinaryTemporalOperatorTrait, RobustnessSemantics, StlOperatorTrait, TemporalOperatorBaseTrait,
    TimeInterval, UnaryTemporalOperatorTrait,
};
use std::fmt::Display;

#[derive(Clone)]
pub struct And<T, Y> {
    pub left: Box<dyn StlOperatorTrait<T, Output = Y>>,
    pub right: Box<dyn StlOperatorTrait<T, Output = Y>>,
}

impl<T, Y> StlOperatorTrait<T> for And<T, Y>
where
    T: Clone + 'static,
    Y: RobustnessSemantics + 'static,
{
    type Output = Y;

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn robustness(&mut self, step: &Step<T>) -> Option<Self::Output> {
        // Get the robustness of the left and right children
        let left_robustness = self.left.robustness(step);
        let right_robustness = self.right.robustness(step);

        left_robustness
            .zip(right_robustness)
            .map(|(l, r)| Y::and(l, r))
    }

    fn get_temporal_depth(&self) -> usize {
        let left_depth = self.left.get_temporal_depth();
        let right_depth = self.right.get_temporal_depth();
        left_depth.max(right_depth)
    }
}

#[derive(Clone)]
pub struct Or<T, Y> {
    pub left: Box<dyn StlOperatorTrait<T, Output = Y>>,
    pub right: Box<dyn StlOperatorTrait<T, Output = Y>>,
}

impl<T, Y> StlOperatorTrait<T> for Or<T, Y>
where
    T: Clone + 'static,
    Y: RobustnessSemantics + 'static,
{
    type Output = Y;

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn robustness(&mut self, step: &Step<T>) -> Option<Self::Output> {
        // Get the robustness of the left and right children
        let left_robustness = self.left.robustness(step);
        let right_robustness = self.right.robustness(step);

        left_robustness
            .zip(right_robustness)
            .map(|(l, r)| Y::or(l, r))
    }

    fn get_temporal_depth(&self) -> usize {
        let left_depth = self.left.get_temporal_depth();
        let right_depth = self.right.get_temporal_depth();
        left_depth.max(right_depth)
    }
}

#[derive(Clone)]
pub struct Not<T, Y> {
    pub operand: Box<dyn StlOperatorTrait<T, Output = Y>>,
}

impl<T, Y> StlOperatorTrait<T> for Not<T, Y>
where
    T: Clone + 'static,
    Y: RobustnessSemantics + 'static,
{
    type Output = Y;

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn robustness(&mut self, step: &Step<T>) -> Option<Self::Output> {
        self.operand.robustness(step).map(Y::not)
    }

    fn get_temporal_depth(&self) -> usize {
        self.operand.get_temporal_depth()
    }
}

#[derive(Clone)]
pub struct Implies<T, Y> {
    pub antecedent: Box<dyn StlOperatorTrait<T, Output = Y>>,
    pub consequent: Box<dyn StlOperatorTrait<T, Output = Y>>,
}

impl<T, Y> StlOperatorTrait<T> for Implies<T, Y>
where
    T: Clone + 'static,
    Y: RobustnessSemantics + 'static,
{
    type Output = Y;

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn robustness(&mut self, step: &Step<T>) -> Option<Self::Output> {
        // Get the robustness of the antecedent and consequent
        let antecedent_robustness = self.antecedent.robustness(step);
        let consequent_robustness = self.consequent.robustness(step);

        antecedent_robustness
            .zip(consequent_robustness)
            .map(|(a, c)| Y::implies(a, c))
    }

    fn get_temporal_depth(&self) -> usize {
        let antecedent_depth = self.antecedent.get_temporal_depth();
        let consequent_depth = self.consequent.get_temporal_depth();
        antecedent_depth.max(consequent_depth)
    }
}

#[derive(Clone)]
pub struct Eventually<T, C, Y> {
    pub interval: TimeInterval,
    pub operand: Box<dyn StlOperatorTrait<T, Output = Y>>,
    pub cache: C,
}

impl<T, C, Y> StlOperatorTrait<T> for Eventually<T, C, Y>
where
    T: Clone + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
    Y: RobustnessSemantics + 'static, // The crucial trait bound
{
    type Output = Y;

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn robustness(&mut self, step: &Step<T>) -> Option<Self::Output> {
        // Use the identity and combining function from the trait
        self.robustness_unary_with(step, Y::eventually_identity(), Y::or)
    }

    fn get_temporal_depth(&self) -> usize {
        self.operand.get_temporal_depth() + self.interval.end.as_secs() as usize
    }
}

impl<T, C, Y> TemporalOperatorBaseTrait<T, C> for Eventually<T, C, Y>
where
    T: Clone + 'static,
    Y: Clone + RobustnessSemantics + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
{
    fn interval(&self) -> TimeInterval {
        self.interval
    }

    fn cache(&mut self) -> &mut C {
        &mut self.cache
    }
}

impl<T, C, Y> UnaryTemporalOperatorTrait<T, C> for Eventually<T, C, Y>
where
    T: Clone + 'static,
    Y: Clone + RobustnessSemantics + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
{
    fn operand(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Output = Self::Output>> {
        &mut self.operand
    }
}

#[derive(Clone)]
pub struct Globally<T, Y, C> {
    pub interval: TimeInterval,
    pub operand: Box<dyn StlOperatorTrait<T, Output = Y> + 'static>,
    pub cache: C,
}

impl<T, C, Y> StlOperatorTrait<T> for Globally<T, Y, C>
where
    T: Clone + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
    Y: RobustnessSemantics + 'static,
{
    type Output = Y;

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn robustness(&mut self, step: &Step<T>) -> Option<Self::Output> {
        self.robustness_unary_with(step, Y::globally_identity(), Y::and)
    }

    fn get_temporal_depth(&self) -> usize {
        self.operand.get_temporal_depth() + self.interval.end.as_secs() as usize
    }
}

impl<T, C, Y> TemporalOperatorBaseTrait<T, C> for Globally<T, Y, C>
where
    T: Clone + 'static,
    Y: Clone + RobustnessSemantics + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
{
    fn interval(&self) -> TimeInterval {
        self.interval
    }

    fn cache(&mut self) -> &mut C {
        &mut self.cache
    }
}

impl<T, C, Y> UnaryTemporalOperatorTrait<T, C> for Globally<T, Y, C>
where
    T: Clone + 'static,
    Y: Clone + RobustnessSemantics + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
{
    fn operand(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Output = Self::Output>> {
        &mut self.operand
    }
}

#[derive(Clone)]
pub struct Until<T, C, Y> {
    pub interval: TimeInterval,
    pub left: Box<dyn StlOperatorTrait<T, Output = Y> + 'static>,
    pub right: Box<dyn StlOperatorTrait<T, Output = Y> + 'static>,
    pub cache: C,
}

impl<T, C, Y> StlOperatorTrait<T> for Until<T, C, Y>
where
    T: Clone + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
    Y: RobustnessSemantics + 'static,
{
    type Output = Y;

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn robustness(&mut self, step: &Step<T>) -> Option<Self::Output> {
        let right_robustness = self.right.robustness(step)?;
        self.cache
            .add_step(self.left.robustness(step), step.timestamp);

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
                        .filter_map(|e| e.value.clone())
                        .fold(Y::globally_identity(), Y::and);

                    Y::and(right_robustness.clone(), min_left_robustness) // OBS: Using clone() here !!!!!!!!!!!!!!!!!!!!!! Should maybe be changed
                })
                .fold(Y::eventually_identity(), Y::or);

            Some(max_robustness)
        } else {
            None // Not enough data to evaluate
        }
    }

    fn get_temporal_depth(&self) -> usize {
        let left_depth = self.left.get_temporal_depth();
        let right_depth = self.right.get_temporal_depth();
        left_depth.max(right_depth) + self.interval.end.as_secs() as usize
    }
}
impl<T, C, Y> TemporalOperatorBaseTrait<T, C> for Until<T, C, Y>
where
    T: Clone + 'static,
    Y: Clone + RobustnessSemantics + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
{
    fn interval(&self) -> TimeInterval {
        self.interval
    }

    fn cache(&mut self) -> &mut C {
        &mut self.cache
    }
}

impl<T, C, Y> BinaryTemporalOperatorTrait<T, C> for Until<T, C, Y>
where
    T: Clone + 'static,
    Y: Clone + RobustnessSemantics + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
{
    fn left(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Output = Self::Output>> {
        &mut self.left
    }

    fn right(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Output = Self::Output>> {
        &mut self.right
    }
}

#[derive(Clone)]
pub enum Atomic<Y> {
    LessThan(f64, std::marker::PhantomData<Y>),
    GreaterThan(f64, std::marker::PhantomData<Y>),
    True(std::marker::PhantomData<Y>),
    False(std::marker::PhantomData<Y>),
}

impl<Y> Atomic<Y> {
    pub fn new_less_than(val: f64) -> Self {
        Atomic::LessThan(val, std::marker::PhantomData)
    }
    pub fn new_greater_than(val: f64) -> Self {
        Atomic::GreaterThan(val, std::marker::PhantomData)
    }
    pub fn new_true() -> Self {
        Atomic::True(std::marker::PhantomData)
    }
    pub fn new_false() -> Self {
        Atomic::False(std::marker::PhantomData)
    }
}

impl<T, Y> StlOperatorTrait<T> for Atomic<Y>
where
    T: Into<f64> + Clone + 'static,
    Y: RobustnessSemantics + 'static,
{
    type Output = Y;
    fn robustness(&mut self, step: &Step<T>) -> Option<Self::Output> {
        let value = step.value.clone().into();
        match self {
            Atomic::True(_) => Some(Y::atomic_true()),
            Atomic::False(_) => Some(Y::atomic_false()),
            Atomic::GreaterThan(c, _) => Some(Y::atomic_greater_than(value, *c)),
            Atomic::LessThan(c, _) => Some(Y::atomic_less_than(value, *c)),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn get_temporal_depth(&self) -> usize {
        0
    }
}

impl<Y> Display for Atomic<Y> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Atomic::LessThan(c, _) => write!(f, "x < {}", c),
            Atomic::GreaterThan(c, _) => write!(f, "x > {}", c),
            Atomic::True(_) => write!(f, "True"),
            Atomic::False(_) => write!(f, "False"),
        }
    }
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
impl<T, Y> Display for Not<T, Y> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "¬({})", self.operand.to_string())
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
