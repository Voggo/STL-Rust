use crate::ring_buffer::{RingBufferTrait, Step};
use crate::stl::core::{
    RobustnessSemantics, StlOperatorTrait, TimeInterval
};
use std::fmt::Display;

#[derive(Clone)]
pub struct And<T, C, Y> {
    pub left: Box<dyn StlOperatorTrait<T, Output = Y>>,
    pub right: Box<dyn StlOperatorTrait<T, Output = Y>>,
    pub eval_buffer: C,
}

impl<T, C, Y> StlOperatorTrait<T> for And<T, C, Y>
where
    T: Clone + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
    Y: RobustnessSemantics + 'static,
{
    type Output = Y;

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>>{
        // Get the robustness of the left and right children
        let left_robustness = self.left.robustness(step);
        let right_robustness = self.right.robustness(step);

        // left_robustness
        //     .zip(right_robustness)
        //     .map(|(l, r)| Y::and(l, r))

        vec![]
    }
}

#[derive(Clone)]
pub struct Or<T, C, Y> {
    pub left: Box<dyn StlOperatorTrait<T, Output = Y>>,
    pub right: Box<dyn StlOperatorTrait<T, Output = Y>>,
    pub eval_buffer: C,
}

impl<T, C, Y> StlOperatorTrait<T> for Or<T, C, Y>
where
    T: Clone + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
    Y: RobustnessSemantics + 'static,
{
    type Output = Y;

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>>{
        // Get the robustness of the left and right children
        let left_robustness = self.left.robustness(step);
        let right_robustness = self.right.robustness(step);

        // left_robustness
        //     .zip(right_robustness)
        //     .map(|(l, r)| Y::or(l, r));

        vec![]
    }
}

#[derive(Clone)]
pub struct Not<T, C, Y> {
    pub operand: Box<dyn StlOperatorTrait<T, Output = Y>>,
    pub eval_buffer: C,
}

impl<T, C, Y> StlOperatorTrait<T> for Not<T, C, Y>
where
    T: Clone + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
    Y: RobustnessSemantics + 'static,
{
    type Output = Y;

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>>{
        // self.operand.robustness(step).map(Y::not)
        vec![]
    }
}

#[derive(Clone)]
pub struct Implies<T, C, Y> {
    pub antecedent: Box<dyn StlOperatorTrait<T, Output = Y>>,
    pub consequent: Box<dyn StlOperatorTrait<T, Output = Y>>,
    pub eval_buffer: C,
}

impl<T, C, Y> StlOperatorTrait<T> for Implies<T, C, Y>
where
    T: Clone + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
    Y: RobustnessSemantics + 'static,
{
    type Output = Y;

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>>{
        // Get the robustness of the antecedent and consequent
        let antecedent_robustness = self.antecedent.robustness(step);
        let consequent_robustness = self.consequent.robustness(step);

        // antecedent_robustness
        //     .zip(consequent_robustness)
        //     .map(|(a, c)| Y::implies(a, c))

        vec![]
    }
}

#[derive(Clone)]
pub struct Eventually<T, C, Y> {
    pub interval: TimeInterval,
    pub operand: Box<dyn StlOperatorTrait<T, Output = Y>>,
    pub cache: C,
    pub eval_buffer: C,
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

    fn robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>>{
        // Use the identity and combining function from the trait
        let sub_robustness_vec = self.operand.robustness(step).to_vec();

        sub_robustness_vec
            .into_iter()
            .for_each(|step| self.cache.add_step(step));

        let t = step.timestamp.saturating_sub(self.interval.end);
        let lower_bound = t + self.interval.start;
        let upper_bound = t + self.interval.end;

        vec![]
    }
}

#[derive(Clone)]
pub struct Globally<T, Y, C> {
    pub interval: TimeInterval,
    pub operand: Box<dyn StlOperatorTrait<T, Output = Y> + 'static>,
    pub cache: C,
    pub eval_buffer: C,
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

    fn robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>>{
        let sub_robustness_vec = self.operand.robustness(step).to_vec();

        // sub_robustness_vec
        //     .into_iter()
        //     .for_each(|step| self.cache().add_step(step));

        // let t = step.timestamp.saturating_sub(self.interval().end);
        // let lower_bound = t + self.interval().start;
        // let upper_bound = t + self.interval().end;

        // if self.is_cache_sufficient(lower_bound, upper_bound, t) {
        //     let result = self
        //         .cache()
        //         .iter()
        //         .filter(|entry| entry.timestamp >= lower_bound && entry.timestamp <= upper_bound)
        //         .filter_map(|entry| entry.value.clone())
        //         .fold(Y::globally_identity(), Y::and);
        // };
        vec![]
    }
}

#[derive(Clone)]
pub struct Until<T, C, Y> {
    pub interval: TimeInterval,
    pub left: Box<dyn StlOperatorTrait<T, Output = Y> + 'static>,
    pub right: Box<dyn StlOperatorTrait<T, Output = Y> + 'static>,
    pub cache: C,
    pub eval_buffer: C,
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

    fn robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>>{
        let right_robustness = self.right.robustness(step);
        // self.cache
        //     .add_step(self.left.robustness(step), step.timestamp);

        // // The window of interest for the left operand's past robustness values
        // let t = step.timestamp.saturating_sub(self.interval.end);
        // let lower_bound_t_prime = t + self.interval.start;
        // let upper_bound_t_prime = t + self.interval.end;

        // // Ensure we have enough data to evaluate the window
        // if self.is_cache_sufficient(lower_bound_t_prime, upper_bound_t_prime, t) {
        //     let max_robustness = self
        //         .cache
        //         .iter()
        //         .filter(|entry| {
        //             entry.timestamp >= lower_bound_t_prime && entry.timestamp <= upper_bound_t_prime
        //         })
        //         .map(|entry| {
        //             let t_prime = entry.timestamp;
        //             let min_left_robustness = self
        //                 .cache
        //                 .iter()
        //                 .filter(|e| e.timestamp >= lower_bound_t_prime && e.timestamp <= t_prime)
        //                 .filter_map(|e| e.value.clone())
        //                 .fold(Y::globally_identity(), Y::and);

        //             Y::and(right_robustness.clone(), min_left_robustness) // OBS: Using clone() here !!!!!!!!!!!!!!!!!!!!!! Should maybe be changed
        //         })
        //         .fold(Y::eventually_identity(), Y::or);

        //     Some(max_robustness)
        // } else {
        //     None // Not enough data to evaluate
        // }
        vec![]
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
    fn robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>>{
        let value = step.value.clone().into();
        match self {
            Atomic::True(_) => vec![Step {
                value: Some(Y::atomic_true()),
                timestamp: step.timestamp,
            }],
            Atomic::False(_) => vec![Step {
                value: Some(Y::atomic_false()),
                timestamp: step.timestamp,
            }],
            Atomic::GreaterThan(c, _) => vec![Step {
                value: Some(Y::atomic_greater_than(value, *c)),
                timestamp: step.timestamp,
            }],
            Atomic::LessThan(c, _) => vec![Step {
                value: Some(Y::atomic_less_than(value, *c)),
                timestamp: step.timestamp,
            }],
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
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
impl<T, C, Y> Display for And<T, C, Y> {
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

impl<T, C, Y> Display for Or<T, C, Y> {
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
impl<T, C, Y> Display for Not<T, C, Y> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "¬({})", self.operand.to_string())
    }
}
impl<T, C, Y> Display for Implies<T, C, Y> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}) → ({})",
            self.antecedent.to_string(),
            self.consequent.to_string()
        )
    }
}
