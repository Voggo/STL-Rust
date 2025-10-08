use crate::ring_buffer::{RingBufferTrait, Step};
use std::time::Duration;

// Time interval type
#[derive(Debug, Copy, Clone)]
pub struct TimeInterval {
    pub start: Duration,
    pub end: Duration,
}

// A generic representation of an STL formula.
pub trait StlOperator<T: Clone> {
    fn robustness(&self, step: &Step<T>) -> f64;
    fn to_string(&self) -> String;
}

pub struct And<T> {
    pub left: Box<dyn StlOperator<T>>,
    pub right: Box<dyn StlOperator<T>>,
}

impl<T: Clone> StlOperator<T> for And<T> {
    fn robustness(&self, step: &Step<T>) -> f64 {
        return self
            .left
            .robustness(step)
            .min(self.right.robustness(step));
    }
    fn to_string(&self) -> String {
        format!("({}) ∧ ({})", self.left.to_string(), self.right.to_string())
    }
}

pub struct Or<T> {
    pub left: Box<dyn StlOperator<T>>,
    pub right: Box<dyn StlOperator<T>>,
}

impl<T: Clone> StlOperator<T> for Or<T> {
    fn robustness(&self, step: &Step<T>) -> f64 {
        return self
            .left
            .robustness(step)
            .max(self.right.robustness(step));
    }
    fn to_string(&self) -> String {
        format!("({}) v ({})", self.left.to_string(), self.right.to_string())
    }
}

pub struct Not<T> {
    pub operand: Box<dyn StlOperator<T>>,
}

impl<T: Clone> StlOperator<T> for Not<T> {
    fn robustness(&self, step: &Step<T>) -> f64 {
        return -self.operand.robustness(step);
    }
    fn to_string(&self) -> String {
        format!("¬({})", self.operand.to_string())
    }
}

pub struct Eventually<T, C>
where
    C: RingBufferTrait<Value = f64>,
{
    pub interval: TimeInterval,
    pub operand: Box<dyn StlOperator<T>>,
    pub cache: C,
}

impl<T: Clone, C: RingBufferTrait<Value = f64>> StlOperator<T> for Eventually<T, C>
where
    C: RingBufferTrait<Value = f64>,
{
    fn robustness(&self, step: &Step<T>) -> f64 {
        todo!("Implement robustness calculation for Eventually operator")
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

pub struct Globally<T, C>
where
    C: RingBufferTrait<Value = f64>,
{
    pub interval: TimeInterval,
    pub operand: Box<dyn StlOperator<T>>,
    pub cache: C,
}

impl<T: Clone, C: RingBufferTrait<Value = f64>> StlOperator<T> for Globally<T, C>
where
    C: RingBufferTrait<Value = f64>,
{
    fn robustness(&self, step: &Step<T>) -> f64 {
        todo!("Implement robustness calculation for Always operator")
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

pub struct Until<T, C>
where
    C: RingBufferTrait<Value = f64>,
{
    pub interval: TimeInterval,
    pub left: Box<dyn StlOperator<T>>,
    pub right: Box<dyn StlOperator<T>>,
    pub cache: C,
}

impl<T: Clone, C: RingBufferTrait<Value = f64>> StlOperator<T> for Until<T, C>
where
    C: RingBufferTrait<Value = f64>,
{
    fn robustness(&self, step: &Step<T>) -> f64 {
        todo!("Implement robustness calculation for Until operator")
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

pub struct Implies<T> {
    pub antecedent: Box<dyn StlOperator<T>>,
    pub consequent: Box<dyn StlOperator<T>>,
}

impl<T: Clone> StlOperator<T> for Implies<T> {
    fn robustness(&self, step: &Step<T>) -> f64 {
        return -self
            .antecedent
            .robustness(step)
            .min(self.consequent.robustness(step));
    }
    fn to_string(&self) -> String {
        format!(
            "({}) -> ({})",
            self.antecedent.to_string(),
            self.consequent.to_string()
        )
    }
}

pub enum Atomic {
    LessThan(f64),
    GreaterThan(f64),
    True,
    False,
}

impl<T: Into<f64> + Clone> StlOperator<T> for Atomic {
    fn robustness(&self, step: &Step<T>) -> f64 
    {
        let value = step.value.clone().into();
        match self {
            Atomic::True => f64::INFINITY,
            Atomic::False => f64::NEG_INFINITY,
            Atomic::GreaterThan(c) => value - c,
            Atomic::LessThan(c) => c - value,
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
