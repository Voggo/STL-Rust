//! Statistically-dispatched incremental operator tree.
//!
//! This internal enum composes concrete operator implementations recursively,
//! removing per-node vtable dispatch from hot `update()` paths while keeping
//! the public monitor API unchanged.

use crate::ring_buffer::{RingBuffer, Step};
use crate::stl::core::{RobustnessSemantics, SignalIdentifier, StlOperatorTrait};
use crate::stl::operators::atomic_operators::Atomic;
use crate::stl::operators::binary_operators::{And, Or};
use crate::stl::operators::not_operator::Not;
use crate::stl::operators::unary_temporal_operators::{Eventually, Globally};
use crate::stl::operators::until_operator::Until;
use std::fmt::{Debug, Display};
use std::time::Duration;

#[derive(Clone)]
pub enum CompiledOperator<T, Y, const IS_EAGER: bool, const IS_ROSI: bool>
where
    T: Into<f64> + Copy + Clone + 'static,
    Y: RobustnessSemantics + Debug + Copy + 'static,
{
    Atomic(Atomic<Y>),
    Not(Not<T, Y, Box<Self>>),
    And(And<T, RingBuffer<Y>, Y, IS_EAGER, IS_ROSI, Box<Self>, Box<Self>>),
    Or(Or<T, RingBuffer<Y>, Y, IS_EAGER, IS_ROSI, Box<Self>, Box<Self>>),
    Eventually(Eventually<T, RingBuffer<Y>, Y, IS_EAGER, IS_ROSI, Box<Self>>),
    Globally(Globally<T, RingBuffer<Y>, Y, IS_EAGER, IS_ROSI, Box<Self>>),
    Until(Until<T, RingBuffer<Y>, Y, IS_EAGER, IS_ROSI, Box<Self>, Box<Self>>),
}

impl<T, Y, const IS_EAGER: bool, const IS_ROSI: bool> StlOperatorTrait<T>
    for CompiledOperator<T, Y, IS_EAGER, IS_ROSI>
where
    T: Into<f64> + Copy + Clone + 'static,
    Y: RobustnessSemantics + Debug + Copy + 'static,
{
    type Output = Y;

    fn update(&mut self, step: &Step<T>) -> Vec<Step<Self::Output>> {
        match self {
            Self::Atomic(op) => <Atomic<Y> as StlOperatorTrait<T>>::update(op, step),
            Self::Not(op) => <Not<T, Y, Box<Self>> as StlOperatorTrait<T>>::update(op, step),
            Self::And(op) => {
                <And<T, RingBuffer<Y>, Y, IS_EAGER, IS_ROSI, Box<Self>, Box<Self>> as StlOperatorTrait<T>>::update(op, step)
            }
            Self::Or(op) => {
                <Or<T, RingBuffer<Y>, Y, IS_EAGER, IS_ROSI, Box<Self>, Box<Self>> as StlOperatorTrait<T>>::update(op, step)
            }
            Self::Eventually(op) => {
                <Eventually<T, RingBuffer<Y>, Y, IS_EAGER, IS_ROSI, Box<Self>> as StlOperatorTrait<T>>::update(op, step)
            }
            Self::Globally(op) => {
                <Globally<T, RingBuffer<Y>, Y, IS_EAGER, IS_ROSI, Box<Self>> as StlOperatorTrait<T>>::update(op, step)
            }
            Self::Until(op) => {
                <Until<T, RingBuffer<Y>, Y, IS_EAGER, IS_ROSI, Box<Self>, Box<Self>> as StlOperatorTrait<T>>::update(op, step)
            }
        }
    }

    fn get_max_lookahead(&self) -> Duration {
        match self {
            Self::Atomic(op) => <Atomic<Y> as StlOperatorTrait<T>>::get_max_lookahead(op),
            Self::Not(op) => {
                <Not<T, Y, Box<Self>> as StlOperatorTrait<T>>::get_max_lookahead(op)
            }
            Self::And(op) => {
                <And<T, RingBuffer<Y>, Y, IS_EAGER, IS_ROSI, Box<Self>, Box<Self>> as StlOperatorTrait<T>>::get_max_lookahead(op)
            }
            Self::Or(op) => {
                <Or<T, RingBuffer<Y>, Y, IS_EAGER, IS_ROSI, Box<Self>, Box<Self>> as StlOperatorTrait<T>>::get_max_lookahead(op)
            }
            Self::Eventually(op) => {
                <Eventually<T, RingBuffer<Y>, Y, IS_EAGER, IS_ROSI, Box<Self>> as StlOperatorTrait<T>>::get_max_lookahead(op)
            }
            Self::Globally(op) => {
                <Globally<T, RingBuffer<Y>, Y, IS_EAGER, IS_ROSI, Box<Self>> as StlOperatorTrait<T>>::get_max_lookahead(op)
            }
            Self::Until(op) => {
                <Until<T, RingBuffer<Y>, Y, IS_EAGER, IS_ROSI, Box<Self>, Box<Self>> as StlOperatorTrait<T>>::get_max_lookahead(op)
            }
        }
    }
}

impl<T, Y, const IS_EAGER: bool, const IS_ROSI: bool> SignalIdentifier
    for CompiledOperator<T, Y, IS_EAGER, IS_ROSI>
where
    T: Into<f64> + Copy + Clone + 'static,
    Y: RobustnessSemantics + Debug + Copy + 'static,
{
    fn get_signal_identifiers(&mut self) -> std::collections::HashSet<&'static str> {
        match self {
            Self::Atomic(op) => op.get_signal_identifiers(),
            Self::Not(op) => op.get_signal_identifiers(),
            Self::And(op) => op.get_signal_identifiers(),
            Self::Or(op) => op.get_signal_identifiers(),
            Self::Eventually(op) => op.get_signal_identifiers(),
            Self::Globally(op) => op.get_signal_identifiers(),
            Self::Until(op) => op.get_signal_identifiers(),
        }
    }
}

impl<T, Y, const IS_EAGER: bool, const IS_ROSI: bool> Display
    for CompiledOperator<T, Y, IS_EAGER, IS_ROSI>
where
    T: Into<f64> + Copy + Clone + 'static,
    Y: RobustnessSemantics + Debug + Copy + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Atomic(op) => write!(f, "{}", op),
            Self::Not(op) => write!(f, "{}", op),
            Self::And(op) => write!(f, "{}", op),
            Self::Or(op) => write!(f, "{}", op),
            Self::Eventually(op) => write!(f, "{}", op),
            Self::Globally(op) => write!(f, "{}", op),
            Self::Until(op) => write!(f, "{}", op),
        }
    }
}
