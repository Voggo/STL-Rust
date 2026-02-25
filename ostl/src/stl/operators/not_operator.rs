//! Unary logical negation operator.
//!
//! This module provides [`Not`], which wraps a single child operator and applies
//! [`RobustnessSemantics::not`] to each emitted value.

use crate::ring_buffer::Step;
use crate::stl::core::{
    RobustnessSemantics, SignalIdentifier, StlOperatorAndSignalIdentifier, StlOperatorTrait,
};
use std::collections::HashSet;
use std::fmt::Display;
use std::marker::PhantomData;
use std::time::Duration;

#[derive(Clone)]
/// STL negation operator `¬φ`.
pub struct Not<T, Y, O = Box<dyn StlOperatorAndSignalIdentifier<T, Y>>> {
    operand: O,
    max_lookahead: Duration,
    _phantom: PhantomData<(T, Y)>,
}

impl<T, Y, O> Not<T, Y, O> {
    /// Creates a new negation operator from a child operand.
    ///
    /// The resulting lookahead equals the child's lookahead.
    pub fn new(operand: O) -> Self
    where
        T: Clone + 'static,
        Y: RobustnessSemantics + 'static,
        O: StlOperatorAndSignalIdentifier<T, Y>,
    {
        let max_lookahead = operand.get_max_lookahead();
        Self {
            operand,
            max_lookahead,
            _phantom: PhantomData,
        }
    }
}

impl<T, Y, O> StlOperatorTrait<T> for Not<T, Y, O>
where
    T: Clone + 'static,
    Y: RobustnessSemantics + 'static,
    O: Clone + StlOperatorAndSignalIdentifier<T, Y>,
{
    type Output = Y;

    fn get_max_lookahead(&self) -> Duration {
        self.max_lookahead
    }

    /// Updates the child operator and negates each produced value.
    fn update(&mut self, step: &Step<T>) -> Vec<Step<Self::Output>> {
        let operand_updates = self.operand.update(step);

        let output_robustness: Vec<Step<Y>> = operand_updates
            .into_iter()
            .map(|step| {
                let negated_value = Y::not(step.value);
                Step {
                    signal: "output",
                    value: negated_value,
                    timestamp: step.timestamp,
                }
            })
            .collect();

        output_robustness
    }
}

impl<T, Y, O> SignalIdentifier for Not<T, Y, O>
where
    T: Clone,
    O: Clone + StlOperatorAndSignalIdentifier<T, Y>,
{
    /// Returns the signal identifiers of the wrapped operand.
    fn get_signal_identifiers(&mut self) -> HashSet<&'static str> {
        self.operand.get_signal_identifiers()
    }
}

impl<T, Y, O> Display for Not<T, Y, O>
where
    T: Clone,
    O: Clone + StlOperatorAndSignalIdentifier<T, Y>,
{
    /// Formats as `¬(operand)`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "¬({})", self.operand)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring_buffer::Step;
    use crate::stl::core::StlOperatorTrait;
    use crate::stl::operators::atomic_operators::Atomic;
    use pretty_assertions::assert_eq;
    use std::time::Duration;

    // Logical operators
    #[test]
    fn not_operator_robustness() {
        let atomic = Atomic::<f64>::new_greater_than("x", 10.0);
        let mut not = Not::new(Box::new(atomic));
        not.get_signal_identifiers();
        let step = Step::new("x", 15.0, Duration::from_secs(5));
        let robustness = not.update(&step);
        assert_eq!(
            robustness,
            vec![Step::new("output", -5.0, Duration::from_secs(5))]
        );
    }

    #[test]
    fn not_signal_identifiers() {
        let atomic = Atomic::<f64>::new_greater_than("x", 10.0);
        let mut not: Not<f64, f64> = Not::new(Box::new(atomic));
        let ids = not.get_signal_identifiers();
        let expected_ids: HashSet<&'static str> = vec!["x"].into_iter().collect();
        assert_eq!(ids, expected_ids);
    }

    #[test]
    fn not_display() {
        let atomic = Atomic::<f64>::new_greater_than("x", 10.0);
        let not: Not<f64, f64> = Not::new(Box::new(atomic));
        let display_str = format!("{}", not);
        assert_eq!(display_str, "¬(x > 10)");
    }
}
