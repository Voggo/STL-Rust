use crate::ring_buffer::{RingBuffer, Step};
use crate::stl::core::{StlOperatorTrait, TimeInterval};
use crate::stl::robustness_cached::{And, Atomic, Eventually, Globally, Not, Or, Until}; // Added Until
use crate::stl::robustness_naive::{StlFormula, StlOperator};
use std::collections::VecDeque;
use std::marker::PhantomData;

// The input definition of the STL formula, independent of implementation.
// This mirrors the structure of the NaiveOperator enum for formula definition.
#[derive(Clone)]
pub enum FormulaDefinition {
    GreaterThan(f64), // signal > c, where c is the threshold
    LessThan(f64),    // signal < c, where c is the threshold
    True,             // Constant True
    False,            // Constant False
    And(Box<FormulaDefinition>, Box<FormulaDefinition>),
    Or(Box<FormulaDefinition>, Box<FormulaDefinition>),
    Not(Box<FormulaDefinition>),
    Eventually(TimeInterval, Box<FormulaDefinition>),
    Globally(TimeInterval, Box<FormulaDefinition>),
    Until(TimeInterval, Box<FormulaDefinition>, Box<FormulaDefinition>), // Added Until
}

/// Defines the monitoring strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MonitoringStrategy {
    Naive,       // O(W) performance, no short-circuiting, time-implicit output
    Incremental, // O(1) amortized performance, uses short-circuiting, time-explicit output
}

/// The final monitor struct that handles the input stream.
pub struct StlMonitor<T: Clone> {
    root_operator: Box<dyn StlOperatorTrait<T, Output = f64>>,
    pub strategy: MonitoringStrategy,
}

impl<T: Clone> StlMonitor<T> {
    /// Creates a new builder instance.
    pub fn builder() -> StlMonitorBuilder<T> {
        StlMonitorBuilder::new()
    }

    /// Processes a single input step and returns all finalized robustness results.
    /// This is the unified public interface.
    pub fn advance_and_get_results(&mut self, step: &Step<T>) -> VecDeque<Step<f64>> {
        // self.root_operator.advance_and_get_results(step)
        todo!("Implement advance_and_get_results for the root operator");
    }

    /// Returns the string representation of the monitor's formula.
    pub fn specification_to_string(&self) -> String {
        self.root_operator.to_string()
    }
}

/// The Builder pattern struct for StlMonitor.
pub struct StlMonitorBuilder<T> {
    formula: Option<FormulaDefinition>,
    strategy: MonitoringStrategy,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> StlMonitorBuilder<T> {
    pub fn new() -> Self {
        StlMonitorBuilder {
            formula: None,
            strategy: MonitoringStrategy::Incremental, // Default to fastest
            _phantom: std::marker::PhantomData,
        }
    }

    /// Sets the formula definition to be monitored.
    pub fn formula(mut self, formula: FormulaDefinition) -> Self {
        self.formula = Some(formula);
        self
    }

    /// Configures the monitoring strategy (Naive or Incremental).
    pub fn strategy(mut self, strategy: MonitoringStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Builds the final StlMonitor by recursively constructing the operator tree.
    pub fn build(self) -> Result<StlMonitor<T>, &'static str>
    where
        T: Into<f64> + Copy + 'static, // Add required bounds
    {
        let formula_def = self
            .formula
            .clone()
            .ok_or("Formula definition is required")?;

        // Factory pattern: Build the correct operator tree based on the strategy
        let root_operator = match self.strategy {
            MonitoringStrategy::Incremental => self.build_incremental_operator(formula_def),
            MonitoringStrategy::Naive => self.build_naive_operator(formula_def),
            _ => return Err("Unsupported monitoring strategy"),
        };

        Ok(StlMonitor {
            root_operator: root_operator,
            strategy: self.strategy,
        })
    }

    // --- Factory Methods ---

    fn build_naive_operator(
        &self,
        formula: FormulaDefinition,
    ) -> Box<dyn StlOperatorTrait<T, Output = f64>>
    where
        T: Into<f64> + Copy + 'static,
    {
        match formula {
            FormulaDefinition::GreaterThan(c) => Box::new(StlFormula::<T, RingBuffer<T>, f64> {
                formula: StlOperator::GreaterThan(c),
                signal: RingBuffer::new(),
                _phantom: std::marker::PhantomData,
            }),
            FormulaDefinition::LessThan(c) => Box::new(StlFormula::<T, RingBuffer<T>, f64> {
                formula: StlOperator::LessThan(c),
                signal: RingBuffer::new(),
                _phantom: std::marker::PhantomData,
            }),
            FormulaDefinition::True => Box::new(StlFormula::<T, RingBuffer<T>, f64> {
                formula: StlOperator::True,
                signal: RingBuffer::new(),
                _phantom: std::marker::PhantomData,
            }),
            FormulaDefinition::False => Box::new(StlFormula::<T, RingBuffer<T>, f64> {
                formula: StlOperator::False,
                signal: RingBuffer::new(),
                _phantom: std::marker::PhantomData,
            }),
            FormulaDefinition::And(l, r) => Box::new(StlFormula::<T, RingBuffer<T>, f64> {
                formula: StlOperator::And(
                    Box::new(
                        self.build_naive_operator(*l)
                            .as_any()
                            .downcast_ref::<StlFormula<T, RingBuffer<T>, f64>>()
                            .unwrap()
                            .formula
                            .clone(),
                    ),
                    Box::new(
                        self.build_naive_operator(*r)
                            .as_any()
                            .downcast_ref::<StlFormula<T, RingBuffer<T>, f64>>()
                            .unwrap()
                            .formula
                            .clone(),
                    ),
                ),
                signal: RingBuffer::new(),
                _phantom: std::marker::PhantomData,
            }),
            FormulaDefinition::Or(l, r) => Box::new(StlFormula::<T, RingBuffer<T>, f64> {
                formula: StlOperator::Or(
                    Box::new(
                        self.build_naive_operator(*l)
                            .as_any()
                            .downcast_ref::<StlFormula<T, RingBuffer<T>, f64>>()
                            .unwrap()
                            .formula
                            .clone(),
                    ),
                    Box::new(
                        self.build_naive_operator(*r)
                            .as_any()
                            .downcast_ref::<StlFormula<T, RingBuffer<T>, f64>>()
                            .unwrap()
                            .formula
                            .clone(),
                    ),
                ),
                signal: RingBuffer::new(),
                _phantom: std::marker::PhantomData,
            }),
            FormulaDefinition::Not(op) => Box::new(StlFormula::<T, RingBuffer<T>, f64> {
                formula: StlOperator::Not(Box::new(
                    self.build_naive_operator(*op)
                        .as_any()
                        .downcast_ref::<StlFormula<T, RingBuffer<T>, f64>>()
                        .unwrap()
                        .formula
                        .clone(),
                )),
                signal: RingBuffer::new(),
                _phantom: std::marker::PhantomData,
            }),
            FormulaDefinition::Eventually(i, op) => Box::new(StlFormula::<T, RingBuffer<T>, f64> {
                formula: StlOperator::Eventually(
                    i,
                    Box::new(
                        self.build_naive_operator(*op)
                            .as_any()
                            .downcast_ref::<StlFormula<T, RingBuffer<T>, f64>>()
                            .unwrap()
                            .formula
                            .clone(),
                    ),
                ),
                signal: RingBuffer::new(),
                _phantom: std::marker::PhantomData,
            }),
            FormulaDefinition::Globally(i, op) => Box::new(StlFormula::<T, RingBuffer<T>, f64> {
                formula: StlOperator::Globally(
                    i,
                    Box::new(
                        self.build_naive_operator(*op)
                            .as_any()
                            .downcast_ref::<StlFormula<T, RingBuffer<T>, f64>>()
                            .unwrap()
                            .formula
                            .clone(),
                    ),
                ),
                signal: RingBuffer::new(),
                _phantom: std::marker::PhantomData,
            }),
            FormulaDefinition::Until(i, l, r) => Box::new(StlFormula::<T, RingBuffer<T>, f64> {
                formula: StlOperator::Until(
                    i,
                    Box::new(
                        self.build_naive_operator(*l)
                            .as_any()
                            .downcast_ref::<StlFormula<T, RingBuffer<T>, f64>>()
                            .unwrap()
                            .formula
                            .clone(),
                    ),
                    Box::new(
                        self.build_naive_operator(*r)
                            .as_any()
                            .downcast_ref::<StlFormula<T, RingBuffer<T>, f64>>()
                            .unwrap()
                            .formula
                            .clone(),
                    ),
                ),
                signal: RingBuffer::new(),
                _phantom: std::marker::PhantomData,
            }),
        }
    }

    fn build_incremental_operator(
        &self,
        formula: FormulaDefinition,
    ) -> Box<dyn StlOperatorTrait<T, Output = f64>>
    where
        T: Into<f64> + Copy + 'static,
    {
        match formula {
            FormulaDefinition::GreaterThan(c) => Box::new(Atomic::GreaterThan(c, PhantomData)),
            FormulaDefinition::LessThan(c) => Box::new(Atomic::LessThan(c, PhantomData)),
            FormulaDefinition::True => Box::new(Atomic::True(PhantomData)),
            FormulaDefinition::False => Box::new(Atomic::False(PhantomData)),
            FormulaDefinition::And(l, r) => Box::new(And {
                left: self.build_incremental_operator(*l),
                right: self.build_incremental_operator(*r),
            }),
            FormulaDefinition::Or(l, r) => Box::new(Or {
                left: self.build_incremental_operator(*l),
                right: self.build_incremental_operator(*r),
            }),
            FormulaDefinition::Not(op) => Box::new(Not {
                operand: self.build_incremental_operator(*op),
            }),
            FormulaDefinition::Eventually(i, op) => Box::new(Eventually {
                interval: i,
                operand: self.build_incremental_operator(*op),
                cache: RingBuffer::new(),
            }),
            FormulaDefinition::Globally(i, op) => Box::new(Globally {
                interval: i,
                operand: self.build_incremental_operator(*op),
                cache: RingBuffer::new(),
            }),
            FormulaDefinition::Until(i, l, r) => Box::new(Until {
                interval: i,
                left: self.build_incremental_operator(*l),
                right: self.build_incremental_operator(*r),
                cache: RingBuffer::new(),
            }),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::stl::monitor::{
        FormulaDefinition, MonitoringStrategy, StlMonitor,
    };
    use std::time::Duration;

    #[test]
    fn test_build() {
        let formula = FormulaDefinition::And(
            Box::new(FormulaDefinition::GreaterThan(5.0)),
            Box::new(FormulaDefinition::Eventually(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(10),
                },
                Box::new(FormulaDefinition::LessThan(10.0)),
            )),
        );

        let monitor: StlMonitor<f64> = StlMonitor::builder()
            .formula(formula.clone())
            .strategy(MonitoringStrategy::Incremental)
            .build()
            .unwrap();

        let monitor_naive: StlMonitor<f64> = StlMonitor::builder()
            .formula(formula)
            .strategy(MonitoringStrategy::Naive)
            .build()
            .unwrap();

        assert_eq!(monitor.strategy, MonitoringStrategy::Incremental);
        assert_eq!(monitor_naive.strategy, MonitoringStrategy::Naive);

        let spec = monitor.specification_to_string();
        println!("Monitor Specification: {}", spec);

        let spec_naive = monitor_naive.specification_to_string();
        println!("Naive Monitor Specification: {}", spec_naive);

        assert_eq!(spec, spec_naive);
    }
}
