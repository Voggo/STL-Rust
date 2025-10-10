use crate::ring_buffer::{RingBuffer, Step};
use crate::stl::core::{RobustnessSemantics, StlOperatorTrait, TimeInterval};
use crate::stl::robustness_cached::{And, Atomic, Eventually, Globally, Implies, Not, Or, Until};
use crate::stl::robustness_naive::{StlFormula, StlOperator};

// The input definition of the STL formula, independent of implementation.
// This mirrors the structure of the NaiveOperator enum for formula definition.
#[derive(Clone)]
pub enum FormulaDefinition {
    GreaterThan(f64),
    LessThan(f64),
    True,
    False,
    And(Box<FormulaDefinition>, Box<FormulaDefinition>),
    Or(Box<FormulaDefinition>, Box<FormulaDefinition>),
    Not(Box<FormulaDefinition>),
    Implies(Box<FormulaDefinition>, Box<FormulaDefinition>),
    Eventually(TimeInterval, Box<FormulaDefinition>),
    Globally(TimeInterval, Box<FormulaDefinition>),
    Until(TimeInterval, Box<FormulaDefinition>, Box<FormulaDefinition>),
}

/// Defines the monitoring strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MonitoringStrategy {
    Naive,
    Incremental,
}

/// The final monitor struct that handles the input stream.
pub struct StlMonitor<T: Clone, Y> {
    root_operator: Box<dyn StlOperatorTrait<T, Output = Y>>,
    pub strategy: MonitoringStrategy,
}

impl<T: Clone, Y> StlMonitor<T, Y> {
    /// Creates a new builder instance.
    pub fn builder() -> StlMonitorBuilder<T, Y> {
        StlMonitorBuilder::new()
    }

    /// Computes the instantaneous robustness for the current step.
    pub fn instantaneous_robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Y>>> {
        self.root_operator.robustness(step)
    }

    /// Returns the string representation of the monitor's formula.
    pub fn specification_to_string(&self) -> String {
        self.root_operator.to_string()
    }
}

/// The Builder pattern struct for StlMonitor.
pub struct StlMonitorBuilder<T, Y> {
    formula: Option<FormulaDefinition>,
    strategy: MonitoringStrategy,
    _phantom: std::marker::PhantomData<(T, Y)>,
}

impl<T, Y> StlMonitorBuilder<T, Y> {
    pub fn new() -> Self {
        StlMonitorBuilder {
            formula: None,
            strategy: MonitoringStrategy::Incremental, // Default
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
    pub fn build(self) -> Result<StlMonitor<T, Y>, &'static str>
    where
        T: Into<f64> + Copy + 'static,           // Add required bounds
        Y: RobustnessSemantics + Copy + 'static, // Add required bounds
    {
        let formula_def = self
            .formula
            .clone()
            .ok_or("Formula definition is required")?;

        // Factory pattern: Build the correct operator tree based on the strategy
        let root_operator = match self.strategy {
            MonitoringStrategy::Incremental => self.build_incremental_operator(formula_def),
            MonitoringStrategy::Naive => self.build_naive_operator(formula_def),
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
    ) -> Box<dyn StlOperatorTrait<T, Output = Y>>
    where
        T: Into<f64> + Copy + 'static,
        Y: RobustnessSemantics + 'static,
    {
        match formula {
            FormulaDefinition::GreaterThan(c) => Box::new(StlFormula::<T, RingBuffer<T>, Y> {
                formula: StlOperator::GreaterThan(c),
                signal: RingBuffer::new(),
                _phantom: std::marker::PhantomData,
            }),
            FormulaDefinition::LessThan(c) => Box::new(StlFormula::<T, RingBuffer<T>, Y> {
                formula: StlOperator::LessThan(c),
                signal: RingBuffer::new(),
                _phantom: std::marker::PhantomData,
            }),
            FormulaDefinition::True => Box::new(StlFormula::<T, RingBuffer<T>, Y> {
                formula: StlOperator::True,
                signal: RingBuffer::new(),
                _phantom: std::marker::PhantomData,
            }),
            FormulaDefinition::False => Box::new(StlFormula::<T, RingBuffer<T>, Y> {
                formula: StlOperator::False,
                signal: RingBuffer::new(),
                _phantom: std::marker::PhantomData,
            }),
            FormulaDefinition::And(l, r) => Box::new(StlFormula::<T, RingBuffer<T>, Y> {
                formula: StlOperator::And(
                    Box::new(
                        self.build_naive_operator(*l)
                            .as_any()
                            .downcast_ref::<StlFormula<T, RingBuffer<T>, Y>>()
                            .unwrap()
                            .formula
                            .clone(),
                    ),
                    Box::new(
                        self.build_naive_operator(*r)
                            .as_any()
                            .downcast_ref::<StlFormula<T, RingBuffer<T>, Y>>()
                            .unwrap()
                            .formula
                            .clone(),
                    ),
                ),
                signal: RingBuffer::new(),
                _phantom: std::marker::PhantomData,
            }),
            FormulaDefinition::Or(l, r) => Box::new(StlFormula::<T, RingBuffer<T>, Y> {
                formula: StlOperator::Or(
                    Box::new(
                        self.build_naive_operator(*l)
                            .as_any()
                            .downcast_ref::<StlFormula<T, RingBuffer<T>, Y>>()
                            .unwrap()
                            .formula
                            .clone(),
                    ),
                    Box::new(
                        self.build_naive_operator(*r)
                            .as_any()
                            .downcast_ref::<StlFormula<T, RingBuffer<T>, Y>>()
                            .unwrap()
                            .formula
                            .clone(),
                    ),
                ),
                signal: RingBuffer::new(),
                _phantom: std::marker::PhantomData,
            }),
            FormulaDefinition::Not(op) => Box::new(StlFormula::<T, RingBuffer<T>, Y> {
                formula: StlOperator::Not(Box::new(
                    self.build_naive_operator(*op)
                        .as_any()
                        .downcast_ref::<StlFormula<T, RingBuffer<T>, Y>>()
                        .unwrap()
                        .formula
                        .clone(),
                )),
                signal: RingBuffer::new(),
                _phantom: std::marker::PhantomData,
            }),
            FormulaDefinition::Implies(l, r) => Box::new(StlFormula::<T, RingBuffer<T>, Y> {
                formula: StlOperator::Implies(
                    Box::new(
                        self.build_naive_operator(*l)
                            .as_any()
                            .downcast_ref::<StlFormula<T, RingBuffer<T>, Y>>()
                            .unwrap()
                            .formula
                            .clone(),
                    ),
                    Box::new(
                        self.build_naive_operator(*r)
                            .as_any()
                            .downcast_ref::<StlFormula<T, RingBuffer<T>, Y>>()
                            .unwrap()
                            .formula
                            .clone(),
                    ),
                ),
                signal: RingBuffer::new(),
                _phantom: std::marker::PhantomData,
            }),
            FormulaDefinition::Eventually(i, op) => Box::new(StlFormula::<T, RingBuffer<T>, Y> {
                formula: StlOperator::Eventually(
                    i,
                    Box::new(
                        self.build_naive_operator(*op)
                            .as_any()
                            .downcast_ref::<StlFormula<T, RingBuffer<T>, Y>>()
                            .unwrap()
                            .formula
                            .clone(),
                    ),
                ),
                signal: RingBuffer::new(),
                _phantom: std::marker::PhantomData,
            }),
            FormulaDefinition::Globally(i, op) => Box::new(StlFormula::<T, RingBuffer<T>, Y> {
                formula: StlOperator::Globally(
                    i,
                    Box::new(
                        self.build_naive_operator(*op)
                            .as_any()
                            .downcast_ref::<StlFormula<T, RingBuffer<T>, Y>>()
                            .unwrap()
                            .formula
                            .clone(),
                    ),
                ),
                signal: RingBuffer::new(),
                _phantom: std::marker::PhantomData,
            }),
            FormulaDefinition::Until(i, l, r) => Box::new(StlFormula::<T, RingBuffer<T>, Y> {
                formula: StlOperator::Until(
                    i,
                    Box::new(
                        self.build_naive_operator(*l)
                            .as_any()
                            .downcast_ref::<StlFormula<T, RingBuffer<T>, Y>>()
                            .unwrap()
                            .formula
                            .clone(),
                    ),
                    Box::new(
                        self.build_naive_operator(*r)
                            .as_any()
                            .downcast_ref::<StlFormula<T, RingBuffer<T>, Y>>()
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
    ) -> Box<dyn StlOperatorTrait<T, Output = Y>>
    where
        T: Into<f64> + Copy + 'static,
        Y: RobustnessSemantics + Copy + 'static,
    {
        match formula {
            FormulaDefinition::GreaterThan(c) => Box::new(Atomic::new_greater_than(c)),
            FormulaDefinition::LessThan(c) => Box::new(Atomic::new_less_than(c)),
            FormulaDefinition::True => Box::new(Atomic::new_true()),
            FormulaDefinition::False => Box::new(Atomic::new_false()),
            FormulaDefinition::And(l, r) => Box::new(And::new(
                self.build_incremental_operator(*l),
                self.build_incremental_operator(*r),
                Some(RingBuffer::new()),
                Some(RingBuffer::new()),
            )),
            FormulaDefinition::Or(l, r) => Box::new(Or::new(
                self.build_incremental_operator(*l),
                self.build_incremental_operator(*r),
                Some(RingBuffer::new()),
                Some(RingBuffer::new()),
            )),
            FormulaDefinition::Not(op) => Box::new(Not::new(self.build_incremental_operator(*op))),
            FormulaDefinition::Implies(l, r) => Box::new(Implies::new(
                self.build_incremental_operator(*l),
                self.build_incremental_operator(*r),
                Some(RingBuffer::new()),
                Some(RingBuffer::new()),
            )),
            FormulaDefinition::Eventually(i, op) => Box::new(Eventually::new(
                i,
                self.build_incremental_operator(*op),
                Some(RingBuffer::new()),
                Some(RingBuffer::new()),
            )),
            FormulaDefinition::Globally(i, op) => Box::new(Globally::new(
                i,
                self.build_incremental_operator(*op),
                Some(RingBuffer::new()),
                Some(RingBuffer::new()),
            )),
            FormulaDefinition::Until(i, l, r) => Box::new(Until::new(
                i,
                self.build_incremental_operator(*l),
                self.build_incremental_operator(*r),
                Some(RingBuffer::new()),
                Some(RingBuffer::new()),
            )),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::stl::monitor::{FormulaDefinition, MonitoringStrategy, StlMonitor};
    use std::time::Duration;

    #[test]
    fn test_build_1() {
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

        let monitor: StlMonitor<f64, f64> = StlMonitor::builder()
            .formula(formula.clone())
            .strategy(MonitoringStrategy::Incremental)
            .build()
            .unwrap();

        let monitor_naive: StlMonitor<f64, f64> = StlMonitor::builder()
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

    #[test]
    fn test_build_2() {
        let formula = FormulaDefinition::Until(
            TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(5),
            },
            Box::new(FormulaDefinition::GreaterThan(3.0)),
            Box::new(FormulaDefinition::LessThan(7.0)),
        );

        let monitor: StlMonitor<f64, f64> = StlMonitor::builder()
            .formula(formula.clone())
            .strategy(MonitoringStrategy::Incremental)
            .build()
            .unwrap();

        let monitor_naive: StlMonitor<f64, f64> = StlMonitor::builder()
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

    #[test]
    fn test_build_3() {
        // test nested temporals
        let formula = FormulaDefinition::Globally(
            TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(20),
            },
            Box::new(FormulaDefinition::Eventually(
                TimeInterval {
                    start: Duration::from_secs(5),
                    end: Duration::from_secs(15),
                },
                Box::new(FormulaDefinition::And(
                    Box::new(FormulaDefinition::GreaterThan(2.0)),
                    Box::new(FormulaDefinition::LessThan(8.0)),
                )),
            )),
        );

        let monitor: StlMonitor<f64, f64> = StlMonitor::builder()
            .formula(formula.clone())
            .strategy(MonitoringStrategy::Incremental)
            .build()
            .unwrap();
        let monitor_naive: StlMonitor<f64, f64> = StlMonitor::builder()
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
