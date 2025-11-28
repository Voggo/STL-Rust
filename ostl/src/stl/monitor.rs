use crate::ring_buffer::{RingBuffer, Step};
use crate::stl::core::{
    RobustnessInterval, RobustnessSemantics, StlOperatorAndSignalIdentifier, StlOperatorTrait,
    TimeInterval,
};
use crate::stl::robustness_cached::{And, Atomic, Eventually, Globally, Not, Or, Until};
use crate::stl::robustness_naive::{StlFormula, StlOperator};
use std::any::TypeId;

#[derive(Clone, Debug)]
pub enum FormulaDefinition {
    GreaterThan(&'static str, f64),
    LessThan(&'static str, f64),
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

/// Defines the evaluation mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvaluationMode {
    Eager,
    Strict,
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
    pub fn update(&mut self, step: &Step<T>) -> Vec<Step<Option<Y>>> {
        self.root_operator.update(step)
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
    evaluation_mode: EvaluationMode,
    _phantom: std::marker::PhantomData<(T, Y)>,
}

impl<T, Y> Default for StlMonitorBuilder<T, Y> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, Y> StlMonitorBuilder<T, Y> {
    pub fn new() -> Self {
        StlMonitorBuilder {
            formula: None,
            strategy: MonitoringStrategy::Incremental, // Default
            evaluation_mode: EvaluationMode::Eager,    // Default
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
    /// Configures the evaluation mode (Eager or Strict).
    pub fn evaluation_mode(mut self, mode: EvaluationMode) -> Self {
        self.evaluation_mode = mode;
        self
    }

    /// Builds the final StlMonitor by recursively constructing the operator tree.
    pub fn build(self) -> Result<StlMonitor<T, Y>, &'static str>
    where
        T: Into<f64> + Copy + 'static, // Add required bounds
        Y: RobustnessSemantics + Copy + 'static + std::fmt::Debug, // Add required bounds
    {
        // The RobustnessSemantics bound guarantees Y is one of the supported types.
        let identifier: &'static str = if TypeId::of::<Y>() == TypeId::of::<bool>() {
            "bool"
        } else if TypeId::of::<Y>() == TypeId::of::<f64>() {
            "f64"
        } else {
            // At this point, only RobustnessInterval remains (guaranteed by the trait bound).
            "RobustnessInterval"
        };

        let formula_def = self
            .formula
            .clone()
            .ok_or("Formula definition is required")?;

        // Factory pattern: Build the correct operator tree based on the strategy
        let root_operator = match (self.strategy, self.evaluation_mode, identifier) {
            (_, EvaluationMode::Eager, "f64") => {
                return Err("Eager evaluation mode is not supported for f64 output type");
            }
            (MonitoringStrategy::Incremental, _, _) => {
                let mut root_operator =
                    build_incremental_operator::<T, Y>(formula_def, self.evaluation_mode);
                root_operator.get_signal_identifiers();
                root_operator
            }
            (MonitoringStrategy::Naive, EvaluationMode::Strict, _) => {
                self.build_naive_operator(formula_def, self.evaluation_mode)
            }
            (_, _, _) => {
                return Err("Chosen strategy and evaluation mode combination is not supported");
            }
        };

        Ok(StlMonitor {
            root_operator,
            strategy: self.strategy,
        })
    }

    /// Top-level factory for the Naive monitor.
    /// It builds the formula enum tree and wraps it in the final StlFormula operator.
    fn build_naive_operator(
        &self,
        formula: FormulaDefinition,
        _mode: EvaluationMode, // Note: mode isn't used by naive, but we keep it for signature consistency
    ) -> Box<dyn StlOperatorTrait<T, Output = Y>>
    where
        T: Into<f64> + Copy + 'static,
        Y: RobustnessSemantics + 'static,
    {
        let formula_enum = build_naive_formula(formula);

        Box::new(StlFormula::<T, RingBuffer<T>, Y>::new(
            formula_enum,
            RingBuffer::new(),
        ))
    }
}

/// Recursively builds the lightweight StlOperator enum from the FormulaDefinition.
fn build_naive_formula(formula: FormulaDefinition) -> StlOperator {
    match formula {
        FormulaDefinition::GreaterThan(s, c) => StlOperator::GreaterThan(s, c),
        FormulaDefinition::LessThan(s, c) => StlOperator::LessThan(s, c),
        FormulaDefinition::True => StlOperator::True,
        FormulaDefinition::False => StlOperator::False,
        FormulaDefinition::And(l, r) => StlOperator::And(
            Box::new(build_naive_formula(*l)),
            Box::new(build_naive_formula(*r)),
        ),
        FormulaDefinition::Or(l, r) => StlOperator::Or(
            Box::new(build_naive_formula(*l)),
            Box::new(build_naive_formula(*r)),
        ),
        FormulaDefinition::Not(op) => StlOperator::Not(Box::new(build_naive_formula(*op))),
        FormulaDefinition::Implies(l, r) => StlOperator::Implies(
            Box::new(build_naive_formula(*l)),
            Box::new(build_naive_formula(*r)),
        ),
        FormulaDefinition::Eventually(i, op) => {
            StlOperator::Eventually(i, Box::new(build_naive_formula(*op)))
        }
        FormulaDefinition::Globally(i, op) => {
            StlOperator::Globally(i, Box::new(build_naive_formula(*op)))
        }
        FormulaDefinition::Until(i, l, r) => StlOperator::Until(
            i,
            Box::new(build_naive_formula(*l)),
            Box::new(build_naive_formula(*r)),
        ),
    }
}

fn build_incremental_operator<T, Y>(
    formula: FormulaDefinition,
    mode: EvaluationMode,
) -> Box<dyn StlOperatorAndSignalIdentifier<T, Y>>
where
    T: Into<f64> + Copy + 'static,
    Y: RobustnessSemantics + Copy + 'static + std::fmt::Debug,
{
    // Determine configuration flags
    let is_eager = matches!(mode, EvaluationMode::Eager);
    let is_rosi = TypeId::of::<Y>() == TypeId::of::<RobustnessInterval>();

    // We use `$( $arg:expr ),*` to capture arguments as a list.
    // We explicitly define <T, RingBuffer<Option<Y>>, Y...> to allow passing 'None' for caches.
    macro_rules! dispatch_operator {
        ($OpType:ident, $( $arg:expr ),* ) => {
            match (is_eager, is_rosi) {
                (true, true) => Box::new($OpType::<T, RingBuffer<Option<Y>>, Y, true, true>::new( $( $arg ),* )),
                (true, false) => Box::new($OpType::<T, RingBuffer<Option<Y>>, Y, true, false>::new( $( $arg ),* )),
                (false, true) => Box::new($OpType::<T, RingBuffer<Option<Y>>, Y, false, true>::new( $( $arg ),* )),
                (false, false) => Box::new($OpType::<T, RingBuffer<Option<Y>>, Y, false, false>::new( $( $arg ),* )),
            }
        };
    }

    match formula {
        // --- Group A: Clean Operators (No Const Generics) ---
        FormulaDefinition::GreaterThan(s, c) => Box::new(Atomic::new_greater_than(s, c)),
        FormulaDefinition::LessThan(s, c) => Box::new(Atomic::new_less_than(s, c)),
        FormulaDefinition::True => Box::new(Atomic::new_true()),
        FormulaDefinition::False => Box::new(Atomic::new_false()),

        FormulaDefinition::Not(op) => {
            let child = build_incremental_operator(*op, mode);
            Box::new(Not::new(child))
        }

        // --- Group B: Optimized Operators (Uses dispatch macro) ---
        FormulaDefinition::And(l, r) => {
            let left = build_incremental_operator(*l, mode);
            let right = build_incremental_operator(*r, mode);
            dispatch_operator!(And, left, right, None, None)
        }

        FormulaDefinition::Or(l, r) => {
            let left = build_incremental_operator(*l, mode);
            let right = build_incremental_operator(*r, mode);
            dispatch_operator!(Or, left, right, None, None)
        }

        FormulaDefinition::Implies(l, r) => {
            let not_left = Box::new(Not::new(build_incremental_operator(*l, mode)));
            let right = build_incremental_operator(*r, mode);
            dispatch_operator!(Or, not_left, right, None, None)
        }

        FormulaDefinition::Eventually(i, op) => {
            let child = build_incremental_operator(*op, mode);
            dispatch_operator!(Eventually, i, child, None, None)
        }

        FormulaDefinition::Globally(i, op) => {
            let child = build_incremental_operator(*op, mode);
            dispatch_operator!(Globally, i, child, None, None)
        }

        FormulaDefinition::Until(i, l, r) => {
            let left = build_incremental_operator(*l, mode);
            let right = build_incremental_operator(*r, mode);
            dispatch_operator!(Until, i, left, right, None, None)
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
            Box::new(FormulaDefinition::GreaterThan("x", 5.0)),
            Box::new(FormulaDefinition::Eventually(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(10),
                },
                Box::new(FormulaDefinition::LessThan("x", 10.0)),
            )),
        );

        let monitor: StlMonitor<f64, f64> = StlMonitor::builder()
            .formula(formula.clone())
            .strategy(MonitoringStrategy::Incremental)
            .evaluation_mode(EvaluationMode::Strict)
            .build()
            .unwrap();

        let monitor_naive: StlMonitor<f64, f64> = StlMonitor::builder()
            .formula(formula)
            .strategy(MonitoringStrategy::Naive)
            .evaluation_mode(EvaluationMode::Strict)
            .build()
            .unwrap();

        assert_eq!(monitor.strategy, MonitoringStrategy::Incremental);
        assert_eq!(monitor_naive.strategy, MonitoringStrategy::Naive);

        let spec = monitor.specification_to_string();
        println!("Monitor Specification: {spec}");

        let spec_naive = monitor_naive.specification_to_string();
        println!("Naive Monitor Specification: {spec_naive}");

        assert_eq!(spec, spec_naive);
    }

    #[test]
    fn test_build_2() {
        let formula = FormulaDefinition::Until(
            TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(5),
            },
            Box::new(FormulaDefinition::GreaterThan("x", 3.0)),
            Box::new(FormulaDefinition::LessThan("x", 7.0)),
        );

        let monitor: StlMonitor<f64, f64> = StlMonitor::builder()
            .formula(formula.clone())
            .strategy(MonitoringStrategy::Incremental)
            .evaluation_mode(EvaluationMode::Strict)
            .build()
            .unwrap();

        let monitor_naive: StlMonitor<f64, f64> = StlMonitor::builder()
            .formula(formula)
            .strategy(MonitoringStrategy::Naive)
            .evaluation_mode(EvaluationMode::Strict)
            .build()
            .unwrap();

        assert_eq!(monitor.strategy, MonitoringStrategy::Incremental);
        assert_eq!(monitor_naive.strategy, MonitoringStrategy::Naive);

        let spec = monitor.specification_to_string();
        println!("Monitor Specification: {spec}");

        let spec_naive = monitor_naive.specification_to_string();
        println!("Naive Monitor Specification: {spec_naive}");

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
                    Box::new(FormulaDefinition::GreaterThan("x", 2.0)),
                    Box::new(FormulaDefinition::LessThan("x", 8.0)),
                )),
            )),
        );

        let monitor: StlMonitor<f64, f64> = StlMonitor::builder()
            .formula(formula.clone())
            .strategy(MonitoringStrategy::Incremental)
            .evaluation_mode(EvaluationMode::Strict)
            .build()
            .unwrap();
        let monitor_naive: StlMonitor<f64, f64> = StlMonitor::builder()
            .formula(formula)
            .strategy(MonitoringStrategy::Naive)
            .evaluation_mode(EvaluationMode::Strict)
            .build()
            .unwrap();
        assert_eq!(monitor.strategy, MonitoringStrategy::Incremental);
        assert_eq!(monitor_naive.strategy, MonitoringStrategy::Naive);

        let spec = monitor.specification_to_string();
        println!("Monitor Specification: {spec}");
        let spec_naive = monitor_naive.specification_to_string();
        println!("Naive Monitor Specification: {spec_naive}");
        assert_eq!(spec, spec_naive);
    }
}
