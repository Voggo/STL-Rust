use crate::ring_buffer::{RingBuffer, Step};
use crate::stl::core::{
    RobustnessSemantics, SignalIdentifier, StlOperatorAndSignalIdentifier, StlOperatorTrait,
};
use crate::stl::formula_definition::FormulaDefinition;
use crate::stl::naive_operators::{StlFormula, StlOperator};
use crate::stl::operators::atomic_operators::Atomic;
use crate::stl::operators::binary_operators::{And, Or};
use crate::stl::operators::not_operator::Not;
use crate::stl::operators::unary_temporal_operators::{Eventually, Globally};
use crate::stl::operators::until_operator::Until;
use crate::synchronizer::{Interpolatable, SynchronizationStrategy, Synchronizer};
use std::any::TypeId;
use std::fmt::Debug;
use std::fmt::Display;
use std::time::Duration;

/// Defines the monitoring strategy.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Algorithm {
    Naive,
    #[default]
    Incremental,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Semantics {
    Rosi,
    #[default]
    Robustness,
    StrictSatisfaction,
    EagerSatisfaction,
}

/// Marker traits and structs for Type-Driven Semantics
pub mod semantic_markers {
    use super::Semantics;
    use crate::stl::core::RobustnessInterval;

    pub trait SemanticType {
        type Output: super::RobustnessSemantics + 'static;
        fn as_enum() -> Semantics;
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Rosi;
    impl SemanticType for Rosi {
        type Output = RobustnessInterval;
        fn as_enum() -> Semantics {
            Semantics::Rosi
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Robustness;
    impl SemanticType for Robustness {
        type Output = f64;
        fn as_enum() -> Semantics {
            Semantics::Robustness
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct StrictSatisfaction;
    impl SemanticType for StrictSatisfaction {
        type Output = bool;
        fn as_enum() -> Semantics {
            Semantics::StrictSatisfaction
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct EagerSatisfaction;
    impl SemanticType for EagerSatisfaction {
        type Output = bool;
        fn as_enum() -> Semantics {
            Semantics::EagerSatisfaction
        }
    }
}

// Re-export markers for easier access like `monitor::StrictSatisfaction`
pub use semantic_markers::{EagerSatisfaction, Robustness, Rosi, StrictSatisfaction};

/// Represents the output of a single monitor update operation.
#[derive(Clone, Debug, PartialEq)]
pub struct MonitorOutput<T, Y> {
    pub input_signal: &'static str,
    pub input_timestamp: Duration,
    pub input_value: T,
    pub evaluations: Vec<SyncStepResult<T, Y>>,
}

impl<Y> Display for MonitorOutput<f64, Y>
where
    Y: Debug + Clone,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let finalized = self.finalize();

        if finalized.is_empty() {
            return write!(f, "No verdicts available");
        }

        for (i, step) in finalized.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "t={:?}: {:?}", step.timestamp, step.value)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SyncStepResult<T, Y> {
    pub sync_step: Step<T>,
    pub outputs: Vec<Step<Y>>,
}

impl<T, Y> SyncStepResult<T, Y> {
    pub fn new(sync_step: Step<T>, outputs: Vec<Step<Y>>) -> Self {
        SyncStepResult { sync_step, outputs }
    }
    pub fn has_outputs(&self) -> bool {
        !self.outputs.is_empty()
    }
}

impl<T, Y> MonitorOutput<T, Y> {
    pub fn new(input: &Step<T>, evaluations: Vec<SyncStepResult<T, Y>>) -> Self
    where
        T: Clone,
    {
        MonitorOutput {
            input_signal: input.signal,
            input_timestamp: input.timestamp,
            input_value: input.value.clone(),
            evaluations,
        }
    }
    pub fn has_outputs(&self) -> bool {
        self.evaluations.iter().any(|e| !e.outputs.is_empty())
    }
    pub fn total_outputs(&self) -> usize {
        self.evaluations.iter().map(|e| e.outputs.len()).sum()
    }
    pub fn is_empty(&self) -> bool {
        self.evaluations.is_empty()
    }
    pub fn outputs_iter(&self) -> impl Iterator<Item = &Step<Y>> {
        self.evaluations.iter().flat_map(|e| e.outputs.iter())
    }
    pub fn latest_verdict_at(&self, timestamp: Duration) -> Option<&Step<Y>> {
        self.outputs_iter()
            .filter(|s| s.timestamp == timestamp)
            .last()
    }
    pub fn all_outputs(&self) -> Vec<Step<Y>>
    where
        Y: Clone,
    {
        self.evaluations
            .iter()
            .flat_map(|e| e.outputs.clone())
            .collect()
    }
    pub fn finalize(&self) -> Vec<Step<Y>>
    where
        Y: Clone,
    {
        let mut latest_map = std::collections::BTreeMap::new();
        for output in self.outputs_iter() {
            latest_map.insert(output.timestamp, output.value.clone());
        }
        latest_map
            .into_iter()
            .map(|(ts, val)| Step::new(self.input_signal, val, ts))
            .collect()
    }
}

/// The final monitor struct that handles the input stream.
/// We do not constrain Y on the struct definition to allow flexible builder patterns,
/// though valid monitors will always have Y: RobustnessSemantics.
pub struct StlMonitor<T: Clone + Interpolatable, Y> {
    root_operator: Box<dyn StlOperatorTrait<T, Output = Y>>,
    synchronizer: Synchronizer<T>,
}

/// Entry point for the builder.
/// This impl block enables `StlMonitor::builder()` to return a builder with default T=f64 and Y=f64.
impl StlMonitor<f64, f64> {
    pub fn builder() -> StlMonitorBuilder<f64, f64> {
        StlMonitorBuilder {
            formula: None,
            algorithm: Algorithm::default(),
            semantics: Semantics::Robustness, // Default, but will be overwritten if semantics() is called
            synchronization_strategy: SynchronizationStrategy::default(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Clone + Interpolatable, Y> StlMonitor<T, Y> {
    pub fn update(&mut self, step: &Step<T>) -> MonitorOutput<T, Y>
    where
        Y: RobustnessSemantics + Debug,
    {
        self.synchronizer.evaluate(step.clone());
        let mut evaluations = Vec::new();
        while let Some(sync_step) = self.synchronizer.pending.pop_front() {
            let op_res = self.root_operator.update(&sync_step);
            evaluations.push(SyncStepResult::new(sync_step, op_res));
        }
        MonitorOutput::new(step, evaluations)
    }

    pub fn specification_to_string(&self) -> String {
        self.root_operator.to_string()
    }

    pub fn get_signal_identifiers(&mut self) -> std::collections::HashSet<&'static str> {
        self.root_operator.get_signal_identifiers()
    }
}

/// The Builder pattern struct for StlMonitor.
pub struct StlMonitorBuilder<T, Y> {
    formula: Option<FormulaDefinition>,
    algorithm: Algorithm,
    /// We store the enum value for logic, and use Y for type safety
    semantics: Semantics,
    synchronization_strategy: SynchronizationStrategy,
    _phantom: std::marker::PhantomData<(T, Y)>,
}

impl<T, Y> StlMonitorBuilder<T, Y> {
    /// Sets the formula definition to be monitored.
    pub fn formula(mut self, formula: FormulaDefinition) -> Self {
        self.formula = Some(formula);
        self
    }

    /// Configures the monitoring algorithm (Naive or Incremental).
    pub fn algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Configures the synchronization strategy for signal synchronization.
    pub fn synchronization_strategy(mut self, strategy: SynchronizationStrategy) -> Self {
        self.synchronization_strategy = strategy;
        self
    }

    /// Applies the semantics, switching the Builder's generic type `Y` to match the semantics.
    /// This allows inference of the output type (bool, f64, RobustnessInterval).
    pub fn semantics<S: semantic_markers::SemanticType>(
        self,
        _marker: S,
    ) -> StlMonitorBuilder<T, S::Output> {
        StlMonitorBuilder {
            formula: self.formula,
            algorithm: self.algorithm,
            semantics: S::as_enum(),
            synchronization_strategy: self.synchronization_strategy,
            _phantom: std::marker::PhantomData,
        }
    }
}

// The build method is only available when Y implements RobustnessSemantics.
// This prevents building when Y is still `Unset`.
impl<T, Y> StlMonitorBuilder<T, Y>
where
    T: Into<f64> + Copy + Interpolatable + 'static,
    Y: RobustnessSemantics + Copy + 'static + std::fmt::Debug,
{
    /// Initializes the operator by collecting signal identifiers and preparing internal state.
    /// This is called automatically during the build process for incremental operators.
    fn initialize_operator(
        &self,
        mut operator: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
    ) -> Box<dyn StlOperatorAndSignalIdentifier<T, Y>> {
        operator.get_signal_identifiers();
        operator
    }

    pub fn build(self) -> Result<StlMonitor<T, Y>, &'static str> {
        let identifier: &'static str = if TypeId::of::<Y>() == TypeId::of::<bool>() {
            "bool"
        } else if TypeId::of::<Y>() == TypeId::of::<f64>() {
            "f64"
        } else {
            "RobustnessInterval"
        };

        let mut formula_def = self
            .formula
            .clone()
            .ok_or("Formula definition is required")?;

        // Validate semantics matches output type Y
        // (This check is theoretically redundant due to the type system now, but kept for safety)
        match (self.semantics, identifier) {
            (Semantics::Rosi, "RobustnessInterval") => {}
            (Semantics::Robustness, "f64") => {}
            (Semantics::StrictSatisfaction | Semantics::EagerSatisfaction, "bool") => {}
            _ => return Err("Semantics does not match output type Y"),
        }

        let root_operator = match (self.algorithm, self.semantics) {
            (Algorithm::Incremental, _) => {
                let operator =
                    build_incremental_operator::<T, Y>(formula_def.clone(), self.semantics);
                self.initialize_operator(operator)
            }
            (Algorithm::Naive, Semantics::StrictSatisfaction | Semantics::Robustness) => {
                self.build_naive_operator(formula_def.clone(), self.semantics)
            }
            (Algorithm::Naive, Semantics::EagerSatisfaction | Semantics::Rosi) => {
                return Err("Naive algorithm does not support RoSI/Eaver evaluation");
            }
        };

        let synchronizer = if formula_def.get_signal_identifiers().len() <= 1 {
            eprintln!(
                "Warning: Only one signal involved, synchronization of signals is disabled for performance."
            );
            Synchronizer::new(SynchronizationStrategy::None)
        } else {
            Synchronizer::new(self.synchronization_strategy)
        };

        Ok(StlMonitor {
            root_operator,
            synchronizer,
        })
    }

    fn build_naive_operator(
        &self,
        formula: FormulaDefinition,
        _semantics: Semantics,
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
    semantics: Semantics,
) -> Box<dyn StlOperatorAndSignalIdentifier<T, Y>>
where
    T: Into<f64> + Copy + 'static,
    Y: RobustnessSemantics + Copy + 'static + std::fmt::Debug,
{
    // Determine configuration flags
    let is_eager = matches!(semantics, Semantics::EagerSatisfaction);
    let is_rosi = matches!(semantics, Semantics::Rosi);

    // We use `$( $arg:expr ),*` to capture arguments as a list.
    // We explicitly define <T, RingBuffer<Option<Y>>, Y...> to allow passing 'None' for caches.
    macro_rules! dispatch_operator {
        ($OpType:ident, $( $arg:expr ),* ) => {
            match (is_eager, is_rosi) {
                (true, true) => Box::new($OpType::<T, RingBuffer<Y>, Y, true, true>::new( $( $arg ),* )),
                (true, false) => Box::new($OpType::<T, RingBuffer<Y>, Y, true, false>::new( $( $arg ),* )),
                (false, true) => Box::new($OpType::<T, RingBuffer<Y>, Y, false, true>::new( $( $arg ),* )),
                (false, false) => Box::new($OpType::<T, RingBuffer<Y>, Y, false, false>::new( $( $arg ),* )),
            }
        };
    }

    match formula {
        FormulaDefinition::GreaterThan(s, c) => Box::new(Atomic::new_greater_than(s, c)),
        FormulaDefinition::LessThan(s, c) => Box::new(Atomic::new_less_than(s, c)),
        FormulaDefinition::True => Box::new(Atomic::new_true()),
        FormulaDefinition::False => Box::new(Atomic::new_false()),

        FormulaDefinition::Not(op) => {
            let child = build_incremental_operator(*op, semantics);
            Box::new(Not::new(child))
        }

        FormulaDefinition::And(l, r) => {
            let left = build_incremental_operator(*l, semantics);
            let right = build_incremental_operator(*r, semantics);
            dispatch_operator!(And, left, right, None, None)
        }

        FormulaDefinition::Or(l, r) => {
            let left = build_incremental_operator(*l, semantics);
            let right = build_incremental_operator(*r, semantics);
            dispatch_operator!(Or, left, right, None, None)
        }

        FormulaDefinition::Implies(l, r) => {
            let not_left = Box::new(Not::new(build_incremental_operator(*l, semantics)));
            let right = build_incremental_operator(*r, semantics);
            dispatch_operator!(Or, not_left, right, None, None)
        }

        FormulaDefinition::Eventually(i, op) => {
            let child = build_incremental_operator(*op, semantics);
            dispatch_operator!(Eventually, i, child, None, None)
        }

        FormulaDefinition::Globally(i, op) => {
            let child = build_incremental_operator(*op, semantics);
            dispatch_operator!(Globally, i, child, None, None)
        }

        FormulaDefinition::Until(i, l, r) => {
            let left = build_incremental_operator(*l, semantics);
            let right = build_incremental_operator(*r, semantics);
            dispatch_operator!(Until, i, left, right, None, None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stl::core::TimeInterval;
    use crate::stl::monitor::{Algorithm, StlMonitor};
    use std::time::Duration;

    #[test]
    fn test_builder_type_inference() {
        let formula = FormulaDefinition::GreaterThan("x", 5.0);

        // This is the syntax you requested:
        // T is defaulted to f64 by StlMonitor::builder()
        // Y is inferred as `bool` because of StrictSatisfaction
        let mut monitor = StlMonitor::builder()
            .formula(formula)
            .semantics(StrictSatisfaction) // Use the marker struct
            .algorithm(Algorithm::Incremental)
            .build()
            .unwrap();

        // The compiler knows this is f64 input, bool output
        let step = Step::new("x", 10.0, Duration::from_secs(1));
        let output = monitor.update(&step);

        // We can assert types in the test
        assert_eq!(output.input_value, 10.0); // T=f64
        // output.latest_verdict_at(...) returns Option<&Step<Option<bool>>>
    }

    #[test]
    fn test_build_1() {
        let formula = FormulaDefinition::And(
            Box::new(FormulaDefinition::GreaterThan("x", 5.0)),
            Box::new(FormulaDefinition::Eventually(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(10),
                },
                Box::new(FormulaDefinition::LessThan("y", 10.0)),
            )),
        );

        // Usage with marker struct implies Y = f64
        let mut monitor = StlMonitor::builder()
            .formula(formula.clone())
            .algorithm(Algorithm::Incremental)
            .semantics(Robustness)
            .build()
            .unwrap();

        let mut monitor_naive = StlMonitor::builder()
            .formula(formula)
            .algorithm(Algorithm::Naive)
            .semantics(Robustness)
            .build()
            .unwrap();

        let spec = monitor.specification_to_string();
        let spec_naive = monitor_naive.specification_to_string();

        let naive_ids = monitor_naive.get_signal_identifiers();
        let inc_ids = monitor.get_signal_identifiers();
        assert_eq!(naive_ids, inc_ids);
        assert!(naive_ids.contains("x"));
        assert!(naive_ids.contains("y"));
        assert!(inc_ids.contains("x"));
        assert!(inc_ids.contains("y"));
        assert_eq!(spec, spec_naive);
    }
}
