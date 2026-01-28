use crate::ring_buffer::{RingBuffer, Step};
use crate::stl::core::{
    RobustnessInterval, RobustnessSemantics, SignalIdentifier, StlOperatorAndSignalIdentifier,
    StlOperatorTrait,
};
use crate::stl::formula_definition::FormulaDefinition;
use crate::stl::naive_operators::{StlFormula, StlOperator};
use crate::stl::operators::atomic_operators::Atomic;
use crate::stl::operators::binary_operators::{And, Or};
use crate::stl::operators::not_operator::Not;
use crate::stl::operators::unary_temporal_operators::{Eventually, Globally};
use crate::stl::operators::until_operator::Until;
use crate::synchronizer::{Interpolatable, InterpolationStrategy, Synchronizer};
use std::any::TypeId;
use std::fmt::Debug;
use std::time::Duration;

/// Defines the monitoring strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MonitoringStrategy {
    Naive,
    Incremental,
}

impl Default for MonitoringStrategy {
    fn default() -> Self {
        MonitoringStrategy::Incremental
    }
}

/// Defines the evaluation mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvaluationMode {
    Eager,
    Strict,
}

impl Default for EvaluationMode {
    fn default() -> Self {
        EvaluationMode::Strict
    }
}

/// Represents the output of a single monitor update operation.
///
/// This struct provides a mapping between the input step that triggered
/// the evaluation and the resulting output steps from the monitor.
#[derive(Clone, PartialEq)]
pub struct MonitorOutput<T, Y> {
    /// The signal name from the input step that triggered this evaluation
    pub input_signal: &'static str,
    /// The timestamp of the input step
    pub input_timestamp: Duration,
    /// The value of the input step
    pub input_value: T,
    /// Each synchronized step and its corresponding evaluation results.
    /// The synchronizer may produce multiple steps (interpolated + real) for a single input.
    pub evaluations: Vec<SyncStepResult<T, Y>>,
}

impl<Y> Debug for MonitorOutput<f64, Y>
where
    Y: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MonitorOutput {{")?;
        writeln!(
            f,
            "  input: signal=\"{}\", timestamp={:?}, value={}",
            self.input_signal, self.input_timestamp, self.input_value
        )?;
        if self.evaluations.is_empty() {
            writeln!(f, "  evaluations: (none)")?;
        } else {
            writeln!(f, "  evaluations:")?;
            for (i, eval) in self.evaluations.iter().enumerate() {
                let is_last = i == self.evaluations.len() - 1;
                let prefix = if is_last { "└──" } else { "├──" };
                let cont = if is_last { "   " } else { "│  " };
                writeln!(
                    f,
                    "  {} sync_step: signal=\"{}\", timestamp={:?}, value={}",
                    prefix, eval.sync_step.signal, eval.sync_step.timestamp, eval.sync_step.value
                )?;
                if eval.outputs.is_empty() {
                    writeln!(f, "  {}     outputs: (none)", cont)?;
                } else {
                    writeln!(f, "  {}     outputs:", cont)?;
                    for (j, output) in eval.outputs.iter().enumerate() {
                        let out_is_last = j == eval.outputs.len() - 1;
                        let out_prefix = if out_is_last {
                            "└──"
                        } else {
                            "├──"
                        };
                        writeln!(
                            f,
                            "  {}       {} t={:?} → {:?}",
                            cont, out_prefix, output.timestamp, output.value
                        )?;
                    }
                }
            }
        }
        write!(f, "}}")
    }
}

/// Represents the evaluation result for a single synchronized step.
#[derive(Clone, PartialEq)]
pub struct SyncStepResult<T, Y> {
    /// The synchronized step that was evaluated (may be interpolated)
    pub sync_step: Step<T>,
    /// The output steps produced by evaluating this synchronized step
    pub outputs: Vec<Step<Option<Y>>>,
}

impl<Y> Debug for SyncStepResult<f64, Y>
where
    Y: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SyncStepResult {{ sync_step: {:?}, outputs: {:?} }}",
            self.sync_step, self.outputs
        )
    }
}

impl<T, Y> SyncStepResult<T, Y> {
    /// Creates a new SyncStepResult.
    pub fn new(sync_step: Step<T>, outputs: Vec<Step<Option<Y>>>) -> Self {
        SyncStepResult { sync_step, outputs }
    }

    /// Returns true if this result has any outputs.
    pub fn has_outputs(&self) -> bool {
        !self.outputs.is_empty()
    }
}

impl<T, Y> MonitorOutput<T, Y> {
    /// Creates a new MonitorOutput from an input step and its evaluation results.
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

    /// Returns true if any evaluation produced outputs.
    pub fn has_outputs(&self) -> bool {
        self.evaluations.iter().any(|e| !e.outputs.is_empty())
    }

    /// Returns the total number of output steps across all evaluations.
    pub fn total_outputs(&self) -> usize {
        self.evaluations.iter().map(|e| e.outputs.len()).sum()
    }

    /// Returns true if there are no evaluations.
    pub fn is_empty(&self) -> bool {
        self.evaluations.is_empty()
    }

    /// Returns an iterator over all output steps (flattened across all evaluations).
    pub fn outputs_iter(&self) -> impl Iterator<Item = &Step<Option<Y>>> {
        self.evaluations.iter().flat_map(|e| e.outputs.iter())
    }

    /// Returns the latest verdict for requested timestamp, if any.
    pub fn latest_verdict_at(&self, timestamp: Duration) -> Option<&Step<Option<Y>>> {
        self.outputs_iter()
            .filter(|s| s.timestamp == timestamp)
            .last()
    }

    /// Collects all outputs into a flat vector.
    pub fn all_outputs(&self) -> Vec<Step<Option<Y>>>
    where
        Y: Clone,
    {
        self.evaluations
            .iter()
            .flat_map(|e| e.outputs.clone())
            .collect()
    }
}

/// The final monitor struct that handles the input stream.
pub struct StlMonitor<T: Clone + Interpolatable, Y> {
    root_operator: Box<dyn StlOperatorTrait<T, Output = Y>>,
    synchronizer: Synchronizer<T>,
}

impl<T: Clone + Interpolatable, Y> StlMonitor<T, Y> {
    /// Creates a new builder instance.
    pub fn builder() -> StlMonitorBuilder<T, Y> {
        StlMonitorBuilder::new()
    }

    /// Returns a `MonitorOutput` that contains both the input step information
    /// and the resulting output steps from the monitor evaluation.
    pub fn update(&mut self, step: &Step<T>) -> MonitorOutput<T, Y>
    where
        Y: RobustnessSemantics + Debug,
    {
        // 1. Push raw step into synchronizer
        self.synchronizer.evaluate(step.clone());
        let mut evaluations = Vec::new();

        // 2. Drain all pending synchronized steps (interpolated + real)
        //    and feed them into the operator tree.
        while let Some(sync_step) = self.synchronizer.pending.pop_front() {
            let op_res = self.root_operator.update(&sync_step);
            evaluations.push(SyncStepResult::new(sync_step, op_res));
        }

        MonitorOutput::new(step, evaluations)
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
    interpolation_strategy: InterpolationStrategy,
    _phantom: std::marker::PhantomData<(T, Y)>,
}

impl<T, Y> Default for StlMonitorBuilder<T, Y> {
    fn default() -> Self {
        Self {
            formula: None,
            strategy: MonitoringStrategy::default(),
            evaluation_mode: EvaluationMode::default(),
            interpolation_strategy: InterpolationStrategy::default(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, Y> StlMonitorBuilder<T, Y> {
    pub fn new() -> Self {
        Self::default()
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

    /// Configures the interpolation strategy for signal synchronization.
    pub fn interpolation_strategy(mut self, strategy: InterpolationStrategy) -> Self {
        self.interpolation_strategy = strategy;
        self
    }

    /// Builds the final StlMonitor by recursively constructing the operator tree.
    pub fn build(self) -> Result<StlMonitor<T, Y>, &'static str>
    where
        T: Into<f64> + Copy + Interpolatable + 'static, // Add required bounds
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

        let mut formula_def = self
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
                    build_incremental_operator::<T, Y>(formula_def.clone(), self.evaluation_mode);
                root_operator.get_signal_identifiers();
                root_operator
            }
            (MonitoringStrategy::Naive, EvaluationMode::Strict, _) => {
                self.build_naive_operator(formula_def.clone(), self.evaluation_mode)
            }
            (_, _, _) => {
                return Err("Chosen strategy and evaluation mode combination is not supported");
            }
        };

        let synchronizer = if formula_def.get_signal_identifiers().len() <= 1 {
            // No need for synchronization if only one signal is involved
            eprintln!(
                "Warning: Only one signal involved, synchronization of signals is disabled for performance."
            );
            Synchronizer::new(InterpolationStrategy::None)
        } else {
            Synchronizer::new(self.interpolation_strategy)
        };

        Ok(StlMonitor {
            root_operator,
            synchronizer,
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
    use crate::stl::core::TimeInterval;
    use crate::stl::monitor::{MonitoringStrategy, StlMonitor};
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

        // assert_eq!(monitor.strategy, MonitoringStrategy::Incremental);
        // assert_eq!(monitor_naive.strategy, MonitoringStrategy::Naive);

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

        // assert_eq!(monitor.strategy, MonitoringStrategy::Incremental);
        // assert_eq!(monitor_naive.strategy, MonitoringStrategy::Naive);

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
        // assert_eq!(monitor.strategy, MonitoringStrategy::Incremental);
        // assert_eq!(monitor_naive.strategy, MonitoringStrategy::Naive);

        let spec = monitor.specification_to_string();
        println!("Monitor Specification: {spec}");
        let spec_naive = monitor_naive.specification_to_string();
        println!("Naive Monitor Specification: {spec_naive}");
        assert_eq!(spec, spec_naive);
    }
}
