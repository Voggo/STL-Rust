use ostl::ring_buffer::Step;
use ostl::stl::core::{RobustnessInterval, TimeInterval};
use ostl::stl::formula_definition::FormulaDefinition;
use ostl::stl::monitor::{
    Algorithm, EagerSatisfaction, MonitorOutput, Robustness, Rosi, StlMonitor, StrictSatisfaction,
};
use ostl::stl::parse_stl;
use ostl::synchronizer::SynchronizationStrategy;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyList, PyTuple};
use std::collections::HashSet;
use std::fmt::Debug;
use std::time::Duration;

// -----------------------------------------------------------------------------
// 1. Formula Parsing Function
// -----------------------------------------------------------------------------

/// Parse an STL formula from a string using the same DSL syntax as the Rust `stl!` macro.
///
/// This allows you to write formulas using the same syntax as Rust, making it easy
/// to port formulas between Python and Rust code.
///
/// # Syntax
///
/// ## Predicates
/// - `signal > value` - Signal greater than value
/// - `signal < value` - Signal less than value
/// - `signal >= value` - Signal greater than or equal to value
/// - `signal <= value` - Signal less than or equal to value
///
/// ## Boolean Constants
/// - `true` - Always true
/// - `false` - Always false
///
/// ## Unary Operators
/// - `!(sub)` or `not(sub)` - Negation
/// - `G[start, end](sub)` or `globally[start, end](sub)` - Globally (always)
/// - `F[start, end](sub)` or `eventually[start, end](sub)` - Eventually (finally)
///
/// ## Binary Operators
/// - `left && right` or `left and right` - Conjunction
/// - `left || right` or `left or right` - Disjunction
/// - `left -> right` or `left implies right` - Implication
/// - `left U[start, end] right` or `left until[start, end] right` - Until
///
/// # Examples
///
/// ```python
/// from ostl_python import parse_formula
///
/// # Simple predicate
/// f = parse_formula("x > 5")
///
/// # Globally operator
/// f = parse_formula("G[0, 10](x > 5)")
///
/// # Complex formula
/// f = parse_formula("G[0, 10](x > 5) && F[0, 5](y < 3)")
///
/// # Using keyword syntax
/// f = parse_formula("globally[0, 10](x > 5) and eventually[0, 5](y < 3)")
/// ```
///
/// # Arguments
/// * `formula_str` - A string containing an STL formula
///
/// # Returns
/// * `Formula` - The parsed formula object
///
/// # Raises
/// * `ValueError` - If the formula string cannot be parsed
#[pyfunction]
#[pyo3(name = "parse_formula")]
fn py_parse_formula(formula_str: &str) -> PyResult<Formula> {
    let formula = parse_stl(formula_str)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
    Ok(Formula { inner: formula })
}

// -----------------------------------------------------------------------------
// 2. Formula Wrapper
// -----------------------------------------------------------------------------

/// A Python wrapper around the Rust FormulaDefinition enum.
#[pyclass]
#[derive(Clone)]
struct Formula {
    inner: FormulaDefinition,
}

#[pymethods]
impl Formula {
    // --- Atomic Propositions ---

    /// Create a greater-than atomic predicate: signal > value
    ///
    /// # Arguments
    /// * `signal` - Name of the signal to compare
    /// * `value` - Threshold value
    ///
    /// # Example
    /// ```python
    /// Formula.gt('x', 0.5)  # x > 0.5
    /// ```
    #[staticmethod]
    // #[pyo3(text_signature = "(signal, value)")]
    fn gt(signal: String, value: f64) -> Self {
        // Leak the string to get &'static str required by ostl
        let sig_ref = Box::leak(signal.into_boxed_str());
        Formula {
            inner: FormulaDefinition::GreaterThan(sig_ref, value),
        }
    }

    /// Create a less-than atomic predicate: signal < value
    ///
    /// # Arguments
    /// * `signal` - Name of the signal to compare
    /// * `value` - Threshold value
    ///
    /// # Example
    /// ```python
    /// Formula.lt('y', 0.8)  # y < 0.8
    /// ```
    #[staticmethod]
    #[pyo3(text_signature = "(signal, value)")]
    fn lt(signal: String, value: f64) -> Self {
        let sig_ref = Box::leak(signal.into_boxed_str());
        Formula {
            inner: FormulaDefinition::LessThan(sig_ref, value),
        }
    }

    /// Create a constant true formula (⊤)
    ///
    /// # Example
    /// ```python
    /// Formula.true_()
    /// ```
    #[staticmethod]
    #[pyo3(text_signature = "()")]
    fn true_() -> Self {
        Formula {
            inner: FormulaDefinition::True,
        }
    }

    /// Create a constant false formula (⊥)
    ///
    /// # Example
    /// ```python
    /// Formula.false_()
    /// ```
    #[staticmethod]
    #[pyo3(text_signature = "()")]
    fn false_() -> Self {
        Formula {
            inner: FormulaDefinition::False,
        }
    }

    // --- Boolean Logic ---

    /// Create conjunction (AND) of two formulas: left ∧ right
    ///
    /// # Arguments
    /// * `left` - First formula
    /// * `right` - Second formula
    #[staticmethod]
    #[pyo3(text_signature = "(left, right)")]
    fn and_(left: &Formula, right: &Formula) -> Self {
        Formula {
            inner: FormulaDefinition::And(
                Box::new(left.inner.clone()),
                Box::new(right.inner.clone()),
            ),
        }
    }

    /// Create disjunction (OR) of two formulas: left ∨ right
    ///
    /// # Arguments
    /// * `left` - First formula
    /// * `right` - Second formula
    #[staticmethod]
    #[pyo3(text_signature = "(left, right)")]
    fn or_(left: &Formula, right: &Formula) -> Self {
        Formula {
            inner: FormulaDefinition::Or(
                Box::new(left.inner.clone()),
                Box::new(right.inner.clone()),
            ),
        }
    }

    /// Create negation (NOT) of a formula: ¬child
    ///
    /// # Arguments
    /// * `child` - Formula to negate
    #[staticmethod]
    #[pyo3(text_signature = "(child)")]
    fn not_(child: &Formula) -> Self {
        Formula {
            inner: FormulaDefinition::Not(Box::new(child.inner.clone())),
        }
    }

    /// Create implication: left → right (if left then right)
    ///
    /// # Arguments
    /// * `left` - Antecedent (condition)
    /// * `right` - Consequent (result)
    #[staticmethod]
    #[pyo3(text_signature = "(left, right)")]
    fn implies(left: &Formula, right: &Formula) -> Self {
        Formula {
            inner: FormulaDefinition::Implies(
                Box::new(left.inner.clone()),
                Box::new(right.inner.clone()),
            ),
        }
    }

    // --- Temporal Logic ---

    /// Create globally (always) temporal formula: G[start,end](child)
    ///
    /// The formula must hold at all time points in [start, end].
    ///
    /// # Arguments
    /// * `start` - Start of time interval (seconds)
    /// * `end` - End of time interval (seconds)
    /// * `child` - Formula that must hold throughout
    #[staticmethod]
    #[pyo3(text_signature = "(start, end, child)")]
    fn always(start: f64, end: f64, child: &Formula) -> Self {
        let interval = TimeInterval {
            start: Duration::from_secs_f64(start),
            end: Duration::from_secs_f64(end),
        };
        Formula {
            inner: FormulaDefinition::Globally(interval, Box::new(child.inner.clone())),
        }
    }

    /// Create eventually (finally) temporal formula: F[start,end](child)
    ///
    /// The formula must hold at some time point in [start, end].
    ///
    /// # Arguments
    /// * `start` - Start of time interval (seconds)
    /// * `end` - End of time interval (seconds)
    /// * `child` - Formula that must hold at some point
    #[staticmethod]
    #[pyo3(text_signature = "(start, end, child)")]
    fn eventually(start: f64, end: f64, child: &Formula) -> Self {
        let interval = TimeInterval {
            start: Duration::from_secs_f64(start),
            end: Duration::from_secs_f64(end),
        };
        Formula {
            inner: FormulaDefinition::Eventually(interval, Box::new(child.inner.clone())),
        }
    }

    /// Create until temporal formula: left U[start,end] right
    ///
    /// Left must hold until right becomes true (within [start, end]).
    ///
    /// # Arguments
    /// * `start` - Start of time interval (seconds)
    /// * `end` - End of time interval (seconds)
    /// * `left` - Formula that must hold until right
    /// * `right` - Formula that must eventually become true
    #[staticmethod]
    #[pyo3(text_signature = "(start, end, left, right)")]
    fn until(start: f64, end: f64, left: &Formula, right: &Formula) -> Self {
        let interval = TimeInterval {
            start: Duration::from_secs_f64(start),
            end: Duration::from_secs_f64(end),
        };
        Formula {
            inner: FormulaDefinition::Until(
                interval,
                Box::new(left.inner.clone()),
                Box::new(right.inner.clone()),
            ),
        }
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Formula({})", self.inner)
    }
}

// -----------------------------------------------------------------------------
// 3. MonitorOutput Wrapper
// -----------------------------------------------------------------------------

/// We need an enum to hold the different output types since Python is dynamic
/// but Rust types are static.
#[derive(Clone)]
enum InnerMonitorOutput {
    Bool(MonitorOutput<f64, bool>),
    Float(MonitorOutput<f64, f64>),
    Interval(MonitorOutput<f64, RobustnessInterval>),
}

/// A wrapper around the Rust MonitorOutput that provides idiomatic Display and Debug formatting.
///
/// This class wraps the output from a monitor update, preserving access to both
/// the structured data (via `to_dict()`) and the Rust Display/Debug formatting
/// (via `__str__()` and `__repr__()`).
///
/// The string representation shows verdicts in the format:
/// ```
/// t={timestamp}: {value}
/// ```
///
/// For multiple verdicts, they are shown on separate lines.
/// If no verdicts are available, it shows "No verdicts available".
#[pyclass(name = "MonitorOutput")]
#[derive(Clone)]
struct PyMonitorOutput {
    inner: InnerMonitorOutput,
}

#[pymethods]
impl PyMonitorOutput {
    /// Convert the monitor output to a dictionary.
    ///
    /// Returns a dictionary containing:
    /// - 'input_signal': the signal name
    /// - 'input_timestamp': the input timestamp
    /// - 'input_value': the input value
    /// - 'evaluations': list of evaluation dictionaries, each containing:
    ///     - 'sync_step_signal': signal name of the synchronized step
    ///     - 'sync_step_timestamp': timestamp of the synchronized step
    ///     - 'sync_step_value': value of the synchronized step
    ///     - 'outputs': list of output dictionaries with:
    ///         - 'timestamp': when the verdict is for
    ///         - 'value': the verdict value (bool, float, or tuple depending on semantics)
    fn to_dict(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| match &self.inner {
            InnerMonitorOutput::Bool(output) => convert_output_to_dict(py, output.clone(), |val| {
                PyBool::new(py, val).to_owned().into_any().unbind()
            }),
            InnerMonitorOutput::Float(output) => {
                convert_output_to_dict(py, output.clone(), |val| {
                    PyFloat::new(py, val).to_owned().into_any().unbind()
                })
            }
            InnerMonitorOutput::Interval(output) => {
                convert_output_to_dict(py, output.clone(), |val| {
                    PyTuple::new(py, [val.0, val.1])
                        .unwrap()
                        .into_any()
                        .unbind()
                })
            }
        })
    }

    /// Get the input signal name.
    #[getter]
    fn input_signal(&self) -> &'static str {
        match &self.inner {
            InnerMonitorOutput::Bool(o) => o.input_signal,
            InnerMonitorOutput::Float(o) => o.input_signal,
            InnerMonitorOutput::Interval(o) => o.input_signal,
        }
    }

    /// Get the input timestamp in seconds.
    #[getter]
    fn input_timestamp(&self) -> f64 {
        match &self.inner {
            InnerMonitorOutput::Bool(o) => o.input_timestamp.as_secs_f64(),
            InnerMonitorOutput::Float(o) => o.input_timestamp.as_secs_f64(),
            InnerMonitorOutput::Interval(o) => o.input_timestamp.as_secs_f64(),
        }
    }

    /// Get the input value.
    #[getter]
    fn input_value(&self) -> f64 {
        match &self.inner {
            InnerMonitorOutput::Bool(o) => o.input_value,
            InnerMonitorOutput::Float(o) => o.input_value,
            InnerMonitorOutput::Interval(o) => o.input_value,
        }
    }

    /// Check if there are any outputs.
    fn has_outputs(&self) -> bool {
        match &self.inner {
            InnerMonitorOutput::Bool(o) => o.has_outputs(),
            InnerMonitorOutput::Float(o) => o.has_outputs(),
            InnerMonitorOutput::Interval(o) => o.has_outputs(),
        }
    }

    /// Get the total number of outputs.
    fn total_outputs(&self) -> usize {
        match &self.inner {
            InnerMonitorOutput::Bool(o) => o.total_outputs(),
            InnerMonitorOutput::Float(o) => o.total_outputs(),
            InnerMonitorOutput::Interval(o) => o.total_outputs(),
        }
    }

    /// Check if the evaluations list is empty.
    fn is_empty(&self) -> bool {
        match &self.inner {
            InnerMonitorOutput::Bool(o) => o.is_empty(),
            InnerMonitorOutput::Float(o) => o.is_empty(),
            InnerMonitorOutput::Interval(o) => o.is_empty(),
        }
    }

    /// Get the finalized verdicts as a list of (timestamp, value) tuples.
    ///
    /// This returns the latest verdict for each unique timestamp,
    /// matching the behavior of Rust's `finalize()` method.
    fn finalize(&self) -> PyResult<Py<PyList>> {
        Python::attach(|py| {
            let list = match &self.inner {
                InnerMonitorOutput::Bool(o) => {
                    let verdicts = o.finalize();
                    let items: Vec<_> = verdicts
                        .iter()
                        .map(|step| {
                            PyTuple::new(
                                py,
                                [
                                    step.timestamp
                                        .as_secs_f64()
                                        .into_pyobject(py)
                                        .unwrap()
                                        .into_any(),
                                    PyBool::new(py, step.value).to_owned().into_any(),
                                ],
                            )
                            .unwrap()
                        })
                        .collect();
                    PyList::new(py, items).unwrap()
                }
                InnerMonitorOutput::Float(o) => {
                    let verdicts = o.finalize();
                    let items: Vec<_> = verdicts
                        .iter()
                        .map(|step| {
                            PyTuple::new(
                                py,
                                [
                                    step.timestamp
                                        .as_secs_f64()
                                        .into_pyobject(py)
                                        .unwrap()
                                        .into_any(),
                                    PyFloat::new(py, step.value).to_owned().into_any(),
                                ],
                            )
                            .unwrap()
                        })
                        .collect();
                    PyList::new(py, items).unwrap()
                }
                InnerMonitorOutput::Interval(o) => {
                    let verdicts = o.finalize();
                    let items: Vec<_> = verdicts
                        .iter()
                        .map(|step| {
                            let interval = PyTuple::new(py, [step.value.0, step.value.1]).unwrap();
                            PyTuple::new(
                                py,
                                [
                                    step.timestamp
                                        .as_secs_f64()
                                        .into_pyobject(py)
                                        .unwrap()
                                        .into_any(),
                                    interval.into_any(),
                                ],
                            )
                            .unwrap()
                        })
                        .collect();
                    PyList::new(py, items).unwrap()
                }
            };
            Ok(list.unbind())
        })
    }

    /// Returns the Display representation of the MonitorOutput.
    ///
    /// This uses the Rust Display implementation which formats verdicts as:
    /// ```
    /// t={timestamp}: {value}
    /// ```
    ///
    /// For multiple verdicts, they are shown on separate lines.
    /// If no verdicts are available, returns "No verdicts available".
    fn __str__(&self) -> String {
        match &self.inner {
            InnerMonitorOutput::Bool(o) => format_monitor_output(o),
            InnerMonitorOutput::Float(o) => format!("{}", o),
            InnerMonitorOutput::Interval(o) => format_monitor_output(o),
        }
    }

    /// Returns the Debug representation of the MonitorOutput.
    ///
    /// This uses the Rust Debug implementation which shows the full
    /// internal structure of the MonitorOutput.
    fn __repr__(&self) -> String {
        match &self.inner {
            InnerMonitorOutput::Bool(o) => format!("{:?}", o),
            InnerMonitorOutput::Float(o) => format!("{:?}", o),
            InnerMonitorOutput::Interval(o) => format!("{:?}", o),
        }
    }
}

/// Helper function to format MonitorOutput using the same logic as Rust's Display impl.
/// This is needed because Display is only implemented for MonitorOutput<f64, Y>.
fn format_monitor_output<Y: Debug + Clone>(output: &MonitorOutput<f64, Y>) -> String {
    let finalized = output.finalize();

    if finalized.is_empty() {
        return "No verdicts available".to_string();
    }

    let mut result = String::new();
    for (i, step) in finalized.iter().enumerate() {
        if i > 0 {
            result.push('\n');
        }
        result.push_str(&format!("t={:?}: {:?}", step.timestamp, step.value));
    }
    result
}

// -----------------------------------------------------------------------------
// 4. Monitor Wrapper
// -----------------------------------------------------------------------------

/// We need an enum to handle the different generics (bool, f64, RobustnessInterval)
/// because Python types are dynamic but Rust types are static.
enum InnerMonitor {
    StrictSatisfaction(StlMonitor<f64, bool>),
    EagerSatisfaction(StlMonitor<f64, bool>),
    Robustness(StlMonitor<f64, f64>),
    Rosi(StlMonitor<f64, RobustnessInterval>),
}

// SAFETY: StlMonitor and its operators are designed to be used from a single thread.
// Python's GIL ensures thread safety across Python threads.
unsafe impl Send for InnerMonitor {}
unsafe impl Sync for InnerMonitor {}

#[pyclass]
struct Monitor {
    inner: InnerMonitor,
    semantics: String,
    algorithm: String,
    synchronization: String,
}

#[pymethods]
impl Monitor {
    /// Create a new monitor.
    ///
    /// Parameters:
    /// -----------
    /// formula : Formula
    ///     The STL formula to monitor
    /// semantics : str, optional
    ///     The output semantics:
    ///     - "StrictSatisfaction": boolean satisfaction with strict evaluation
    ///     - "EagerSatisfaction": boolean satisfaction with eager evaluation
    ///     - "Robustness": quantitative robustness as a single float value (default)
    ///     - "Rosi": robustness as an interval (min, max)
    /// algorithm : str, optional
    ///     The monitoring algorithm:
    ///     - "Incremental": efficient incremental monitoring (default)
    ///     - "Naive": simple but less efficient
    /// synchronization : str, optional
    ///     The signal synchronization method:
    ///     - "ZeroOrderHold": zero-order hold (default)
    ///     - "Linear": linear interpolation
    ///     - "None": no interpolation
    /// Returns:
    /// --------
    /// Monitor
    #[new]
    #[pyo3(signature = (formula, semantics="Robustness", algorithm="Incremental", synchronization="ZeroOrderHold"))]
    fn new(
        formula: &Formula,
        semantics: &str,
        algorithm: &str,
        synchronization: &str,
    ) -> PyResult<Self> {
        // Parse algorithm
        let algo = match algorithm {
            "Incremental" => Algorithm::Incremental,
            "Naive" => Algorithm::Naive,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid algorithm. Use 'Incremental' or 'Naive'",
                ));
            }
        };

        let synchronization_strategy = match synchronization {
            "ZeroOrderHold" => SynchronizationStrategy::ZeroOrderHold,
            "Linear" => SynchronizationStrategy::Linear,
            "None" => SynchronizationStrategy::None,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid synchronization. Use 'ZeroOrderHold', 'Linear', or 'None'",
                ));
            }
        };

        // Build monitor based on semantics
        match semantics {
            "StrictSatisfaction" => {
                let m = StlMonitor::builder()
                    .formula(formula.inner.clone())
                    .algorithm(algo)
                    .semantics(StrictSatisfaction)
                    .synchronization_strategy(synchronization_strategy)
                    .build()
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
                Ok(Monitor {
                    inner: InnerMonitor::StrictSatisfaction(m),
                    semantics: semantics.to_string(),
                    algorithm: algorithm.to_string(),
                    synchronization: synchronization.to_string(),
                })
            }
            "EagerSatisfaction" => {
                let m = StlMonitor::builder()
                    .formula(formula.inner.clone())
                    .algorithm(algo)
                    .semantics(EagerSatisfaction)
                    .synchronization_strategy(synchronization_strategy)
                    .build()
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
                Ok(Monitor {
                    inner: InnerMonitor::EagerSatisfaction(m),
                    semantics: semantics.to_string(),
                    algorithm: algorithm.to_string(),
                    synchronization: synchronization.to_string(),
                })
            }
            "Robustness" => {
                let m = StlMonitor::builder()
                    .formula(formula.inner.clone())
                    .algorithm(algo)
                    .semantics(Robustness)
                    .synchronization_strategy(synchronization_strategy)
                    .build()
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
                Ok(Monitor {
                    inner: InnerMonitor::Robustness(m),
                    semantics: semantics.to_string(),
                    algorithm: algorithm.to_string(),
                    synchronization: synchronization.to_string(),
                })
            }
            "Rosi" => {
                let m = StlMonitor::builder()
                    .formula(formula.inner.clone())
                    .algorithm(algo)
                    .semantics(Rosi)
                    .synchronization_strategy(synchronization_strategy)
                    .build()
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
                Ok(Monitor {
                    inner: InnerMonitor::Rosi(m),
                    semantics: semantics.to_string(),
                    algorithm: algorithm.to_string(),
                    synchronization: synchronization.to_string(),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid semantics. Use 'StrictSatisfaction', 'EagerSatisfaction', 'Robustness', or 'Rosi'",
            )),
        }
    }

    /// Update the monitor with a new data point.
    ///
    /// Parameters:
    /// -----------
    /// signal : str
    ///     The signal name
    /// value : float
    ///     The signal value
    /// timestamp : float
    ///     The timestamp in seconds
    ///
    /// Returns:
    /// --------
    /// MonitorOutput
    ///     An object containing the monitor output with:
    ///     - Display/Debug formatting via __str__() and __repr__()
    ///     - Structured data access via to_dict() method
    ///     - Properties: input_signal, input_timestamp, input_value
    ///     - Methods: has_outputs(), total_outputs(), is_empty(), finalize()
    fn update(&mut self, signal: String, value: f64, timestamp: f64) -> PyMonitorOutput {
        let sig_ref = Box::leak(signal.into_boxed_str());
        let step = Step::new(sig_ref, value, Duration::from_secs_f64(timestamp));

        match &mut self.inner {
            InnerMonitor::Robustness(m) => {
                let output = m.update(&step);
                PyMonitorOutput {
                    inner: InnerMonitorOutput::Float(output),
                }
            }
            InnerMonitor::EagerSatisfaction(m) | InnerMonitor::StrictSatisfaction(m) => {
                let output = m.update(&step);
                PyMonitorOutput {
                    inner: InnerMonitorOutput::Bool(output),
                }
            }
            InnerMonitor::Rosi(m) => {
                let output = m.update(&step);
                PyMonitorOutput {
                    inner: InnerMonitorOutput::Interval(output),
                }
            }
        }
    }

    /// Get the set of signal identifiers used in the monitor's formula.
    /// Returns:
    /// --------
    /// Set[str]
    ///     A set of signal names (identifiers) used in the formula. 
    fn get_signal_identifiers(&mut self) -> HashSet<&'static str> {
        match &mut self.inner {
            InnerMonitor::StrictSatisfaction(m) => m.get_signal_identifiers(),
            InnerMonitor::EagerSatisfaction(m) => m.get_signal_identifiers(),
            InnerMonitor::Robustness(m) => m.get_signal_identifiers(),
            InnerMonitor::Rosi(m) => m.get_signal_identifiers(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Monitor(semantics='{}', algorithm='{}', synchronization='{}')",
            self.semantics, self.algorithm, self.synchronization
        )
    }
}

/// Helper function to convert Rust MonitorOutput to a Python Dictionary
fn convert_output_to_dict<Y, F>(
    py: Python,
    output: MonitorOutput<f64, Y>,
    val_mapper: F,
) -> PyResult<Py<PyAny>>
where
    F: Fn(Y) -> Py<PyAny>,
{
    let dict = PyDict::new(py);
    dict.set_item("input_signal", output.input_signal)?;
    dict.set_item("input_timestamp", output.input_timestamp.as_secs_f64())?;
    dict.set_item("input_value", output.input_value)?;

    // Preserve the structure of individual evaluations/sync steps
    let mut evaluations_list = Vec::new();

    // Iterate over all evaluations triggered by this input
    for eval in output.evaluations {
        let eval_dict = PyDict::new(py);

        // Add sync step information
        eval_dict.set_item("sync_step_signal", eval.sync_step.signal)?;
        eval_dict.set_item(
            "sync_step_timestamp",
            eval.sync_step.timestamp.as_secs_f64(),
        )?;
        eval_dict.set_item("sync_step_value", eval.sync_step.value)?;

        let mut outputs_list = Vec::new();

        for out_step in eval.outputs {
            let val = out_step.value;
            let output_dict = PyDict::new(py);
            output_dict.set_item("timestamp", out_step.timestamp.as_secs_f64())?;
            output_dict.set_item("value", val_mapper(val))?;
            outputs_list.push(output_dict);
        }

        eval_dict.set_item("outputs", outputs_list)?;
        evaluations_list.push(eval_dict);
    }

    dict.set_item("evaluations", evaluations_list)?;
    Ok(dict.into_any().unbind())
}

// -----------------------------------------------------------------------------
// 4. Module Definition
// -----------------------------------------------------------------------------

#[pymodule]
fn ostl_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add("__doc__", "Online Signal Temporal Logic (STL) monitoring library.\n\n\
        This library provides efficient online monitoring of STL formulas with multiple semantics:\n\
        - StrictSatisfaction/EagerSatisfaction: true/false evaluation\n\
        - Robustness: robustness as a single float value\n\
        - Rosi: robustness as an interval (min, max)\n\n\
        Example using parse_formula (recommended):\n\
        -----------------------------------------\n\
        >>> import ostl_python\n\
        >>> # Parse formula using the same DSL syntax as Rust's stl! macro\n\
        >>> phi = ostl_python.parse_formula('G[0, 5](x > 0.5)')\n\
        >>> # Create monitor with Robustness semantics\n\
        >>> monitor = ostl_python.Monitor(phi, semantics='Robustness')\n\
        >>> # Feed data\n\
        >>> output = monitor.update('x', 1.0, 0.0)\n\
        >>> # Print using Rust's Display formatting\n\
        >>> print(output)\n\
        >>> # Access structured data\n\
        >>> print(output.to_dict())\n\n\
        Example using Formula builder methods:\n\
        --------------------------------------\n\
        >>> import ostl_python\n\
        >>> # Create formula: Always[0,5](x > 0.5)\n\
        >>> phi = ostl_python.Formula.always(0, 5, ostl_python.Formula.gt('x', 0.5))\n\
        >>> # Create monitor with Robustness semantics\n\
        >>> monitor = ostl_python.Monitor(phi, semantics='Robustness')\n\
        >>> # Feed data\n\
        >>> output = monitor.update('x', 1.0, 0.0)\n\
        >>> # Use __str__ and __repr__ for Rust-style formatting\n\
        >>> print(str(output))  # Display format\n\
        >>> print(repr(output)) # Debug format\n\
    ")?;

    m.add_function(wrap_pyfunction!(py_parse_formula, m)?)?;
    m.add_class::<Formula>()?;
    m.add_class::<PyMonitorOutput>()?;
    m.add_class::<Monitor>()?;

    Ok(())
}
