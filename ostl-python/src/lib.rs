use ostl::ring_buffer::Step;
use ostl::stl::core::{RobustnessInterval, TimeInterval};
use ostl::stl::formula_definition::FormulaDefinition;
use ostl::stl::monitor::{EvaluationMode, MonitorOutput, MonitoringStrategy, StlMonitor};
use ostl::synchronizer::InterpolationStrategy;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyTuple};
use std::time::Duration;

// -----------------------------------------------------------------------------
// 1. Formula Wrapper
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
        format!("Formula({})", self.inner.to_string())
    }
}

// -----------------------------------------------------------------------------
// 2. Monitor Wrapper
// -----------------------------------------------------------------------------

/// We need an enum to handle the different generics (bool, f64, RobustnessInterval)
/// because Python types are dynamic but Rust types are static.
enum InnerMonitor {
    Qualitative(StlMonitor<f64, bool>),
    Quantitative(StlMonitor<f64, f64>),
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
    strategy: String,
    mode: String,
    interpolation: String,
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
    ///     - "qualitative": returns True/False (default)
    ///     - "quantitative": returns robustness as a single float value
    ///     - "rosi": returns robustness as an interval (min, max)
    /// strategy : str, optional
    ///     The monitoring strategy:
    ///     - "incremental": efficient incremental monitoring (default)
    ///     - "naive": simple but less efficient
    /// mode : str, optional
    ///     The evaluation mode:
    ///     - "eager": produces verdicts as soon as possible (default for robustness)
    ///     - "strict": waits for complete information (default for qualitative/quantitative)
    /// interpolation : str, optional
    ///     The signal interpolation method:
    ///     - "zoh": zero-order hold (default)
    ///     - "linear": linear interpolation
    ///     - "none": no interpolation
    /// Returns:
    /// --------
    /// Monitor
    #[new]
    #[pyo3(signature = (formula, semantics="qualitative", strategy="incremental", mode=None, interpolation="zoh"))]
    fn new(
        formula: &Formula,
        semantics: &str,
        strategy: &str,
        mode: Option<&str>,
        interpolation: &str,
    ) -> PyResult<Self> {
        // Parse strategy
        let monitoring_strategy = match strategy {
            "incremental" => MonitoringStrategy::Incremental,
            "naive" => MonitoringStrategy::Naive,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid strategy. Use 'incremental' or 'naive'",
                ));
            }
        };

        // Determine default evaluation mode based on semantics
        let default_mode = if semantics == "qualitative" || semantics == "quantitative" {
            "strict"
        } else {
            "eager"
        };

        let mode_str = mode.unwrap_or(default_mode);
        let evaluation_mode = match mode_str {
            "eager" => EvaluationMode::Eager,
            "strict" => EvaluationMode::Strict,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid mode. Use 'eager' or 'strict'",
                ));
            }
        };

        let interpolation_strategy = match interpolation {
            "zoh" => ostl::synchronizer::InterpolationStrategy::ZeroOrderHold,
            "linear" => ostl::synchronizer::InterpolationStrategy::Linear,
            "none" => ostl::synchronizer::InterpolationStrategy::None,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid interpolation. Use 'zoh', 'linear', or 'none'",
                ));
            }
        };

        // Build monitor based on semantics
        match semantics {
            "qualitative" => {
                let m = StlMonitor::<f64, bool>::builder()
                    .formula(formula.inner.clone())
                    .strategy(monitoring_strategy)
                    .evaluation_mode(evaluation_mode)
                    .interpolation_strategy(interpolation_strategy)
                    .build()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
                Ok(Monitor {
                    inner: InnerMonitor::Qualitative(m),
                    semantics: semantics.to_string(),
                    strategy: strategy.to_string(),
                    mode: mode_str.to_string(),
                    interpolation: interpolation.to_string(),
                })
            }
            "quantitative" => {
                // Quantitative doesn't support Eager mode
                if evaluation_mode == EvaluationMode::Eager {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Eager evaluation mode is not supported for quantitative semantics. Use 'strict' mode.",
                    ));
                }
                let m = StlMonitor::<f64, f64>::builder()
                    .formula(formula.inner.clone())
                    .strategy(monitoring_strategy)
                    .evaluation_mode(evaluation_mode)
                    .interpolation_strategy(interpolation_strategy)
                    .build()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
                Ok(Monitor {
                    inner: InnerMonitor::Quantitative(m),
                    semantics: semantics.to_string(),
                    strategy: strategy.to_string(),
                    mode: mode_str.to_string(),
                    interpolation: interpolation.to_string(),
                })
            }
            "rosi" => {
                let m = StlMonitor::<f64, RobustnessInterval>::builder()
                    .formula(formula.inner.clone())
                    .strategy(monitoring_strategy)
                    .evaluation_mode(evaluation_mode)
                    .interpolation_strategy(interpolation_strategy)
                    .build()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
                Ok(Monitor {
                    inner: InnerMonitor::Rosi(m),
                    semantics: semantics.to_string(),
                    strategy: strategy.to_string(),
                    mode: mode_str.to_string(),
                    interpolation: interpolation.to_string(),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid semantics. Use 'qualitative', 'quantitative', or 'rosi'",
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
    /// dict
    ///     A dictionary containing:
    ///     - 'input_signal': the signal name
    ///     - 'input_timestamp': the input timestamp
    ///     - 'evaluations': list of evaluation dictionaries, each containing:
    ///         - 'sync_step_signal': signal name of the synchronized step
    ///         - 'sync_step_timestamp': timestamp of the synchronized step
    ///         - 'sync_step_value': value of the synchronized step
    ///         - 'outputs': list of output dictionaries with:
    ///             - 'timestamp': when the verdict is for
    ///             - 'value': the verdict value (bool, float, or tuple depending on semantics)
    fn update(&mut self, signal: String, value: f64, timestamp: f64) -> PyResult<Py<PyAny>> {
        let sig_ref = Box::leak(signal.into_boxed_str());
        let step = Step::new(sig_ref, value, Duration::from_secs_f64(timestamp));

        Python::attach(|py| match &mut self.inner {
            InnerMonitor::Qualitative(m) => {
                let output = m.update(&step);
                convert_output(py, output, |val| {
                    PyBool::new(py, val).to_owned().into_any().unbind()
                })
            }
            InnerMonitor::Quantitative(m) => {
                let output = m.update(&step);
                convert_output(py, output, |val| PyFloat::new(py, val).into_any().unbind())
            }
            InnerMonitor::Rosi(m) => {
                let output = m.update(&step);
                convert_output(py, output, |val| {
                    PyTuple::new(py, &[val.0, val.1])
                        .unwrap()
                        .into_any()
                        .unbind()
                })
            }
        })
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            InnerMonitor::Qualitative(_) => format!(
                "Monitor(semantics='qualitative', strategy='{}', mode='{}', interpolation='{}')",
                self.strategy, self.mode, self.interpolation
            ),
            InnerMonitor::Quantitative(_) => format!(
                "Monitor(semantics='quantitative', strategy='{}', mode='{}', interpolation='{}')",
                self.strategy, self.mode, self.interpolation
            ),
            InnerMonitor::Rosi(_) => format!(
                "Monitor(semantics='rosi', strategy='{}', mode='{}', interpolation='{}')",
                self.strategy, self.mode, self.interpolation
            ),
        }
    }
}

/// Helper function to convert Rust MonitorOutput to a Python Dictionary
fn convert_output<Y, F>(
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
            if let Some(val) = out_step.value {
                let output_dict = PyDict::new(py);
                output_dict.set_item("timestamp", out_step.timestamp.as_secs_f64())?;
                output_dict.set_item("value", val_mapper(val))?;
                outputs_list.push(output_dict);
            }
        }

        eval_dict.set_item("outputs", outputs_list)?;
        evaluations_list.push(eval_dict);
    }

    dict.set_item("evaluations", evaluations_list)?;
    Ok(dict.into_any().unbind())
}

// -----------------------------------------------------------------------------
// 3. Module Definition
// -----------------------------------------------------------------------------

#[pymodule]
fn ostl_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add("__doc__", "Online Signal Temporal Logic (STL) monitoring library.\n\n\
        This library provides efficient online monitoring of STL formulas with multiple semantics:\n\
        - Boolean: classic true/false evaluation\n\
        - Quantitative: robustness as a single float value\n\
        - Robustness (RoSI): robustness as an interval (min, max)\n\n\
        Example:\n\
        --------\n\
        >>> import ostl_python\n\
        >>> # Create formula: Always[0,5](x > 0.5)\n\
        >>> phi = ostl_python.Formula.always(0, 5, ostl_python.Formula.gt('x', 0.5))\n\
        >>> # Create monitor with robustness semantics\n\
        >>> monitor = ostl_python.Monitor(phi, semantics='robustness')\n\
        >>> # Feed data\n\
        >>> result = monitor.update('x', 1.0, 0.0)\n\
        >>> print(result['verdicts'])\n\
    ")?;

    m.add_class::<Formula>()?;
    m.add_class::<Monitor>()?;

    Ok(())
}
