use std::fmt::Display;
use std::time::Duration;

use crate::ring_buffer::RingBufferTrait;
use crate::ring_buffer::Step;
use crate::stl::core::{RobustnessSemantics, StlOperatorTrait, TimeInterval};

#[derive(Debug, Clone)]
// A generic representation of an STL formula.
pub enum StlOperator {
    // Boolean operators
    Not(Box<StlOperator>),
    And(Box<StlOperator>, Box<StlOperator>),
    Or(Box<StlOperator>, Box<StlOperator>),

    // Temporal operators
    Globally(TimeInterval, Box<StlOperator>),
    Eventually(TimeInterval, Box<StlOperator>),
    Until(TimeInterval, Box<StlOperator>, Box<StlOperator>),

    // logical operators
    Implies(Box<StlOperator>, Box<StlOperator>),

    // Atomic propositions
    True,
    False,
    GreaterThan(&'static str, f64), // signal name, threshold
    LessThan(&'static str, f64),    // signal name, threshold
}

#[derive(Debug, Clone)]
pub struct StlFormula<T, C, Y>
where
    T: 'static,
    C: RingBufferTrait<Value = T> + 'static,
    Y: RobustnessSemantics + 'static,
{
    pub formula: StlOperator,
    pub signal: C,
    pub _phantom: std::marker::PhantomData<Y>,
}

impl<T, C, Y> StlFormula<T, C, Y>
where
    T: 'static,
    C: RingBufferTrait<Value = T> + 'static,
    Y: RobustnessSemantics + 'static,
{
    pub fn new(formula: StlOperator, signal: C) -> Self {
        Self {
            formula,
            signal,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, C, Y> Display for StlFormula<T, C, Y>
where
    Y: RobustnessSemantics,
    C: RingBufferTrait<Value = T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.formula.to_string())
    }
}

impl<T, C, Y> StlOperatorTrait<T> for StlFormula<T, C, Y>
where
    T: Clone + Copy + Into<f64> + 'static,
    C: RingBufferTrait<Value = T> + Clone + 'static,
    Y: RobustnessSemantics + 'static,
{
    type Output = Y;

    fn get_max_lookahead(&self) -> Duration {
        self.formula.get_max_lookahead()
    }

    fn robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        self.signal.add_step(step.clone());

        // The top-level monitor is responsible for calculating the evaluation time.
        let max_lookahead = self.formula.get_max_lookahead();
        if step.timestamp < max_lookahead {
            return vec![]; // Not enough data to evaluate
        }

        // This is the correct, past timestamp we are evaluating for.
        let t_eval = step.timestamp.saturating_sub(max_lookahead);

        // Pass t_eval to the naive evaluator.
        let robustness = self
            .formula
            .robustness_naive(step.signal, &self.signal, t_eval);

        //if None, return empty vec; wrap inner Y into Option<Y> so types match Vec<Step<Option<Y>>>
        match robustness {
            Some(robustness_step) => {
                // The step from robustness_naive should have the t_eval timestamp
                debug_assert_eq!(robustness_step.timestamp, t_eval);
                vec![Step::new(
                    robustness_step.signal,
                    Some(robustness_step.value),
                    robustness_step.timestamp,
                )]
            }
            None => vec![],
        }
    }
}

impl StlOperator {
    /// Computes the robustness of the formula at a specific time `t_eval`.
    /// This function assumes all necessary signal data (up to `t_eval + max_lookahead`) is present.
    pub fn robustness_naive<T, C, Y>(
        &self,
        signal_name: &str,
        signal: &C,
        t_eval: Duration,
    ) -> Option<Step<Y>>
    where
        C: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
        Y: RobustnessSemantics,
    {
        match self {
            StlOperator::True => self.eval_true(t_eval),
            StlOperator::False => self.eval_false(t_eval),
            StlOperator::GreaterThan(name, c) => {
                if *name == signal_name {
                    self.eval_greater_than(*c, signal, t_eval)
                } else {
                    None
                }
            }
            StlOperator::LessThan(name, c) => {
                if *name == signal_name {
                    self.eval_less_than(*c, signal, t_eval)
                } else {
                    None
                }
            }
            StlOperator::Not(phi) => self.eval_not(phi, signal_name, signal, t_eval),
            StlOperator::And(phi, psi) => self.eval_and(phi, psi, signal_name, signal, t_eval),
            StlOperator::Or(phi, psi) => self.eval_or(phi, psi, signal_name, signal, t_eval),
            StlOperator::Implies(phi, psi) => {
                self.eval_implies(phi, psi, signal_name, signal, t_eval)
            }
            StlOperator::Eventually(interval, phi) => {
                self.eval_eventually(interval, phi, signal_name, signal, t_eval)
            }
            StlOperator::Globally(interval, phi) => {
                self.eval_globally(interval, phi, signal_name, signal, t_eval)
            }
            StlOperator::Until(interval, phi, psi) => {
                self.eval_until(interval, phi, psi, signal_name, signal, t_eval)
            }
        }
    }

    /// Recursively computes the maximum lookahead time required for the formula.
    pub fn get_max_lookahead(&self) -> Duration {
        match self {
            StlOperator::Globally(interval, f) | StlOperator::Eventually(interval, f) => {
                interval.end + f.get_max_lookahead()
            }
            StlOperator::Until(interval, f1, f2) => {
                interval.end + f1.get_max_lookahead().max(f2.get_max_lookahead())
            }
            StlOperator::Not(f) => f.get_max_lookahead(),
            StlOperator::And(f1, f2) | StlOperator::Or(f1, f2) | StlOperator::Implies(f1, f2) => {
                f1.get_max_lookahead().max(f2.get_max_lookahead())
            }
            StlOperator::True
            | StlOperator::False
            | StlOperator::GreaterThan(_, _)
            | StlOperator::LessThan(_, _) => Duration::ZERO,
        }
    }

    fn eval_true<Y: RobustnessSemantics>(&self, t_eval: Duration) -> Option<Step<Y>> {
        Some(Step::new("output", Y::atomic_true(), t_eval))
    }

    fn eval_false<Y: RobustnessSemantics>(&self, t_eval: Duration) -> Option<Step<Y>> {
        Some(Step::new("output", Y::atomic_false(), t_eval))
    }

    fn eval_greater_than<T, S, Y>(&self, c: f64, signal: &S, t_eval: Duration) -> Option<Step<Y>>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
        Y: RobustnessSemantics,
    {
        // Find the signal step at t_eval

        // only consider if signal name matches
        signal.iter().find(|s| s.timestamp == t_eval).map(|step| {
            Step::new(
                "output",
                Y::atomic_greater_than(step.value.clone().into(), c),
                t_eval,
            )
        })
    }

    fn eval_less_than<T, S, Y>(&self, c: f64, signal: &S, t_eval: Duration) -> Option<Step<Y>>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
        Y: RobustnessSemantics,
    {
        // Find the signal step at t_eval
        signal.iter().find(|s| s.timestamp == t_eval).map(|step| {
            Step::new(
                "output",
                Y::atomic_less_than(step.value.clone().into(), c),
                t_eval,
            )
        })
    }

    fn eval_not<T, S, Y>(
        &self,
        phi: &StlOperator,
        signal_name: &str,
        signal: &S,
        t_eval: Duration,
    ) -> Option<Step<Y>>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
        Y: RobustnessSemantics,
    {
        // Recursively call for the *same* t_eval
        phi.robustness_naive(signal_name, signal, t_eval)
            .map(|step| Step::new("output", Y::not(step.value), step.timestamp))
    }

    fn eval_and<T, S, Y>(
        &self,
        phi: &StlOperator,
        psi: &StlOperator,
        signal_name: &str,
        signal: &S,
        t_eval: Duration,
    ) -> Option<Step<Y>>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
        Y: RobustnessSemantics,
    {
        // Recursively call both children for the *same* t_eval
        phi.robustness_naive(signal_name, signal, t_eval)
            .zip(psi.robustness_naive(signal_name, signal, t_eval))
            .map(|(step1, step2)| {
                // Timestamps are guaranteed to be equal (both are t_eval)
                debug_assert_eq!(step1.timestamp, step2.timestamp);
                Step::new("output", Y::and(step1.value, step2.value), step1.timestamp)
            })
    }

    fn eval_or<T, S, Y>(
        &self,
        phi: &StlOperator,
        psi: &StlOperator,
        signal_name: &str,
        signal: &S,
        t_eval: Duration,
    ) -> Option<Step<Y>>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
        Y: RobustnessSemantics,
    {
        phi.robustness_naive(signal_name, signal, t_eval)
            .zip(psi.robustness_naive(signal_name, signal, t_eval))
            .map(|(step1, step2)| {
                debug_assert_eq!(step1.timestamp, step2.timestamp);
                Step::new("output", Y::or(step1.value, step2.value), step1.timestamp)
            })
    }

    fn eval_implies<T, S, Y>(
        &self,
        phi: &StlOperator,
        psi: &StlOperator,
        signal_name: &str,
        signal: &S,
        t_eval: Duration,
    ) -> Option<Step<Y>>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
        Y: RobustnessSemantics,
    {
        phi.robustness_naive(signal_name, signal, t_eval)
            .zip(psi.robustness_naive(signal_name, signal, t_eval))
            .map(|(step1, step2)| {
                debug_assert_eq!(step1.timestamp, step2.timestamp);
                Step::new(
                    "output",
                    Y::implies(step1.value, step2.value),
                    step1.timestamp,
                )
            })
    }

    fn eval_eventually<T, S, Y>(
        &self,
        interval: &TimeInterval,
        phi: &StlOperator,
        signal_name: &str,
        signal: &S,
        t_eval: Duration,
    ) -> Option<Step<Y>>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
        Y: RobustnessSemantics,
    {
        // We are evaluating for t_eval. We need data in the window [t_eval+a, t_eval+b]
        let lower_bound_t_prime = t_eval + interval.start;
        let upper_bound_t_prime = t_eval + interval.end;

        // Check if we have enough data in the signal
        if let Some(back_step) = signal.get_back() {
            if back_step.timestamp < upper_bound_t_prime {
                return None; // Not enough data to evaluate yet
            }
        } else {
            return None; // Signal is empty
        }

        let result = signal
            .iter()
            .filter(|step| {
                step.timestamp >= lower_bound_t_prime && step.timestamp <= upper_bound_t_prime
            })
            // Recursively call for *each step's timestamp* in the window
            .map(|step| phi.robustness_naive(signal_name, signal, step.timestamp))
            .fold(Some(Y::eventually_identity()), |acc, x| match (acc, x) {
                (Some(a), Some(current_step)) => Some(Y::or(a, current_step.value)),
                (Some(a), None) => Some(a), // Or should this be None? If one fails, all fail?
                (None, Some(current_step)) => Some(current_step.value),
                (None, None) => None,
            })?;

        // Return the result for the original t_eval
        Some(Step::new("output", result, t_eval))
    }

    fn eval_globally<T, S, Y>(
        &self,
        interval: &TimeInterval,
        phi: &StlOperator,
        signal_name: &str,
        signal: &S,
        t_eval: Duration,
    ) -> Option<Step<Y>>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
        Y: RobustnessSemantics,
    {
        let lower_bound_t_prime = t_eval + interval.start;
        let upper_bound_t_prime = t_eval + interval.end;

        if let Some(back_step) = signal.get_back() {
            if back_step.timestamp < upper_bound_t_prime {
                return None; // Not enough data to evaluate yet
            }
        } else {
            return None; // Signal is empty
        }

        let result = signal
            .iter()
            .filter(|step| {
                step.timestamp >= lower_bound_t_prime && step.timestamp <= upper_bound_t_prime
            })
            .map(|step| phi.robustness_naive(signal_name, signal, step.timestamp))
            .fold(Some(Y::globally_identity()), |acc, x| match (acc, x) {
                (Some(a), Some(current_step)) => Some(Y::and(a, current_step.value)),
                (Some(a), None) => Some(a),
                (None, Some(current_step)) => Some(current_step.value),
                (None, None) => None,
            })?;

        Some(Step::new("output", result, t_eval))
    }

    fn eval_until<T, S, Y>(
        &self,
        interval: &TimeInterval,
        phi: &StlOperator,
        psi: &StlOperator,
        signal_name: &str,
        signal: &S,
        t_eval: Duration,
    ) -> Option<Step<Y>>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
        Y: RobustnessSemantics,
    {
        let lower_bound_t_prime = t_eval + interval.start;
        let upper_bound_t_prime = t_eval + interval.end;

        if let Some(back_step) = signal.get_back() {
            if back_step.timestamp < upper_bound_t_prime {
                return None; // Not enough data to evaluate yet
            }
        } else {
            return None; // Signal is empty
        }

        let result = signal
            .iter()
            .filter(|step| {
                step.timestamp >= lower_bound_t_prime && step.timestamp <= upper_bound_t_prime
            })
            .map(|step_t_prime| {
                let t_prime = step_t_prime.timestamp;
                let robustness_psi = psi.robustness_naive(signal_name, signal, t_prime);

                let robustness_phi_g = signal
                    .iter()
                    .filter(|s| s.timestamp >= lower_bound_t_prime && s.timestamp < t_prime) // G is up to t_prime
                    .map(|s| phi.robustness_naive(signal_name, signal, s.timestamp)) // This is Option<Step<Y>>
                    .fold(Some(Y::globally_identity()), |acc, x| match (acc, x) {
                        (Some(a), Some(current_step)) => Some(Y::and(a, current_step.value)),
                        (Some(a), None) => Some(a),
                        (None, Some(current_step)) => Some(current_step.value),
                        (None, None) => None,
                    });

                robustness_psi
                    .zip(robustness_phi_g)
                    .map(|(r_psi, r_phi_val)| {
                        // r_psi is Step<Y>, r_phi_val is Y
                        Y::and(r_psi.value, r_phi_val) // r_psi.value
                    })
            })
            .fold(Some(Y::eventually_identity()), |acc, x| match (acc, x) {
                (Some(a), Some(robustness_value)) => Some(Y::or(a, robustness_value)),
                (Some(a), None) => Some(a),
                (None, Some(robustness_value)) => Some(robustness_value),
                (None, None) => None,
            })?;

        Some(Step::new("output", result, t_eval))
    }

    // ... (to_tree_string and Display are unchanged) ...
    /// Recursively generate a tree-like string representation of the formula.
    pub fn to_tree_string(&self, indent: usize) -> String {
        let padding = " ".repeat(indent);
        match self {
            StlOperator::True => format!("{}True", padding),
            StlOperator::False => format!("{}False", padding),
            StlOperator::Not(f) => format!("{}Not\n{}", padding, f.to_tree_string(indent + 2)),
            StlOperator::And(f1, f2) => format!(
                "{}And\n{}\n{}",
                padding,
                f1.to_tree_string(indent + 2),
                f2.to_tree_string(indent + 2)
            ),
            StlOperator::Or(f1, f2) => format!(
                "{}Or\n{}\n{}",
                padding,
                f1.to_tree_string(indent + 2),
                f2.to_tree_string(indent + 2)
            ),
            StlOperator::Globally(interval, f) => format!(
                "{}Always [{} - {}]\n{}",
                padding,
                interval.start.as_secs_f64(),
                interval.end.as_secs_f64(),
                f.to_tree_string(indent + 2)
            ),
            StlOperator::Eventually(interval, f) => format!(
                "{}Eventually [{} - {}]\n{}",
                padding,
                interval.start.as_secs_f64(),
                interval.end.as_secs_f64(),
                f.to_tree_string(indent + 2)
            ),
            StlOperator::Until(interval, f1, f2) => format!(
                "{}Until [{} - {}]\n{}\n{}",
                padding,
                interval.start.as_secs_f64(),
                interval.end.as_secs_f64(),
                f1.to_tree_string(indent + 2),
                f2.to_tree_string(indent + 2)
            ),
            StlOperator::Implies(f1, f2) => format!(
                "{}Implies\n{}\n{}",
                padding,
                f1.to_tree_string(indent + 2),
                f2.to_tree_string(indent + 2)
            ),
            StlOperator::GreaterThan(s, val) => format!("{}{} > {}", padding, s, val),
            StlOperator::LessThan(s, val) => format!("{}{} < {}", padding, s, val),
        }
    }
}
impl Display for StlOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                StlOperator::True => "True".to_string(),
                StlOperator::False => "False".to_string(),
                StlOperator::Not(f) => format!("¬({})", f.to_string()),
                StlOperator::And(f1, f2) => format!("({}) ∧ ({})", f1.to_string(), f2.to_string()),
                StlOperator::Or(f1, f2) => format!("({}) v ({})", f1.to_string(), f2.to_string()),
                StlOperator::Globally(interval, f) => format!(
                    "G[{}, {}]({})",
                    interval.start.as_secs_f64(),
                    interval.end.as_secs_f64(),
                    f.to_string()
                ),
                StlOperator::Eventually(interval, f) => format!(
                    "F[{}, {}]({})",
                    interval.start.as_secs_f64(),
                    interval.end.as_secs_f64(),
                    f.to_string()
                ),
                StlOperator::Until(interval, f1, f2) => format!(
                    "({}) U[{}, {}] ({})",
                    f1.to_string(),
                    interval.start.as_secs_f64(),
                    interval.end.as_secs_f64(),
                    f2.to_string()
                ),
                StlOperator::Implies(f1, f2) =>
                    format!("({}) → ({})", f1.to_string(), f2.to_string()),
                StlOperator::GreaterThan(s, val) => format!("{} > {}", s, val),
                StlOperator::LessThan(s, val) => format!("{} < {}", s, val),
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring_buffer::{RingBuffer, Step};
    use std::time::Duration;

    // Helper to create a signal from a vector of values and timestamps
    fn create_signal(values: Vec<f64>, timestamps: Vec<u64>) -> RingBuffer<f64> {
        let mut signal = RingBuffer::new();
        for (val, ts) in values.into_iter().zip(timestamps.into_iter()) {
            signal.add_step(Step::new("x", val, Duration::from_secs(ts)));
        }
        signal
    }

    #[test]
    fn atomic_greater_than_robustness() {
        let formula = StlOperator::GreaterThan("x", 10.0);
        let signal = create_signal(vec![15.0], vec![5]);
        let robustness =
            formula.robustness_naive::<f64, _, f64>("x", &signal, Duration::from_secs(5));
        assert_eq!(
            robustness,
            Some(Step::new("output", 5.0, Duration::from_secs(5)))
        );

        let signal2 = create_signal(vec![8.0], vec![5]);
        let robustness2 =
            formula.robustness_naive::<f64, _, f64>("x", &signal2, Duration::from_secs(5));
        assert_eq!(
            robustness2,
            Some(Step::new("output", -2.0, Duration::from_secs(5)))
        );
    }

    #[test]
    fn atomic_less_than_robustness() {
        let formula = StlOperator::LessThan("x", 10.0);
        let signal = create_signal(vec![5.0], vec![5]);
        let robustness =
            formula.robustness_naive::<f64, _, f64>("x", &signal, Duration::from_secs(5));
        assert_eq!(
            robustness,
            Some(Step::new("output", 5.0, Duration::from_secs(5)))
        );

        let signal2 = create_signal(vec![12.0], vec![5]);
        let robustness2 =
            formula.robustness_naive::<f64, _, f64>("x", &signal2, Duration::from_secs(5));
        assert_eq!(
            robustness2,
            Some(Step::new("output", -2.0, Duration::from_secs(5)))
        );
    }

    #[test]
    fn atomic_true_robustness() {
        let formula = StlOperator::True;
        let signal = create_signal(vec![0.0], vec![5]);
        let robustness =
            formula.robustness_naive::<f64, _, f64>("x", &signal, Duration::from_secs(5));
        assert_eq!(
            robustness,
            Some(Step::new("output", f64::INFINITY, Duration::from_secs(5)))
        );
    }

    #[test]
    fn atomic_false_robustness() {
        let formula = StlOperator::False;
        let signal = create_signal(vec![0.0], vec![5]);
        let robustness =
            formula.robustness_naive::<f64, _, f64>("x", &signal, Duration::from_secs(5));
        assert_eq!(
            robustness,
            Some(Step::new(
                "output",
                f64::NEG_INFINITY,
                Duration::from_secs(5)
            ))
        );
    }

    #[test]
    fn not_operator_robustness() {
        let formula = StlOperator::Not(Box::new(StlOperator::GreaterThan("x", 10.0)));
        let signal = create_signal(vec![15.0], vec![5]);
        let robustness =
            formula.robustness_naive::<f64, _, f64>("x", &signal, Duration::from_secs(5));
        assert_eq!(
            robustness,
            Some(Step::new("output", -5.0, Duration::from_secs(5)))
        );
    }

    #[test]
    fn and_operator_robustness() {
        let formula = StlOperator::And(
            Box::new(StlOperator::GreaterThan("x", 10.0)),
            Box::new(StlOperator::LessThan("x", 20.0)),
        );
        let signal = create_signal(vec![15.0], vec![5]);
        let robustness =
            formula.robustness_naive::<f64, _, f64>("x", &signal, Duration::from_secs(5));
        assert_eq!(
            robustness,
            Some(Step::new("output", 5.0, Duration::from_secs(5)))
        );
    }

    #[test]
    fn or_operator_robustness() {
        let formula = StlOperator::Or(
            Box::new(StlOperator::GreaterThan("x", 10.0)),
            Box::new(StlOperator::LessThan("x", 5.0)),
        );
        let signal = create_signal(vec![15.0], vec![5]);
        let robustness =
            formula.robustness_naive::<f64, _, f64>("x", &signal, Duration::from_secs(5));
        assert_eq!(
            robustness,
            Some(Step::new("output", 5.0, Duration::from_secs(5)))
        );
    }

    #[test]
    fn implies_operator_robustness() {
        let formula = StlOperator::Implies(
            Box::new(StlOperator::GreaterThan("x", 10.0)),
            Box::new(StlOperator::LessThan("x", 20.0)),
        );
        let signal = create_signal(vec![15.0], vec![5]);
        let robustness =
            formula.robustness_naive::<f64, _, f64>("x", &signal, Duration::from_secs(5));
        assert_eq!(
            robustness,
            Some(Step::new("output", 5.0, Duration::from_secs(5)))
        );
    }

    #[test]
    fn eventually_operator_robustness() {
        let formula = StlOperator::Eventually(
            TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(4),
            },
            Box::new(StlOperator::GreaterThan("x", 10.0)),
        );

        // Case 1: Not enough data
        let signal1 = create_signal(vec![15.0, 12.0], vec![0, 2]); // Max timestamp is 2, need up to 4
        let robustness1 =
            formula.robustness_naive::<f64, _, f64>("x", &signal1, Duration::from_secs(0));
        assert_eq!(robustness1, None);
        // Case 2: Just enough data for t_eval=0
        let signal2 = create_signal(vec![15.0, 12.0, 8.0], vec![0, 2, 4]); // Max timestamp 4
        // eval at t=0, window [0,4], values are 15, 12, 8. Robustness values: 5, 2, -2. Max is 5.
        let robustness2 =
            formula.robustness_naive::<f64, _, f64>("x", &signal2, Duration::from_secs(0));
        assert_eq!(
            robustness2,
            Some(Step::new("output", 5.0, Duration::from_secs(0)))
        );

        // Case 3: More data, eval at t=2
        let signal3 = create_signal(vec![15.0, 12.0, 8.0, 5.0], vec![0, 2, 4, 6]); // Max timestamp 6
        // eval at t=2, window [2,6], values are 12, 8, 5. Robustness values: 2, -2, -5. Max is 2.
        let robustness3 =
            formula.robustness_naive::<f64, _, f64>("x", &signal3, Duration::from_secs(2));
        assert_eq!(
            robustness3,
            Some(Step::new("output", 2.0, Duration::from_secs(2)))
        );

        // Case 4: Full signal, eval at t=4
        let signal4 = create_signal(vec![15.0, 12.0, 8.0, 5.0, 12.0], vec![0, 2, 4, 6, 8]); // Max timestamp 8
        // eval at t=4, window [4,8], values are 8, 5, 12. Robustness values: -2, -5, 2. Max is 2.
        let robustness4 =
            formula.robustness_naive::<f64, _, f64>("x", &signal4, Duration::from_secs(4));
        assert_eq!(
            robustness4,
            Some(Step::new("output", 2.0, Duration::from_secs(4)))
        );
    }

    #[test]
    fn globally_operator_robustness() {
        let formula = StlOperator::Globally(
            TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(4),
            },
            Box::new(StlOperator::GreaterThan("x", 10.0)),
        );

        // Case 1: Not enough data
        let signal1 = create_signal(vec![15.0, 12.0], vec![0, 2]); // Max timestamp is 2, need up to 4
        let robustness1 =
            formula.robustness_naive::<f64, _, f64>("x", &signal1, Duration::from_secs(0));
        assert_eq!(robustness1, None);

        // Case 2: Just enough data for t_eval=0
        let signal2 = create_signal(vec![15.0, 12.0, 8.0], vec![0, 2, 4]); // Max timestamp 4
        // eval at t=0, window [0,4], values are 15, 12, 8. Robustness values: 5, 2, -2. Min is -2.
        let robustness2 =
            formula.robustness_naive::<f64, _, f64>("x", &signal2, Duration::from_secs(0));
        assert_eq!(
            robustness2,
            Some(Step::new("output", -2.0, Duration::from_secs(0)))
        );

        // Case 3: More data, eval at t=2
        let signal3 = create_signal(vec![15.0, 12.0, 8.0, 5.0], vec![0, 2, 4, 6]); // Max timestamp 6
        // eval at t=2, window [2,6], values are 12, 8, 5. Robustness values: 2, -2, -5. Min is -5.
        let robustness3 =
            formula.robustness_naive::<f64, _, f64>("x", &signal3, Duration::from_secs(2));
        assert_eq!(
            robustness3,
            Some(Step::new("output", -5.0, Duration::from_secs(2)))
        );

        // Case 4: Full signal, eval at t=4
        let signal4 = create_signal(vec![15.0, 12.0, 8.0, 5.0, 12.0], vec![0, 2, 4, 6, 8]); // Max timestamp 8
        // eval at t=4, window [4,8], values are 8, 5, 12. Robustness values: -2, -5, 2. Min is -5.
        let robustness4 =
            formula.robustness_naive::<f64, _, f64>("x", &signal4, Duration::from_secs(4));
        assert_eq!(
            robustness4,
            Some(Step::new("output", -5.0, Duration::from_secs(4)))
        );
    }

    #[test]
    fn multi_signal_input() {
        let formula = StlOperator::And(
            Box::new(StlOperator::Globally(
                TimeInterval { start: Duration::from_secs(0), end: Duration::from_secs(2) },
                Box::new(StlOperator::GreaterThan("x", 0.0)),
            )),
            Box::new(StlOperator::LessThan("y", 5.0)),
        );

        let mut f = StlFormula::new(
            formula.clone(),
            RingBuffer::new(), // Placeholder, not used in naive evaluation
        );

        let signal_x = create_signal(vec![0.0, 6.0, 1.0, 3.0, 8.0, 1.0, 7.0], vec![0, 1, 2, 3, 4, 5, 6]);
        let signal_y = create_signal(vec![4.0, 3.0, 6.0, 7.0, 2.0, 1.0, 0.0], vec![0, 1, 2, 3, 4, 5, 6]);
        let expected_f64 = vec![
            vec![],
            vec![],
            vec![Step::new("output", 0.0, Duration::from_secs(0))],
            vec![Step::new("output", 2.0, Duration::from_secs(1))],
            vec![Step::new("output", -1.0, Duration::from_secs(2))],
            vec![Step::new("output", -2.0, Duration::from_secs(3))],
            vec![Step::new("output", 1.0, Duration::from_secs(4))],
        ];
        let mut results: Vec<Vec<Step<f64>>> = vec![];

    }
}
