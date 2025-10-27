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
    GreaterThan(f64),
    LessThan(f64),
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_max_lookahead(&self) -> Duration {
        self.formula.get_max_lookahead()
    }

    fn robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        self.signal.add_step(step.clone());
        let robustness = self.formula.robustness_naive(&self.signal, step); // robustness for signal at step.timestamp

        //if None, return empty vec; wrap inner Y into Option<Y> so types match Vec<Step<Option<Y>>>
        match robustness {
            Some(robustness_step) => vec![Step::new(Some(robustness_step.value), robustness_step.timestamp)],
            None => vec![],
        }
    }
}

impl StlOperator {
    /// Computes the robustness of the formula at a given time step using a naive recursive approach.
    /// This method directly implements the STL semantics for robustness calculation.
    /// It may not be efficient for large signals or complex formulas due to its recursive nature.
    /// Returns `None` if the signal does not have enough data to evaluate the formula at the given time step.
    pub fn robustness_naive<T, C, Y>(&self, signal: &C, current_step: &Step<T>) -> Option<Step<Y>>
    where
        C: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
        Y: RobustnessSemantics,
    {
        match self {
            StlOperator::True => self.eval_true(current_step),
            StlOperator::False => self.eval_false(current_step),
            StlOperator::GreaterThan(c) => self.eval_greater_than(*c, current_step),
            StlOperator::LessThan(c) => self.eval_less_than(*c, current_step),
            StlOperator::Not(phi) => self.eval_not(phi, signal, current_step),
            StlOperator::And(phi, psi) => self.eval_and(phi, psi, signal, current_step),
            StlOperator::Or(phi, psi) => self.eval_or(phi, psi, signal, current_step),
            StlOperator::Implies(phi, psi) => self.eval_implies(phi, psi, signal, current_step),
            StlOperator::Eventually(interval, phi) => {
                self.eval_eventually(interval, phi, signal, current_step)
            }
            StlOperator::Globally(interval, phi) => {
                self.eval_globally(interval, phi, signal, current_step)
            }
            StlOperator::Until(interval, phi, psi) => {
                self.eval_until(interval, phi, psi, signal, current_step)
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
            | StlOperator::GreaterThan(_)
            | StlOperator::LessThan(_) => Duration::ZERO,
        }
    }

    fn eval_true<T, Y: RobustnessSemantics>(&self, current_step: &Step<T>) -> Option<Step<Y>> 
    where
        T: Clone + Into<f64>,
        Y: RobustnessSemantics,
    {
        Some(Step::new(Y::atomic_true(), current_step.timestamp))
    }

    fn eval_false<T, Y: RobustnessSemantics>(&self, current_step: &Step<T>) -> Option<Step<Y>> 
    where
        T: Clone + Into<f64>,
        Y: RobustnessSemantics,
    {
        Some(Step::new(Y::atomic_false(), current_step.timestamp))
    }

    fn eval_greater_than<T, Y>(&self, c: f64, current_step: &Step<T>) -> Option<Step<Y>>
    where
        T: Clone + Into<f64>,
        Y: RobustnessSemantics,
    {
        Some(Step::new(
            Y::atomic_greater_than(current_step.value.clone().into(), c),
            current_step.timestamp,
        ))
    }

    fn eval_less_than<T, Y>(&self, c: f64, current_step: &Step<T>) -> Option<Step<Y>>
    where
        T: Clone + Into<f64>,
        Y: RobustnessSemantics,
    {
        Some(Step::new(
            Y::atomic_less_than(current_step.value.clone().into(), c),
            current_step.timestamp,
        ))
    }

    fn eval_not<T, S, Y>(
        &self,
        phi: &StlOperator,
        signal: &S,
        current_step: &Step<T>,
    ) -> Option<Step<Y>>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
        Y: RobustnessSemantics,
    {
        phi.robustness_naive(signal, current_step)
            .map(|step| Step::new(Y::not(step.value), step.timestamp))
    }

    fn eval_and<T, S, Y>(
        &self,
        phi: &StlOperator,
        psi: &StlOperator,
        signal: &S,
        current_step: &Step<T>,
    ) -> Option<Step<Y>>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
        Y: RobustnessSemantics,
    {
        phi.robustness_naive(signal, current_step)
            .zip(psi.robustness_naive(signal, current_step))
            .map(|(step1, step2)| {
                // Timestamps must be equal since they originate from the same current_step
                debug_assert_eq!(step1.timestamp, step2.timestamp);
                Step::new(Y::and(step1.value, step2.value), step1.timestamp)
            })
    }

    fn eval_or<T, S, Y>(
        &self,
        phi: &StlOperator,
        psi: &StlOperator,
        signal: &S,
        current_step: &Step<T>,
    ) -> Option<Step<Y>>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
        Y: RobustnessSemantics,
    {
        phi.robustness_naive(signal, current_step)
            .zip(psi.robustness_naive(signal, current_step))
            .map(|(step1, step2)| {
                // Timestamps must be equal since they originate from the same current_step
                debug_assert_eq!(step1.timestamp, step2.timestamp);
                Step::new(Y::or(step1.value, step2.value), step1.timestamp)
            })
    }

    fn eval_implies<T, S, Y>(
        &self,
        phi: &StlOperator,
        psi: &StlOperator,
        signal: &S,
        current_step: &Step<T>,
    ) -> Option<Step<Y>>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
        Y: RobustnessSemantics,
    {
        phi.robustness_naive(signal, current_step)
            .zip(psi.robustness_naive(signal, current_step))
            .map(|(step1, step2)| {
                // Timestamps must be equal since they originate from the same current_step
                debug_assert_eq!(step1.timestamp, step2.timestamp);
                Step::new(Y::implies(step1.value, step2.value), step1.timestamp)
            })
    }

    fn eval_eventually<T, S, Y>(
        &self,
        interval: &TimeInterval,
        phi: &StlOperator,
        signal: &S,
        current_step: &Step<T>,
    ) -> Option<Step<Y>>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
        Y: RobustnessSemantics,
    {
        let max_lookahead = self.get_max_lookahead();
        if current_step.timestamp < max_lookahead {
            return None; // Not enough data to evaluate
        }
        let t = current_step.timestamp.saturating_sub(max_lookahead);
        let lower_bound_t_prime = t + interval.start;
        let upper_bound_t_prime = t + interval.end;
        let back = signal.get_back()?.timestamp;
        if signal.is_empty() || upper_bound_t_prime - lower_bound_t_prime > back - t {
            return None; // Not enough data to evaluate
        }

        let result = signal
            .iter()
            .filter(|step| {
                step.timestamp >= lower_bound_t_prime && step.timestamp <= upper_bound_t_prime
            })
            .map(|step| phi.robustness_naive(signal, step))
            .fold(Some(Y::eventually_identity()), |acc, x| match (acc, x) {
                (Some(a), Some(current_step)) => Some(Y::or(a, current_step.value)),
                (Some(a), None) => Some(a),
                (None, Some(current_step)) => Some(current_step.value),
                (None, None) => None,
            })?;

        // FIXED: Return a Step with the calculated evaluation timestamp 't'
        Some(Step::new(result, t))
    }

    fn eval_globally<T, S, Y>(
        &self,
        interval: &TimeInterval,
        phi: &StlOperator,
        signal: &S,
        current_step: &Step<T>,
    ) -> Option<Step<Y>>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
        Y: RobustnessSemantics,
    {
        let max_lookahead = self.get_max_lookahead();

        if current_step.timestamp < max_lookahead {
            return None; // Not enough data to evaluate
        }

        let t = current_step.timestamp.saturating_sub(max_lookahead);
        
        let lower_bound_t_prime = t + interval.start;
        let upper_bound_t_prime = t + interval.end;
        let back = signal.get_back()?.timestamp;
        if signal.is_empty() || upper_bound_t_prime - lower_bound_t_prime > back - t {
            return None; // Not enough data to evaluate
        }

        let result = signal
            .iter()
            .filter(|step| {
                step.timestamp >= lower_bound_t_prime && step.timestamp <= upper_bound_t_prime
            })
            .map(|step| phi.robustness_naive(signal, step))
            .fold(Some(Y::globally_identity()), |acc, x| match (acc, x) {
                (Some(a), Some(current_step)) => Some(Y::and(a, current_step.value)),
                (Some(a), None) => Some(a),
                (None, Some(current_step)) => Some(current_step.value),
                (None, None) => None,
            })?;
        Some(Step::new(result, t))
    }

    fn eval_until<T, S, Y>(
        &self,
        interval: &TimeInterval,
        phi: &StlOperator,
        psi: &StlOperator,
        signal: &S,
        current_step: &Step<T>,
    ) -> Option<Step<Y>>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
        Y: RobustnessSemantics,
    {
        let max_lookahead = self.get_max_lookahead();
        if current_step.timestamp < max_lookahead {
            return None; // Not enough data to evaluate
        }
        let t = current_step.timestamp.saturating_sub(max_lookahead);
        let lower_bound_t_prime = t + interval.start;
        let upper_bound_t_prime = t + interval.end;
        let back = signal.get_back()?.timestamp;
        if signal.is_empty() || upper_bound_t_prime - lower_bound_t_prime > back - t {
            return None; // Not enough data to evaluate
        }

        let result = signal
            .iter()
            .filter(|step| {
                step.timestamp >= lower_bound_t_prime && step.timestamp <= upper_bound_t_prime
            })
            .map(|step| {
                let t_prime = step.timestamp;
                let robustness_psi = psi.robustness_naive(signal, step);
                let robustness_phi = signal
                    .iter()
                    .filter(|s| s.timestamp >= lower_bound_t_prime && s.timestamp <= t_prime)
                    .map(|s| phi.robustness_naive(signal, s)) // This is Option<Step<Y>>
                    .fold(Some(Y::globally_identity()), |acc, x| match (acc, x) {
                        (Some(a), Some(current_step)) => Some(Y::and(a, current_step.value)), // FIXED
                        (Some(a), None) => Some(a),
                        (None, Some(current_step)) => Some(current_step.value), // FIXED
                        (None, None) => None,
                    });

                robustness_psi
                    .zip(robustness_phi)
                    .map(|(r_psi, r_phi_val)| {
                        // r_psi is Step<Y>, r_phi_val is Y
                        Y::and(r_psi.value, r_phi_val) // r_psi.value
                    })
            })
            .fold(Some(Y::eventually_identity()), |acc, x| match (acc, x) {
                (Some(a), Some(robustness_value)) => Some(Y::or(a, robustness_value)), // FIXED (robustness_value is Y)
                (Some(a), None) => Some(a),
                (None, Some(robustness_value)) => Some(robustness_value), // FIXED (robustness_value is Y)
                (None, None) => None,
            })?;

        Some(Step::new(result, t)) // FIXED
    }

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
            StlOperator::GreaterThan(val) => format!("{}x > {}", padding, val),
            StlOperator::LessThan(val) => format!("{}x < {}", padding, val),
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
                StlOperator::GreaterThan(val) => format!("x > {}", val),
                StlOperator::LessThan(val) => format!("x < {}", val),
            }
        )
    }
}
