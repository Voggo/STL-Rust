use std::time::Duration;
use std::fmt::Display;

use crate::ring_buffer::RingBufferTrait;
use crate::ring_buffer::Step;
use crate::stl::core::{StlOperatorTrait, TimeInterval};

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
pub struct StlFormula<T, C>
where
    C: RingBufferTrait<Value = T>,
{
    pub formula: StlOperator,
    pub signal: C,
}
impl <T, C> Display for StlFormula<T, C>
where
    C: RingBufferTrait<Value = T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.formula.to_string())
    }
}

impl<T, C> StlOperatorTrait<T, f64> for StlFormula<T, C>
where
    T: Clone + Copy + Into<f64>,
    C: RingBufferTrait<Value = T> + Clone,
{
    fn robustness(&mut self, step: &Step<T>) -> Option<f64> {
        self.signal.add_step(step.value.clone(), step.timestamp);
        self.formula.robustness_naive(&self.signal, step) // robustness for signal at step.timestamp
    }
}

impl<T, C> StlOperatorTrait<T, bool> for StlFormula<T, C>
where
    T: Clone + Copy + Into<f64>,
    C: RingBufferTrait<Value = T> + Clone,
{
    fn robustness(&mut self, step: &Step<T>) -> Option<bool> {
        self.signal.add_step(step.value.clone(), step.timestamp);
        self.formula.robustness_naive(&self.signal, step).map(|r| r > 0.0) // robustness for signal at step.timestamp
    }
}

impl StlOperator {
    /// Computes the robustness of the formula at a given time step using a naive recursive approach.
    /// This method directly implements the STL semantics for robustness calculation.
    /// It may not be efficient for large signals or complex formulas due to its recursive nature.
    /// Returns `None` if the signal does not have enough data to evaluate the formula at the given time step.
    pub fn robustness_naive<T, C>(&self, signal: &C, current_step: &Step<T>) -> Option<f64>
    where
        C: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
    {
        match self {
            StlOperator::True => self.eval_true::<T>(),
            StlOperator::False => self.eval_false::<T>(),
            StlOperator::GreaterThan(c) => self.eval_greater_than(*c, current_step),
            StlOperator::LessThan(c) => self.eval_less_than(*c, current_step),
            StlOperator::Not(phi) => self.eval_not(phi, signal, current_step),
            StlOperator::And(phi, psi) => self.eval_and(phi, psi, signal, current_step),
            StlOperator::Or(phi, psi) => self.eval_or(phi, psi, signal, current_step),
            StlOperator::Implies(phi, psi) => self.eval_implies(phi, psi, signal, current_step),
            StlOperator::Eventually(interval, phi) => self.eval_eventually(interval, phi, signal, current_step),
            StlOperator::Globally(interval, phi) => self.eval_globally(interval, phi, signal, current_step),
            StlOperator::Until(interval, phi, psi) => self.eval_until(interval, phi, psi, signal, current_step),
        }
    }

    /// Recursively computes the maximum lookahead time required for the formula.
    pub fn get_max_lookahead(&self) -> Duration {
        match self {
            StlOperator::Globally(interval, f)
            | StlOperator::Eventually(interval, f)
            | StlOperator::Until(interval, f, _) => interval.end.max(f.get_max_lookahead()),
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

    fn eval_true<T>(&self) -> Option<f64> {
        Some(f64::INFINITY)
    }

    fn eval_false<T>(&self) -> Option<f64> {
        Some(f64::NEG_INFINITY)
    }

    fn eval_greater_than<T: Clone + Into<f64>>(&self, c: f64, current_step: &Step<T>) -> Option<f64> {
        Some(current_step.value.clone().into() - c)
    }

    fn eval_less_than<T: Clone + Into<f64>>(&self, c: f64, current_step: &Step<T>) -> Option<f64> {
        Some(c - current_step.value.clone().into())
    }

    fn eval_not<T, S>(&self, phi: &StlOperator, signal: &S, current_step: &Step<T>) -> Option<f64>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
    {
        phi.robustness_naive(signal, current_step).map(|r| -r)
    }

    fn eval_and<T, S>(&self, phi: &StlOperator, psi: &StlOperator, signal: &S, current_step: &Step<T>) -> Option<f64>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
    {
        phi.robustness_naive(signal, current_step)
            .zip(psi.robustness_naive(signal, current_step))
            .map(|(r1, r2)| r1.min(r2))
    }

    fn eval_or<T, S>(&self, phi: &StlOperator, psi: &StlOperator, signal: &S, current_step: &Step<T>) -> Option<f64>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
    {
        phi.robustness_naive(signal, current_step)
            .zip(psi.robustness_naive(signal, current_step))
            .map(|(r1, r2)| r1.max(r2))
    }

    fn eval_implies<T, S>(&self, phi: &StlOperator, psi: &StlOperator, signal: &S, current_step: &Step<T>) -> Option<f64>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
    {
        phi.robustness_naive(signal, current_step)
            .zip(psi.robustness_naive(signal, current_step))
            .map(|(r1, r2)| (-r1).max(r2))
    }

    fn eval_eventually<T, S>(&self, interval: &TimeInterval, phi: &StlOperator, signal: &S, current_step: &Step<T>) -> Option<f64>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
    {
        let t = current_step.timestamp.saturating_sub(interval.end);
        let lower_bound_t_prime = t + interval.start;
        let upper_bound_t_prime = t + interval.end;
        let back = signal.get_back()?.timestamp;
        if signal.is_empty() || upper_bound_t_prime - lower_bound_t_prime > back - t {
            return None; // Not enough data to evaluate
        }

        let result = signal
            .iter()
            .filter(|step| step.timestamp >= lower_bound_t_prime && step.timestamp <= upper_bound_t_prime)
            .map(|step| phi.robustness_naive(signal, step))
            .fold(f64::NEG_INFINITY, |acc, x| acc.max(x.unwrap_or(f64::NEG_INFINITY)));
        Some(result)
    }

    fn eval_globally<T, S>(&self, interval: &TimeInterval, phi: &StlOperator, signal: &S, current_step: &Step<T>) -> Option<f64>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
    {
        let t = current_step.timestamp.saturating_sub(interval.end);
        let lower_bound_t_prime = t + interval.start;
        let upper_bound_t_prime = t + interval.end;
        let back = signal.get_back()?.timestamp;
        if signal.is_empty() || upper_bound_t_prime - lower_bound_t_prime > back - t {
            return None; // Not enough data to evaluate
        }

        let result = signal
            .iter()
            .filter(|step| step.timestamp >= lower_bound_t_prime && step.timestamp <= upper_bound_t_prime)
            .map(|step| phi.robustness_naive(signal, step))
            .fold(f64::INFINITY, |acc, x| acc.min(x.unwrap_or(f64::INFINITY)));
        Some(result)
    }

    fn eval_until<T, S>(&self, interval: &TimeInterval, phi: &StlOperator, psi: &StlOperator, signal: &S, current_step: &Step<T>) -> Option<f64>
    where
        S: RingBufferTrait<Value = T>,
        T: Clone + Copy + Into<f64>,
    {
        let t = current_step.timestamp.saturating_sub(interval.end);
        let lower_bound_t_prime = t + interval.start;
        let upper_bound_t_prime = t + interval.end;
        let back = signal.get_back()?.timestamp;
        if signal.is_empty() || upper_bound_t_prime - lower_bound_t_prime > back - t {
            return None; // Not enough data to evaluate
        }

        let result = signal
            .iter()
            .filter(|step| step.timestamp >= lower_bound_t_prime && step.timestamp <= upper_bound_t_prime)
            .map(|step| {
                let t_prime = step.timestamp;
                let robustness_psi = psi.robustness_naive(signal, step);
                let robustness_phi = signal
                    .iter()
                    .filter(|s| s.timestamp >= lower_bound_t_prime && s.timestamp <= t_prime)
                    .map(|s| phi.robustness_naive(signal, s))
                    .fold(f64::NEG_INFINITY, |acc, x| acc.max(x.unwrap_or(f64::NEG_INFINITY)));
                robustness_psi.map(|r_psi| r_psi.min(robustness_phi))
            })
            .fold(f64::INFINITY, |acc, x| acc.min(x.unwrap_or(f64::INFINITY)));
        Some(result)
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
        write!(f, "{}", match self {
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
            StlOperator::Implies(f1, f2) => format!("({}) → ({})", f1.to_string(), f2.to_string()),
            StlOperator::GreaterThan(val) => format!("x > {}", val),
            StlOperator::LessThan(val) => format!("x < {}", val),
        })
    }
}
