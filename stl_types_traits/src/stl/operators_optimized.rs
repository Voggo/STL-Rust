use std::time::Duration;
use crate::signal::{Signal, SignalTrait, Step};
use std::ops::Index;

// Time interval type
#[derive(Debug, Copy, Clone)]
pub struct TimeInterval {
    pub start: Duration,
    pub end: Duration,
}

#[derive(Debug, Clone)]
// A generic representation of an STL formula.
pub enum STLFormula {
    // Boolean operators
    Not(Box<STLFormula>, Box<Cache>),
    And(Box<STLFormula>, Box<STLFormula>, Box<Cache>),
    Or(Box<STLFormula>, Box<STLFormula>, Box<Cache>),

    // Temporal operators
    Always(TimeInterval, Box<STLFormula>, Box<Cache>),
    Eventually(TimeInterval, Box<STLFormula>, Box<Cache>),
    Until(TimeInterval, Box<STLFormula>, Box<STLFormula>, Box<Cache>),

    // logical operators
    Implies(Box<STLFormula>, Box<STLFormula>, Box<Cache>),

    // Atomic propositions
    True(Box<Cache>),
    False(Box<Cache>),
    GreaterThan(f64, Box<Cache>),
    LessThan(f64, Box<Cache>),
}

impl STLFormula {
    /// Recursively generates a pretty-printed string representation of the formula.
    pub fn to_string(&self) -> String {
        match self {
            STLFormula::True(_) => "True".to_string(),
            STLFormula::False(_) => "False".to_string(),
            STLFormula::Not(f, _) => format!("¬({})", f.to_string()),
            STLFormula::And(f1, f2, _) => format!("({}) /\\ ({})", f1.to_string(), f2.to_string()),
            STLFormula::Or(f1, f2, _) => format!("({}) \\/ ({})", f1.to_string(), f2.to_string()),
            STLFormula::Always(interval, f, _) => format!(
                "G[{}, {}]({})",
                interval.start.as_secs_f64(),
                interval.end.as_secs_f64(),
                f.to_string()
            ),
            STLFormula::Eventually(interval, f, _) => format!(
                "F[{}, {}]({})",
                interval.start.as_secs_f64(),
                interval.end.as_secs_f64(),
                f.to_string()
            ),
            STLFormula::Until(interval, f1, f2, _) => format!(
                "({}) U[{}, {}] ({})",
                f1.to_string(),
                interval.start.as_secs_f64(),
                interval.end.as_secs_f64(),
                f2.to_string()
            ),
            STLFormula::Implies(f1, f2, _) => format!("({}) -> ({})", f1.to_string(), f2.to_string()),
            STLFormula::GreaterThan(val, _) => format!("x > {}", val),
            STLFormula::LessThan(val, _) => format!("x < {}", val),
        }
    }

    /// Recursively generate a tree-like string representation of the formula.
    pub fn to_tree_string(&self, indent: usize) -> String {
        let padding = " ".repeat(indent);
        match self {
            STLFormula::True(_) => format!("{}True", padding),
            STLFormula::False(_) => format!("{}False", padding),
            STLFormula::Not(f, _) => format!(
                "{}Not\n{}",
                padding,
                f.to_tree_string(indent + 2)
            ),
            STLFormula::And(f1, f2, _) => format!(
                "{}And\n{}\n{}",
                padding,
                f1.to_tree_string(indent + 2),
                f2.to_tree_string(indent + 2)
            ),
            STLFormula::Or(f1, f2, _) => format!(
                "{}Or\n{}\n{}",
                padding,
                f1.to_tree_string(indent + 2),
                f2.to_tree_string(indent + 2)
            ),
            STLFormula::Always(interval, f, _) => format!(
                "{}Always [{} - {}]\n{}",
                padding,
                interval.start.as_secs_f64(),
                interval.end.as_secs_f64(),
                f.to_tree_string(indent + 2)
            ),
            STLFormula::Eventually(interval, f, _) => format!(
                "{}Eventually [{} - {}]\n{}",
                padding,
                interval.start.as_secs_f64(),
                interval.end.as_secs_f64(),
                f.to_tree_string(indent + 2)
            ),
            STLFormula::Until(interval, f1, f2, _) => format!(
                "{}Until [{} - {}]\n{}\n{}",
                padding,
                interval.start.as_secs_f64(),
                interval.end.as_secs_f64(),
                f1.to_tree_string(indent + 2),
                f2.to_tree_string(indent + 2)
            ),
            STLFormula::Implies(f1, f2, _) => format!(
                "{}Implies\n{}\n{}",
                padding,
                f1.to_tree_string(indent + 2),
                f2.to_tree_string(indent + 2)
            ),
            STLFormula::GreaterThan(val, _) => format!("{}x > {}", padding, val),
            STLFormula::LessThan(val, _) => format!("{}x < {}", padding, val),
        }
    }
    

    /// Recursively computes the maximum lookahead time required for the formula.
    pub fn get_max_lookahead(&self) -> Duration {
        match self {
            STLFormula::Always(interval, f, _)
            | STLFormula::Eventually(interval, f, _) => {
                interval.end.max(f.get_max_lookahead())
            },
            STLFormula::Until(.., f1, f2, _)
            | STLFormula::And(f1, f2, _)
            | STLFormula::Or(f1, f2, _)
            | STLFormula::Implies(f1, f2, _) => {
                f1.get_max_lookahead().max(f2.get_max_lookahead())
            }
            STLFormula::Not(f, _) => f.get_max_lookahead(),
            STLFormula::True(_) | STLFormula::False(_) | STLFormula::GreaterThan(_, _) | STLFormula::LessThan(_, _) => Duration::ZERO,
        }
    }
}

/// A cache for storing the robustness signal of a subformula.
/// The `last_input_timestamp` is used to check if the cache is stale
/// when new data arrives in the input signal.
#[derive(Debug, Clone)]
pub struct Cache {
    pub robustness_signal: Signal<f64>,
    pub last_input_timestamp: Duration,
}

impl Default for Cache {
    fn default() -> Self {
        Self {
            robustness_signal: Signal::new(),
            last_input_timestamp: Duration::ZERO,
        }
    }
}
