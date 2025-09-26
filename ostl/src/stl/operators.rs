use std::time::Duration;

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
    Not(Box<STLFormula>),
    And(Box<STLFormula>, Box<STLFormula>),
    Or(Box<STLFormula>, Box<STLFormula>),

    // Temporal operators
    Always(TimeInterval, Box<STLFormula>),
    Eventually(TimeInterval, Box<STLFormula>),
    Until(TimeInterval, Box<STLFormula>, Box<STLFormula>),

    // logical operators
    Implies(Box<STLFormula>, Box<STLFormula>),

    // Atomic propositions
    True,
    False,
    GreaterThan(f64),
    LessThan(f64),
}

impl STLFormula {
    /// Recursively generates a pretty-printed string representation of the formula.
    pub fn to_string(&self) -> String {
        match self {
            STLFormula::True => "True".to_string(),
            STLFormula::False => "False".to_string(),
            STLFormula::Not(f) => format!("¬({})", f.to_string()),
            STLFormula::And(f1, f2) => format!("({}) /\\ ({})", f1.to_string(), f2.to_string()),
            STLFormula::Or(f1, f2) => format!("({}) /\\ ({})", f1.to_string(), f2.to_string()),
            STLFormula::Always(interval, f) => format!(
                "G[{}, {}]({})",
                interval.start.as_secs_f64(),
                interval.end.as_secs_f64(),
                f.to_string()
            ),
            STLFormula::Eventually(interval, f) => format!(
                "F[{}, {}]({})",
                interval.start.as_secs_f64(),
                interval.end.as_secs_f64(),
                f.to_string()
            ),
            STLFormula::Until(interval, f1, f2) => format!(
                "({}) U[{}, {}] ({})",
                f1.to_string(),
                interval.start.as_secs_f64(),
                interval.end.as_secs_f64(),
                f2.to_string()
            ),
            STLFormula::Implies(f1, f2) => format!("({}) -> ({})", f1.to_string(), f2.to_string()),
            STLFormula::GreaterThan(val) => format!("x > {}", val),
            STLFormula::LessThan(val) => format!("x < {}", val),
        }
    }

    /// Recursively generate a tree-like string representation of the formula.
    pub fn to_tree_string(&self, indent: usize) -> String {
        let padding = " ".repeat(indent);
        match self {
            STLFormula::True => format!("{}True", padding),
            STLFormula::False => format!("{}False", padding),
            STLFormula::Not(f) => format!(
                "{}Not\n{}",
                padding,
                f.to_tree_string(indent + 2)
            ),
            STLFormula::And(f1, f2) => format!(
                "{}And\n{}\n{}",
                padding,
                f1.to_tree_string(indent + 2),
                f2.to_tree_string(indent + 2)
            ),
            STLFormula::Or(f1, f2) => format!(
                "{}Or\n{}\n{}",
                padding,
                f1.to_tree_string(indent + 2),
                f2.to_tree_string(indent + 2)
            ),
            STLFormula::Always(interval, f) => format!(
                "{}Always [{} - {}]\n{}",
                padding,
                interval.start.as_secs_f64(),
                interval.end.as_secs_f64(),
                f.to_tree_string(indent + 2)
            ),
            STLFormula::Eventually(interval, f) => format!(
                "{}Eventually [{} - {}]\n{}",
                padding,
                interval.start.as_secs_f64(),
                interval.end.as_secs_f64(),
                f.to_tree_string(indent + 2)
            ),
            STLFormula::Until(interval, f1, f2) => format!(
                "{}Until [{} - {}]\n{}\n{}",
                padding,
                interval.start.as_secs_f64(),
                interval.end.as_secs_f64(),
                f1.to_tree_string(indent + 2),
                f2.to_tree_string(indent + 2)
            ),
            STLFormula::Implies(f1, f2) => format!(
                "{}Implies\n{}\n{}",
                padding,
                f1.to_tree_string(indent + 2),
                f2.to_tree_string(indent + 2)
            ),
            STLFormula::GreaterThan(val) => format!("{}x > {}", padding, val),
            STLFormula::LessThan(val) => format!("{}x < {}", padding, val),
        }
    }
    

    /// Recursively computes the maximum lookahead time required for the formula.
    pub fn get_max_lookahead(&self) -> Duration {
        match self {
            STLFormula::Always(interval, f)
            | STLFormula::Eventually(interval, f)
            | STLFormula::Until(interval, f, _) => {
                interval.end.max(f.get_max_lookahead())
            }
            STLFormula::Not(f) => f.get_max_lookahead(),
            STLFormula::And(f1, f2)
            | STLFormula::Or(f1, f2)
            | STLFormula::Implies(f1, f2) => {
                f1.get_max_lookahead().max(f2.get_max_lookahead())
            }
            STLFormula::True | STLFormula::False | STLFormula::GreaterThan(_) | STLFormula::LessThan(_) => Duration::ZERO,
        }
    }
}
