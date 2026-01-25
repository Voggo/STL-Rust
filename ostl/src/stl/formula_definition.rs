use crate::stl::core::TimeInterval;

use std::fmt::Display;

/// An enum representing the definition of an STL formula.
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

impl Display for FormulaDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                FormulaDefinition::True => "True".to_string(),
                FormulaDefinition::False => "False".to_string(),
                FormulaDefinition::Not(f) => format!("¬({f})"),
                FormulaDefinition::And(f1, f2) => format!("({f1}) ∧ ({f2})"),
                FormulaDefinition::Or(f1, f2) => format!("({f1}) v ({f2})"),
                FormulaDefinition::Globally(interval, f) => format!(
                    "G[{}, {}]({})",
                    interval.start.as_secs_f64(),
                    interval.end.as_secs_f64(),
                    f
                ),
                FormulaDefinition::Eventually(interval, f) => format!(
                    "F[{}, {}]({})",
                    interval.start.as_secs_f64(),
                    interval.end.as_secs_f64(),
                    f
                ),
                FormulaDefinition::Until(interval, f1, f2) => format!(
                    "({}) U[{}, {}] ({})",
                    f1,
                    interval.start.as_secs_f64(),
                    interval.end.as_secs_f64(),
                    f2
                ),
                FormulaDefinition::Implies(f1, f2) => format!("({f1}) → ({f2})"),
                FormulaDefinition::GreaterThan(s, val) => format!("{s} > {val}"),
                FormulaDefinition::LessThan(s, val) => format!("{s} < {val}"),
            }
        )
    }
}

impl FormulaDefinition {
    pub fn to_tree_string(&self, indent: usize) -> String {
        // Produce a tree-like multi-line representation using characters similar to `tree`:
        // ├──  branch
        // └──  last branch
        // │    vertical continuation
        let padding = " ".repeat(indent);

        fn label(node: &FormulaDefinition) -> String {
            match node {
                FormulaDefinition::True => "True".to_string(),
                FormulaDefinition::False => "False".to_string(),
                FormulaDefinition::Not(_) => "Not".to_string(),
                FormulaDefinition::And(_, _) => "And".to_string(),
                FormulaDefinition::Or(_, _) => "Or".to_string(),
                FormulaDefinition::Globally(interval, _) => format!(
                    "Globally[{},{}]",
                    interval.start.as_secs_f64(),
                    interval.end.as_secs_f64()
                ),
                FormulaDefinition::Eventually(interval, _) => format!(
                    "Eventually[{},{}]",
                    interval.start.as_secs_f64(),
                    interval.end.as_secs_f64()
                ),
                FormulaDefinition::Until(interval, _, _) => format!(
                    "Until[{},{}]",
                    interval.start.as_secs_f64(),
                    interval.end.as_secs_f64()
                ),
                FormulaDefinition::Implies(_, _) => "Implies".to_string(),
                FormulaDefinition::GreaterThan(s, val) => format!("{} > {}", s, val),
                FormulaDefinition::LessThan(s, val) => format!("{} < {}", s, val),
            }
        }

        fn write_node(
            node: &FormulaDefinition,
            prefix: &str,
            is_root: bool,
            is_last: bool,
            out: &mut String,
        ) {
            if is_root {
                out.push_str(&format!("{}{}\n", prefix, label(node)));
            } else {
                let connector = if is_last { "└── " } else { "├── " };
                out.push_str(&format!("{}{}{}\n", prefix, connector, label(node)));
            }

            // prepare prefix for children
            let child_prefix = if is_root {
                // root's children prefix depends on whether root had initial padding
                if prefix.trim().is_empty() {
                    if is_last {
                        format!("{}    ", prefix)
                    } else {
                        format!("{}│   ", prefix)
                    }
                } else {
                    if is_last {
                        format!("{}    ", prefix)
                    } else {
                        format!("{}│   ", prefix)
                    }
                }
            } else {
                if is_last {
                    format!("{}    ", prefix)
                } else {
                    format!("{}│   ", prefix)
                }
            };

            match node {
                FormulaDefinition::Not(child) => {
                    write_node(child, &child_prefix, false, true, out);
                }
                FormulaDefinition::And(l, r)
                | FormulaDefinition::Or(l, r)
                | FormulaDefinition::Implies(l, r)
                | FormulaDefinition::Until(_, l, r) => {
                    write_node(l, &child_prefix, false, false, out);
                    write_node(r, &child_prefix, false, true, out);
                }
                FormulaDefinition::Eventually(_, child) | FormulaDefinition::Globally(_, child) => {
                    write_node(child, &child_prefix, false, true, out);
                }
                // leaves: True, False, GreaterThan, LessThan - nothing to do
                FormulaDefinition::True
                | FormulaDefinition::False
                | FormulaDefinition::GreaterThan(_, _)
                | FormulaDefinition::LessThan(_, _) => {}
            }
        }

        let mut out = String::new();
        write_node(self, &padding, true, true, &mut out);
        // trim trailing newline for tidiness
        out.trim_end().to_string()
    }
}
