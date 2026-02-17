use ostl::ring_buffer::Step;
use ostl::stl::core::TimeInterval;
use ostl::stl::formula_definition::FormulaDefinition;
use ostl::stl::monitor::{
    Algorithm, DelayedQualitative, DelayedQuantitative, EagerQualitative, Rosi, StlMonitor,
};
use std::env;
use std::time::Duration;

// WARNING: UNUSED FILE

// This file is a simple binary used in combination with heaptrack to monitor memory usage
// of different monitoring strategies and evaluation modes.
// The output is hard to use so there is also a more integrated benchmarking in benches/monitor_benchmark.rs

// ---
// Copy-paste your formula/signal fixtures here
// (Fixtures from `tests/` aren't visible to `benches/`)
// ---
fn formula_1() -> FormulaDefinition {
    // (G[0,2] (x > 0)) U[0,6] (F[0,2] (x > 3))
    FormulaDefinition::Until(
        TimeInterval {
            start: Duration::from_secs(0),
            end: Duration::from_secs(6),
        },
        Box::new(FormulaDefinition::Globally(
            TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(2),
            },
            Box::new(FormulaDefinition::GreaterThan("x", 0.0)),
        )),
        Box::new(FormulaDefinition::Eventually(
            TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(2),
            },
            Box::new(FormulaDefinition::GreaterThan("x", 3.0)),
        )),
    )
}

fn _formula_2() -> FormulaDefinition {
    // A very long formula for memory benchmarking
    // This formula is designed to create a deep and wide operator tree.
    // (G[0,1] (x > 0)) AND (F[0,1] (x < 10)) AND (G[0,1] (x != 5)) AND ... (repeated many times)
    let mut formula = FormulaDefinition::GreaterThan("x", 0.0); // Start with a simple atomic predicate

    for i in 0..50 {
        // Add a Globally operator
        let g_op = FormulaDefinition::Globally(
            TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(1),
            },
            Box::new(FormulaDefinition::GreaterThan("x", i as f64)),
        );
        formula = FormulaDefinition::And(Box::new(formula), Box::new(g_op));

        // Add an Eventually operator
        let f_op = FormulaDefinition::Eventually(
            TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(1),
            },
            Box::new(FormulaDefinition::LessThan("x", (100 - i) as f64)),
        );
        formula = FormulaDefinition::And(Box::new(formula), Box::new(f_op));

        // Add another Globally operator with a different predicate
        let g_op_2 = FormulaDefinition::Globally(
            TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(1),
            },
            Box::new(FormulaDefinition::Not(Box::new(
                FormulaDefinition::GreaterThan("x", (50 + i) as f64),
            ))),
        );
        formula = FormulaDefinition::And(Box::new(formula), Box::new(g_op_2));
    }
    formula
}

// IMPORTANT: Create a *long* signal for meaningful benchmarks
fn get_long_signal(size: usize) -> Vec<Step<f64>> {
    (0..size)
        .map(|i| {
            let t = Duration::from_secs(i as u64);
            let val = (i % 10) as f64; // Simple predictable signal
            Step::new("x", val, t)
        })
        .collect()
}

fn print_usage(program: &str) {
    eprintln!("Usage: {program} [strategy] [evaluation_mode]");
    eprintln!();
    eprintln!("Supported values:");
    eprintln!("  strategy: naive, incremental");
    eprintln!("  evaluation_mode: eager, strict");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  {program}             # defaults to: naive eager");
    eprintln!("  {program} naive eager");
}

fn parse_strategy(s: &str) -> Result<Algorithm, String> {
    if s.eq_ignore_ascii_case("naive") {
        Ok(Algorithm::Naive)
    } else if s.eq_ignore_ascii_case("incremental") {
        Ok(Algorithm::Incremental)
    } else {
        Err(format!("unknown strategy: {s}"))
    }
}

fn parse_semantics(s: &str) -> Result<SemanticChoice, String> {
    if s.eq_ignore_ascii_case("eager") {
        Ok(SemanticChoice::Eager)
    } else if s.eq_ignore_ascii_case("strict") {
        Ok(SemanticChoice::Strict)
    } else if s.eq_ignore_ascii_case("robustness") {
        Ok(SemanticChoice::Robustness)
    } else if s.eq_ignore_ascii_case("rosi") {
        Ok(SemanticChoice::Rosi)
    } else {
        Err(format!("unknown semantics: {s}"))
    }
}

#[derive(Debug)]
enum SemanticChoice {
    Eager,
    Strict,
    Robustness,
    Rosi,
}

fn main() {
    let mut args = env::args();
    let program = args.next().unwrap_or_else(|| "monitor_memory".to_string());

    // Positional args:
    //   1) algorithm (default: naive)
    //   2) semantics (default: strict)
    let algorithm_arg = args.next();
    let semantics_arg = args.next();

    if args.len() < 2 {
        print_usage(&program);
    }

    let algorithm = match algorithm_arg {
        Some(s) => match parse_strategy(&s) {
            Ok(st) => st,
            Err(e) => {
                eprintln!("{e}");
                Algorithm::Naive
            }
        },
        None => Algorithm::Naive,
    };

    let semantics = match semantics_arg {
        Some(s) => match parse_semantics(&s) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{e}");
                SemanticChoice::Strict
            }
        },
        None => SemanticChoice::Strict,
    };

    println!("Using algorithm: {algorithm:?}, semantics: {semantics:?}");

    let formula = formula_1();
    let signal_size = 1000; // Size of the signal for benchmarking
    let signal = get_long_signal(signal_size);

    // Create the monitor using args - dispatch based on semantics
    match semantics {
        SemanticChoice::Strict => {
            let mut monitor = StlMonitor::builder()
                .formula(formula)
                .algorithm(algorithm)
                .semantics(DelayedQualitative)
                .build()
                .unwrap();
            for step in signal {
                let _result = monitor.update(&step);
            }
        }
        SemanticChoice::Eager => {
            let mut monitor = StlMonitor::builder()
                .formula(formula)
                .algorithm(algorithm)
                .semantics(EagerQualitative)
                .build()
                .unwrap();
            for step in signal {
                let _result = monitor.update(&step);
            }
        }
        SemanticChoice::Robustness => {
            let mut monitor = StlMonitor::builder()
                .formula(formula)
                .algorithm(algorithm)
                .semantics(DelayedQuantitative)
                .build()
                .unwrap();
            for step in signal {
                let _result = monitor.update(&step);
            }
        }
        SemanticChoice::Rosi => {
            let mut monitor = StlMonitor::builder()
                .formula(formula)
                .algorithm(algorithm)
                .semantics(Rosi)
                .build()
                .unwrap();
            for step in signal {
                let _result = monitor.update(&step);
            }
        }
    }
}
