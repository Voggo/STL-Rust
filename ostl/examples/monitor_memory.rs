use ostl::ring_buffer::Step;
use ostl::stl::core::TimeInterval;
use ostl::stl::monitor::{EvaluationMode, FormulaDefinition, MonitoringStrategy, StlMonitor};
use std::time::Duration;
use std::env;

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

fn parse_strategy(s: &str) -> Result<MonitoringStrategy, String> {
    if s.eq_ignore_ascii_case("naive") {
        Ok(MonitoringStrategy::Naive)
    } else if s.eq_ignore_ascii_case("incremental") {
        Ok(MonitoringStrategy::Incremental)
    } else {
        Err(format!("unknown strategy: {s}"))
    }
}

fn parse_eval_mode(s: &str) -> Result<EvaluationMode, String> {
    if s.eq_ignore_ascii_case("eager") {
        Ok(EvaluationMode::Eager)
    } else if s.eq_ignore_ascii_case("strict") {
        Ok(EvaluationMode::Strict)
    } else {
        Err(format!("unknown evaluation mode: {s}"))
    }
}

fn main() {
    let mut args = env::args();
    let program = args.next().unwrap_or_else(|| "monitor_memory".to_string());

    // Positional args:
    //   1) strategy (default: naive)
    //   2) evaluation_mode (default: eager)
    let strategy_arg = args.next();
    let eval_mode_arg = args.next();

    if args.len() < 2 {
        print_usage(&program);
    }

    let strategy = match strategy_arg {
        Some(s) => match parse_strategy(&s) {
            Ok(st) => st,
            Err(e) => {
                eprintln!("{e}");
                MonitoringStrategy::Naive
            }
        },
        None => MonitoringStrategy::Naive,
    };

    let evaluation_mode = match eval_mode_arg {
        Some(s) => match parse_eval_mode(&s) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{e}");
                EvaluationMode::Strict
            }
        },
        None => EvaluationMode::Strict,
    };

    println!(
        "Using strategy: {strategy:?}, evaluation_mode: {evaluation_mode:?}"
    );

    let formula = formula_1();
    let signal_size = 1000; // Size of the signal for benchmarking
    let signal = get_long_signal(signal_size);

    // Create the monitor using args
    let mut monitor: StlMonitor<f64, bool> = StlMonitor::builder()
        .formula(formula)
        .strategy(strategy)
        .evaluation_mode(evaluation_mode)
        .build()
        .unwrap();

    // Run the monitor on the signal
    for step in signal {
        let _result = monitor.update(&step);
        // For benchmarking purposes, we ignore the result
    }
}
