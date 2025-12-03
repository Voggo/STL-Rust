// In benches/monitor_benchmark.rs
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use ostl::ring_buffer::Step;
use ostl::stl;
use ostl::stl::core::{RobustnessInterval, TimeInterval};
use ostl::stl::monitor::{EvaluationMode, FormulaDefinition, MonitoringStrategy, StlMonitor};
use std::time::Duration;


/// Returns the vector of Signal Temporal Logic formulas specified in formula.csv
pub fn get_formulas() -> Vec<FormulaDefinition> {
    let mut formulas = Vec::new();

    // --- Basic Formulas (Lines 1-12) ---
    // Using the stl! macro directly as intended for standard expressions.
    // Note: 'x[t]' in CSV is mapped to signal 'x'.

    formulas.push(stl!((x < 0.5) and (x > -0.5)));

    // 2. Or 
    formulas.push(stl!((x < 0.5) or (x > -0.5)));

    // 3. Not 
    formulas.push(stl!(not (x < 0.5)));

    // 4-6. Globally (Always) 
    formulas.push(stl!(alw[0, 10] (x < 0.5)));
    formulas.push(stl!(alw[0, 100] (x < 0.5)));
    formulas.push(stl!(alw[0, 1000] (x < 0.5)));

    // 7-9. Eventually 
    formulas.push(stl!(ev[0, 10] (x < 0.5)));
    formulas.push(stl!(ev[0, 100] (x < 0.5)));
    formulas.push(stl!(ev[0, 1000] (x < 0.5)));

    // 10-12. Until 
    formulas.push(stl!((x < 0.5) until[0, 10] (x > -0.5)));
    formulas.push(stl!((x < 0.5) until[0, 100] (x > -0.5)));
    formulas.push(stl!((x < 0.5) until[0, 1000] (x > -0.5)));


    // --- Complex Nested Formulas (Lines 13-21) ---
    // These patterns (Depth 10, 20, 30) are too deep for clean single-line macros.
    // We construct them recursively, using stl! for the atomic leaves.

    // helper to create Interval [0, 10]
    let zero_ten = TimeInterval { start: Duration::from_secs(0), end: Duration::from_secs(10) };

    // Pattern A: ((x < 0.5)) and (ev_[0,10] ( ... ))
    // Ends with ((x > 0.0))
    let make_and_ev_chain = |depth: usize| -> FormulaDefinition {
        let mut curr = stl!(x > 0.0); // Base case
        for _ in 0..depth {
            // Replicates: (x < 0.5) and ev[0,10](curr)
            curr = FormulaDefinition::And(
                Box::new(stl!(x < 0.5)),
                Box::new(FormulaDefinition::Eventually(zero_ten.clone(), Box::new(curr)))
            );
        }
        curr
    };

    formulas.push(make_and_ev_chain(10)); 
    formulas.push(make_and_ev_chain(20)); 
    formulas.push(make_and_ev_chain(30)); 

    // Pattern B: ev_[0,10] (alw_[0,10] ( ... ))
    // Alternates Ev/Alw. Ends with ((x > 0.0))
    let make_ev_alw_chain = |depth: usize| -> FormulaDefinition {
        let mut curr = stl!(x > 0.0); // Base case
        // The CSV shows sequences like ev(alw(...)). 
        // We assume 'depth' represents the number of Ev/Alw pairs or layers.
        // Based on the pattern, we add layers inside out.
        for _ in 0..depth {
            // Inner: alw[0,10](curr)
            let alw_layer = FormulaDefinition::Globally(zero_ten.clone(), Box::new(curr));
            // Outer: ev[0,10](alw_layer)
            curr = FormulaDefinition::Eventually(zero_ten.clone(), Box::new(alw_layer));
        }
        curr
    };

    formulas.push(make_ev_alw_chain(10)); 
    formulas.push(make_ev_alw_chain(20)); 
    formulas.push(make_ev_alw_chain(30)); 

    // Pattern C: ((x < 0.5)) until_[0,10] ( ... )
    // Chained Until. Ends with ((x > 0.0))
    let make_until_chain = |depth: usize| -> FormulaDefinition {
        let mut curr = stl!(x > 0.0); // Base case (Right side of last until)
        for _ in 0..depth {
            // Replicates: (x < 0.5) until[0,10] (curr)
            curr = FormulaDefinition::Until(
                zero_ten.clone(),
                Box::new(stl!(x < 0.5)),
                Box::new(curr)
            );
        }
        curr
    };

    formulas.push(make_until_chain(10));
    formulas.push(make_until_chain(20)); 
    formulas.push(make_until_chain(30)); 

    formulas
}

// IMPORTANT: Create a *long* signal for meaningful benchmarks
fn get_long_signal(size: usize, freq: u64) -> Vec<Step<f64>> {
    (0..size)
        .map(|i| {
            let t = Duration::from_secs_f64(i as f64 / freq as f64);
            let val = (i % 10) as f64; // Simple predictable signal
            Step::new("x", val, t)
        })
        .collect()
}

// ---
// The Benchmark Function
// ---
fn benchmark_monitors(c: &mut Criterion) {
    let formulas = get_formulas();
    let signal_size = 1000; // 1,000 steps
    let freq = 1000; // 1 kHz signal
    let signal = get_long_signal(signal_size, freq);

    for formula in formulas[9..] {
        run_performance_benchmark(c, formula, signal_size, &signal);
    }
}

macro_rules! create_benchmarks {
    ($group:expr, $formula:expr, $signal:expr, [$(($name:expr, $output_ty:ty, $strategy:expr, $eval_mode:expr)),*]) => {
        $(
            $group.bench_function($name, |b| {
                b.iter_batched(
                    || {
                        let (f_clone, s_clone) = ($formula.clone(), $signal.clone());
                        let monitor: StlMonitor<f64, $output_ty> = StlMonitor::builder()
                            .formula(f_clone)
                            .strategy($strategy)
                            .evaluation_mode($eval_mode)
                            .build()
                            .unwrap();
                        (monitor, s_clone)
                    },
                    |(mut monitor, signal)| {
                        for step in signal {
                            monitor.update(&step);
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            });
        )*
    };
}

fn run_performance_benchmark(
    c: &mut Criterion,
    formula: FormulaDefinition,
    signal_size: usize,
    signal: &Vec<Step<f64>>,
) {
    // Create a benchmark group to compare implementations
    let temp: StlMonitor<f64, bool> = StlMonitor::builder()
        .formula(formula.clone())
        .build()
        .unwrap();
    let group_name = format!(
        "STL Monitor Performance, Formula: {:?}, Signal Size: {}",
        temp.specification_to_string(),
        signal_size
    );
    let mut group = c.benchmark_group(group_name);
    group.throughput(Throughput::Elements(signal_size as u64));

    // --- Benchmark Naive (f64, Strict) ---
    // To add the naive benchmark back, uncomment the following line in the array
    // ("Naive_f64_Strict", f64, MonitoringStrategy::Naive, EvaluationMode::Strict),
    group.sample_size(10);

    create_benchmarks!(group, &formula, &signal, [
        ("Incremental_bool_Strict", bool, MonitoringStrategy::Incremental, EvaluationMode::Strict),
        ("Incremental_f64_Strict", f64, MonitoringStrategy::Incremental, EvaluationMode::Strict),
        ("Incremental_bool_eager", bool, MonitoringStrategy::Incremental, EvaluationMode::Eager),
        ("Incremental_RobustnessInterval_eager", RobustnessInterval, MonitoringStrategy::Incremental, EvaluationMode::Eager)
    ]);

    group.finish();
}

criterion_group!(benches, benchmark_monitors);
criterion_main!(benches);