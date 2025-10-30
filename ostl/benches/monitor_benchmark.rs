// In benches/monitor_benchmark.rs
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use ostl::ring_buffer::Step;
use ostl::stl::core::TimeInterval;
use ostl::stl::monitor::{EvaluationMode, FormulaDefinition, MonitoringStrategy, StlMonitor};
use std::time::Duration;

// ---
// Copy-paste your formula/signal fixtures here
// (Fixtures from `tests/` aren't visible to `benches/`)
// ---
fn formula_2() -> FormulaDefinition {
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
            Box::new(FormulaDefinition::GreaterThan("x",3.0)),
        )),
    )
}

// IMPORTANT: Create a *long* signal for meaningful benchmarks
fn get_long_signal(size: usize) -> Vec<Step<f64>> {
    (0..size)
        .map(|i| {
            let t = Duration::from_secs(i as u64);
            let val = (i % 10) as f64; // Simple predictable signal
            Step::new("x",val, t)
        })
        .collect()
}

// ---
// The Benchmark Function
// ---
fn benchmark_monitors(c: &mut Criterion) {
    let formula = formula_2();
    let signal_size = 1000; // 1,000 steps
    let signal = get_long_signal(signal_size);

    // Create a benchmark group to compare implementations
    let mut group = c.benchmark_group("Formula 2 (f64, Strict) 1k steps");
    group.throughput(Throughput::Elements(signal_size as u64));

    // --- Benchmark Naive (f64, Strict) ---
    group.bench_function("Naive_f64_Strict", |b| {
        // `iter_batched` is essential for stateful objects.
        // `setup` creates a fresh monitor for *each* iteration.
        // `routine` runs the code to be measured.
        b.iter_batched(
            || {
                // SETUP: Clone the inputs
                let (f_clone, s_clone) = (formula.clone(), signal.clone());
                // SETUP: Build the monitor
                let monitor: StlMonitor<f64, f64> = StlMonitor::builder()
                    .formula(f_clone)
                    .strategy(MonitoringStrategy::Naive)
                    .evaluation_mode(EvaluationMode::Strict)
                    .build()
                    .unwrap();
                (monitor, s_clone)
            },
            |(mut monitor, signal)| {
                // ROUTINE: This is the part being timed
                for step in signal {
                    monitor.instantaneous_robustness(&step);
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // --- Benchmark Incremental (f64, Strict) ---
    group.bench_function("Incremental_f64_Strict", |b| {
        b.iter_batched(
            || {
                let (f_clone, s_clone) = (formula.clone(), signal.clone());
                let monitor: StlMonitor<f64, f64> = StlMonitor::builder()
                    .formula(f_clone)
                    .strategy(MonitoringStrategy::Incremental)
                    .evaluation_mode(EvaluationMode::Strict)
                    .build()
                    .unwrap();
                (monitor, s_clone)
            },
            |(mut monitor, signal)| {
                for step in signal {
                    monitor.instantaneous_robustness(&step);
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // --- Benchmark Eager Incremental (bool, eager) ---
    group.bench_function("Incremental_bool_eager", |b| {
        b.iter_batched(
            || {
                let (f_clone, s_clone) = (formula.clone(), signal.clone());
                let monitor: StlMonitor<f64, bool> = StlMonitor::builder()
                    .formula(f_clone)
                    .strategy(MonitoringStrategy::Incremental)
                    .evaluation_mode(EvaluationMode::Eager)
                    .build()
                    .unwrap();
                (monitor, s_clone)
            },
            |(mut monitor, signal)| {
                for step in signal {
                    monitor.instantaneous_robustness(&step);
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, benchmark_monitors);
criterion_main!(benches);
