// In benches/monitor_benchmark.rs
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
#[cfg(feature = "dhat-heap")]
use dhat;
use ostl::ring_buffer::Step;
use ostl::stl::core::TimeInterval;
use ostl::stl::monitor::{EvaluationMode, FormulaDefinition, MonitoringStrategy, StlMonitor};
use std::time::Duration;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

// ---
// Copy-paste your formula/signal fixtures here
// (Fixtures from `tests/` aren't visible to `benches/`)
// ---
fn _formula_1() -> FormulaDefinition {
    //0x < 30 && x > -30) && (x < 0.5 && x > -0.5) -> F[0, 50]G[0, 20](x<0.5 && x>-0.5))
    FormulaDefinition::Globally(
        TimeInterval {
            start: Duration::from_secs(30),
            end: Duration::from_secs(100),
        },
        Box::new(FormulaDefinition::Implies(
            Box::new(FormulaDefinition::And(
                Box::new(FormulaDefinition::And(
                    Box::new(FormulaDefinition::LessThan("x", 30.0)),
                    Box::new(FormulaDefinition::GreaterThan("x", -30.0)),
                )),
                Box::new(FormulaDefinition::Or(
                    Box::new(FormulaDefinition::GreaterThan("x", 0.5)),
                    Box::new(FormulaDefinition::LessThan("x", -0.5)),
                )),
            )),
            Box::new(FormulaDefinition::Eventually(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(50),
                },
                Box::new(FormulaDefinition::Globally(
                    TimeInterval {
                        start: Duration::from_secs(0),
                        end: Duration::from_secs(20),
                    },
                    Box::new(FormulaDefinition::And(
                        Box::new(FormulaDefinition::LessThan("x", 0.5)),
                        Box::new(FormulaDefinition::GreaterThan("x", -0.5)),
                    )),
                )),
            )),
        )),
    )
}

fn _formula_2() -> FormulaDefinition {
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

fn formula_3() -> FormulaDefinition {
    // (x > 5) /\ (x < 10)
    FormulaDefinition::And(
        Box::new(FormulaDefinition::GreaterThan("x", 5.0)),
        Box::new(FormulaDefinition::LessThan("x", 10.0)),
    )
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

// ---
// The Benchmark Function
// ---
fn benchmark_monitors(c: &mut Criterion) {
    let formula = formula_3();
    let signal_size = 1000; // 1,000 steps
    let signal = get_long_signal(signal_size);

    #[cfg(feature = "dhat-heap")]
    run_memory_profiling(&formula, &signal);

    run_performance_benchmark(c, formula, signal_size, signal);
}

fn run_performance_benchmark(
    c: &mut Criterion,
    formula: FormulaDefinition,
    signal_size: usize,
    signal: Vec<Step<f64>>,
) {
    // Create a benchmark group to compare implementations
    let temp:StlMonitor<f64, bool> = StlMonitor::builder().formula(formula.clone()).build().unwrap();
    let group_name = format!("STL Monitor Performance, Formula: {:?}, Signal Size: {}", temp.specification_to_string(), signal_size);
    let mut group = c.benchmark_group(group_name);
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
                    monitor.update(&step);
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
    // --- Benchmark Incremental (bool, Strict) ---
    group.bench_function("Incremental_bool_Strict", |b| {
        b.iter_batched(
            || {
                let (f_clone, s_clone) = (formula.clone(), signal.clone());
                let monitor: StlMonitor<f64, bool> = StlMonitor::builder()
                    .formula(f_clone)
                    .strategy(MonitoringStrategy::Incremental)
                    .evaluation_mode(EvaluationMode::Strict)
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
                    monitor.update(&step);
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
                    monitor.update(&step);
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
    // --- Benchmark Eager Incremental (f64, eager) ---
    // group.bench_function("Incremental_RobustnessInterval_eager", |b| {
    //     b.iter_batched(
    //         || {
    //             let (f_clone, s_clone) = (formula.clone(), signal.clone());
    //             let monitor: StlMonitor<f64, RobustnessInterval> = StlMonitor::builder()
    //                 .formula(f_clone)
    //                 .strategy(MonitoringStrategy::Incremental)
    //                 .evaluation_mode(EvaluationMode::Eager)
    //                 .build()
    //                 .unwrap();
    //             (monitor, s_clone)
    //         },
    //         |(mut monitor, signal)| {
    //             for step in signal {
    //                 monitor.update(&step);
    //             }
    //         },
    //         criterion::BatchSize::SmallInput,
    //     );
    // });
    group.finish();
}

#[cfg(feature = "dhat-heap")]
fn run_memory_profiling(formula: &FormulaDefinition, signal: &Vec<Step<f64>>) {
    // Start memory profiling
    let _profiler = dhat::Profiler::builder().testing().build();
    // SETUP: Build the monitor for naive and strict evaluation
    let mut monitor: StlMonitor<f64, f64> = StlMonitor::builder()
        .formula(formula.clone())
        .strategy(MonitoringStrategy::Naive)
        .evaluation_mode(EvaluationMode::Strict)
        .build()
        .unwrap();
    for step in signal {
        monitor.update(step);
    }
    print_heap_stats("Naive Strict f64");
    drop(_profiler); // Stop profiling
    let _profiler = dhat::Profiler::builder().testing().build();

    // SETUP: Build the monitor for incremental and strict evaluation
    let mut monitor: StlMonitor<f64, f64> = StlMonitor::builder()
        .formula(formula.clone())
        .strategy(MonitoringStrategy::Incremental)
        .evaluation_mode(EvaluationMode::Strict)
        .build()
        .unwrap();
    for step in signal {
        monitor.update(step);
    }
    print_heap_stats("Incremental Strict f64");
    drop(_profiler); // Stop profiling
    let _profiler = dhat::Profiler::builder().testing().build();
    // SETUP: Build the monitor for incremental and eager evaluation for bool
    let mut monitor: StlMonitor<f64, bool> = StlMonitor::builder()
        .formula(formula.clone())
        .strategy(MonitoringStrategy::Incremental)
        .evaluation_mode(EvaluationMode::Eager)
        .build()
        .unwrap();
    for step in signal {
        monitor.update(step);
    }
    print_heap_stats("Incremental Eager bool");
}

#[cfg(feature = "dhat-heap")]
fn print_heap_stats(test_name: &'static str) {
    // use serde_json to write dhat profile to file
    let heap_stats = dhat::HeapStats::get();
    let measures = serde_json::json!({
        "Final Blocks": {
            "value": heap_stats.curr_blocks,
        },
        "Final Bytes": {
            "value": heap_stats.curr_bytes,
        },
        "Max Blocks": {
            "value": heap_stats.max_blocks,
        },
        "Max Bytes": {
            "value": heap_stats.max_bytes,
        },
        "Total Blocks": {
            "value": heap_stats.total_blocks,
        },
        "Total Bytes": {
            "value": heap_stats.total_bytes,
        },
    });
    let pretty_measures =
        serde_json::to_string_pretty(&measures).expect("serialize heap statistics");
    println!("Dhat heap profile for {}:\n{}", test_name, pretty_measures);
}

criterion_group!(benches, benchmark_monitors);
criterion_main!(benches);
