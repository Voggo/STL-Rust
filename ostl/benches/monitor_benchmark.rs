// In benches/monitor_benchmark.rs
use criterion::{Criterion, Throughput, criterion_group, criterion_main, PlotConfiguration, AxisScale};
use ostl::ring_buffer::Step;
use ostl::stl::core::RobustnessInterval;
use ostl::stl::monitor::{EvaluationMode, MonitoringStrategy, StlMonitor};
use std::time::Duration;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

// in order to get a json file with all the needed info do:
// Install it: cargo install cargo-criterion
// Run: cargo criterion --message-format=json > results.json

// Helper to read a single signal CSV into a Vec<Step<f64>>
fn read_signal_from_csv<P>(filename: P) -> io::Result<Vec<Step<f64>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename).expect("failed to read file: {filename}");
    let reader = io::BufReader::new(file);
    let mut signal = Vec::new();
    // we skip the first line which is the header
    for (i, line) in reader.lines().skip(1).enumerate() {
        if let Ok(value_str) = line {
            // we split the line by the comma and take the second element
            let columns: Vec<&str> = value_str.split(',').collect();
            if columns.len() == 2 {
                if let Ok(val) = columns[1].trim().parse::<f64>() {
                    let t = Duration::from_secs_f64(i as f64);
                    signal.push(Step::new("x", val, t));
                }
            }
        }
    }
    Ok(signal)
}

/// Reads signals from the specified CSV files.
pub fn get_signals_from_csv() -> Vec<Vec<Step<f64>>> {
    let filenames = [
        "benches/signal_5000.csv",
        "benches/signal_10000.csv",
        "benches/signal_20000.csv",
    ];
    let mut signals = Vec::new();

    for filename in &filenames {
        match read_signal_from_csv(filename) {
            Ok(signal) => signals.push(signal),
            Err(e) => panic!("Failed to read signal from {}: {}", filename, e),
        }
    }

    signals
}

// ---
// The Benchmark Function
// ---
fn benchmark_monitors(c: &mut Criterion) {
    let formulas = ostl::stl::formulas::get_formulas(&[]);
    let signals = get_signals_from_csv();

    for (id, formula) in formulas {
        // Create a temp monitor to get the specification string for the group name
        let temp: StlMonitor<f64, bool> = StlMonitor::builder()
            .formula(formula.clone())
            .build()
            .unwrap();
        let spec_str = temp.specification_to_string();
        let formula_name = format!("{}: {}", id, spec_str);
        

        let mut group = c.benchmark_group(&formula_name);
        group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
        
        // Increase sample size for better statistical significance
        group.sample_size(50);
        // Increase measurement time
        group.measurement_time(Duration::from_secs(3));

        for signal in &signals {
            let signal_size = signal.len();
            group.throughput(Throughput::Elements(signal_size as u64));

            // 1. Incremental bool Strict
            group.bench_with_input(
                criterion::BenchmarkId::new("Incremental_bool_Strict", signal_size),
                signal,
                |b, signal| {
                    b.iter_batched(
                        || {
                            let monitor: StlMonitor<f64, bool> = StlMonitor::builder()
                                .formula(formula.clone())
                                .strategy(MonitoringStrategy::Incremental)
                                .evaluation_mode(EvaluationMode::Strict)
                                .build()
                                .unwrap();
                            (monitor, signal.clone())
                        },
                        |(mut monitor, signal)| {
                            for step in signal {
                                monitor.update(&step);
                            }
                        },
                        // Use LargeInput to exclude the significant drop/clone time of the signal
                        criterion::BatchSize::LargeInput,
                    );
                },
            );

            // 2. Incremental f64 Strict
            group.bench_with_input(
                criterion::BenchmarkId::new("Incremental_f64_Strict", signal_size),
                signal,
                |b, signal| {
                    b.iter_batched(
                        || {
                             let monitor: StlMonitor<f64, f64> = StlMonitor::builder()
                                .formula(formula.clone())
                                .strategy(MonitoringStrategy::Incremental)
                                .evaluation_mode(EvaluationMode::Strict)
                                .build()
                                .unwrap();
                            (monitor, signal.clone())
                        },
                        |(mut monitor, signal)| {
                            for step in signal {
                                monitor.update(&step);
                            }
                        },
                        criterion::BatchSize::LargeInput,
                    );
                },
            );

            // If this formula is among the first 12, also run Naive strategy (strict) benchmarks
            if id <= 11 {
                // Naive bool Strict
                group.bench_with_input(
                    criterion::BenchmarkId::new("Naive_bool_Strict", signal_size),
                    signal,
                    |b, signal| {
                        b.iter_batched(
                            || {
                                let monitor: StlMonitor<f64, bool> = StlMonitor::builder()
                                    .formula(formula.clone())
                                    .strategy(MonitoringStrategy::Naive)
                                    .evaluation_mode(EvaluationMode::Strict)
                                    .build()
                                    .unwrap();
                                (monitor, signal.clone())
                            },
                            |(mut monitor, signal)| {
                                for step in signal {
                                    monitor.update(&step);
                                }
                            },
                            criterion::BatchSize::LargeInput,
                        );
                    },
                );

                // Naive f64 Strict
                group.bench_with_input(
                    criterion::BenchmarkId::new("Naive_f64_Strict", signal_size),
                    signal,
                    |b, signal| {
                        b.iter_batched(
                            || {
                                let monitor: StlMonitor<f64, f64> = StlMonitor::builder()
                                    .formula(formula.clone())
                                    .strategy(MonitoringStrategy::Naive)
                                    .evaluation_mode(EvaluationMode::Strict)
                                    .build()
                                    .unwrap();
                                (monitor, signal.clone())
                            },
                            |(mut monitor, signal)| {
                                for step in signal {
                                    monitor.update(&step);
                                }
                            },
                            criterion::BatchSize::LargeInput,
                        );
                    },
                );
            }

             // 3. Incremental bool Eager
            group.bench_with_input(
                criterion::BenchmarkId::new("Incremental_bool_Eager", signal_size),
                signal,
                |b, signal| {
                    b.iter_batched(
                        || {
                            let monitor: StlMonitor<f64, bool> = StlMonitor::builder()
                                .formula(formula.clone())
                                .strategy(MonitoringStrategy::Incremental)
                                .evaluation_mode(EvaluationMode::Eager)
                                .build()
                                .unwrap();
                            (monitor, signal.clone())
                        },
                        |(mut monitor, signal)| {
                            for step in signal {
                                monitor.update(&step);
                            }
                        },
                        criterion::BatchSize::LargeInput,
                    );
                },
            );

            // 4. Incremental RobustnessInterval Eager
            group.bench_with_input(
                criterion::BenchmarkId::new("Incremental_RoSI_Eager", signal_size),
                signal,
                |b, signal| {
                    b.iter_batched(
                        || {
                             let monitor: StlMonitor<f64, RobustnessInterval> = StlMonitor::builder()
                                .formula(formula.clone())
                                .strategy(MonitoringStrategy::Incremental)
                                .evaluation_mode(EvaluationMode::Eager)
                                .build()
                                .unwrap();
                            (monitor, signal.clone())
                        },
                        |(mut monitor, signal)| {
                            for step in signal {
                                monitor.update(&step);
                            }
                        },
                        criterion::BatchSize::LargeInput,
                    );
                },
            );
        }
        group.finish();
    }
}

criterion_group!(benches, benchmark_monitors);
criterion_main!(benches);