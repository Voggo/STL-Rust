use ostl::ring_buffer::Step;
use ostl::stl::core::RobustnessInterval;
use ostl::stl::formula_definition::FormulaDefinition;
use ostl::stl::monitor::{EvaluationMode, MonitoringStrategy, StlMonitor};
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::Duration;

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
pub fn get_signals_from_csv() -> Vec<(usize, Vec<Step<f64>>)> {
    let filenames = [
        ("benches/signal_5000.csv", 5000),
        // ("benches/signal_10000.csv", 10000),
        // ("benches/signal_20000.csv", 20000),
    ];
    let mut signals = Vec::new();

    for (filename, size) in &filenames {
        match read_signal_from_csv(filename) {
            Ok(signal) => signals.push((*size, signal)),
            Err(e) => eprintln!("Failed to read signal from {}: {}", filename, e),
        }
    }

    signals
}

/// Format RobustnessInterval as a string suitable for CSV output
fn format_robustness_interval(interval: RobustnessInterval) -> String {
    format!("[{},{}]", interval.0, interval.1)
}

/// Run evaluation for a single formula on a signal and save to CSV
fn evaluate_formula(
    formula_id: usize,
    formula: &FormulaDefinition,
    signal: &[Step<f64>],
    signal_size: usize,
    output_dir: &str,
) -> io::Result<()> {
    // Prepare output directories
    let output_path = Path::new(output_dir);
    let rosi_dir = output_path.join("rosi");
    let strict_dir = output_path.join("strict");
    std::fs::create_dir_all(&rosi_dir)?;
    std::fs::create_dir_all(&strict_dir)?;

    println!(
        "Evaluating formula{}:\n {}\n with signal size {}...",
        formula_id,
        formula.to_string(),
        signal_size
    );

    // Build monitors
    let mut monitor_rosi: StlMonitor<f64, RobustnessInterval> = StlMonitor::builder()
        .formula(formula.clone())
        .strategy(MonitoringStrategy::Incremental)
        .evaluation_mode(EvaluationMode::Eager)
        .build()
        .expect("Failed to build monitor");

    let mut monitor_strict: StlMonitor<f64, f64> = StlMonitor::builder()
        .formula(formula.clone())
        .strategy(MonitoringStrategy::Incremental)
        .evaluation_mode(EvaluationMode::Strict)
        .build()
        .expect("Failed to build monitor");

    let filename_rosi = rosi_dir.join(format!(
        "eval_approach_F{}_sizeN={}_appr=RoSI.csv",
        formula_id, signal_size
    ));
    let filename_strict = strict_dir.join(format!(
        "eval_approach_F{}_sizeN={}_appr=Strict.csv",
        formula_id, signal_size
    ));

    let mut file_rosi = File::create(&filename_rosi)?;
    let mut file_strict = File::create(&filename_strict)?;

    writeln!(file_rosi, "input_time_step,output_times,output_values")?;
    writeln!(file_strict, "input_time_step,output_times,output_values")?;

    // Helper to format outputs generically
    fn format_output<T, F>(output: &[Step<T>], value_fmt: F) -> (String, String)
    where
        F: Fn(&Step<T>) -> Option<String>,
    {
        let mut times = Vec::new();
        let mut values = Vec::new();

        for s in output.iter().filter(|s| s.signal == "output") {
            times.push(s.timestamp.as_secs().to_string());
            if let Some(v) = value_fmt(s) {
                values.push(v);
            }
        }

        let times_str = if times.is_empty() {
            "[]".to_string()
        } else {
            format!("[{}]", times.join(","))
        };

        let values_str = if values.is_empty() {
            "[]".to_string()
        } else {
            format!("[{}]", values.join(","))
        };

        (times_str, values_str)
    }

    // Iterate through signal steps and write CSV rows
    for step in signal {
        let output_rosi = monitor_rosi.update(step);
        let output_strict = monitor_strict.update(step);

        let (times_rosi, values_rosi) = format_output(&output_rosi.all_outputs(), |s| {
            s.value.map(|iv| format_robustness_interval(iv))
        });

        let (times_strict, values_strict) = format_output(&output_strict.all_outputs(), |s| {
            s.value.map(|v| format!("{}", v))
        });

        let input_time_secs = step.timestamp.as_secs();
        writeln!(
            file_rosi,
            "{},\"{}\",\"{}\"",
            input_time_secs, times_rosi, values_rosi
        )?;
        writeln!(
            file_strict,
            "{},\"{}\",\"{}\"",
            input_time_secs, times_strict, values_strict
        )?;
    }

    println!(
        "Saved evaluation for formula {} with signal size {} to {}",
        formula_id,
        signal_size,
        filename_rosi.display()
    );

    Ok(())
}

fn main() -> io::Result<()> {
    let formulas = ostl::stl::formulas::get_formulas(&[1]);
    let signals = get_signals_from_csv();

    let output_dir = "examples/sandbox";

    for (formula_id, formula) in formulas {
        for (signal_size, signal) in &signals {
            evaluate_formula(formula_id, &formula, signal, *signal_size, output_dir)?;
        }
    }

    println!("Evaluation completed!");
    Ok(())
}
