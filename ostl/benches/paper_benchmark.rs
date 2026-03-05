use ostl::ring_buffer::Step;
use ostl::stl::formula_definition::FormulaDefinition;
use ostl::stl::monitor::{
	Algorithm, DelayedQualitative, DelayedQuantitative, EagerQualitative, Rosi, StlMonitor,
};
use ostl::stl::parse_stl;
use std::fs::{File, create_dir_all};
use std::hint::black_box;
use std::io::{self, BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Configuration (env-overridable)
// ---------------------------------------------------------------------------
const DEFAULT_M_RUNS: usize = 50;
const DEFAULT_INTERVAL_START: usize = 1;
const DEFAULT_INTERVAL_END: usize = 5_000;
const DEFAULT_INTERVAL_STRIDE: usize = 100;
const DEFAULT_SIGNAL_PATH: &str = "benches/signal_generation/signals/signal_20000.csv";
const DEFAULT_OUTPUT_CSV: &str = "benches/results/paper_native_benchmark_results.csv";

#[derive(Copy, Clone)]
enum SemanticsKind {
	DelayedQuantitative,
	DelayedQualitative,
	EagerQualitative,
	Rosi,
}

impl SemanticsKind {
	fn name(self) -> &'static str {
		match self {
			Self::DelayedQuantitative => "DelayedQuantitative",
			Self::DelayedQualitative => "DelayedQualitative",
			Self::EagerQualitative => "EagerQualitative",
			Self::Rosi => "Rosi",
		}
	}
}

#[derive(Clone)]
struct BenchItem {
	formula_id: usize,
	spec: String,
	benchmark_kind: &'static str,
	interval_len: usize,
	mode: FormulaMode,
}

#[derive(Clone)]
enum FormulaMode {
	Single(FormulaDefinition),
}

struct BenchResult {
	formula_id: usize,
	spec: String,
	semantics: &'static str,
	algorithm: &'static str,
	mode: &'static str,
	n_samples: usize,
	m_runs: usize,
	avg_total_s: f64,
	avg_per_sample_s: f64,
	avg_per_sample_us: f64,
	benchmark_kind: &'static str,
	interval_len: usize,
}

fn env_usize_or_default(key: &str, default: usize) -> usize {
	std::env::var(key)
		.ok()
		.and_then(|v| v.parse::<usize>().ok())
		.filter(|v| *v > 0)
		.unwrap_or(default)
}

fn env_string_or_default(key: &str, default: &str) -> String {
	std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn interval_lengths() -> Vec<usize> {
	let start = env_usize_or_default("PAPER_INTERVAL_START", DEFAULT_INTERVAL_START);
	let end = env_usize_or_default("PAPER_INTERVAL_END", DEFAULT_INTERVAL_END);
	let stride = env_usize_or_default("PAPER_INTERVAL_STRIDE", DEFAULT_INTERVAL_STRIDE);

	let (start, end) = if start <= end {
		(start, end)
	} else {
		(end, start)
	};

	let mut intervals = Vec::new();
	let mut current = start;
	while current <= end {
		intervals.push(current);
		current = current.saturating_add(stride);
		if current == 0 {
			break;
		}
	}

	if intervals.last().copied() != Some(end) {
		intervals.push(end);
	}

	intervals
}

fn read_signal_from_csv<P>(filename: P) -> io::Result<Vec<Step<f64>>>
where
	P: AsRef<Path>,
{
	let file = File::open(filename)?;
	let reader = io::BufReader::new(file);
	let mut signal = Vec::new();

	for line in reader.lines().skip(1) {
		let value_str = line?;
		let columns: Vec<&str> = value_str.split(',').collect();
		if columns.len() == 2
			&& let (Ok(ts), Ok(val)) = (
				columns[0].trim().parse::<f64>(),
				columns[1].trim().parse::<f64>(),
			)
		{
			let t = Duration::from_secs_f64(ts);
			signal.push(Step::new("x", val, t));
		}
	}

	Ok(signal)
}

fn run_single_formula(semantics: SemanticsKind, signal: &[Step<f64>], formula: &FormulaDefinition) {

	match semantics {
		SemanticsKind::DelayedQuantitative => {
			let mut monitor: StlMonitor<f64, f64> = StlMonitor::builder()
				.formula(formula.clone())
				.algorithm(Algorithm::Incremental)
				.semantics(DelayedQuantitative)
				.build()
				.unwrap();

			for step in signal {
				black_box(monitor.update(step));
			}
		}
		SemanticsKind::DelayedQualitative => {
			let mut monitor: StlMonitor<f64, bool> = StlMonitor::builder()
				.formula(formula.clone())
				.algorithm(Algorithm::Incremental)
				.semantics(DelayedQualitative)
				.build()
				.unwrap();

			for step in signal {
				black_box(monitor.update(step));
			}
		}
		SemanticsKind::EagerQualitative => {
			let mut monitor: StlMonitor<f64, bool> = StlMonitor::builder()
				.formula(formula.clone())
				.algorithm(Algorithm::Incremental)
				.semantics(EagerQualitative)
				.build()
				.unwrap();

			for step in signal {
				black_box(monitor.update(step));
			}
		}
		SemanticsKind::Rosi => {
			let mut monitor = StlMonitor::builder()
				.formula(formula.clone())
				.algorithm(Algorithm::Incremental)
				.semantics(Rosi)
				.build()
				.unwrap();

			for step in signal {
				black_box(monitor.update(step));
			}
		}
	}
}

fn bench_item(item: &BenchItem, signal: &[Step<f64>], m_runs: usize, semantics: SemanticsKind) -> BenchResult {
    if (item.interval_len > 1000 && semantics == SemanticsKind::Rosi) {
        println!("Skipping formula_id={} kind={} interval={} spec={} for Rosi semantics (too long interval)",
            item.formula_id, item.benchmark_kind, item.interval_len, item.spec);
        return BenchResult {
            formula_id: item.formula_id,
            spec: item.spec.clone(),
            semantics: semantics.name(),
            algorithm: "Incremental",
            mode: "online",
            n_samples: signal.len(),
            m_runs,
            avg_total_s: f64::NAN,
            avg_per_sample_s: f64::NAN,
            avg_per_sample_us: f64::NAN,
            benchmark_kind: item.benchmark_kind,
            interval_len: item.interval_len,
        };
    }
	let n_samples = signal.len();
	let mut total_time = 0.0;

	for _ in 0..m_runs {
		let t0 = Instant::now();
		match &item.mode {
			FormulaMode::Single(formula) => run_single_formula(semantics, signal, formula),
		}
		total_time += t0.elapsed().as_secs_f64();
	}

	let avg_total = total_time / m_runs as f64;
	let avg_per_sample = avg_total / n_samples as f64;

	BenchResult {
		formula_id: item.formula_id,
		spec: item.spec.clone(),
		semantics: semantics.name(),
		algorithm: "Incremental",
		mode: "online",
		n_samples,
		m_runs,
		avg_total_s: avg_total,
		avg_per_sample_s: avg_per_sample,
		avg_per_sample_us: avg_per_sample * 1e6,
		benchmark_kind: item.benchmark_kind,
		interval_len: item.interval_len,
	}
}

fn build_bench_items(intervals: &[usize]) -> Vec<BenchItem> {
	let mut items = Vec::new();

    for &bnd in intervals {
        let u_spec = format!("(x > 0.5) U[0,{bnd}] (x > 0.8)");
        let u = parse_stl(&u_spec).unwrap_or_else(|e| panic!("Invalid formula '{}': {}", u_spec, e));

        items.push(BenchItem {
            formula_id: 4,
            spec: u_spec,
            benchmark_kind: "until",
            interval_len: bnd,
            mode: FormulaMode::Single(u),
        });
    }

	for &bnd in intervals {
		let g_spec = format!("G[0,{bnd}] (x > 0.5)");
		let g = parse_stl(&g_spec).unwrap_or_else(|e| panic!("Invalid formula '{}': {}", g_spec, e));

		items.push(BenchItem {
			formula_id: 5,
			spec: g_spec,
			benchmark_kind: "globally",
			interval_len: bnd,
			mode: FormulaMode::Single(g),
		});

		let f_spec = format!("F[0,{bnd}] (x > 0.5)");
		let f = parse_stl(&f_spec).unwrap_or_else(|e| panic!("Invalid formula '{}': {}", f_spec, e));

		items.push(BenchItem {
			formula_id: 6,
			spec: f_spec,
			benchmark_kind: "eventually",
			interval_len: bnd,
			mode: FormulaMode::Single(f),
		});
	}

	items
}

fn csv_escape(field: &str) -> String {
	if field.contains(',') || field.contains('"') || field.contains('\n') {
		format!("\"{}\"", field.replace('"', "\"\""))
	} else {
		field.to_string()
	}
}

fn write_csv_header(w: &mut BufWriter<File>) -> io::Result<()> {
	writeln!(
		w,
		"formula_id,spec,semantics,algorithm,mode,n_samples,m_runs,avg_total_s,avg_per_sample_s,avg_per_sample_us,benchmark_kind,interval_len"
	)
}

fn write_result_row(w: &mut BufWriter<File>, r: &BenchResult) -> io::Result<()> {
	writeln!(
		w,
		"{},{},{},{},{},{},{},{:.12},{:.12},{:.6},{},{}",
		r.formula_id,
		csv_escape(&r.spec),
		r.semantics,
		r.algorithm,
		r.mode,
		r.n_samples,
		r.m_runs,
		r.avg_total_s,
		r.avg_per_sample_s,
		r.avg_per_sample_us,
		r.benchmark_kind,
		r.interval_len
	)
}

fn ensure_parent_dir(path: &Path) -> io::Result<()> {
	if let Some(parent) = path.parent()
		&& !parent.as_os_str().is_empty()
	{
		create_dir_all(parent)?;
	}
	Ok(())
}

fn main() -> io::Result<()> {
	let m_runs = env_usize_or_default("PAPER_M_RUNS", DEFAULT_M_RUNS);
	let signal_path = env_string_or_default("PAPER_SIGNAL_PATH", DEFAULT_SIGNAL_PATH);
	let output_path = PathBuf::from(env_string_or_default("PAPER_OUTPUT_CSV", DEFAULT_OUTPUT_CSV));

	let intervals = interval_lengths();
	let items = build_bench_items(&intervals);
	let signal = read_signal_from_csv(&signal_path)?;

	println!("Loaded signal with {} samples from {}", signal.len(), signal_path);
	println!("Interval count: {}", intervals.len());
	println!("Averaging over M = {} runs", m_runs);

	ensure_parent_dir(&output_path)?;
	let file = File::create(&output_path)?;
	let mut writer = BufWriter::new(file);
	write_csv_header(&mut writer)?;

	let semantics = [
		SemanticsKind::DelayedQuantitative,
		SemanticsKind::DelayedQualitative,
		SemanticsKind::EagerQualitative,
		SemanticsKind::Rosi,
	];

	for sem in semantics {
		println!("\n--- Semantics: {} ---", sem.name());
		for item in &items {
			println!(
				"formula_id={} kind={} interval={} spec={}",
				item.formula_id, item.benchmark_kind, item.interval_len, item.spec
			);
			let result = bench_item(item, &signal, m_runs, sem);
            if result.avg_total_s.is_nan() {
                println!("  Skipped (too long interval for Rosi semantics)");
            } else {
                write_result_row(&mut writer, &result)?;
            }
			writer.flush()?;
		}
	}

	println!("\nResults saved to {}", output_path.display());
	Ok(())
}
