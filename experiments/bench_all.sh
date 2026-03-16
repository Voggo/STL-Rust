#!/usr/bin/env sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CRATE_DIR="$PROJECT_ROOT/ostl"
SIGNAL_DIR="$SCRIPT_DIR/BENCH_RESULTS/signal_generation/signals"
OUTPUT_DIR="$SCRIPT_DIR/BENCH_RESULTS/outputs"
ANALYSIS_DIR="$SCRIPT_DIR/data_analysis"

M1_RESULTS="$OUTPUT_DIR/results_M=1.csv" # for measuring cache sizes
M50_RESULTS="$OUTPUT_DIR/results_M=50.csv" # for performance comparison
REGRESSION_OUT="$OUTPUT_DIR/regression_fit_results.csv"
FIGURE_OUT="$OUTPUT_DIR/performance_comparison.pdf"
PY_OSTL_RESULTS="$OUTPUT_DIR/ostlpython_benchmark_results.csv"
RTAMT_RESULTS="$OUTPUT_DIR/rtamt_benchmark_results.csv"

mkdir -p "$SIGNAL_DIR" "$OUTPUT_DIR"

# Generate signal for benchmarks
python "$SCRIPT_DIR/signal_generation/signal_generator.py" --num-samples 20000 --output-path "$SIGNAL_DIR/signal_20000_chirp.csv" --signal-type chirp

# Run Python benchmark scripts in experiments/
python "$SCRIPT_DIR/ostl_python_results.py" \
	--signal-csv "$SIGNAL_DIR/signal_20000_chirp.csv" \
	--m-runs 50 \
	--output "$PY_OSTL_RESULTS" \
	# --overwrite

python "$SCRIPT_DIR/test_rtamt.py" \
	--signal-csv "$SIGNAL_DIR/signal_20000_chirp.csv" \
	--m-runs 50 \
	--output "$RTAMT_RESULTS" \
	# --overwrite

(
	cd "$CRATE_DIR" || exit 1
	PAPER_M_RUNS=1 PAPER_SIGNAL_PATH="$SIGNAL_DIR/signal_20000_chirp.csv" PAPER_OUTPUT_CSV="$M1_RESULTS" cargo bench --bench paper_benchmark --features track-cache-size
	PAPER_M_RUNS=50 PAPER_SIGNAL_PATH="$SIGNAL_DIR/signal_20000_chirp.csv" PAPER_OUTPUT_CSV="$M50_RESULTS" cargo bench --bench paper_benchmark

	python "$ANALYSIS_DIR/regression_analysis.py" \
		--native-csv "$M50_RESULTS" \
		--output "$REGRESSION_OUT"

	python "$ANALYSIS_DIR/performance_comparison.py" \
		--benchmark-csv "$M50_RESULTS" \
		--regression-csv "$REGRESSION_OUT" \
		--output "$FIGURE_OUT"
)