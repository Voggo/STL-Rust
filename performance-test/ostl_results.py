import ostl_python.ostl_python as ostl
import csv
import time
import os
import sys
import json
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
M = 1  # Number of runs to average over
SIGNAL_CSV = os.path.join(
    os.path.dirname(__file__),
    "..",
    "ostl",
    "benches",
    "signal_generation",
    "signals",
    "signal_20000.csv",
)

# Formulas matching the ostl benchmark catalog (IDs 1-12)
# Uses the same DSL syntax as Rust's stl! macro (parsed via parse_formula)
FORMULAS: dict[int, str] = {}
b = np.arange(0, 5001, 100)
b[0] += 1  # avoid zero bound for the first formula

formula_id = 1

# globally formulas
for bnd in b:
    FORMULAS[formula_id] = f"G[0,{bnd:.1f}] (x > 0.0)"
    formula_id += 1

# eventually formulas
for bnd in b:
    FORMULAS[formula_id] = f"F[0,{bnd:.1f}] (x > 0.0)"
    formula_id += 1

# with until, only 10 formulas before it gets too slow
for bnd in b:
    FORMULAS[formula_id] = f"(x < 0.0) U[0,{bnd:.1f}] (x > 0.0)"
    formula_id += 1

# phi1
FORMULAS[formula_id] = "(x < 0.5) and (x > -0.5)"

# phi2
FORMULAS[formula_id + 1] = "G[0,1000] (x > 0.5 -> F[0,100] (x < 0.0))"

# phi3
FORMULAS[formula_id + 2] = "(G[0,100] (x < 0.5)) or (G[100,150] (x > 0.0))"


def load_signal(path: str) -> list[tuple[float, float]]:
    """Return list of (timestep, value) from a CSV file."""
    signal: list[tuple[float, float]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            signal.append((float(row["timestep"]), float(row["value"])))
    return signal


def bench_formula(
    formula_id: int,
    spec: str,
    signal: list[tuple[float, float]],
    m: int,
    semantics: str = "DelayedQuantitative",
    algorithm: str = "Incremental",
) -> dict:
    """Run *m* passes of the formula and return per-sample timing stats.

    Each run creates a fresh monitor, then feeds the signal one sample at
    a time via monitor.update() — true online monitoring.
    """
    n_samples = len(signal)
    total_time = 0.0
    parsed_formula = ostl.parse_formula(spec)
    temporal_depth = 0

    for _ in range(m):
        monitor = ostl.Monitor(
            parsed_formula,
            semantics=semantics,
            algorithm=algorithm,
            synchronization="None",
        )
        t0 = time.perf_counter()
        for ts, val in signal:
            monitor.update("x", val, ts)
        t1 = time.perf_counter()
        total_time += t1 - t0
        temporal_depth = monitor.get_temporal_depth()

    avg_total = total_time / m
    avg_per_sample = avg_total / n_samples

    return {
        "formula_id": formula_id,
        "spec": spec,
        "semantics": semantics,
        "algorithm": algorithm,
        "mode": "online",
        "n_samples": n_samples,
        "m_runs": m,
        "avg_total_s": avg_total,
        "avg_per_sample_s": avg_per_sample,
        "avg_per_sample_us": avg_per_sample * 1e6,
        "temporal_depth": temporal_depth,
    }


def should_process_formula(spec: str, semantics: str) -> bool:
    """Check if formula should be processed for given semantics.

    For RoSI semantics, only process formulas with temporal bounds <= 1000.
    """
    if semantics != "Rosi":
        return True

    # Extract temporal bound from spec (e.g., "G[0,500]" -> 500)
    import re

    match = re.search(r"\[0,(\d+(?:\.\d+)?)\]", spec)
    if match:
        bound = float(match.group(1))
        return bound <= 1000

    # If no temporal operator with bound, include the formula
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    signal = load_signal(SIGNAL_CSV)
    print(f"Loaded signal with {len(signal)} samples from {SIGNAL_CSV}")
    print(f"Averaging over M = {M} runs\n")

    semantics = [
        # "DelayedQuantitative",
        # "DelayedQualitative",
        # "EagerQualitative",
        "Rosi",
    ]
    algorithm = "Incremental"

    # Initialize CSV file with headers
    out_path = os.path.join(os.path.dirname(__file__), "ostl_benchmark_results.csv")

    # Check if file exists and ask user for confirmation
    if os.path.exists(out_path):
        print(f"Warning: File '{out_path}' already exists.")
        response = input("Do you want to overwrite it? (yes/no): ").strip().lower()
        if response not in ["yes", "y"]:
            out_path = input("Enter a new filename (or path): ").strip()
            if not out_path.endswith(".csv"):
                out_path += ".csv"
            print(f"Using new filename: {out_path}")

    csv_file = open(out_path, "w", newline="")
    fieldnames = [
        "formula_id",
        "spec",
        "semantics",
        "algorithm",
        "mode",
        "n_samples",
        "m_runs",
        "avg_total_s",
        "avg_per_sample_s",
        "avg_per_sample_us",
        "temporal_depth",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for sem in semantics:
        print(f"\n--- Semantics: {sem} ---")
        pbar = tqdm(FORMULAS.items(), desc=f"Processing {sem}")
        for fid, spec in pbar:
            if not should_process_formula(spec, sem):
                continue
            pbar.set_postfix({"fid": fid, "spec": spec[:30] + "..."})
            res = bench_formula(
                fid, spec, signal, M, semantics=sem, algorithm=algorithm
            )
            # Write result to CSV file
            writer.writerow(res)
            csv_file.flush()

    csv_file.close()
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
