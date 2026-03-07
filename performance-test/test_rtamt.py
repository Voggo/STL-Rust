"""
Comprehensive benchmark suite for rtamt monitoring.

Tests formula families with varying temporal bounds and records performance metrics,
matching the structure and approach of ostl_results.py.
"""

import rtamt
import csv
import time
import os
import json
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
M = 50  # Number of runs to average over
SIGNAL_CSV = os.path.join(
    os.path.dirname(__file__),
    "..",
    "ostl",
    "benches",
    "signal_generation",
    "signals",
    "signal_20000.csv",
)

# Generate formulas matching the ostl benchmark catalog
# Uses the same structure: globally, eventually, until with varying bounds
FORMULAS: dict[int, str] = {}
b = np.arange(0, 5001, 100)
b[0] += 1  # avoid zero bound for the first formula

formula_id = 1

# # Globally formulas with varying bounds
# for bnd in b:
#     FORMULAS[formula_id] = f"always[0,{bnd:.1f}] (x < 0.5)"
#     formula_id += 1

# # Eventually formulas with varying bounds
# for bnd in b:
#     FORMULAS[formula_id] = f"eventually[0,{bnd:.1f}] (x < 0.5)"
#     formula_id += 1

# # Until formulas with varying bounds
# for bnd in b:
#     FORMULAS[formula_id] = f"(x < 0.5) until[0,{bnd:.1f}] (x > -0.5)"
#     formula_id += 1

# Simple formulas (no temporal operators)
FORMULAS[formula_id] = "(x < 0.5) and (x > -0.5)"
FORMULAS[formula_id + 1] = "G[0,1000] (x > 0.5 -> F[0,100] (x < 0.0))"
FORMULAS[formula_id + 2] = "(G[0,100] (x < 0.5)) or (G[100,150] (x > 0.0))"


def load_signal(path: str) -> list[tuple[float, float]]:
    """Return list of (timestep, value) from a CSV file."""
    signal: list[tuple[float, float]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            signal.append((float(row["timestep"]), float(row["value"])))
    return signal


# ---------------------------------------------------------------------------
# Dense-Time Online Monitoring
# ---------------------------------------------------------------------------

def make_dense_monitor(spec: str) -> rtamt.StlDenseTimeSpecification:
    """Create a fresh rtamt dense-time online monitor."""
    monitor = rtamt.StlDenseTimeSpecification()
    monitor.declare_var("x", "float")
    monitor.spec = spec
    monitor.parse()
    monitor.pastify()
    return monitor


def bench_dense_online(
    formula_id: int,
    spec: str,
    signal: list[tuple[float, float]],
    m: int,
) -> dict:
    """Benchmark dense-time online monitoring (one sample at a time)."""
    try:
        n_samples = len(signal)
        total_time = 0.0

        for _ in range(m):
            monitor = make_dense_monitor(spec)
            t0 = time.perf_counter()
            for ts, val in signal:
                monitor.update(["x", [[ts, val]]])
            t1 = time.perf_counter()
            total_time += t1 - t0

        avg_total = total_time / m
        avg_per_sample = avg_total / n_samples

        return {
            "formula_id": formula_id,
            "spec": spec,
            "monitor_type": "dense-time",
            "mode": "online",
            "n_samples": n_samples,
            "m_runs": m,
            "avg_total_s": avg_total,
            "avg_per_sample_s": avg_per_sample,
            "avg_per_sample_us": avg_per_sample * 1e6,
        }
    except Exception as e:
        print(f"ERROR: Formula {formula_id} (dense-time online) failed: {str(e)[:100]}")
        return None


# ---------------------------------------------------------------------------
# Dense-Time Offline Monitoring
# ---------------------------------------------------------------------------

def make_dense_monitor_offline(spec: str) -> rtamt.StlDenseTimeSpecification:
    """Create a fresh rtamt dense-time monitor for offline evaluation."""
    monitor = rtamt.StlDenseTimeSpecification()
    monitor.declare_var("x", "float")
    monitor.spec = spec
    monitor.parse()
    return monitor


def bench_dense_offline(
    formula_id: int,
    spec: str,
    signal: list[tuple[float, float]],
    m: int,
) -> dict:
    """Benchmark dense-time offline evaluation (whole signal at once)."""
    try:
        n_samples = len(signal)
        total_time = 0.0
        # dense-time offline expects: ['x', [(t0,v0), (t1,v1), ...]]
        dataset = ["x", [[ts, val] for ts, val in signal]]

        for _ in range(m):
            monitor = make_dense_monitor_offline(spec)
            t0 = time.perf_counter()
            monitor.evaluate(dataset)
            t1 = time.perf_counter()
            total_time += t1 - t0

        avg_total = total_time / m
        avg_per_sample = avg_total / n_samples

        return {
            "formula_id": formula_id,
            "spec": spec,
            "monitor_type": "dense-time",
            "mode": "offline",
            "n_samples": n_samples,
            "m_runs": m,
            "avg_total_s": avg_total,
            "avg_per_sample_s": avg_per_sample,
            "avg_per_sample_us": avg_per_sample * 1e6,
        }
    except Exception as e:
        print(f"ERROR: Formula {formula_id} (dense-time offline) failed: {str(e)[:100]}")
        return None


# ---------------------------------------------------------------------------
# Discrete-Time Online Monitoring
# ---------------------------------------------------------------------------

def make_discrete_monitor(spec: str) -> rtamt.StlDiscreteTimeSpecification:
    """Create a fresh rtamt discrete-time monitor for online monitoring."""
    monitor = rtamt.StlDiscreteTimeSpecification()
    monitor.declare_var("x", "float")
    monitor.spec = spec
    monitor.parse()
    monitor.pastify()
    return monitor


def bench_discrete_online(
    formula_id: int,
    spec: str,
    signal: list[tuple[float, float]],
    m: int,
) -> dict:
    """Benchmark discrete-time online monitoring (one sample at a time)."""
    try:
        n_samples = len(signal)
        total_time = 0.0

        for _ in range(m):
            monitor = make_discrete_monitor(spec)
            t0 = time.perf_counter()
            for ts, val in signal:
                monitor.update(ts, [["x", val]])
            t1 = time.perf_counter()
            total_time += t1 - t0

        avg_total = total_time / m
        avg_per_sample = avg_total / n_samples

        return {
            "formula_id": formula_id,
            "spec": spec,
            "monitor_type": "discrete-time",
            "mode": "online",
            "n_samples": n_samples,
            "m_runs": m,
            "avg_total_s": avg_total,
            "avg_per_sample_s": avg_per_sample,
            "avg_per_sample_us": avg_per_sample * 1e6,
        }
    except Exception as e:
        print(f"ERROR: Formula {formula_id} (discrete-time online) failed: {str(e)[:100]}")
        return None


# ---------------------------------------------------------------------------
# Discrete-Time Offline Monitoring
# ---------------------------------------------------------------------------

def make_discrete_monitor_offline(spec: str) -> rtamt.StlDiscreteTimeSpecification:
    """Create a fresh rtamt discrete-time monitor for offline evaluation."""
    monitor = rtamt.StlDiscreteTimeSpecification()
    monitor.declare_var("x", "float")
    monitor.spec = spec
    monitor.parse()
    return monitor


def bench_discrete_offline(
    formula_id: int,
    spec: str,
    signal: list[tuple[float, float]],
    m: int,
) -> dict:
    """Benchmark discrete-time offline evaluation (whole signal at once)."""
    try:
        n_samples = len(signal)
        total_time = 0.0
        # discrete-time offline expects: {'time': [...], 'x': [...]}
        dataset = {
            "time": [ts for ts, _ in signal],
            "x": [val for _, val in signal],
        }

        for _ in range(m):
            monitor = make_discrete_monitor_offline(spec)
            t0 = time.perf_counter()
            monitor.evaluate(dataset)
            t1 = time.perf_counter()
            total_time += t1 - t0

        avg_total = total_time / m
        avg_per_sample = avg_total / n_samples

        return {
            "formula_id": formula_id,
            "spec": spec,
            "monitor_type": "discrete-time",
            "mode": "offline",
            "n_samples": n_samples,
            "m_runs": m,
            "avg_total_s": avg_total,
            "avg_per_sample_s": avg_per_sample,
            "avg_per_sample_us": avg_per_sample * 1e6,
        }
    except Exception as e:
        print(f"ERROR: Formula {formula_id} (discrete-time offline) failed: {str(e)[:100]}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    signal = load_signal(SIGNAL_CSV)
    print(f"Loaded signal with {len(signal)} samples from {SIGNAL_CSV}")
    print(f"Averaging over M = {M} runs\n")

    # Initialize CSV file with headers
    out_path = os.path.join(os.path.dirname(__file__), "rtamt_benchmark_results.csv")

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
        "monitor_type",
        "mode",
        "n_samples",
        "m_runs",
        "avg_total_s",
        "avg_per_sample_s",
        "avg_per_sample_us",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # --- Dense-Time Online ---
    # print("Dense-Time — Online")
    # pbar = tqdm(FORMULAS.items(), desc="Dense-Time Online")
    # for fid, spec in pbar:
    #     pbar.set_postfix({"fid": fid, "spec": spec[:40] + "..."})
    #     res = bench_dense_online(fid, spec, signal, M)
    #     if res is not None:
    #         writer.writerow(res)
    #         csv_file.flush()

    # # --- Dense-Time Offline ---
    # print("\nDense-Time — Offline")
    # pbar = tqdm(FORMULAS.items(), desc="Dense-Time Offline")
    # for fid, spec in pbar:
    #     pbar.set_postfix({"fid": fid, "spec": spec[:40] + "..."})
    #     res = bench_dense_offline(fid, spec, signal, M)
    #     if res is not None:
    #         writer.writerow(res)
    #         csv_file.flush()

    # --- Discrete-Time Online ---
    print("\nDiscrete-Time — Online")
    pbar = tqdm(FORMULAS.items(), desc="Discrete-Time Online")
    for fid, spec in pbar:
        pbar.set_postfix({"fid": fid, "spec": spec[:40] + "..."})
        res = bench_discrete_online(fid, spec, signal, M)
        if res is not None:
            writer.writerow(res)
            csv_file.flush()

    # --- Discrete-Time Offline ---
    # print("\nDiscrete-Time — Offline")
    # pbar = tqdm(FORMULAS.items(), desc="Discrete-Time Offline")
    # for fid, spec in pbar:
    #     pbar.set_postfix({"fid": fid, "spec": spec[:40] + "..."})
    #     res = bench_discrete_offline(fid, spec, signal, M)
    #     if res is not None:
    #         writer.writerow(res)
    #         csv_file.flush()

    csv_file.close()
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
