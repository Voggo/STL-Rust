import ostl_python.ostl_python as ostl
import csv
import time
import os
import sys
import json

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
M = 100  # Number of runs to average over
M_SLOW = 5  # Fewer runs for expensive formulas
SLOW_IDS = {12}  # U[0,1000] is very expensive
SIGNAL_CSV = os.path.join(
    os.path.dirname(__file__),
    "..", "ostl", "benches", "signal_generation", "signals", "signal_5000.csv",
)

# Formulas matching the ostl benchmark catalog (IDs 1-12)
# Uses the same DSL syntax as Rust's stl! macro (parsed via parse_formula)
FORMULAS: dict[int, str] = {
    1: "(x < 0.5) and (x > -0.5)",
    2: "(x < 0.5) or  (x > -0.5)",
    3: "not(x < 0.5)",
    4: "G[0, 10]  (x < 0.5)",
    5: "G[0, 100] (x < 0.5)",
    6: "G[0, 1000](x < 0.5)",
    7: "F[0, 10]  (x < 0.5)",
    8: "F[0, 100] (x < 0.5)",
    9: "F[0, 1000](x < 0.5)",
    10: "(x < 0.5) U[0, 10]  (x > -0.5)",
    11: "(x < 0.5) U[0, 100] (x > -0.5)",
    12: "(x < 0.5) U[0, 1000](x > -0.5)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_signal(path: str) -> list[tuple[float, float]]:
    """Return list of (timestep, value) from a CSV file."""
    signal: list[tuple[float, float]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            signal.append((float(row["timestep"]), float(row["value"])))
    return signal


def suppress_stderr():
    """Redirect stderr to /dev/null to suppress Rust-side warnings."""
    devnull = open(os.devnull, "w")
    old_fd = os.dup(2)
    os.dup2(devnull.fileno(), 2)
    return old_fd, devnull


def restore_stderr(old_fd, devnull):
    """Restore stderr after suppression."""
    os.dup2(old_fd, 2)
    os.close(old_fd)
    devnull.close()


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

    old_fd, devnull = suppress_stderr()
    try:
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
    finally:
        restore_stderr(old_fd, devnull)

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
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    signal = load_signal(SIGNAL_CSV)
    print(f"Loaded signal with {len(signal)} samples from {SIGNAL_CSV}")
    print(f"Averaging over M = {M} runs\n")

    semantics = ["DelayedQuantitative", "DelayedQualitative", "EagerQualitative", "Rosi"]
    algorithm = "Incremental"

    print(f"  Algorithm: {algorithm}  |  Semantics: {semantics}")
    print(f"{'ID':>3}  {'avg/sample (µs)':>16}  {'avg total (ms)':>15}  Formula")
    print("-" * 80)

    results = []
    for sem in semantics:
        print(f"\n--- Semantics: {sem} ---")
        for fid, spec in FORMULAS.items():
            m = M_SLOW if fid in SLOW_IDS else M
            print(f"  Running formula {fid} (M={m}) ...", end="", flush=True)
            res = bench_formula(fid, spec, signal, m, semantics=sem, algorithm=algorithm)
            results.append(res)
            print(
                f"\r{res['formula_id']:>3}  "
                f"{res['avg_per_sample_us']:>16.3f}  "
                f"{res['avg_total_s'] * 1e3:>15.3f}  "
                f"{res['spec']}"
            )

    # Save results to JSON
    out_path = os.path.join(os.path.dirname(__file__), "ostl_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
