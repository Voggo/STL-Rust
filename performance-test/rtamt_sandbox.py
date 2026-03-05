import rtamt
import csv
import time
import os
import json

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
M = 1  # Number of runs to average over (reduced for slow formulas)
# Formulas with large Until windows are extremely slow in rtamt offline;
# use fewer runs so the benchmark finishes in reasonable time.
M_SLOW = 5  # runs for IDs in SLOW_IDS
SLOW_IDS = {11, 12}  # until[0,100] and until[0,1000]
SIGNAL_CSV = os.path.join(
    os.path.dirname(__file__),
    "..", "ostl", "benches", "signal_generation", "signals", "signal_20.csv",
)

# Formulas matching the ostl benchmark catalog (IDs 1-12)
FORMULAS: dict[int, str] = {
    # 1: "(x < 0.5) and (x > -0.5)",
    # 2: "(x < 0.5) or  (x > -0.5)",
    # 3: "not (x < 0.5)",
    # 4: "always[0,10]  (x < 0.5)",
    # 5: "always[0,100] (x < 0.5)",
    # 6: "always[0,1000](x < 0.5)",
    7: "eventually[0,10]  (x < 0.5)",
    # 8: "eventually[0,100] (x < 0.5)",
    # 9: "eventually[0,1000](x < 0.5)",
    # 10: "(x < 0.5) until[0,10]  (x > -0.5)",
    # 11: "(x < 0.5) until[0,100] (x > -0.5)",
    # 12: "(x < 0.5) until[0,1000](x > -0.5)",
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


def make_online_monitor(spec: str):
    """Create and configure a fresh rtamt discrete-time online monitor."""
    monitor = rtamt.StlDiscreteTimeSpecification()
    monitor.declare_var("x", "float")
    monitor.spec = spec
    monitor.parse()
    monitor.pastify()  # Convert to past-time for online monitoring
    return monitor

def bench_formula_online(
    formula_id: int,
    spec: str,
    signal: list[tuple[float, float]],
    m: int,
) -> dict:
    """Benchmark using online monitoring (one sample at a time)."""
    n_samples = len(signal)
    total_time = 0.0

    for _ in range(m):
        monitor = make_online_monitor(spec)
        t0 = time.perf_counter()
        for ts, val in signal:
            # v = monitor.update(["x", [[ts, val]]])
            v = monitor.update(ts, [["x", val]])
            print(f"t={ts:.2f}, x={val:.3f} => {v}")
        t1 = time.perf_counter()
        total_time += t1 - t0

    avg_total = total_time / m
    avg_per_sample = avg_total / n_samples

    return {
        "formula_id": formula_id,
        "spec": spec,
        "mode": "online",
        "n_samples": n_samples,
        "m_runs": m,
        "avg_total_s": avg_total,
        "avg_per_sample_s": avg_per_sample,
        "avg_per_sample_us": avg_per_sample * 1e6,
    }


def bench_formula(
    formula_id: int,
    spec: str,
    signal: list[tuple[float, float]],
    m: int,
) -> dict:
    """Pick online or offline mode depending on operator support."""
    return bench_formula_online(formula_id, spec, signal, m)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    signal = load_signal(SIGNAL_CSV)
    print(f"Loaded signal with {len(signal)} samples from {SIGNAL_CSV}")
    print(f"Averaging over M = {M} runs\n")
    print(f"{'ID':>3}  {'Mode':>7}  {'avg/sample (µs)':>16}  {'avg total (ms)':>15}  Formula")
    print("-" * 90)

    results = []
    for fid, spec in FORMULAS.items():
        m = M_SLOW if fid in SLOW_IDS else M
        print(f"  Running formula {fid} ({m} runs) ...", end="", flush=True)
        res = bench_formula(fid, spec, signal, m)
        results.append(res)
        print(
            f"\r{res['formula_id']:>3}  "
            f"{res['mode']:>7}  "
            f"{res['avg_per_sample_us']:>16.3f}  "
            f"{res['avg_total_s'] * 1e3:>15.3f}  "
            f"{res['spec']}"
        )

    # Save results to JSON
    out_path = os.path.join(os.path.dirname(__file__), "rtamt_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

