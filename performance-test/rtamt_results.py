import rtamt
import csv
import time
import os
import json

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
M = 50  # Number of runs to average over
M_SLOW = 5  # Fewer runs for expensive formulas
SLOW_IDS = {11, 12}  # until[0,100] and until[0,1000]
SIGNAL_CSV = os.path.join(
    os.path.dirname(__file__),
    "..", "ostl", "benches", "signal_generation", "signals", "signal_5000.csv",
)

# Formulas matching the ostl benchmark catalog (IDs 1-12)
FORMULAS: dict[int, str] = {
    1: "(x < 0.5) and (x > -0.5)",
    2: "(x < 0.5) or  (x > -0.5)",
    3: "not (x < 0.5)",
    4: "always[0,10]  (x < 0.5)",
    5: "always[0,100] (x < 0.5)",
    6: "always[0,1000](x < 0.5)",
    7: "eventually[0,10]  (x < 0.5)",
    8: "eventually[0,100] (x < 0.5)",
    9: "eventually[0,1000](x < 0.5)",
    10: "(x < 0.5) until[0,10]  (x > -0.5)",
    11: "(x < 0.5) until[0,100] (x > -0.5)",
    12: "(x < 0.5) until[0,1000](x > -0.5)",
}

# Discrete-time online monitor does NOT support bounded temporal operators,
# so only pure boolean formulas (1-3) can run in discrete-time online mode.
DISCRETE_ONLINE_IDS = {1, 2, 3}


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


# ---- Dense-time monitor (pastified, supports all operators online) --------

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
        "monitor": "dense-time",
        "mode": "online",
        "n_samples": n_samples,
        "m_runs": m,
        "avg_total_s": avg_total,
        "avg_per_sample_s": avg_per_sample,
        "avg_per_sample_us": avg_per_sample * 1e6,
    }


# ---- Discrete-time monitor -----------------------------------------------

def make_discrete_monitor(spec: str) -> rtamt.StlDiscreteTimeSpecification:
    """Create a fresh rtamt discrete-time monitor."""
    monitor = rtamt.StlDiscreteTimeSpecification()
    monitor.declare_var("x", "float")
    monitor.spec = spec
    monitor.parse()
    monitor.pastify()  # Convert to past-time for online monitoring
    return monitor


def bench_discrete_online(
    formula_id: int,
    spec: str,
    signal: list[tuple[float, float]],
    m: int,
) -> dict:
    """Benchmark discrete-time online monitoring (one sample at a time).

    Only works for formulas without bounded temporal operators (IDs 1-3).
    """
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
        "monitor": "discrete-time",
        "mode": "online",
        "n_samples": n_samples,
        "m_runs": m,
        "avg_total_s": avg_total,
        "avg_per_sample_s": avg_per_sample,
        "avg_per_sample_us": avg_per_sample * 1e6,
    }


# ---- Offline evaluation ---------------------------------------------------

def make_dense_monitor_offline(spec: str) -> rtamt.StlDenseTimeSpecification:
    """Create a fresh rtamt dense-time monitor for offline evaluation."""
    monitor = rtamt.StlDenseTimeSpecification()
    monitor.declare_var("x", "float")
    monitor.spec = spec
    monitor.parse()
    return monitor


def make_discrete_monitor_offline(spec: str) -> rtamt.StlDiscreteTimeSpecification:
    """Create a fresh rtamt discrete-time monitor for offline evaluation."""
    monitor = rtamt.StlDiscreteTimeSpecification()
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
        "monitor": "dense-time",
        "mode": "offline",
        "n_samples": n_samples,
        "m_runs": m,
        "avg_total_s": avg_total,
        "avg_per_sample_s": avg_per_sample,
        "avg_per_sample_us": avg_per_sample * 1e6,
    }


def bench_discrete_offline(
    formula_id: int,
    spec: str,
    signal: list[tuple[float, float]],
    m: int,
) -> dict:
    """Benchmark discrete-time offline evaluation (whole signal at once)."""
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
        "monitor": "discrete-time",
        "mode": "offline",
        "n_samples": n_samples,
        "m_runs": m,
        "avg_total_s": avg_total,
        "avg_per_sample_s": avg_per_sample,
        "avg_per_sample_us": avg_per_sample * 1e6,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def print_header(title: str) -> None:
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")
    print(f"{'ID':>3}  {'Mode':>7}  {'avg/sample (µs)':>16}  {'avg total (ms)':>15}  Formula")
    print("-" * 90)


def print_result(res: dict) -> None:
    print(
        f"\r{res['formula_id']:>3}  "
        f"{res['mode']:>7}  "
        f"{res['avg_per_sample_us']:>16.3f}  "
        f"{res['avg_total_s'] * 1e3:>15.3f}  "
        f"{res['spec']}"
    )


def main() -> None:
    signal = load_signal(SIGNAL_CSV)
    print(f"Loaded signal with {len(signal)} samples from {SIGNAL_CSV}")
    print(f"Averaging over M = {M} runs (M_SLOW = {M_SLOW} for IDs {SLOW_IDS})")

    all_results = []

    # --- Dense-time (pastified, online, all formulas) ----------------------
    print_header("Dense-Time (pastified) — Online")
    for fid, spec in FORMULAS.items():
        if fid in [10, 11, 12]:
            print(f"  Skipping formula {fid} for dense-time online mode (unsupported until).")
            continue
        m = M_SLOW if fid in SLOW_IDS else M
        print(f"  Running formula {fid} (M={m}) ...", end="", flush=True)
        res = bench_dense_online(fid, spec, signal, m)
        all_results.append(res)
        print_result(res)

    # --- Discrete-time — Online -------------------
    print_header("Discrete-Time — Online")
    for fid, spec in FORMULAS.items():
        m = M_SLOW if fid in SLOW_IDS else M
        print(f"  Running formula {fid} (M={m}) ...", end="", flush=True)
        res = bench_discrete_online(fid, spec, signal, m)
        all_results.append(res)
        print_result(res)

    # --- Dense-time — Offline ------------------------------------------
    print_header("Dense-Time — Offline")
    for fid, spec in FORMULAS.items():
        m = M_SLOW if fid in SLOW_IDS else M
        print(f"  Running formula {fid} (M={m}) ...", end="", flush=True)
        res = bench_dense_offline(fid, spec, signal, m)
        all_results.append(res)
        print_result(res)

    # --- Discrete-time — Offline ---------------------------------------
    print_header("Discrete-Time — Offline")
    for fid, spec in FORMULAS.items():
        m = M_SLOW if fid in SLOW_IDS else M
        print(f"  Running formula {fid} (M={m}) ...", end="", flush=True)
        res = bench_discrete_offline(fid, spec, signal, m)
        all_results.append(res)
        print_result(res)

    # Save all results to JSON
    out_path = os.path.join(os.path.dirname(__file__), "rtamt_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

