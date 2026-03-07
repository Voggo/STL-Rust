"""
Data loading utilities for benchmark CSV files.

Loads and structures data from:
  - paper_native_benchmark_results.csv   (Rust native)
  - ostlpython_benchmark_results.csv     (Python binding)
  - rtamt_benchmark_results.csv          (rtamt baseline)
"""

import csv
import os
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Paths  (relative to the performance-test directory)
# ---------------------------------------------------------------------------
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NATIVE_CSV = os.path.join(
    _BASE, "..", "ostl", "benches", "results", "paper_native_benchmark_results.csv"
)
PYTHON_CSV = os.path.join(_BASE, "ostlpython_benchmark_results.csv")
RTAMT_CSV = os.path.join(_BASE, "rtamt_benchmark_results.csv")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkRow:
    formula_id: str
    spec: str
    semantics: str
    avg_per_sample_us: float
    interval_len: Optional[int] = None
    benchmark_kind: Optional[str] = None


# ---------------------------------------------------------------------------
# Generic CSV loader
# ---------------------------------------------------------------------------
def _load_csv(path: str) -> list[dict]:
    """Load a CSV file and return a list of row dicts."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Native Rust benchmark data
# ---------------------------------------------------------------------------
def load_native() -> list[BenchmarkRow]:
    """Load native Rust benchmark results."""
    rows = _load_csv(NATIVE_CSV)
    result = []
    for r in rows:
        ilen = r.get("interval_len", "0")
        result.append(BenchmarkRow(
            formula_id=r["formula_id"],
            spec=r["spec"],
            semantics=r["semantics"],
            avg_per_sample_us=float(r["avg_per_sample_us"]),
            interval_len=int(ilen) if ilen else 0,
            benchmark_kind=r.get("benchmark_kind", ""),
        ))
    return result


def load_native_general() -> dict[int, dict[str, float]]:
    """Return {formula_id: {semantics: avg_per_sample_us}} for general formulas (1-3)."""
    data = load_native()
    result: dict[int, dict[str, float]] = {}
    for r in data:
        fid = int(r.formula_id)
        if fid in (1, 2, 3):
            result.setdefault(fid, {})[r.semantics] = r.avg_per_sample_us
    return result


def load_native_line(formula_id: str, semantics: str) -> list[tuple[int, float]]:
    """Return sorted [(interval_len, avg_per_sample_us)] for a given formula_id and semantics."""
    data = load_native()
    pairs = []
    for r in data:
        if r.formula_id == formula_id and r.semantics == semantics:
            pairs.append((r.interval_len, r.avg_per_sample_us))
    pairs.sort(key=lambda x: x[0])
    return pairs


# ---------------------------------------------------------------------------
# Python (ostl_python) benchmark data
# ---------------------------------------------------------------------------
# In ostl_results.py the formula IDs for general formulas are 154, 155, 156.
# Map them back to our canonical IDs 1, 2, 3.
_PYTHON_GENERAL_SPECS = {
    "(x < 0.5) and (x > -0.5)": 1,
    "G[0,1000] (x > 0.5 -> F[0,100] (x < 0.0))": 2,
    "(G[0,100] (x < 0.5)) or (G[100,150] (x > 0.0))": 3,
}


def load_python_general() -> dict[int, dict[str, float]]:
    """Return {canonical_fid: {semantics: avg_per_sample_us}} for general formulas."""
    rows = _load_csv(PYTHON_CSV)
    result: dict[int, dict[str, float]] = {}
    for r in rows:
        spec = r["spec"]
        if spec in _PYTHON_GENERAL_SPECS:
            fid = _PYTHON_GENERAL_SPECS[spec]
            sem = r["semantics"]
            result.setdefault(fid, {})[sem] = float(r["avg_per_sample_us"])
    return result


# ---------------------------------------------------------------------------
# rtamt benchmark data
# ---------------------------------------------------------------------------
_RTAMT_GENERAL_SPECS = {
    "(x < 0.5) and (x > -0.5)": 1,
    "G[0,1000] (x > 0.5 -> F[0,100] (x < 0.0))": 2,
    "(G[0,100] (x < 0.5)) or (G[100,150] (x > 0.0))": 3,
}


def load_rtamt_general() -> dict[int, float]:
    """Return {canonical_fid: avg_per_sample_us} for rtamt discrete-time online."""
    rows = _load_csv(RTAMT_CSV)
    result: dict[int, float] = {}
    for r in rows:
        spec = r["spec"]
        if spec in _RTAMT_GENERAL_SPECS:
            fid = _RTAMT_GENERAL_SPECS[spec]
            # Use discrete-time online as the rtamt baseline
            if r["monitor_type"] == "discrete-time" and r["mode"] == "online":
                result[fid] = float(r["avg_per_sample_us"])
    return result
