"""
RoSI interval comparison logic.
"""

import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum, auto

from .helpers import parse_json_list_with_inf, to_float


class ViolationType(Enum):
    
    NONE = auto()
    MISSING_VERDICT = auto()
    CONTRADICTION = auto()  # Intervals are disjoint
    BREACH_DECIDED_ROSI_UNKNOWN = auto()  # Breach has verdict, RoSI has 0 (unknown)
    ROSI_DECIDED_BREACH_UNKNOWN = auto()  # RoSI has verdict, Breach has 0 (unknown)
    BREACH_TIGHTER = auto()  # Breach interval is strictly contained in RoSI
    ROSI_TIGHTER = auto()  # RoSI interval is strictly contained in Breach
    BOUNDS_MISMATCH = auto()  # Overlapping, consistent on zero, but bounds differ


@dataclass
class Interval:
    """Represents a robustness interval."""

    lower: float
    upper: float

    def contains_zero(self) -> bool:
        """Check if interval contains 0."""
        return self.lower <= 0 <= self.upper

    def is_point(self) -> bool:
        """Check if interval is a single point (lower == upper)."""
        return abs(self.lower - self.upper) < 1e-10

    def get_value(self) -> Optional[float]:
        """Get the concluded value if interval is a point."""
        if self.is_point():
            return (self.lower + self.upper) / 2
        return None

    def __str__(self) -> str:
        return f"[{self.lower}, {self.upper}]"


@dataclass
class ComparisonResult:
    """Result of comparing two intervals."""

    formula_id: str
    time_step: int
    breach_interval: Optional[Interval] = None
    rosi_interval: Optional[Interval] = None

    tolerance: float = 1e-1

    violation_type: ViolationType = ViolationType.NONE
    violations: List[str] = field(default_factory=list)

    def check_consistency(self):
        """Check for violations and classify them."""
        self.violations.clear()
        self.violation_type = ViolationType.NONE

        # 1. Check Missing Verdicts
        if self.breach_interval is None or self.rosi_interval is None:
            self.violation_type = ViolationType.MISSING_VERDICT
            missing_side = "Breach" if self.breach_interval is None else "RoSI"
            self.violations.append(f"Missing verdict on {missing_side} side")
            return

        b = self.breach_interval
        r = self.rosi_interval

        # 2. Check Contradiction (Disjoint Intervals)
        inter_low = max(b.lower, r.lower)
        inter_high = min(b.upper, r.upper)

        if inter_low > inter_high + self.tolerance:
            self.violation_type = ViolationType.CONTRADICTION
            self.violations.append(f"Contradiction: Breach {b} vs RoSI {r} (Disjoint)")
            return

        # 3. Check Decision Disagreement (Zero Containment)
        b_zero = b.contains_zero()
        r_zero = r.contains_zero()

        if b_zero != r_zero:
            if b_zero:  # Breach has zero (Unknown), RoSI doesn't (Decided)
                self.violation_type = ViolationType.ROSI_DECIDED_BREACH_UNKNOWN
                self.violations.append(
                    f"RoSI Decided / Breach Unknown: Breach {b} vs RoSI {r}"
                )
            else:  # RoSI has zero (Unknown), Breach doesn't (Decided)
                self.violation_type = ViolationType.BREACH_DECIDED_ROSI_UNKNOWN
                self.violations.append(
                    f"Breach Decided / RoSI Unknown: Breach {b} vs RoSI {r}"
                )
            return

        # 4. Check Bounds Mismatch & Tightness
        low_diff = abs(b.lower - r.lower) > self.tolerance
        high_diff = abs(b.upper - r.upper) > self.tolerance

        if low_diff or high_diff:
            # Check strict containment
            b_in_r = (b.lower >= r.lower - self.tolerance) and (
                b.upper <= r.upper + self.tolerance
            )
            r_in_b = (r.lower >= b.lower - self.tolerance) and (
                r.upper <= b.upper + self.tolerance
            )

            if b_in_r and not r_in_b:
                self.violation_type = ViolationType.BREACH_TIGHTER
                self.violations.append(f"Breach Tighter: Breach {b} vs RoSI {r}")
            elif r_in_b and not b_in_r:
                self.violation_type = ViolationType.ROSI_TIGHTER
                self.violations.append(f"RoSI Tighter: Breach {b} vs RoSI {r}")
            else:
                self.violation_type = ViolationType.BOUNDS_MISMATCH
                self.violations.append(f"Bounds Mismatch: Breach {b} vs RoSI {r}")
        else:
            self.violation_type = ViolationType.NONE

    def has_violations(self) -> bool:
        """Check if there are any violations."""
        return self.violation_type != ViolationType.NONE

    def get_summary(self) -> str:
        """Get a one-line summary."""
        if not self.has_violations():
            return "✓ OK"
        return f"[{self.violation_type.name}] " + " | ".join(self.violations)


@dataclass
class FormulaComparison:
    """Results for a single formula."""

    formula_id: str
    total_comparisons: int = 0
    total_with_verdicts: int = 0
    violations_count: int = 0

    # Violation Counts
    missing_verdict_count: int = 0
    contradiction_count: int = 0
    breach_decided_count: int = 0
    rosi_decided_count: int = 0
    breach_tighter_count: int = 0
    rosi_tighter_count: int = 0
    bounds_mismatch_count: int = 0

    comparison_details: List[ComparisonResult] = field(default_factory=list)

    def add_comparison(self, result: ComparisonResult):
        """Add a comparison result."""
        self.comparison_details.append(result)
        self.total_comparisons += 1

        if result.breach_interval is not None and result.rosi_interval is not None:
            self.total_with_verdicts += 1

        if result.has_violations():
            self.violations_count += 1
            vt = result.violation_type
            if vt == ViolationType.MISSING_VERDICT:
                self.missing_verdict_count += 1
            elif vt == ViolationType.CONTRADICTION:
                self.contradiction_count += 1
            elif vt == ViolationType.BREACH_DECIDED_ROSI_UNKNOWN:
                self.breach_decided_count += 1
            elif vt == ViolationType.ROSI_DECIDED_BREACH_UNKNOWN:
                self.rosi_decided_count += 1
            elif vt == ViolationType.BREACH_TIGHTER:
                self.breach_tighter_count += 1
            elif vt == ViolationType.ROSI_TIGHTER:
                self.rosi_tighter_count += 1
            elif vt == ViolationType.BOUNDS_MISMATCH:
                self.bounds_mismatch_count += 1

    def __str__(self) -> str:
        verdict_info = (
            f"{self.total_with_verdicts}/{self.total_comparisons} w/ verdicts"
        )

        if self.violations_count == 0:
            return f"F{self.formula_id}: {verdict_info}, 0 violations"

        details = []
        if self.missing_verdict_count:
            details.append(f"Missing: {self.missing_verdict_count}")
        if self.contradiction_count:
            details.append(f"Contradiction: {self.contradiction_count}")
        if self.breach_decided_count:
            details.append(f"Breach Decided: {self.breach_decided_count}")
        if self.rosi_decided_count:
            details.append(f"RoSI Decided: {self.rosi_decided_count}")
        if self.breach_tighter_count:
            details.append(f"Breach Tighter: {self.breach_tighter_count}")
        if self.rosi_tighter_count:
            details.append(f"RoSI Tighter: {self.rosi_tighter_count}")
        if self.bounds_mismatch_count:
            details.append(f"Mismatch: {self.bounds_mismatch_count}")

        return (
            f"F{self.formula_id}: {verdict_info}, "
            f"{self.violations_count} violations | {', '.join(details)}"
        )


def read_results(file_path: Path, value_column: str) -> Dict[int, Interval]:
    """Read evaluation results and return the final interval for each time step.

    Only takes the LAST (most recent) interval value for each time step across all rows.
    This ensures we get the converged final value, not intermediate values.
    Interpolated time steps (non-integers or floats from intermediate values) are ignored.
    """
    final_intervals = {}  # time_step -> Interval

    try:
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    times_str = row.get("output_times", "[]")
                    values_str = row.get(value_column, "[]")

                    times = parse_json_list_with_inf(times_str)
                    values = parse_json_list_with_inf(values_str)

                    if not times or not values or len(times) != len(values):
                        continue

                    for t, v in zip(times, values):
                        # v is [lower, upper]
                        if not isinstance(v, list) or len(v) < 2:
                            continue

                        lower = to_float(v[0])
                        upper = to_float(v[1])

                        # Convert time to int; skip if it's a float (interpolated value)
                        t_int = int(t) if isinstance(t, (int, float)) else None
                        if t_int is None:
                            continue

                        # Skip interpolated times (where t != int(t))
                        if isinstance(t, float) and abs(t - t_int) > 1e-6:
                            continue

                        # Always overwrite with the latest value (last occurrence wins)
                        final_intervals[t_int] = Interval(lower, upper)

                except (ValueError, TypeError):
                    continue
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")

    return final_intervals
