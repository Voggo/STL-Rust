"""
Strict value comparison logic.
"""

import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum, auto

from .helpers import parse_json_list_with_inf, to_float


class StrictViolationType(Enum):
    NONE = auto()
    MISSING_VERDICT = auto()
    VALUE_MISMATCH = auto()  # Values differ


@dataclass
class StrictComparisonResult:
    """Result of comparing two strict values."""

    formula_id: str
    time_step: int
    breach_value: Optional[float] = None
    rosi_value: Optional[float] = None

    tolerance: float = 1e-6

    violation_type: StrictViolationType = StrictViolationType.NONE
    violations: List[str] = field(default_factory=list)

    def check_consistency(self):
        """Check for violations in strict comparison."""
        self.violations.clear()
        self.violation_type = StrictViolationType.NONE

        # Check Missing Verdicts
        if self.breach_value is None or self.rosi_value is None:
            self.violation_type = StrictViolationType.MISSING_VERDICT
            missing_side = "Breach" if self.breach_value is None else "RoSI"
            self.violations.append(f"Missing verdict on {missing_side} side")
            return

        # Check Value Mismatch (different values)
        if abs(self.breach_value - self.rosi_value) > self.tolerance:
            self.violation_type = StrictViolationType.VALUE_MISMATCH
            self.violations.append(
                f"Value Mismatch: Breach {self.breach_value:.6f} vs RoSI {self.rosi_value:.6f}"
            )
            return

    def has_violations(self) -> bool:
        """Check if there are any violations."""
        return self.violation_type != StrictViolationType.NONE

    def get_summary(self) -> str:
        """Get a one-line summary."""
        if not self.has_violations():
            return "✓ OK"
        return f"[{self.violation_type.name}] " + " | ".join(self.violations)


@dataclass
class StrictFormulaComparison:
    """Results for a single formula (Strict approach)."""

    formula_id: str
    total_comparisons: int = 0
    total_with_verdicts: int = 0
    violations_count: int = 0

    # Violation Counts
    missing_verdict_count: int = 0
    value_mismatch_count: int = 0

    comparison_details: List[StrictComparisonResult] = field(default_factory=list)

    def add_comparison(self, result: StrictComparisonResult):
        """Add a comparison result."""
        self.comparison_details.append(result)
        self.total_comparisons += 1

        if result.breach_value is not None and result.rosi_value is not None:
            self.total_with_verdicts += 1

        if result.has_violations():
            self.violations_count += 1
            vt = result.violation_type
            if vt == StrictViolationType.MISSING_VERDICT:
                self.missing_verdict_count += 1
            elif vt == StrictViolationType.VALUE_MISMATCH:
                self.value_mismatch_count += 1

    def __str__(self) -> str:
        verdict_info = (
            f"{self.total_with_verdicts}/{self.total_comparisons} w/ verdicts"
        )

        if self.violations_count == 0:
            return f"F{self.formula_id}: {verdict_info}, 0 violations"

        details = []
        if self.missing_verdict_count:
            details.append(f"Missing: {self.missing_verdict_count}")
        if self.value_mismatch_count:
            details.append(f"Value Mismatch: {self.value_mismatch_count}")

        return (
            f"F{self.formula_id}: {verdict_info}, "
            f"{self.violations_count} violations | {', '.join(details)}"
        )


def read_strict_results(file_path: Path) -> Dict[int, float]:
    """Read strict evaluation results and return the final value for each time step.
    
    Only takes the LAST (most recent) value for each time step across all rows.
    This ensures we get the converged final value, not intermediate values.
    Interpolated time steps (non-integers or floats from intermediate values) are ignored.
    """
    final_values = {}  # time_step -> value

    try:
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    times_str = row.get("output_times", "[]")
                    values_str = row.get("output_values", "[]")

                    times = parse_json_list_with_inf(times_str)
                    values = parse_json_list_with_inf(values_str)

                    if not times or not values or len(times) != len(values):
                        continue

                    for t, v in zip(times, values):
                        # v is a single value (already parsed as a number)
                        try:
                            if isinstance(v, str):
                                if v == "-inf":
                                    value = float("-inf")
                                elif v == "inf":
                                    value = float("inf")
                                else:
                                    value = float(v)
                            else:
                                value = float(v)
                            
                            # Convert time to int; skip if it's a float (interpolated value)
                            t_int = int(t) if isinstance(t, (int, float)) else None
                            if t_int is None:
                                continue
                            
                            # Skip interpolated times (where t != int(t))
                            if isinstance(t, float) and abs(t - t_int) > 1e-6:
                                continue
                            
                            # Always overwrite with the latest value (last occurrence wins)
                            final_values[t_int] = value
                        except (ValueError, TypeError):
                            continue

                except (ValueError, TypeError):
                    continue
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")

    return final_values
