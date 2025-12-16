"""
Table writing utilities for summary reports.
"""

from pathlib import Path
from typing import Dict

from .rosi_comparison import FormulaComparison
from .strict_comparison import StrictFormulaComparison


def write_summary_table(
    formula_comparisons: Dict[str, FormulaComparison], output_file: Path
):
    """Write a summary table for RoSI comparisons to a file."""
    headers = [
        "Formula",
        "Total",
        "Violations",
        "Rate (%)",
        "Contradiction",
        "Breach Decided",
        "RoSI Decided",
        "Breach Tighter",
        "RoSI Tighter",
        "Mismatch",
    ]

    # Calculate column widths
    widths = [len(h) for h in headers]
    rows = []

    for formula_num in sorted(formula_comparisons.keys(), key=lambda x: int(x)):
        cmp = formula_comparisons[formula_num]

        rate = 0.0
        if cmp.total_with_verdicts > 0:
            # Only count violations where we had verdicts
            relevant_violations = cmp.violations_count - cmp.missing_verdict_count
            rate = (relevant_violations / cmp.total_with_verdicts) * 100

        row = [
            f"F{formula_num}",
            str(cmp.total_comparisons),
            str(cmp.violations_count),
            f"{rate:.2f}",
            str(cmp.contradiction_count),
            str(cmp.breach_decided_count),
            str(cmp.rosi_decided_count),
            str(cmp.breach_tighter_count),
            str(cmp.rosi_tighter_count),
            str(cmp.bounds_mismatch_count),
        ]
        rows.append(row)

        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    with open(output_file, "w") as f:
        # Header
        f.write(
            "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)) + " |\n"
        )
        # Separator
        f.write("|-" + "-|-".join("-" * w for w in widths) + "-|\n")
        # Rows
        for row in rows:
            f.write(
                "| " + " | ".join(f"{val:<{w}}" for val, w in zip(row, widths)) + " |\n"
            )

        f.write("\n\n### Violation Types Legend\n\n")
        f.write(
            "- **Contradiction**: The intervals from Breach and RoSI are disjoint (e.g., one says satisfied, the other violated).\n"
        )
        f.write(
            "- **Breach Decided**: Breach provides a definitive verdict (Sat/Unsat), while RoSI returns Unknown (interval contains 0).\n"
        )
        f.write(
            "- **RoSI Decided**: RoSI provides a definitive verdict, while Breach returns Unknown.\n"
        )
        f.write(
            "- **Breach Tighter**: Breach's interval is strictly contained within RoSI's interval (Breach is more precise).\n"
        )
        f.write(
            "- **RoSI Tighter**: RoSI's interval is strictly contained within Breach's interval (RoSI is more precise).\n"
        )
        f.write(
            "- **Mismatch**: Intervals overlap and agree on the verdict (or both unknown), but the bounds differ significantly.\n"
        )

    print(f"✓ Summary table written to {output_file}")


def write_strict_summary_table(
    formula_comparisons: Dict[str, StrictFormulaComparison], output_file: Path
):
    """Write a summary table for Strict comparisons to a file."""
    headers = [
        "Formula",
        "Total",
        "Violations",
        "Rate (%)",
        "Value Mismatch",
    ]

    # Calculate column widths
    widths = [len(h) for h in headers]
    rows = []

    for formula_num in sorted(formula_comparisons.keys(), key=lambda x: int(x)):
        cmp = formula_comparisons[formula_num]

        rate = 0.0
        if cmp.total_with_verdicts > 0:
            # Only count violations where we had verdicts
            relevant_violations = cmp.violations_count - cmp.missing_verdict_count
            rate = (relevant_violations / cmp.total_with_verdicts) * 100

        row = [
            f"F{formula_num}",
            str(cmp.total_comparisons),
            str(cmp.violations_count),
            f"{rate:.2f}",
            str(cmp.value_mismatch_count),
        ]
        rows.append(row)

        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    with open(output_file, "w") as f:
        # Header
        f.write(
            "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)) + " |\n"
        )
        # Separator
        f.write("|-" + "-|-".join("-" * w for w in widths) + "-|\n")
        # Rows
        for row in rows:
            f.write(
                "| " + " | ".join(f"{val:<{w}}" for val, w in zip(row, widths)) + " |\n"
            )

        f.write("\n\n### Violation Types Legend\n\n")
        f.write(
            "- **Value Mismatch**: The strict values from Breach and RoSI differ beyond tolerance.\n"
        )

    print(f"✓ Strict summary table written to {output_file}")
