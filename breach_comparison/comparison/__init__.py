"""
Evaluation Results Comparison Package.

Modules for comparing evaluation results from Breach and RoSI approaches.
"""

from .helpers import parse_json_list_with_inf, to_float, extract_formula_number, find_matching_files
from .rosi_comparison import (
    ViolationType,
    Interval,
    ComparisonResult,
    FormulaComparison,
    read_results,
)
from .strict_comparison import (
    StrictViolationType,
    StrictComparisonResult,
    StrictFormulaComparison,
    read_strict_results,
)
from .table_writer import write_summary_table, write_strict_summary_table

__all__ = [
    "parse_json_list_with_inf",
    "to_float",
    "extract_formula_number",
    "find_matching_files",
    "ViolationType",
    "Interval",
    "ComparisonResult",
    "FormulaComparison",
    "read_results",
    "StrictViolationType",
    "StrictComparisonResult",
    "StrictFormulaComparison",
    "read_strict_results",
    "write_summary_table",
    "write_strict_summary_table",
]
