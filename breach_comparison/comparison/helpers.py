"""
Common helper functions for evaluation result comparison.
"""

import json
import csv
import re
from pathlib import Path
from typing import Dict, List, Any, Optional


def parse_json_list_with_inf(json_str: str) -> List[Any]:
    """Parse JSON string that might contain -inf/inf."""
    try:
        # Remove outer brackets and spaces
        json_str = json_str.strip()

        # Replace inf values
        json_str = json_str.replace("-inf", "NEG_INF")
        json_str = json_str.replace("inf", "POS_INF")
        json_str = json_str.replace("NEG_INF", '"-inf"')
        json_str = json_str.replace("POS_INF", '"inf"')

        return json.loads(json_str)
    except Exception:
        return []


def to_float(val: Any) -> float:
    """Convert a value to float, handling inf strings."""
    if isinstance(val, str):
        if val == "-inf":
            return float("inf") # Changed to inf from -inf since Breach has problems!
        if val == "inf":
            return float("inf")
    
    f_val = float(val)
    if f_val == 100.0:
        return float("inf")
    if f_val == -100.0:
        return float("inf") ## CHanged to inf from -inf since Breach has problems!
    return f_val


def extract_formula_number(filename: str) -> Optional[str]:
    """Extract formula number from filename."""
    match = re.search(r"_F(\d+)_", filename)
    if match:
        return match.group(1)
    return None


def find_matching_files(directory: Path, pattern: str) -> List[Path]:
    """Find all files matching a glob pattern in a directory."""
    return sorted(list(directory.glob(pattern)))
