"""Common utilities for benchmark visualization scripts."""

import os
import re
import pandas as pd

# Font size constants for consistent styling across all plots
FONT_SIZE_TITLE = 16
FONT_SIZE_LEGEND = 12
FONT_SIZE_LABEL = 12


def extract_time_from_stats(stats_str):
    """Parse execution time in seconds from statistics string.

    Args:
        stats_str: Statistics string in format "mean=X; lb=Y; ub=Z" where X is in nanoseconds.

    Returns:
        Time in seconds as float, or None if parsing fails.
    """
    if pd.isna(stats_str):
        return None
    match = re.search(r"mean=([0-9\.]+)", str(stats_str))
    if match:
        return float(match.group(1)) * 1e-9  # Convert ns to seconds
    return None


def load_benchmark_data(csv_path):
    """Load benchmark CSV and extract timing information.

    Args:
        csv_path: Path to benchmark CSV file.

    Returns:
        DataFrame with added 'time_s' column.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["time_s"] = df["statistics"].apply(extract_time_from_stats)
    return df


def load_formulas(formulas_csv_path):
    """Load formula definitions and create lookup dictionaries.

    Args:
        formulas_csv_path: Path to formulas CSV file.

    Returns:
        Tuple of (id_to_formula, id_to_short_name, formula_to_id, formula_order).
    """
    if not os.path.exists(formulas_csv_path):
        raise FileNotFoundError(f"File not found: {formulas_csv_path}")

    df = pd.read_csv(formulas_csv_path)

    id_to_formula = {row["id"]: row["formula"] for _, row in df.iterrows()}
    id_to_short_name = {row["id"]: row["short_name"] for _, row in df.iterrows()}
    formula_to_id = {row["formula"]: row["id"] for _, row in df.iterrows()}

    # Create ordering based on formula IDs for consistent sorting
    formula_order = {formula: idx for idx, formula in enumerate(id_to_formula.values())}

    return id_to_formula, id_to_short_name, formula_to_id, formula_order


def create_config_label(df):
    """Add config label column combining approach, eval_mode, and out_type.

    Args:
        df: DataFrame with 'approach', 'eval_mode', and 'out_type' columns.

    Returns:
        DataFrame with added 'config' column.
    """
    df = df.copy()
    df["config"] = df["approach"] + "/" + df["eval_mode"] + "/" + df["out_type"]
    return df


def ensure_output_folder(folder_path="plots"):
    """Create output folder if it doesn't exist.

    Args:
        folder_path: Path to output folder.
    """
    os.makedirs(folder_path, exist_ok=True)


def get_formulas_for_ids(id_list, id_to_formula):
    """Get formulas for given formula IDs.

    Args:
        id_list: List of formula IDs.
        id_to_formula: Dictionary mapping ID to formula string.

    Returns:
        List of formula strings.
    """
    return [id_to_formula[id_] for id_ in id_list if id_ in id_to_formula]


def extract_depth_from_short_name(short_name):
    """Extract depth value from short_name column.

    Args:
        short_name: Short name string in format "... (Depth N)".

    Returns:
        Depth as int, or None if not found.
    """
    match = re.search(r"\(Depth (\d+)\)", str(short_name))
    if match:
        return int(match.group(1))
    return None


def extract_window_size(formula):
    """Extract window size from formula string.

    Args:
        formula: Formula string potentially containing "[0, N]" pattern.

    Returns:
        Window size as int, or None if not found.
    """
    match = re.search(r"\[0,\s*(\d+)\]", str(formula))
    if match:
        return int(match.group(1))
    return None


def extract_operator_type(formula):
    """Extract primary operator type from formula.

    Args:
        formula: Formula string.

    Returns:
        Operator type string ("G (Global)", "F (Eventually)", "U (Until)"), or None.
    """
    formula = str(formula)
    if formula.startswith("G["):
        return "G (Global)"
    elif formula.startswith("F[") and "G[" not in formula:
        return "F (Eventually)"
    elif formula.startswith("(") and ("U[" in formula or ")U[" in formula):
        return "U (Until)"
    return None
