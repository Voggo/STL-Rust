"""Standalone configuration for regression-fit plotting.

This configuration is intentionally separate from plotting/config.py to keep
regression visualization styling independent from benchmark comparison plots.
"""

from pathlib import Path

# Output
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "figures"

# Figure/layout
FIGSIZE = (14, 8)
DPI = 300
POINT_SIZE = 10
POINT_ALPHA = 0.45
LINE_WIDTH = 1.8
GRID_ALPHA = 0.25
LEGEND_FONT = 8

# Typography (LaTeX-like, but self-contained)
FONT_FAMILY = "serif"
FONT_SERIF = ["DejaVu Serif"]
FONT_SANS = ["DejaVu Sans"]

# Semantic colors
SEMANTICS_COLORS = {
    "DelayedQuantitative": "#1f77b4",
    "DelayedQualitative": "#2ca02c",
    "EagerQualitative": "#ff7f0e",
    "Rosi": "#9467bd",
}

SEMANTICS_SHORT = {
    "DelayedQuantitative": "Del. Quant.",
    "DelayedQualitative": "Del. Qual.",
    "EagerQualitative": "Eager Qual.",
    "Rosi": "RoSI",
}

# Operator line/marker style
OPERATOR_STYLE = {
    "G": {"linestyle": "-", "marker": "o"},
    "F": {"linestyle": "--", "marker": "s"},
    "U": {"linestyle": "-.", "marker": "^"},
}

# Panel grouping
NON_ROSI_SEMANTICS = ["DelayedQuantitative", "DelayedQualitative", "EagerQualitative"]
ROSI_SEMANTICS = ["Rosi"]
OPERATORS = ["G", "F", "U"]


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
