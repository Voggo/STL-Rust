"""
RoSI semantics performance (general formulas).

Bar chart showing RoSI execution time across general formulas,
with Python binding overhead visualization.
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    COLORS, LABELS_FULL, FORMULA_NAMES, USE_LOG_Y_AXIS,
    FIGURE_WIDTH, FIGURE_HEIGHT,
    PYTHON_OVERHEAD_ALPHA, PYTHON_OVERHEAD_HATCH, PYTHON_OVERHEAD_LABEL,
    get_output_dir,
)
from data_loader import load_native_general, load_python_general


def _save_figure(fig: plt.Figure, name: str) -> None:
    """Save figure to PDF and PNG formats."""
    output_dir = get_output_dir()
    for ext in ["pdf", "png"]:
        filepath = output_dir / f"{name}.{ext}"
        fig.savefig(filepath)
        print(f"Saved: {filepath}")


def plot_bar_rosi(save: bool = True, show: bool = False) -> plt.Figure:
    """Create bar chart for RoSI semantics across general formulas."""
    # Load data: {formula_id: {semantics: time_µs}}
    native_data = load_native_general()
    python_data = load_python_general()

    formula_ids = sorted(native_data.keys())
    bar_width = 0.5
    
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    x_positions = np.arange(len(formula_ids))
    semantic = "Rosi"
    color = COLORS[semantic]

    # Extract RoSI values for each formula
    rust_values = [native_data.get(fid, {}).get(semantic, 0) for fid in formula_ids]
    python_values = [python_data.get(fid, {}).get(semantic, 0) for fid in formula_ids]

    # Rust native bars
    ax.bar(x_positions, rust_values, bar_width, color=color,
           label=f"{LABELS_FULL[semantic]} (Native)", edgecolor="white", linewidth=0.5)

    # Python binding overhead
    overhead_values = [max(0, py - ru) for py, ru in zip(python_values, rust_values)]
    ax.bar(x_positions, overhead_values, bar_width, bottom=rust_values,
           color=color, alpha=PYTHON_OVERHEAD_ALPHA, hatch=PYTHON_OVERHEAD_HATCH,
           edgecolor="white", linewidth=0.5, label=PYTHON_OVERHEAD_LABEL, zorder=4)

    # Configure axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels([FORMULA_NAMES[fid] for fid in formula_ids])
    ax.set_xlabel("Formula")
    ax.set_ylabel("Time per sample (µs)")
    ax.set_title("RoSI Semantics Performance")
    
    if USE_LOG_Y_AXIS:
        ax.set_yscale("log")
    
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    if save:
        _save_figure(fig, "plot2_rosi_performance")
    
    if show:
        plt.show()
    
    return fig


if __name__ == "__main__":
    plot_bar_rosi(save=True, show=True)
