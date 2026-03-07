"""
Performance comparison across semantics (general formulas).

Bar chart showing execution time for three general formulas 
across different STL semantics, with Python binding overhead.
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    COLORS, LABELS_FULL, SEMANTICS_BAR, FORMULA_NAMES,
    FIGURE_WIDTH, FIGURE_HEIGHT, USE_LOG_Y_AXIS,
    PYTHON_OVERHEAD_ALPHA, PYTHON_OVERHEAD_HATCH, PYTHON_OVERHEAD_LABEL,
    get_output_dir,
)
from data_loader import load_native_general, load_python_general, load_rtamt_general


def _save_figure(fig: plt.Figure, name: str) -> None:
    """Save figure to PDF and PNG formats."""
    output_dir = get_output_dir()
    for ext in ["pdf", "png"]:
        filepath = output_dir / f"{name}.{ext}"
        fig.savefig(filepath)
        print(f"Saved: {filepath}")


def plot_bar_general(save: bool = True, show: bool = False) -> plt.Figure:
    """Create bar chart comparing semantics across general formulas."""
    # Load data: {formula_id: {semantics: time_µs}}
    native_data = load_native_general()
    python_data = load_python_general()
    rtamt_data = load_rtamt_general()

    formula_ids = sorted(native_data.keys())
    n_formulas = len(formula_ids)
    n_semantics = len(SEMANTICS_BAR)
    
    # Layout parameters
    bar_width = 0.18
    group_spacing = 0.3
    
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    group_positions = []
    python_overhead_labeled = False

    for group_idx, fid in enumerate(formula_ids):
        group_start = group_idx * (n_semantics * bar_width + group_spacing)
        group_center = group_start + (n_semantics - 1) * bar_width / 2
        group_positions.append(group_center)

        for sem_idx, semantic in enumerate(SEMANTICS_BAR):
            x_pos = group_start + sem_idx * bar_width
            color = COLORS[semantic]
            label = LABELS_FULL[semantic] if group_idx == 0 else None

            if semantic == "rtamt":
                # RTAMT has single value per formula
                value = rtamt_data.get(fid, 0)
                ax.bar(x_pos, value, bar_width * 0.9, color=color, label=label,
                       edgecolor="white", linewidth=0.5)
            else:
                # Rust native performance
                rust_value = native_data.get(fid, {}).get(semantic, 0)
                ax.bar(x_pos, rust_value, bar_width * 0.9, color=color, label=label,
                       edgecolor="white", linewidth=0.5)

                # Python binding overhead (stacked on top)
                python_value = python_data.get(fid, {}).get(semantic, 0)
                overhead = max(0, python_value - rust_value)
                
                if overhead > 0:
                    overhead_label = PYTHON_OVERHEAD_LABEL if not python_overhead_labeled else None
                    ax.bar(x_pos, overhead, bar_width * 0.9, bottom=rust_value,
                           color=color, alpha=PYTHON_OVERHEAD_ALPHA,
                           hatch=PYTHON_OVERHEAD_HATCH, edgecolor="white",
                           linewidth=0.5, label=overhead_label, zorder=4)
                    python_overhead_labeled = True

    # Configure axes
    ax.set_xticks(group_positions)
    ax.set_xticklabels([FORMULA_NAMES[fid] for fid in formula_ids])
    ax.set_xlabel("Formula")
    ax.set_ylabel("Time per sample (µs)")
    ax.set_title("Execution Time Across Semantics")
    
    if USE_LOG_Y_AXIS:
        ax.set_yscale("log")
    
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    if save:
        _save_figure(fig, "plot1_semantics_comparison")
    
    if show:
        plt.show()
    
    return fig


if __name__ == "__main__":
    plot_bar_general(save=True, show=True)
