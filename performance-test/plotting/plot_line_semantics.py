"""
Scalability analysis for Delayed and Eager semantics: execution time vs. temporal depth.

Line chart comparing how different STL semantics scale with temporal bound,
for temporal operators (G, F, U).
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    COLORS, LABELS_SHORT, SEMANTICS_LINE, OPERATOR_MARKERS, OPERATOR_LINESTYLES,
    FORMULAS_LINE, FIGURE_WIDTH, FIGURE_HEIGHT,
    USE_AVERAGING_FORMULA, USE_LOG_Y_AXIS,
    get_output_dir,
)
from data_loader import load_native_line


def _save_figure(fig: plt.Figure, name: str) -> None:
    """Save figure to PDF and PNG formats."""
    output_dir = get_output_dir()
    for ext in ["pdf", "png"]:
        filepath = output_dir / f"{name}.{ext}"
        fig.savefig(filepath)
        print(f"Saved: {filepath}")


def plot_line_semantics(save: bool = True, show: bool = False) -> plt.Figure:
    """Create line chart comparing semantic scalability across temporal depth."""
    from config import OPERATOR_FORMULA_IDS
    
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    if USE_AVERAGING_FORMULA:
        # Average F and G results for each semantic
        for semantic in SEMANTICS_LINE:
            f_pairs = load_native_line(OPERATOR_FORMULA_IDS["F"], semantic)
            g_pairs = load_native_line(OPERATOR_FORMULA_IDS["G"], semantic)
            
            if f_pairs and g_pairs:
                # Create mappings and find common temporal bounds
                f_dict = {b: v for b, v in f_pairs}
                g_dict = {b: v for b, v in g_pairs}
                common_bounds = sorted(set(f_dict.keys()) & set(g_dict.keys()))
                
                if common_bounds:
                    # Average the two operators
                    avg_times = [(f_dict[b] + g_dict[b]) / 2 for b in common_bounds]
                    label = f"{LABELS_SHORT[semantic]}: {FORMULAS_LINE['6']}"  # Show F formula
                    ax.plot(
                        common_bounds, avg_times,
                        color=COLORS[semantic],
                        marker="D",
                        linestyle="-",
                        markersize=4,
                        linewidth=1.2,
                        label=label,
                    )
            
            # Always plot U separately
            u_pairs = load_native_line(OPERATOR_FORMULA_IDS["U"], semantic)
            if u_pairs:
                bounds, times = zip(*u_pairs)
                label = f"{LABELS_SHORT[semantic]}: {FORMULAS_LINE['4']}"
                ax.plot(
                    bounds, times,
                    color=COLORS[semantic],
                    marker=OPERATOR_MARKERS["U"],
                    linestyle=OPERATOR_LINESTYLES["U"],
                    markersize=4,
                    linewidth=1.2,
                    label=label,
                )
    else:
        # Plot all operators separately
        op_labels = {
            "U": FORMULAS_LINE["4"],
            "G": FORMULAS_LINE["5"],
            "F": FORMULAS_LINE["6"],
        }
        
        for semantic in SEMANTICS_LINE:
            for op, fid in OPERATOR_FORMULA_IDS.items():
                pairs = load_native_line(fid, semantic)
                if not pairs:
                    continue
                
                bounds, times = zip(*pairs)
                label = f"{LABELS_SHORT[semantic]}: {op_labels[op]}"
                ax.plot(
                    bounds, times,
                    color=COLORS[semantic],
                    marker=OPERATOR_MARKERS[op],
                    linestyle=OPERATOR_LINESTYLES[op],
                    markersize=4,
                    linewidth=1.2,
                    label=label,
                )

    # Configure axes and labels
    ax.set_xlabel("Temporal bound $b$")
    ax.set_ylabel("Time per sample ($\mu$s)")
    ax.set_title("Scalability with Temporal Depth")
    
    if USE_LOG_Y_AXIS:
        ax.set_yscale("log")
        ax.set_xscale("log")
    
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(alpha=0.3)

    if save:
        _save_figure(fig, "plot3_temporaldepth_scalability")
    
    if show:
        plt.show()
    
    return fig


if __name__ == "__main__":
    plot_line_semantics(save=True, show=True)
