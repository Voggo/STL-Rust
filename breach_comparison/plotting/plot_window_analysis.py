"""Generate window size analysis plots showing temporal operator behavior."""

import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

from plotting_utils import (
    load_benchmark_data,
    create_config_label,
    ensure_output_folder,
    extract_window_size,
    extract_operator_type,
    FONT_SIZE_TITLE,
    FONT_SIZE_LEGEND,
    FONT_SIZE_LABEL,
)


def generate_window_analysis(
    own_csv_path="results/own/benchmark_results_own.csv",
    output_folder="plots",
):
    """Generate plots showing execution time vs window size for temporal operators.

    Args:
        own_csv_path: Path to own benchmark results CSV.
        output_folder: Output folder for plots.
    """
    print("Loading data...")

    # Load data
    df_own = load_benchmark_data(own_csv_path)
    df_own = create_config_label(df_own)

    ensure_output_folder(output_folder)

    # Extract metadata
    df_own["window"] = df_own["formula"].apply(extract_window_size)
    df_own["operator"] = df_own["formula"].apply(extract_operator_type)

    # Filter and clean
    df_windowed = df_own.dropna(subset=["window", "operator", "time_s"]).copy()
    df_windowed["window"] = df_windowed["window"].astype(int)

    # Filter to only simple window-based formulas (G[...], F[...], or base U[...])
    df_windowed = df_windowed[
        (df_windowed["formula"].str.startswith("G["))
        | (df_windowed["formula"].str.startswith("F["))
        | (
            (df_windowed["formula"].str.startswith("("))
            & ~(df_windowed["formula"].str.contains(r"\) U\[.*\) U\[", regex=True))
            & (df_windowed["formula"].str.contains(r"\) U\[", regex=True))
        )
    ].copy()

    if df_windowed.empty:
        print("No valid windowed formulas found.")
        return

    print(f"\nFound {len(df_windowed)} records.")
    print(f"Signal Sizes: {sorted(df_windowed['sizeN'].unique())}")
    print(f"Window Sizes: {sorted(df_windowed['window'].unique())}")

    # Generate plots
    unique_sizes = sorted(df_windowed["sizeN"].unique())

    for size in unique_sizes:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        df_size = df_windowed[df_windowed["sizeN"] == size]
        operators = ["G (Global)", "F (Eventually)", "U (Until)"]

        for idx, op in enumerate(operators):
            ax = axes[idx]
            df_op = df_size[df_size["operator"] == op]

            pivot_data = df_op.pivot_table(
                index="window", columns="config", values="time_s", aggfunc="mean"
            )
            pivot_data = pivot_data.sort_index()

            if not pivot_data.empty:
                markers = ["o", "s", "^", "D", "v", "p", "x", "*"]
                colors = plt.cm.tab10(np.linspace(0, 1, len(pivot_data.columns)))

                for i, col in enumerate(pivot_data.columns):
                    series = pivot_data[col].dropna()
                    ax.plot(
                        series.index,
                        series.values,
                        marker=markers[i % len(markers)],
                        label=col,
                        color=colors[i % len(colors)],
                        linewidth=2,
                        markersize=8,
                        alpha=0.8,
                    )

                # operators = ["G (Global)", "F (Eventually)", "U (Until)"]

                if op == "G (Global)":
                    op_label = "\u25A1 (Globally)"  # Unicode white square
                elif op == "F (Eventually)":
                    op_label = "\u25C7 (Eventually)"  # Unicode white diamond
                else:
                    op_label = "U (Until)"
                ax.set_title(
                    f"{op_label} - Window Size Effect\n(Signal Size: {size})",
                    fontsize=FONT_SIZE_TITLE,
                    fontweight="bold",
                )
                ax.set_xlabel("Window Size", fontsize=FONT_SIZE_LABEL)
                ax.set_ylabel("Execution Time (s)", fontsize=FONT_SIZE_LABEL)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xticks(pivot_data.index)
                ax.get_xaxis().set_major_formatter(ScalarFormatter())
                ax.grid(True, which="both", linestyle="--", alpha=0.4)
                ax.legend(fontsize=FONT_SIZE_LEGEND, loc="best")
            else:
                ax.text(0.5, 0.5, "No Data", ha="center", transform=ax.transAxes)

        # Set equal y-axis limits
        y_limits = [
            min(ax.get_ylim()[0] for ax in axes),
            max(ax.get_ylim()[1] for ax in axes),
        ]
        for ax in axes:
            ax.set_ylim(y_limits)

        plt.tight_layout()
        out_path = os.path.join(output_folder, f"window_effect_size_{size}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {out_path}")

    print("\nWindow analysis complete.")


if __name__ == "__main__":
    generate_window_analysis()
