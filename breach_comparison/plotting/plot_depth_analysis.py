"""Generate depth analysis plots showing formula complexity effects."""

import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

from plotting_utils import (
    load_benchmark_data,
    load_formulas,
    create_config_label,
    ensure_output_folder,
    extract_depth_from_short_name,
    FONT_SIZE_TITLE,
    FONT_SIZE_LEGEND,
    FONT_SIZE_LABEL,
)


def generate_depth_analysis(
    own_csv_path="results/own/benchmark_results_own.csv",
    formulas_csv_path="results/own/formulas_own.csv",
    output_folder="plots",
):
    """Generate plots showing execution time vs formula depth complexity.

    Args:
        own_csv_path: Path to own benchmark results CSV.
        formulas_csv_path: Path to formulas CSV.
        output_folder: Output folder for plots.
    """
    print("Loading data...")

    # Load data
    df_own = load_benchmark_data(own_csv_path)
    df_own = create_config_label(df_own)

    # Load formulas
    _, id_to_short_name, formula_to_id, _ = load_formulas(formulas_csv_path)

    ensure_output_folder(output_folder)

    # Add depth and short_name to dataframe
    df_own["short_name"] = df_own["formula"].map(
        lambda f: (
            id_to_short_name.get(formula_to_id.get(f)) if formula_to_id.get(f) else None
        )
    )
    df_own["depth"] = df_own["short_name"].apply(extract_depth_from_short_name)

    # Define formula groups based on formula patterns
    # Branching: formulas 13-15 with ∧ structure
    mask_branching = df_own["short_name"].str.contains(
        "branching|∧", regex=True, na=False
    )
    # Alternating: formulas 16-18 with F[0, 10](G[0, 10]... structure
    mask_alternating = df_own["short_name"].str.contains(
        r"alternating|F\[0, 10\]\(G\[0, 10\]", regex=True, na=False
    )
    # Until: formulas 19-21 with ((x>0)U[0, 10](x>0))U[0, 10]... structure
    mask_until = df_own["short_name"].str.contains(
        r"until|U\[0, 10\]", regex=True, na=False
    ) & df_own["short_name"].str.contains("Depth", na=False)

    df_branching = df_own[mask_branching].copy()
    df_alternating = df_own[mask_alternating].copy()
    df_until = df_own[mask_until].copy()

    # Remove rows with NaN time or depth
    for df in [df_branching, df_alternating, df_until]:
        df.dropna(subset=["time_s", "depth"], inplace=True)
        df["depth"] = df["depth"].astype(int)

    print(f"Branching formulas: {len(df_branching)} records")
    print(f"Alternating formulas: {len(df_alternating)} records")
    print(f"Until formulas: {len(df_until)} records")

    # Get union of all signal sizes
    all_sizes = set()
    for df in [df_branching, df_alternating, df_until]:
        all_sizes.update(df["sizeN"].unique())

    unique_sizes = sorted(all_sizes)
    print(f"Signal Sizes: {unique_sizes}")

    for size in unique_sizes:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        data_groups = [
            (df_branching, "Branching Structure", axes[0]),
            (df_alternating, "Alternating F/G Structure", axes[1]),
            (df_until, "Until-Nesting Structure", axes[2]),
        ]

        for df_group, group_title, ax in data_groups:
            
            df_size = df_group[df_group["sizeN"] == size]

            if not df_size.empty:
                pivot_data = df_size.pivot_table(
                    index="depth", columns="config", values="time_s", aggfunc="mean"
                )
                pivot_data = pivot_data.sort_index()

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

                ax.set_title(
                    f"{group_title}\n(Signal Size: {size})",
                    fontsize=FONT_SIZE_TITLE,
                    fontweight="bold",
                )
                ax.set_xlabel("Depth", fontsize=FONT_SIZE_LABEL)
                ax.set_ylabel("Execution Time (s)", fontsize=FONT_SIZE_LABEL)
                ax.set_yscale("log")
                # ax.set_xscale("log")
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
        out_path = os.path.join(output_folder, f"depth_effect_size_{size}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {out_path}")

    print("\nDepth analysis complete.")


if __name__ == "__main__":
    generate_depth_analysis()
