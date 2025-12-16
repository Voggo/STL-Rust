"""Generate approach comparison plots showing Incremental vs Naive performance."""

import os
import matplotlib.pyplot as plt

from plotting_utils import (
    load_benchmark_data,
    create_config_label,
    ensure_output_folder,
    FONT_SIZE_TITLE,
    FONT_SIZE_LEGEND,
    FONT_SIZE_LABEL,
)


def generate_approach_comparison(
    own_csv_path="results/own/benchmark_results_own.csv",
    output_folder="plots",
):
    """Generate approach comparison plots for Incremental vs Naive.

    Args:
        own_csv_path: Path to own benchmark results CSV.
        output_folder: Output folder for plots.
    """
    print("Loading data...")

    # Load data
    df_own = load_benchmark_data(own_csv_path)
    df_own = create_config_label(df_own)

    ensure_output_folder(output_folder)

    # APPROACH COMPARISON: Incremental vs Naive across all output types and eval modes
    print("\nGenerating approach comparison plots...")

    for size in sorted(df_own["sizeN"].unique()):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        size_data = df_own[df_own["sizeN"] == size]

        # Find formulas that have data for BOTH approaches
        formulas_with_both = []
        for formula in size_data["formula"].unique():
            formula_data = size_data[size_data["formula"] == formula]
            approaches_present = set(formula_data["approach"].unique())
            if "Incremental" in approaches_present and "Naive" in approaches_present:
                formulas_with_both.append(formula)

        # Filter to only common formulas
        common_data = size_data[size_data["formula"].isin(formulas_with_both)]
        n_common = len(formulas_with_both)

        # Subplot 1: Time by approach and eval_mode
        ax = axes[0]

        # Create custom grouping for eval modes
        common_data_custom = common_data.copy()
        common_data_custom["eval_mode_custom"] = common_data_custom.apply(
            lambda row: (
                f"Eager ({row['out_type']})"
                if row["eval_mode"] == "Eager" and row["approach"] == "Incremental"
                else row["eval_mode"]
            ),
            axis=1,
        )

        pivot_approach_eval = (
            common_data_custom.groupby(["approach", "eval_mode_custom"])["time_s"]
            .mean()
            .unstack()
        )

        # Define colors
        color_map = {
            "Strict": "#1f77b4",  # Blue
            "Eager (bool)": "#ff7f0e",  # Orange
            "Eager (rosi)": "#ffbb78",  # Light orange
        }
        colors_eval = [
            color_map.get(col, "#1f77b4") for col in pivot_approach_eval.columns
        ]

        pivot_approach_eval.plot(
            kind="bar", ax=ax, color=colors_eval, alpha=0.8, edgecolor="black"
        )
        ax.set_yscale("log")
        ax.set_xlabel("Approach", fontsize=FONT_SIZE_LABEL, fontweight="bold")
        ax.set_ylabel(
            "Mean Execution Time (seconds, log scale)", fontsize=FONT_SIZE_LABEL, fontweight="bold"
        )
        ax.set_title(
            f"Approach & Eval Mode - Size {size}\n(Averaged across {n_common} formulas with both approaches)",
            fontsize=FONT_SIZE_TITLE,
            fontweight="bold",
        )
        ax.legend(title="Eval Mode", fontsize=FONT_SIZE_LEGEND)
        ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.4)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        eval_ylim = ax.get_ylim()

        # Subplot 2: Time by approach and output type
        ax = axes[1]
        pivot_approach_out = (
            common_data.groupby(["approach", "out_type"])["time_s"].mean().unstack()
        )

        # Define distinct colors for output types
        color_map_out = {
            "bool": "#2ca02c",  # Green
            "f64": "#9467bd",  # Purple
            "rosi": "#8c564b",  # Brown
        }
        colors_out = [
            color_map_out.get(col, "#1f77b4") for col in pivot_approach_out.columns
        ]

        pivot_approach_out.plot(
            kind="bar", ax=ax, color=colors_out, alpha=0.8, edgecolor="black"
        )
        ax.set_yscale("log")
        ax.set_xlabel("Approach", fontsize=FONT_SIZE_LABEL, fontweight="bold")
        ax.set_ylabel(
            "Mean Execution Time (seconds, log scale)", fontsize=FONT_SIZE_LABEL, fontweight="bold"
        )
        ax.set_title(
            f"Approach & Output Type - Size {size}\n(Averaged across {n_common} formulas with both approaches)",
            fontsize=FONT_SIZE_TITLE,
            fontweight="bold",
        )
        ax.legend(title="Output Type", fontsize=FONT_SIZE_LEGEND)
        ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.4)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        # Set equal y-axis limits
        axes[0].set_ylim(eval_ylim)
        axes[1].set_ylim(eval_ylim)

        out_path = os.path.join(output_folder, f"approach_comparison_size_{size}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved: {out_path}")

    print("\nApproach comparison complete.")


if __name__ == "__main__":
    generate_approach_comparison()
