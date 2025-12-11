"""Generate grouped bar charts for all formulas and configurations."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from plotting_utils import (
    load_benchmark_data,
    load_formulas,
    create_config_label,
    ensure_output_folder,
)


def generate_bar_charts(
    own_csv_path="results/own/benchmark_results_own.csv",
    formulas_csv_path="results/own/formulas_own.csv",
    output_folder="plots",
):
    """Generate a grouped bar chart for all formulas with all configurations.

    Args:
        own_csv_path: Path to own benchmark results CSV.
        formulas_csv_path: Path to formulas CSV.
        output_folder: Output folder for plots.
    """
    print("Loading data...")

    # Load data
    df_own = load_benchmark_data(own_csv_path)
    df_own = create_config_label(df_own)

    _, id_to_short_name, formula_to_id, formula_order = load_formulas(
        formulas_csv_path
    )

    # Get unique formulas and signal sizes
    unique_formulas = sorted(
        df_own["formula"].unique(), key=lambda x: formula_order.get(x, 999)
    )
    unique_sizes = sorted(df_own["sizeN"].unique())
    unique_configs = sorted(df_own["config"].unique())

    print(
        f"Found {len(unique_formulas)} unique formulas, {len(unique_configs)} configurations, "
        f"and {len(unique_sizes)} signal sizes"
    )

    ensure_output_folder(output_folder)

    # Generate a chart for each signal size
    for size in unique_sizes:
        print(f"\nProcessing signal size: {size}")

        # Filter data for this signal size
        df_size = df_own[df_own["sizeN"] == size]

        # Prepare data: calculate mean time for each formula-config pair
        chart_data = {config: [] for config in unique_configs}
        formula_labels = []

        for formula in unique_formulas:
            if formula == "Unknown Formula" or pd.isna(formula):
                continue

            # Get formula ID for short name
            short_name = formula_to_id.get(formula)
            if short_name and short_name in id_to_short_name:
                short_name = id_to_short_name[short_name]
            else:
                short_name = formula

            formula_labels.append(short_name)

            # Get mean time for each config for this formula
            formula_data = df_size[df_size["formula"] == formula]
            for config in unique_configs:
                config_data = formula_data[formula_data["config"] == config]
                if not config_data.empty:
                    mean_time = config_data["time_s"].mean()
                    chart_data[config].append(mean_time)
                else:
                    chart_data[config].append(np.nan)

        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(20, 8))

        x = np.arange(len(formula_labels))
        n_configs = len(unique_configs)
        width = 0.8 / n_configs

        # Color palette for configurations
        colors = plt.cm.tab20(np.linspace(0, 1, n_configs))

        for i, config in enumerate(unique_configs):
            offset = (i - n_configs / 2 + 0.5) * width
            ax.bar(
                x + offset,
                chart_data[config],
                width,
                label=config,
                color=colors[i],
                alpha=0.85,
                edgecolor="black",
                linewidth=0.5,
            )

        # Set log scale on y-axis
        ax.set_yscale("log")

        # Labels and title
        ax.set_xlabel("Formula", fontsize=13, fontweight="bold")
        ax.set_ylabel(
            "Mean Execution Time (seconds, log scale)", fontsize=13, fontweight="bold"
        )
        ax.set_title(
            f"All Configurations - Signal Size: {size}\n(Per-formula performance)",
            fontsize=14,
            fontweight="bold",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(formula_labels, rotation=45, ha="right", fontsize=9)

        ax.legend(fontsize=9, loc="upper left", ncol=2)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

        # Save
        out_path = os.path.join(output_folder, f"bar_chart_size_{size}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    print("\nBar chart generation complete.")


if __name__ == "__main__":
    generate_bar_charts()
