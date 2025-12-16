"""Generate performance ranking plot showing fastest configuration per formula."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from plotting_utils import (
    load_benchmark_data,
    load_formulas,
    create_config_label,
    ensure_output_folder,
    FONT_SIZE_TITLE,
    FONT_SIZE_LEGEND,
    FONT_SIZE_LABEL,
)


def generate_ranking(
    own_csv_path="results/own/benchmark_results_own.csv",
    formulas_csv_path="results/own/formulas_own.csv",
    output_folder="plots",
):
    """Generate performance ranking plot showing fastest configuration per formula.

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
    _, _, _, formula_order = load_formulas(formulas_csv_path)

    ensure_output_folder(output_folder)

    # PERFORMANCE RANKING: Fastest configuration per formula
    print("\nGenerating performance ranking analysis...")

    unique_configs = sorted(df_own["config"].unique())

    fig, ax = plt.subplots(figsize=(14, 7))

    # Count which configuration is fastest for each formula at each size
    ranking_counts = {}
    for config in unique_configs:
        ranking_counts[config] = 0

    # Sort formulas by ID order
    all_formulas = sorted(
        df_own["formula"].unique(), key=lambda x: formula_order.get(x, 999)
    )

    for size in sorted(df_own["sizeN"].unique()):
        size_data = df_own[df_own["sizeN"] == size]

        for formula in all_formulas:
            if formula == "Unknown Formula" or pd.isna(formula):
                continue

            formula_data = size_data[size_data["formula"] == formula]

            # Find fastest config for this formula
            min_time = float("inf")
            fastest_config = None

            for config in unique_configs:
                config_data = formula_data[formula_data["config"] == config]
                if not config_data.empty:
                    mean_time = config_data["time_s"].mean()
                    if mean_time < min_time:
                        min_time = mean_time
                        fastest_config = config

            if fastest_config:
                ranking_counts[fastest_config] += 1

    configs_sorted = sorted(ranking_counts.items(), key=lambda x: x[1], reverse=True)
    configs_names = [c[0] for c in configs_sorted]
    configs_wins = [c[1] for c in configs_sorted]

    colors_ranking = plt.cm.viridis(np.linspace(0, 1, len(configs_names)))
    bars = ax.barh(
        configs_names,
        configs_wins,
        color=colors_ranking,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    ax.set_xlabel("Number of Formulas Where Fastest", fontsize=FONT_SIZE_LABEL, fontweight="bold")
    ax.set_title(
        "Performance Ranking - Fastest Configuration per Formula\n(Across all 21 formulas × 3 signal sizes = 63 tests)",
        fontsize=FONT_SIZE_TITLE,
        fontweight="bold",
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, configs_wins)):
        ax.text(
            val + 0.1, i, str(int(val)), va="center", fontsize=10, fontweight="bold"
        )

    out_path = os.path.join(output_folder, "performance_ranking.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {out_path}")

    print("\nPerformance ranking complete.")


if __name__ == "__main__":
    generate_ranking()
