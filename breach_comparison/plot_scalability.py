"""Generate scalability analysis plots showing execution time vs signal size."""

import os
import matplotlib.pyplot as plt
import numpy as np

from plotting_utils import (
    load_benchmark_data,
    create_config_label,
    ensure_output_folder,
)


def generate_scalability(
    own_csv_path="results/own/benchmark_results_own.csv",
    output_folder="plots",
):
    """Generate scalability plots showing execution time vs signal size.

    Args:
        own_csv_path: Path to own benchmark results CSV.
        output_folder: Output folder for plots.
    """
    print("Loading data...")

    # Load data
    df_own = load_benchmark_data(own_csv_path)
    df_own = create_config_label(df_own)

    ensure_output_folder(output_folder)

    # SCALABILITY ANALYSIS: Execution time vs Signal Size for each configuration
    print("\n1. Generating scalability plots...")
    unique_configs = sorted(df_own["config"].unique())

    # 1a. Scalability plot for Incremental-only configurations
    fig, ax = plt.subplots(figsize=(12, 7))

    markers = ["o", "s", "^", "D", "v", "p"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_configs)))

    incremental_configs = [c for c in unique_configs if c.startswith("Incremental")]

    # Find formulas that have data for ALL incremental configs
    formulas_all_incremental = []
    for formula in df_own["formula"].unique():
        formula_data = df_own[df_own["formula"] == formula]
        configs_present = set(
            formula_data[formula_data["config"].isin(incremental_configs)][
                "config"
            ].unique()
        )
        if len(configs_present) == len(incremental_configs):
            formulas_all_incremental.append(formula)

    # Filter to only formulas with complete data
    df_incremental_complete = df_own[df_own["formula"].isin(formulas_all_incremental)]

    for i, config in enumerate(incremental_configs):
        config_data = (
            df_incremental_complete[df_incremental_complete["config"] == config]
            .groupby("sizeN")["time_s"]
            .mean()
            .sort_index()
        )
        ax.plot(
            config_data.index,
            config_data.values,
            marker=markers[i % len(markers)],
            label=config,
            color=colors[i],
            linewidth=2,
            markersize=8,
            alpha=0.8,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Signal Size (N)", fontsize=12, fontweight="bold")
    ax.set_ylabel(
        "Mean Execution Time (seconds, log scale)", fontsize=12, fontweight="bold"
    )
    ax.set_title(
        f"Scalability Analysis - Incremental Configurations\n(Averaged across {len(formulas_all_incremental)} formulas with complete data)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    out_path = os.path.join(output_folder, "scalability_incremental_only.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {out_path}")

    # 1b. Scalability plot comparing Incremental vs Naive
    fig, ax = plt.subplots(figsize=(12, 7))

    # Find formulas that have data for both approaches at all signal sizes
    all_sizes = sorted(df_own["sizeN"].unique())
    formulas_with_both = []
    for formula in df_own["formula"].unique():
        has_all = True
        for size in all_sizes:
            formula_size_data = df_own[
                (df_own["formula"] == formula) & (df_own["sizeN"] == size)
            ]
            approaches_present = set(formula_size_data["approach"].unique())
            if (
                "Incremental" not in approaches_present
                or "Naive" not in approaches_present
            ):
                has_all = False
                break
        if has_all:
            formulas_with_both.append(formula)

    n_common = len(formulas_with_both)
    common_data = df_own[df_own["formula"].isin(formulas_with_both)]

    for i, config in enumerate(unique_configs):
        config_data = (
            common_data[common_data["config"] == config]
            .groupby("sizeN")["time_s"]
            .mean()
            .sort_index()
        )
        if not config_data.empty:
            ax.plot(
                config_data.index,
                config_data.values,
                marker=markers[i % len(markers)],
                label=config,
                color=colors[i],
                linewidth=2,
                markersize=8,
                alpha=0.8,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Signal Size (N)", fontsize=12, fontweight="bold")
    ax.set_ylabel(
        "Mean Execution Time (seconds, log scale)", fontsize=12, fontweight="bold"
    )
    ax.set_title(
        f"Scalability Analysis - All Configurations\n(Averaged across {n_common} formulas tested by both approaches)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    out_path = os.path.join(output_folder, "scalability_all_configs.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {out_path}")

    print("\nScalability analysis complete.")


if __name__ == "__main__":
    generate_scalability()
