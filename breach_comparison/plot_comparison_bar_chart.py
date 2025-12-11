"""Generate comparison bar charts between Breach and our approaches."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from plotting_utils import (
    load_benchmark_data,
    load_formulas,
    ensure_output_folder,
)


def generate_comparison_charts(
    own_csv_path="results/own/benchmark_results_own.csv",
    breach_csv_path="results/breach/breach_results_all_updated.csv",
    formulas_csv_path="results/own/formulas_own.csv",
    output_folder="plots",
):
    """Generate comparison bar charts between Breach and our approaches.

    Args:
        own_csv_path: Path to own benchmark results CSV.
        breach_csv_path: Path to Breach results CSV.
        formulas_csv_path: Path to formulas CSV.
        output_folder: Output folder for plots.
    """
    print("Loading data...")

    # Load data
    df_own = load_benchmark_data(own_csv_path)

    # Load Breach data (different time column name)
    df_breach = pd.read_csv(breach_csv_path, skipinitialspace=True)
    df_breach["time_s"] = df_breach["mean_elapsed_s"]

    # Load formulas
    _, id_to_short_name, formula_to_id, formula_order = load_formulas(formulas_csv_path)

    ensure_output_folder(output_folder)

    # Get unique formulas and signal sizes
    unique_formulas = sorted(
        df_own["formula"].unique(), key=lambda x: formula_order.get(x, 999)
    )
    unique_sizes = sorted(df_own["sizeN"].unique())

    print(
        f"Found {len(unique_formulas)} unique formulas and {len(unique_sizes)} signal sizes"
    )

    # Define comparison groups
    comparisons = [
        {
            "name": "online_vs_rosi",
            "title": "Breach Online vs Ours (Rosi)",
            "configurations": [
                ("Breach (online)", lambda df: df[df["approach"] == "online"]),
                (
                    "Ours (Rosi)",
                    lambda df: df[
                        (df["approach"] == "Incremental")
                        & (df["eval_mode"] == "Eager")
                        & (df["out_type"] == "rosi")
                    ],
                ),
            ],
            "colors": ["#d62728", "#1f77b4"],
        },
        {
            "name": "thom_classic_vs_incremental",
            "title": "Breach (Thom+Classic) vs Ours (Incremental Configurations)",
            "configurations": [
                ("Breach (thom)", lambda df: df[df["approach"] == "thom"]),
                ("Breach (classic)", lambda df: df[df["approach"] == "classic"]),
                (
                    "Ours (Inc Strict F64)",
                    lambda df: df[
                        (df["approach"] == "Incremental")
                        & (df["eval_mode"] == "Strict")
                        & (df["out_type"] == "f64")
                    ],
                ),
                (
                    "Ours (Inc Strict Bool)",
                    lambda df: df[
                        (df["approach"] == "Incremental")
                        & (df["eval_mode"] == "Strict")
                        & (df["out_type"] == "bool")
                    ],
                ),
            ],
            "colors": ["#d62728", "#ff6b6b", "#1f77b4", "#ff7f0e"],
        },
    ]

    # Generate charts for each comparison group
    for comp in comparisons:
        print(f"\nGenerating comparison: {comp['name']}")

        # Generate a chart for each signal size
        for size in unique_sizes:
            print(f"  Processing signal size: {size}")

            # Prepare data: calculate mean time for each formula-config pair
            chart_data = {}
            config_labels = [cfg[0] for cfg in comp["configurations"]]

            for config_label, config_filter in comp["configurations"]:
                chart_data[config_label] = []

            formula_labels = []

            for formula in unique_formulas:
                if formula == "Unknown Formula" or pd.isna(formula):
                    continue

                # Get formula ID for short name using ID-based lookup
                f_id = formula_to_id.get(formula)
                short_name = id_to_short_name.get(f_id, formula) if f_id else formula
                formula_labels.append(short_name)

                # Get data for this formula and size
                formula_own = df_own[
                    (df_own["formula"] == formula) & (df_own["sizeN"] == size)
                ]
                formula_breach = df_breach[
                    (df_breach["formula"] == formula) & (df_breach["sizeN"] == size)
                ]

                for config_label, config_filter in comp["configurations"]:
                    # Determine which dataframe to use
                    if "Breach" in config_label:
                        config_data = config_filter(formula_breach)
                    else:
                        config_data = config_filter(formula_own)

                    if not config_data.empty:
                        mean_time = config_data["time_s"].mean()
                        chart_data[config_label].append(mean_time)
                    else:
                        chart_data[config_label].append(np.nan)

            # Skip if no data
            if all(
                all(
                    np.isnan(v) if isinstance(v, (int, float)) else False
                    for v in chart_data[k]
                )
                for k in chart_data
            ):
                print(f"    Skipping: No data for {comp['name']} at size {size}")
                continue

            # Create grouped bar chart
            fig, ax = plt.subplots(figsize=(20, 8))

            x = np.arange(len(formula_labels))
            n_configs = len(config_labels)
            width = 0.8 / n_configs

            bars_list = []
            for i, config_label in enumerate(config_labels):
                offset = (i - n_configs / 2 + 0.5) * width
                bars = ax.bar(
                    x + offset,
                    chart_data[config_label],
                    width,
                    label=config_label,
                    color=comp["colors"][i],
                    alpha=0.85,
                    edgecolor="black",
                    linewidth=0.5,
                )
                bars_list.append(bars)

            # Set log scale on y-axis
            ax.set_yscale("log")

            # Labels and title
            ax.set_xlabel("Formula", fontsize=13, fontweight="bold")
            ax.set_ylabel(
                "Mean Execution Time (seconds, log scale)",
                fontsize=13,
                fontweight="bold",
            )
            ax.set_title(
                f"{comp['title']} - Signal Size: {size}\n(Per-formula performance)",
                fontsize=14,
                fontweight="bold",
            )

            ax.set_xticks(x)
            ax.set_xticklabels(formula_labels, rotation=45, ha="right", fontsize=9)

            ax.legend(fontsize=11, loc="upper left")
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)

            # Save
            out_path = os.path.join(
                output_folder, f"comparison_{comp['name']}_size_{size}.png"
            )
            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved: {out_path}")

    print("\nComparison chart generation complete.")


if __name__ == "__main__":
    generate_comparison_charts()
