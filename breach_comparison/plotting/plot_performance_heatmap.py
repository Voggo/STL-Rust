"""Generate heatmap plots showing configuration performance across formulas."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.ma import masked_invalid

from plotting_utils import (
    load_benchmark_data,
    load_formulas,
    create_config_label,
    ensure_output_folder,
    FONT_SIZE_TITLE,
    FONT_SIZE_LEGEND,
    FONT_SIZE_LABEL,
)


def generate_heatmap(
    own_csv_path="results/own/benchmark_results_own.csv",
    formulas_csv_path="results/own/formulas_own.csv",
    output_folder="plots",
    signal_sizes=None,
):
    """Generate heatmap plots showing configuration performance across formulas.

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
    _, id_to_short_name, formula_to_id, formula_order = load_formulas(formulas_csv_path)

    ensure_output_folder(output_folder)

    # HEATMAP: Configuration performance across formulas
    print("\nGenerating heatmap of configurations across formulas...")

    unique_configs = sorted(df_own["config"].unique())

    # Sort formulas by ID order
    unique_formulas = sorted(
        df_own["formula"].unique(), key=lambda x: formula_order.get(x, 999)
    )

    unique_sizes = sorted(df_own["sizeN"].unique() if signal_sizes is None else signal_sizes)
    for size in unique_sizes:
        size_data = df_own[df_own["sizeN"] == size]

        # Create matrix: rows=formulas, cols=configurations
        heatmap_data = np.zeros((len(unique_formulas), len(unique_configs)))
        missing_mask = np.zeros((len(unique_formulas), len(unique_configs)), dtype=bool)

        for f_idx, formula in enumerate(unique_formulas):
            if formula == "Unknown Formula" or pd.isna(formula):
                continue

            formula_data = size_data[size_data["formula"] == formula]

            for c_idx, config in enumerate(unique_configs):
                config_data = formula_data[formula_data["config"] == config]
                if not config_data.empty:
                    heatmap_data[f_idx, c_idx] = config_data["time_s"].mean()
                else:
                    heatmap_data[f_idx, c_idx] = np.nan
                    missing_mask[f_idx, c_idx] = True

        # Get short formula names using ID-based lookup
        formula_labels = []
        for formula in unique_formulas:
            if formula == "Unknown Formula" or pd.isna(formula):
                continue
            f_id = formula_to_id.get(formula)
            short_name = id_to_short_name.get(f_id, formula) if f_id else formula
            formula_labels.append(short_name)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 14))

        # Use log scale for heatmap
        heatmap_display = np.log10(heatmap_data).copy()

        # Create masked array for missing entries

        heatmap_masked = masked_invalid(heatmap_display)

        im = ax.imshow(heatmap_masked, cmap="YlOrRd", aspect="auto")
        # Add grey color for missing data
        ax.imshow(
            np.where(missing_mask, 1, np.nan),
            cmap="Greys",
            aspect="auto",
            alpha=0.7,
            vmin=0,
            vmax=1,
        )

        ax.set_xticks(np.arange(len(unique_configs)))
        ax.set_yticks(np.arange(len(formula_labels)))
        ax.set_xticklabels(unique_configs, rotation=45, ha="right", fontsize=FONT_SIZE_LABEL)
        ax.set_yticklabels(formula_labels, fontsize=FONT_SIZE_LABEL)

        # Add 'X' markers for missing data
        for f_idx in range(len(formula_labels)):
            for c_idx in range(len(unique_configs)):
                if missing_mask[f_idx, c_idx]:
                    ax.text(
                        c_idx,
                        f_idx,
                        "✗",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=12,
                        fontweight="bold",
                    )

        ax.set_xlabel("Configuration", fontsize=FONT_SIZE_LABEL, fontweight="bold")
        ax.set_ylabel("Formula", fontsize=FONT_SIZE_LABEL, fontweight="bold")
        ax.set_title(
            f"Configuration Performance Heatmap (log scale) - Size {size}\n✗ indicates missing data (too slow to run)",
            fontsize=FONT_SIZE_TITLE,
            fontweight="bold",
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(
            "log10(Time in seconds)\n(Darkest = Missing)",
            fontsize=10,
            fontweight="bold",
        )

        out_path = os.path.join(output_folder, f"heatmap_size_{size}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved: {out_path}")

    print("\nHeatmap generation complete.")


if __name__ == "__main__":
    signal_sizes = [20000]
    generate_heatmap(signal_sizes=signal_sizes)
