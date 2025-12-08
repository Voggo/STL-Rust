import pandas as pd
import matplotlib.pyplot as plt
import os
import re

def extract_own_time(stats_str):
    """Parses the time in seconds from the statistics string."""
    if pd.isna(stats_str):
        return None
    # Look for "mean=X" where X is in nanoseconds
    match = re.search(r"mean=([0-9\.]+)", str(stats_str))
    if match:
        return float(match.group(1)) * 1e-9  # Convert ns to seconds
    return None


def make_own_legend(row):
    """Creates a readable legend label for 'Our' results."""
    app = str(row["approach"]).lower().replace("incremental", "inc")
    mode = str(row["eval_mode"]).lower()
    out = str(row["out_type"]).lower()
    return f"{app}/{mode}/{out}"


def plot_dataset(ax, df, color, label_col, title_prefix=""):
    """Helper function to plot a dataset onto an axes."""
    if df.empty:
        return False

    # distinct markers for different configurations
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "X", "d"]

    groups = df.groupby(label_col)
    for i, (label, group) in enumerate(groups):
        # Sort by size for correct line connection
        group = group.sort_values("sizeN")

        # Determine marker
        marker = markers[i % len(markers)]

        ax.plot(
            group["sizeN"],
            group["time_s"],
            color=color,
            linestyle=":",
            marker=marker,
            label=f"{title_prefix}{label}",
            markersize=6,
        )
    return True


def generate_plots():
    # --- Configuration ---
    own_csv_path = os.path.join("results", "own", "benchmark_results_own.csv")
    breach_csv_path = os.path.join(
        "results", "breach", "breach_results_all_updated.csv"
    )
    base_output_folder = "plots"

    print("Loading data...")

    # 1. Load "Own" Data
    try:
        df_own = pd.read_csv(own_csv_path)
        df_own["time_s"] = df_own["statistics"].apply(extract_own_time)
        df_own["legend_label"] = df_own.apply(make_own_legend, axis=1)
    except Exception as e:
        print(f"Error processing Own results: {e}")
        df_own = pd.DataFrame()

    # 2. Load "Breach" Data
    try:
        df_breach = pd.read_csv(breach_csv_path, skipinitialspace=True)
        # Breach time is already in seconds
        df_breach["time_s"] = df_breach["mean_elapsed_s"]
        # Legend is the approach column
        df_breach["legend_label"] = df_breach["approach"]
    except Exception as e:
        print(f"Error processing Breach results: {e}")
        df_breach = pd.DataFrame()

    # 3. Identify Unique Formulas
    own_formulas = (
        set(df_own["formula"].dropna().unique()) if not df_own.empty else set()
    )
    breach_formulas = (
        set(df_breach["formula"].dropna().unique()) if not df_breach.empty else set()
    )
    all_formulas = own_formulas.union(breach_formulas)

    print(f"Found {len(all_formulas)} unique formulas in union between {len(own_formulas)} (Own) and {len(breach_formulas)} (Breach).")


    formulas = pd.read_csv(
        os.path.join("results", "own", "formulas_own.csv")
    )
    formula_dict = {row["id"]: row["formula"] for _, row in formulas.iterrows()}
    short_name_dict = {row["id"]: row["short_name"] for _, row in formulas.iterrows()}


    # 4. Generate Plots for each formula
    for formula in all_formulas:
        if formula == "Unknown Formula":
            continue

        f_id = [k for k, v in formula_dict.items() if v == formula][0]

        # Prepare folder for this formula
        safe_name = "formula_" + "_" + str(
            f_id
        )
        formula_folder = os.path.join(base_output_folder, safe_name)
        os.makedirs(formula_folder, exist_ok=True)

        # Filter data
        data_own = (
            df_own[df_own["formula"] == formula] if not df_own.empty else pd.DataFrame()
        )
        data_breach = (
            df_breach[df_breach["formula"] == formula]
            if not df_breach.empty
            else pd.DataFrame()
        )

        # Define the 3 plot configurations
        # (Filename, Title Suffix, Include_Breach, Include_Own)
        plot_configs = [
            ("breach.png", "Breach Results", True, False),
            ("ours.png", "Our Results", False, True),
            ("comparison.png", "Comparison", True, True),
        ]

        for filename, title_suffix, inc_breach, inc_own in plot_configs:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))

            has_data = False

            # Plot Breach (Red)
            if inc_breach and not data_breach.empty:
                if plot_dataset(
                    ax,
                    data_breach,
                    "tab:red",
                    "legend_label",
                    title_prefix="Breach: " if inc_own else "",
                ):
                    has_data = True

            # Plot Ours (Blue)
            if inc_own and not data_own.empty:
                if plot_dataset(
                    ax,
                    data_own,
                    "tab:blue",
                    "legend_label",
                    title_prefix="Ours: " if inc_breach else "",
                ):
                    has_data = True

            if has_data:
                ax.set_xlabel("Signal Size (N)", fontsize=12)
                ax.set_ylabel("Time (seconds)", fontsize=12)
                ax.set_title(f"{short_name_dict[f_id]} - {title_suffix}", fontsize=10, wrap=True)
                ax.legend()
                ax.grid(True, linestyle="--", alpha=0.7)

                # Save
                out_path = os.path.join(formula_folder, filename)
                plt.tight_layout()
                plt.savefig(out_path, dpi=300)
                plt.close(fig)
                # print(f"Saved: {out_path}")
            else:
                plt.close(fig)

    print("\nProcessing complete.")


if __name__ == "__main__":
    generate_plots()
