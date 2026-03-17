import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ── Academic style ────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.linewidth": 1.2,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "legend.fontsize": 10,
        "legend.title_fontsize": 8,
        "figure.dpi": 150,
    }
)

# Default tuned for single-column page layout where the figure should span most
# of the horizontal text area without becoming cluttered.
FIG_SIZE = (6.6, 4.2)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Create final runtime regression plot")
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=root.parent / "results" / "paper_native_benchmark_results_final.csv",
        help="Path to native benchmark CSV",
    )
    parser.add_argument(
        "--regression-csv",
        type=Path,
        default=root / "regression_fit_results.csv",
        help="Path to regression-fit CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "final_plot.pdf",
        help="Output plot path",
    )
    parser.add_argument(
        "--fg-mode",
        choices=["global", "eventually", "average", "both"],
        default="global",
        help=(
            "How to handle formulas F/G: "
            "'global' shows only G, "
            "'eventually' shows only F, "
            "'average' shows F/G average, "
            "'both' shows F and G separately"
        ),
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=FIG_SIZE[0],
        help="Figure width in inches (default tuned for one-column page-width fit)",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=FIG_SIZE[1],
        help="Figure height in inches",
    )
    return parser.parse_args()


def eval_fit(p, x):
    return p["intercept"] + p["coef_b"] * x + p["coef_b2"] * x**2


def adjust_log_label_positions(y_values, y_min, y_max, min_delta=0.06):
    """Spread label y positions on a log-scale axis to reduce overlaps."""
    if len(y_values) == 0:
        return np.array([])

    y_logs = np.log10(np.asarray(y_values, dtype=float))
    order = np.argsort(y_logs)
    pos = y_logs[order].copy()

    lo = np.log10(y_min)
    hi = np.log10(y_max)
    margin = (hi - lo) * 0.02
    lo += margin
    hi -= margin

    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_delta)

    if len(pos) > 0 and pos[-1] > hi:
        pos[-1] = hi
        for i in range(len(pos) - 2, -1, -1):
            pos[i] = min(pos[i], pos[i + 1] - min_delta)

    if len(pos) > 0 and pos[0] < lo:
        pos[0] = lo
        for i in range(1, len(pos)):
            pos[i] = max(pos[i], pos[i - 1] + min_delta)

    adjusted = np.empty_like(y_logs)
    adjusted[order] = 10 ** pos
    return adjusted


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.benchmark_csv)
    df = df[df["formula_id"].isin([4, 5, 6])].copy()
    formula_operator = {4: "U", 5: "G", 6: "F"}
    df["operator"] = df["formula_id"].map(formula_operator)

    fits_orig = pd.read_csv(args.regression_csv)
    fits_orig = fits_orig[fits_orig["source"] == "native"]

    df_u = df[df["operator"] == "U"].copy()

    if args.fg_mode == "average":
        df_fg = (
            df[df["operator"].isin(["F", "G"])]
            .groupby(["semantics", "interval_len"], as_index=False)["avg_per_sample_us"]
            .mean()
        )
        df_fg["operator"] = "F/G"
        df_plot = pd.concat([df_u, df_fg], ignore_index=True)
        operators_to_plot = ["U", "F/G"]
    elif args.fg_mode == "both":
        df_plot = pd.concat([df_u, df[df["operator"].isin(["F", "G"])].copy()], ignore_index=True)
        operators_to_plot = ["U", "F", "G"]
    elif args.fg_mode == "eventually":
        df_plot = pd.concat([df_u, df[df["operator"] == "F"].copy()], ignore_index=True)
        operators_to_plot = ["U", "F"]
    else:  # global
        df_plot = pd.concat([df_u, df[df["operator"] == "G"].copy()], ignore_index=True)
        operators_to_plot = ["U", "G"]

    fit_params = {}

    for _, row in fits_orig[fits_orig["operator"] == "U"].iterrows():
        key = (row["semantics"], "U")
        fit_params[key] = row

    if args.fg_mode == "average":
        df_fg = df_plot[df_plot["operator"] == "F/G"].copy()
        for sem in df_plot["semantics"].unique():
            fg_rows = fits_orig[(fits_orig["semantics"] == sem) & (fits_orig["operator"].isin(["F", "G"]))]
            model_names = fg_rows["model_name"].unique()

            g_avg = df_fg[df_fg["semantics"] == sem].sort_values("interval_len")
            x_raw = g_avg["interval_len"].values
            y = g_avg["avg_per_sample_us"].values

            if len(model_names) == 1 and model_names[0] == "constant":
                avg_intercept = fg_rows["intercept"].mean()
                key = (sem, "F/G")
                fit_params[key] = {
                    "model_name": "constant",
                    "intercept": avg_intercept,
                    "coef_b": 0.0,
                    "coef_b2": 0.0,
                }
            else:
                X = x_raw.reshape(-1, 1)
                reg = LinearRegression().fit(X, y)
                key = (sem, "F/G")
                fit_params[key] = {
                    "model_name": "linear",
                    "intercept": reg.intercept_,
                    "coef_b": reg.coef_[0],
                    "coef_b2": 0.0,
                }
    else:
        fg_ops = [op for op in operators_to_plot if op in {"F", "G"}]
        for _, row in fits_orig[fits_orig["operator"].isin(fg_ops)].iterrows():
            key = (row["semantics"], row["operator"])
            fit_params[key] = row

    semantics_colors = {
        "DelayedQuantitative": "#64baaa",
        "DelayedQualitative": "#6bae48",
        "EagerQualitative": "#e52740",
        "Rosi": "#3c0701",
    }
    semantics_display = {
        "DelayedQuantitative": "Del. Quant.",
        "DelayedQualitative": "Del. Qual.",
        "EagerQualitative": "Eager Qual.",
        "Rosi": "RoSI",
    }
    operator_markers = {"U": "^", "F": "o", "G": "s", "F/G": "D"}
    operator_display = {"U": "U", "F": "F", "G": "G", "F/G": "F/G"}

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    direct_labels = []

    for (semantics, operator), group in df_plot.groupby(["semantics", "operator"]):
        g = group.sort_values("interval_len")
        color = semantics_colors[semantics]
        marker = operator_markers[operator]
        fit = fit_params.get((semantics, operator))

        ax.scatter(
            g["interval_len"],
            g["avg_per_sample_us"],
            color=color,
            marker=marker,
            s=40,
            zorder=3,
            linewidths=0.7,
            edgecolors="white",
            alpha=0.9,
        )

        if fit is not None:
            x_min, x_max = g["interval_len"].min(), g["interval_len"].max()
            if fit["model_name"] == "constant":
                ax.plot(
                    [x_min, x_max],
                    [fit["intercept"], fit["intercept"]],
                    color=color,
                    linewidth=1.8,
                    linestyle=":",
                    alpha=0.6,
                    zorder=2,
                )
            else:
                x_fit = np.linspace(x_min, x_max, 500)
                y_fit = eval_fit(fit, x_fit)
                y_data_min = g["avg_per_sample_us"].min()
                mask = y_fit >= y_data_min
                if mask.any():
                    ax.plot(
                        x_fit[mask],
                        y_fit[mask],
                        color=color,
                        linewidth=1.8,
                        linestyle=":",
                        alpha=0.6,
                        zorder=2,
                    )

        x_last = g["interval_len"].iloc[-1]
        y_last = g["avg_per_sample_us"].iloc[-1]
        direct_labels.append(
            {
                "x": x_last,
                "y": y_last,
                "semantics": semantics,
                "operator": operator,
                "label": f"{semantics_display[semantics]}, {operator_display[operator]}",
                "color": color,
            }
        )

    ax.set_yscale("log")
    ax.set_xlabel("Temporal depth ($b$)", labelpad=5)
    ax.set_ylabel("Average time per sample (\u00b5s, log scale)", labelpad=5)
    # ax.set_title("Performance scaling of temporal operators", pad=5)
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.55)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.tick_params(which="both", top=True, right=True, width=1.1)

    x_data_max = df_plot["interval_len"].max()
    x_label = x_data_max * 1.015
    x_right = x_data_max * 1.35
    ax.set_xlim(right=x_right)

    y_min, y_max = ax.get_ylim()

    # Spread non-RoSI labels per operator to avoid near-overlapping stacks,
    # especially for G labels which often cluster tightly.
    y_targets = {i: info["y"] for i, info in enumerate(direct_labels)}
    non_rosi_indices = [i for i, info in enumerate(direct_labels) if info["semantics"] != "RoSI"]
    operators = sorted({direct_labels[i]["operator"] for i in non_rosi_indices})
    for op in operators:
        idx = [i for i in non_rosi_indices if direct_labels[i]["operator"] == op]
        if not idx:
            continue
        y_vals = [direct_labels[i]["y"] for i in idx]
        min_delta = 0.4 if op == "G" else 0.3
        adjusted = adjust_log_label_positions(y_vals, y_min, y_max, min_delta=min_delta)
        for i, y_adj in zip(idx, adjusted):
            y_targets[i] = y_adj

    for i, label_info in enumerate(direct_labels):
        y_target = y_targets[i]
        if label_info["semantics"] == "Rosi":
            ax.annotate(
                label_info["label"],
                xy=(label_info["x"], label_info["y"]),
                xytext=(5, 0),
                textcoords="offset points",
                color=label_info["color"],
                fontsize=11,
                va="center",
                ha="left",
                clip_on=False,
                bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.75},
            )
        else:
            ax.annotate(
                label_info["label"],
                xy=(label_info["x"], label_info["y"]),
                xytext=(x_label, y_target),
                textcoords="data",
                color=label_info["color"],
                fontsize=11,
                va="center",
                ha="left",
                clip_on=False,
                bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.75},
                arrowprops={
                    "arrowstyle": "-",
                    "lw": 1.0,
                    "color": label_info["color"],
                    "alpha": 0.8,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=600, bbox_inches="tight")
    print(f"Plot saved successfully to: {args.output}")


if __name__ == "__main__":
    main()