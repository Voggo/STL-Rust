import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ── Academic style ────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 12,
        "axes.titlesize": 16,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.fontsize": 12,
        "legend.title_fontsize": 8,
        "figure.dpi": 150,
    }
)

FIG_SIZE = (14, 10)


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
    return parser.parse_args()


def fmt_model(name, intercept, b, b2, r2, rmse, is_constant):
    if is_constant:
        eq = f"$y = {intercept:.3f}$"
        return f"{eq}, RMSE={rmse:.3f}"
    if name == "linear":
        eq = f"$y \\approx {abs(b):.4f}b$" if b >= 0 else f"$y \\approx -{abs(b):.4f}b$"
    elif name == "quadratic":
        sb2 = "+" if b2 >= 0 else "-"
        eq = f"$y \\approx {abs(b):.4f}b {sb2} {abs(b2):.5f}b^2$"
    else:
        eq = name
    return f"{eq},  $R^2$={r2:.4f}"


def eval_fit(p, x):
    return p["intercept"] + p["coef_b"] * x + p["coef_b2"] * x**2


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.benchmark_csv)
    df = df[df["formula_id"].isin([4, 5, 6])].copy()
    formula_operator = {4: "U", 5: "G", 6: "F"}
    df["operator"] = df["formula_id"].map(formula_operator)

    fits_orig = pd.read_csv(args.regression_csv)
    fits_orig = fits_orig[fits_orig["source"] == "native"]

    df_fg = (
        df[df["operator"].isin(["F", "G"])]
        .groupby(["semantics", "interval_len"], as_index=False)["avg_per_sample_us"]
        .mean()
    )
    df_fg["operator"] = "F/G"

    df_u = df[df["operator"] == "U"].copy()
    df_plot = pd.concat([df_u, df_fg], ignore_index=True)

    fit_params = {}
    fit_info = {}

    for _, row in fits_orig[fits_orig["operator"] == "U"].iterrows():
        key = (row["semantics"], "U")
        fit_params[key] = row
        fit_info[key] = fmt_model(
            row["model_name"],
            row["intercept"],
            row["coef_b"],
            row["coef_b2"],
            row["r2"],
            row["rmse"],
            is_constant=(row["model_name"] == "constant"),
        )

    for sem in df_plot["semantics"].unique():
        fg_rows = fits_orig[(fits_orig["semantics"] == sem) & (fits_orig["operator"].isin(["F", "G"]))]
        model_names = fg_rows["model_name"].unique()

        g_avg = df_fg[df_fg["semantics"] == sem].sort_values("interval_len")
        x_raw = g_avg["interval_len"].values
        y = g_avg["avg_per_sample_us"].values

        if len(model_names) == 1 and model_names[0] == "constant":
            avg_intercept = fg_rows["intercept"].mean()
            avg_rmse = fg_rows["rmse"].mean()
            key = (sem, "F/G")
            fit_params[key] = {
                "model_name": "constant",
                "intercept": avg_intercept,
                "coef_b": 0.0,
                "coef_b2": 0.0,
                "r2": 0.0,
                "rmse": avg_rmse,
            }
            fit_info[key] = fmt_model("constant", avg_intercept, 0, 0, 0, avg_rmse, is_constant=True)
        else:
            X = x_raw.reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)
            r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))
            key = (sem, "F/G")
            fit_params[key] = {
                "model_name": "linear",
                "intercept": reg.intercept_,
                "coef_b": reg.coef_[0],
                "coef_b2": 0.0,
                "r2": r2,
                "rmse": rmse,
            }
            fit_info[key] = fmt_model("linear", reg.intercept_, reg.coef_[0], 0, r2, rmse, is_constant=False)

    semantics_colors = {
        "DelayedQuantitative": "#1f77b4",
        "DelayedQualitative": "#d62728",
        "EagerQualitative": "#2ca02c",
        "Rosi": "#ff7f0e",
    }
    operator_markers = {"U": "^", "F/G": "D"}

    fig, ax = plt.subplots(figsize=FIG_SIZE)

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
            s=18,
            zorder=3,
            linewidths=0.4,
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
                    linewidth=1.4,
                    linestyle=":",
                    alpha=0.5,
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
                        linewidth=1.4,
                        linestyle=":",
                        alpha=0.5,
                        zorder=2,
                    )

    ax.set_yscale("log")
    ax.set_xlabel("Temporal depth ($b$)", labelpad=6)
    ax.set_ylabel("Average time per sample (\u00b5s, log scale)", labelpad=6)
    ax.set_title("Runtime scaling of formulas 4\u20136 across semantics (native)", pad=10)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.3)
    ax.tick_params(which="both", top=True, right=True)

    op_label = {"U": "U", "F/G": "F/G avg."}
    handles = []
    for op in ["U", "F/G"]:
        for sem, color in semantics_colors.items():
            info = fit_info.get((sem, op), "—")
            label = f"{sem},  {op_label[op]}:  {info}"
            import matplotlib.colors as mcolors

            rgba = list(mcolors.to_rgba(color))
            line_color = tuple(rgba[:3] + [0.5])
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=line_color,
                    marker=operator_markers[op],
                    markersize=6,
                    linewidth=1.4,
                    linestyle=":",
                    markeredgewidth=0.4,
                    markeredgecolor="white",
                    markerfacecolor=color,
                    label=label,
                )
            )

    ax.legend(
        handles=handles,
        ncols=2,
        loc="upper right",
        framealpha=0.95,
        edgecolor="0.7",
        handlelength=2.2,
        borderpad=0.8,
        labelspacing=0.5,
        columnspacing=1.5,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=1000, bbox_inches="tight")
    print(f"Plot saved successfully to: {args.output}")


if __name__ == "__main__":
    main()