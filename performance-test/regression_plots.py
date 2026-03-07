"""Plot regression fits for scalability benchmarks.

Creates publication-style figures with raw datapoints and fitted curves under
the requested policy:
- Non-RoSI semantics: constant for G/F, linear for U.
- RoSI semantics: linear for G/F, quadratic for U.

Each figure contains four panels:
- Native / Non-RoSI semantics
- Native / RoSI semantics
- Python / Non-RoSI semantics
- Python / RoSI semantics
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import regression_plot_config as cfg
from regression_analysis import (
    DEFAULT_OUTPUT_CSV,
    NATIVE_CSV,
    PYTHON_CSV,
    DataPoint,
    RegressionResult,
    load_scalability_points,
    run_regression,
    save_results_csv,
)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": cfg.FONT_FAMILY,
            "font.serif": cfg.FONT_SERIF,
            "font.sans-serif": cfg.FONT_SANS,
            "figure.dpi": cfg.DPI,
            "savefig.dpi": cfg.DPI,
            "savefig.bbox": "tight",
            "figure.constrained_layout.use": True,
        }
    )


def _predict_from_result(result: RegressionResult, x: np.ndarray) -> np.ndarray:
    return result.intercept + result.coef_x * x + result.coef_x2 * (x**2)


def _fmt_coef(value: float) -> str:
    if np.isclose(value, 0.0):
        return "0"

    mantissa_str, exp_str = f"{value:.2e}".split("e")
    mantissa = float(mantissa_str)
    exponent = int(exp_str)

    if exponent == 0:
        return f"{mantissa:.2f}"

    return f"{mantissa:.2f}\\times 10^{{{exponent}}}"


def _formula_label(result: RegressionResult) -> str:
    if result.model_degree == 0:
        return f"$y={_fmt_coef(result.intercept)}$"
    if result.model_degree == 1:
        return f"$y={_fmt_coef(result.intercept)} + {_fmt_coef(result.coef_x)}x$"
    return (
        f"$y={_fmt_coef(result.intercept)}"
        f" + {_fmt_coef(result.coef_x)}x"
        f" + {_fmt_coef(result.coef_x2)}x^2$"
    )


def _group_points(points: list[DataPoint]) -> dict[tuple[str, str, str], list[DataPoint]]:
    grouped: dict[tuple[str, str, str], list[DataPoint]] = defaultdict(list)
    for p in points:
        grouped[(p.source, p.semantics, p.operator)].append(p)
    for key in grouped:
        grouped[key].sort(key=lambda d: d.temporal_depth)
    return grouped


def _group_results(
    results: list[RegressionResult],
) -> dict[tuple[str, str, str], RegressionResult]:
    return {(r.source, r.semantics, r.operator): r for r in results}


def _plot_panel(
    ax: plt.Axes,
    source: str,
    semantics: list[str],
    points_by_key: dict[tuple[str, str, str], list[DataPoint]],
    result_by_key: dict[tuple[str, str, str], RegressionResult],
) -> None:
    for sem in semantics:
        for op in cfg.OPERATORS:
            key = (source, sem, op)
            points = points_by_key.get(key)
            fit = result_by_key.get(key)
            if not points or fit is None:
                continue

            x = np.array([p.temporal_depth for p in points], dtype=float)
            y = np.array([p.time_us for p in points], dtype=float)

            # Raw datapoints
            ax.scatter(
                x,
                y,
                s=cfg.POINT_SIZE,
                alpha=cfg.POINT_ALPHA,
                color=cfg.SEMANTICS_COLORS[sem],
                marker=cfg.OPERATOR_STYLE[op]["marker"],
            )

            # Fit curve
            x_fit = np.linspace(x.min(), x.max(), 200)
            y_fit = _predict_from_result(fit, x_fit)

            r2_text = "n/a" if fit.model_degree == 0 else f"{fit.r2:.3f}"
            label = (
                f"{cfg.SEMANTICS_SHORT[sem]}-{op} "
                f"({_formula_label(fit)}, $R^2$={r2_text}, RMSE={fit.rmse:.3f})"
            )
            ax.plot(
                x_fit,
                y_fit,
                color=cfg.SEMANTICS_COLORS[sem],
                linestyle=cfg.OPERATOR_STYLE[op]["linestyle"],
                linewidth=cfg.LINE_WIDTH,
                label=label,
            )

    ax.set_xlabel("Temporal bound $b$")
    ax.set_ylabel("Time per sample ($\\mu$s)")
    ax.grid(alpha=cfg.GRID_ALPHA)


def _plot_policy_figure(
    points: list[DataPoint],
    results: list[RegressionResult],
    output_stem: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=cfg.FIGSIZE)

    points_by_key = _group_points(points)
    result_by_key = _group_results(results)

    # Row 1: native
    _plot_panel(axes[0, 0], "native", cfg.NON_ROSI_SEMANTICS, points_by_key, result_by_key)
    _plot_panel(axes[0, 1], "native", cfg.ROSI_SEMANTICS, points_by_key, result_by_key)

    # Row 2: python
    _plot_panel(axes[1, 0], "python", cfg.NON_ROSI_SEMANTICS, points_by_key, result_by_key)
    _plot_panel(axes[1, 1], "python", cfg.ROSI_SEMANTICS, points_by_key, result_by_key)

    axes[0, 0].set_title("Native: Delayed/Eager semantics")
    axes[0, 1].set_title("Native: RoSI semantics")
    axes[1, 0].set_title("Python: Delayed/Eager semantics")
    axes[1, 1].set_title("Python: RoSI semantics")

    for ax in axes.flat:
        ax.legend(loc="upper left", fontsize=cfg.LEGEND_FONT, framealpha=0.9)

    policy_text = "Non-RoSI (G/F constant, U linear); RoSI (G/F linear, U quadratic)"

    fig.suptitle(f"Scalability Regression Fits — {policy_text}")

    out_dir = cfg.ensure_output_dir()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{output_stem}.{ext}")

    plt.close(fig)


def main() -> None:
    _configure_matplotlib()

    points = load_scalability_points(NATIVE_CSV, PYTHON_CSV)

    # Fixed policy: constant G/F for non-RoSI
    results = run_regression(native_csv=NATIVE_CSV, python_csv=PYTHON_CSV)
    save_results_csv(results, DEFAULT_OUTPUT_CSV)
    _plot_policy_figure(
        points=points,
        results=results,
        output_stem="regression_fits_constant_fg_non_rosi",
    )

    print(f"Saved regression CSV: {DEFAULT_OUTPUT_CSV}")
    print(f"Saved regression figures in: {cfg.ensure_output_dir()}")


if __name__ == "__main__":
    main()
