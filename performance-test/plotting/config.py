"""
Configuration for performance evaluation plots.

Provides centralized settings for colors, labels, markers, and styling
applied consistently across all plots. Modify settings here to update
all visualizations globally.
"""

from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================================
# Display and Output Configuration
# ============================================================================

# Figure dimensions and resolution
FIGURE_WIDTH = 8
FIGURE_HEIGHT = 6
DPI = 1000

# Font sizes (in points)
FONT_SIZE_BASE = 14
FONT_SIZE_TITLE = 16
FONT_SIZE_TICK = 11
FONT_SIZE_LEGEND = 11

# Font configuration: "serif" for LaTeX-like appearance (default)
# Alternative: "sans-serif" for modern appearance
FONT_FAMILY = "serif"

# ============================================================================
# Matplotlib Style Setup
# ============================================================================

def _configure_matplotlib() -> None:
    """Configure matplotlib for consistent, publication-quality plots."""
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans"],
        "font.serif": ["DejaVu Serif"],
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE_BASE,
        "axes.titlesize": FONT_SIZE_TITLE,
        "axes.labelsize": FONT_SIZE_BASE,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": True,
    })

_configure_matplotlib()

# ============================================================================
# Semantics Configuration (colors, labels, markers)
# ============================================================================

# Color mapping for semantics (primary visual encoding)
COLORS = {
    "DelayedQuantitative": "#1f77b4",
    "DelayedQualitative": "#2ca02c",
    "EagerQualitative": "#ff7f0e",
    "Rosi": "#9467bd",
    "rtamt": "#d62728",
}

# Full semantic names (for legends)
LABELS_FULL = {
    "DelayedQuantitative": "Delayed Quantitative",
    "DelayedQualitative": "Delayed Qualitative",
    "EagerQualitative": "Eager Qualitative",
    "Rosi": "RoSI",
    "rtamt": "RTAMT",
}

# Abbreviated labels (for compact display)
LABELS_SHORT = {
    "DelayedQuantitative": "Del. Quant.",
    "DelayedQualitative": "Del. Qual.",
    "EagerQualitative": "Eager Qual.",
    "Rosi": "RoSI",
    "rtamt": "RTAMT",
}

# Operator styling (marker and line style for temporal operators)
OPERATOR_MARKERS = {
    "G": "o",     # circle
    "F": "s",     # square
    "U": "^",     # triangle
}

OPERATOR_LINESTYLES = {
    "G": "-",     # solid
    "F": "--",    # dashed
    "U": "-.",    # dash-dot
}

OPERATOR_LABELS = {
    "G": r"$\mathbf{G}[0,b]$",
    "F": r"$\mathbf{F}[0,b]$",
    "U": r"$\mathbf{U}[0,b]$",
}

# ============================================================================
# Plotting Options
# ============================================================================

# Average F and G operator results together (reduces visual clutter)
USE_AVERAGING_FORMULA = True

# Use logarithmic y-axis
USE_LOG_Y_AXIS = False

# ============================================================================
# Formula Definitions
# ============================================================================

FORMULAS = {
    1: r"$\varphi_1$: $(x < 0.5) \wedge (x > -0.5)$",
    2: r"$\varphi_2$: $\mathbf{G}_{[0,1000]}(x > 0.5 \to \mathbf{F}_{[0,100]}(x < 0))$",
    3: r"$\varphi_3$: $\mathbf{G}_{[0,100]}(x < 0.5) \vee \mathbf{G}_{[100,150]}(x > 0)$",
}

FORMULA_NAMES = {
    1: r"$\varphi_1$",
    2: r"$\varphi_2$",
    3: r"$\varphi_3$",
}

# Detailed formula descriptions for line charts (operators with varying bounds)
FORMULAS_LINE = {
    "4": r"$\psi_1 \mathbf{U}_{[0,b]}\psi_2$",      # Until with inner predicates
    "5": r"$\mathbf{G}_{[0,b]}\psi_1$",              # Globally with inner predicate
    "6": r"$\mathbf{F}_{[0,b]}\psi_1$",                # Eventually with inner predicate
}

FORMULA_SHORT_LINE = {
    "4": r"$\mathbf{U}$",
    "5": r"$\mathbf{G}$",
    "6": r"$\mathbf{F}$",
}

# ============================================================================
# Python Binding Visualization
# ============================================================================

# Python binding overhead appearance in stacked bar charts
PYTHON_OVERHEAD_ALPHA = 0.45
PYTHON_OVERHEAD_HATCH = "//"
PYTHON_OVERHEAD_LABEL = "Python binding overhead"

# ============================================================================
# Data Organization
# ============================================================================

# Semantics to display in bar charts
SEMANTICS_BAR = [
    "DelayedQuantitative",
    "DelayedQualitative",
    "EagerQualitative",
    "rtamt",
]

# Semantics to display in line charts (excluding RoSI)
SEMANTICS_LINE = [
    "DelayedQuantitative",
    "DelayedQualitative",
    "EagerQualitative",
]

# Benchmark formula IDs for each operator
OPERATOR_FORMULA_IDS = {
    "U": "4",
    "G": "5",
    "F": "6",
}

# ============================================================================
# Output Configuration
# ============================================================================

OUTPUT_DIR = "figures"


def get_output_dir() -> Path:
    """Get output directory path, creating it if necessary."""
    path = Path(OUTPUT_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_figure_size(plot_type: str = "default") -> tuple:
    """Get figure size for a specific plot type.
    
    Args:
        plot_type: Type of plot ("bar" or "line", defaults to "default").
        
    Returns:
        Tuple of (width, height) in inches.
    """
    return (FIGURE_WIDTH, FIGURE_HEIGHT)
