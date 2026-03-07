#!/usr/bin/env python3
"""
Generate all four performance-comparison plots.

Usage:
    python plotting/plot_all.py          # save only
    python plotting/plot_all.py --show   # save + interactive display
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_bar_general import plot_bar_general
from plot_bar_rosi import plot_bar_rosi
from plot_line_semantics import plot_line_semantics
from plot_line_rosi import plot_line_rosi


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all performance plots.")
    parser.add_argument("--show", action="store_true", help="Display plots interactively.")
    args = parser.parse_args()

    show = args.show

    print("Generating Plot 1: Bar chart – General formulas (all semantics + rtamt)...")
    plot_bar_general(save=True, show=show)

    print("Generating Plot 2: Bar chart – General formulas (RoSI only)...")
    plot_bar_rosi(save=True, show=show)

    print("Generating Plot 3: Line chart – Scalability (Delayed & Eager semantics)...")
    plot_line_semantics(save=True, show=show)

    print("Generating Plot 4: Line chart – Scalability (RoSI semantics)...")
    plot_line_rosi(save=True, show=show)

    from config import OUTPUT_DIR
    out_dir = os.path.join(os.path.dirname(__file__), "..", OUTPUT_DIR)
    print(f"\nAll plots saved to {os.path.abspath(out_dir)}/")


if __name__ == "__main__":
    main()
