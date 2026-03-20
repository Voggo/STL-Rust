# FeSTL Python (`festl-python`)

Python bindings for FeSTL, the `festl` online Signal Temporal Logic (STL) monitoring engine.

- PyPI package: `festl-python`
- Python import name: `festl_python`
- Rust core crate: `festl`

## What this package provides

`festl-python` exposes the high-performance Rust monitoring engine through a Pythonic API so you can parse STL formulas and evaluate streaming signals from Python applications and notebooks.

Supported monitoring semantics include:

- Delayed Qualitative
- Delayed Quantitative
- Eager Qualitative
- RoSI (Robust Satisfaction Intervals)

## Installation

From PyPI:

```bash
pip install festl-python
```

## Quick Start

```python
import festl_python as festl

phi = festl.parse_formula("G[0, 10](x > 5)")
monitor = festl.Monitor(phi, semantics="DelayedQuantitative")

output = monitor.update("x", 6.0, 0.5)
print(output)
print(output.to_dict())
```

## Project links

- Repository: [github.com/Voggo/FeSTL](https://github.com/Voggo/FeSTL)
- Documentation: [voggo.github.io/FeSTL](https://voggo.github.io/FeSTL/)
- Rust API docs: [docs.rs/festl](https://docs.rs/festl)

## License

This package is distributed under the INTO-CPS Association Public License (ICAPL), with GPL v3 as a supported subsidiary mode. See `LICENSE` in the repository root and `ICA-USAGE-MODE.txt` for the selected mode.

## Developer notes

For local development, wheel building, and docs generation instructions, see the repository documentation.
