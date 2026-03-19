# CANARIFY Python (`ostl-python`)

Python bindings for CANARIFY, the `ostl` online Signal Temporal Logic (STL) monitoring engine.

- PyPI package: `ostl-python`
- Python import name: `ostl_python`
- Rust core crate: `ostl`

## What this package provides

`ostl-python` exposes the high-performance Rust monitoring engine through a Pythonic API so you can parse STL formulas and evaluate streaming signals from Python applications and notebooks.

Supported monitoring semantics include:

- Delayed Qualitative
- Delayed Quantitative
- Eager Qualitative
- RoSI (Robust Satisfaction Intervals)

## Installation

From PyPI:

```bash
pip install ostl-python
```

## Quick Start

```python
import ostl_python as ostl

phi = ostl.parse_formula("G[0, 10](x > 5)")
monitor = ostl.Monitor(phi, semantics="DelayedQuantitative")

output = monitor.update("x", 6.0, 0.5)
print(output)
print(output.to_dict())
```

## Project links

- Repository: [github.com/Voggo/STL-Rust](https://github.com/Voggo/STL-Rust)
- Documentation: [voggo.github.io/STL-Rust](https://voggo.github.io/STL-Rust/)
- Rust API docs: [docs.rs/ostl](https://docs.rs/ostl)

## License

This package is distributed under the INTO-CPS Association Public License (ICAPL), with GPL v3 as a supported subsidiary mode. See `LICENSE` in the repository root and `ICA-USAGE-MODE.txt` for the selected mode.

## Developer notes

For local development, wheel building, and docs generation instructions, see the repository documentation.
