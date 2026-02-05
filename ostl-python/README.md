# OSTL Python Interface

Python bindings for the Online Signal Temporal Logic (STL) monitoring library.

## Installation

### Prerequisites

- Rust toolchain (install from [rustup.rs](https://rustup.rs/))
- Maturin ([install guide](https://github.com/PyO3/maturin))

### Steps

Build and install the package using maturin:

```bash
cd ostl-python
maturin develop --release
```

For production deployment:

```bash
maturin build --release
pip install target/wheels/ostl_python-*.whl
```

## Generating Documentation

### Prerequisites

Install `mkdocs` and `mkdocstrings-python`:

```bash
pip install mkdocs mkdocstrings-python
```

### Steps

Generate the documentation:

```bash
cd ostl-python
mkdocs build
```

## Quick Start

See the docs.
