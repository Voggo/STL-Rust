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

## Quick Start

```python
import ostl_python.ostl_python as ostl

# Define an STL formula: Always[0,5](x > 0.5)
formula = ostl.Formula.always(0, 5, ostl.Formula.gt("x", 0.5))

# Create a monitor with Rosi semantics
monitor = ostl.Monitor(formula, semantics="Rosi")

# Feed data and get verdicts
result = monitor.update("x", 1.0, 0.0)
for evaluation in result['evaluations']:
    print(evaluation['outputs'])
```

## Features

### Multiple Semantics

The library supports four types of monitoring semantics:

1. **StrictSatisfaction** (`semantics="StrictSatisfaction"`): Boolean satisfaction with strict evaluation
   - Returns: `True` or `False`
   - Waits for complete information before producing verdicts

2. **EagerSatisfaction** (`semantics="EagerSatisfaction"`): Boolean satisfaction with eager evaluation
   - Returns: `True` or `False`
   - Produces verdicts as soon as possible

3. **Robustness** (`semantics="Robustness"`, default): Quantitative robustness as a single value
   - Returns: Float value (positive = satisfied, negative = violated)

4. **Rosi** (`semantics="Rosi"`): Robustness as an interval
   - Returns: Tuple `(min, max)` representing robustness interval

### Monitoring Algorithms

- **Incremental** (`algorithm="Incremental"`, default): Efficient online monitoring using sliding windows
- **Naive** (`algorithm="Naive"`): Simple but less efficient approach

### Signal Synchronization

- **ZeroOrderHold** (`synchronization="ZeroOrderHold"`, default): Zero-order hold interpolation
- **Linear** (`synchronization="Linear"`): Linear interpolation
- **None** (`synchronization="None"`): No interpolation

## Formula Construction

### Atomic Predicates

```python
# Greater than
ostl.Formula.gt("signal_name", threshold)

# Less than
ostl.Formula.lt("signal_name", threshold)

# Constants
ostl.Formula.true_()
ostl.Formula.false_()
```

### Boolean Operators

```python
# Conjunction (AND)
ostl.Formula.and_(formula1, formula2)

# Disjunction (OR)
ostl.Formula.or_(formula1, formula2)

# Negation (NOT)
ostl.Formula.not_(formula)

# Implication
ostl.Formula.implies(formula1, formula2)
```

### Temporal Operators

```python
# Globally (Always): G[a,b](φ)
ostl.Formula.always(start, end, child_formula)

# Eventually (Finally): F[a,b](φ)
ostl.Formula.eventually(start, end, child_formula)

# Until: φ1 U[a,b] φ2
ostl.Formula.until(start, end, formula1, formula2)
```
