# OSTL Python Interface

Python bindings for the Online Signal Temporal Logic (STL) monitoring library.

## Installation

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

# Create a monitor with robustness semantics
monitor = ostl.Monitor(formula, semantics="robustness")

# Feed data and get verdicts
result = monitor.update("x", 1.0, 0.0)
print(result['verdicts'])
```

## Features

### Multiple Semantics

The library supports three types of monitoring semantics:

1. **Boolean** (`semantics="boolean"`): Classic true/false evaluation
   - Returns: `True` or `False`
   - Best for: Binary decision making

2. **Quantitative** (`semantics="quantitative"`): Robustness as a single value
   - Returns: Float value (positive = satisfied, negative = violated)
   - Best for: Optimization and measuring "how well" a formula is satisfied

3. **Robustness Interval** (`semantics="robustness"`): RoSI semantics
   - Returns: Tuple `(min, max)` representing robustness interval
   - Best for: Handling uncertainty and eager evaluation

### Monitoring Strategies

- **Incremental** (`strategy="incremental"`, default): Efficient online monitoring using sliding windows
- **Naive** (`strategy="naive"`): Simple but less efficient approach

### Evaluation Modes

- **Eager** (`mode="eager"`): Produces verdicts as soon as possible (default for robustness)
- **Strict** (`mode="strict"`): Waits for complete information (default for boolean/quantitative)

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

## Examples

### Example 1: Boolean Monitoring

```python
import ostl_python.ostl_python as ostl

# Formula: Always[0,5](x > 0.5)
phi = ostl.Formula.always(0, 5, ostl.Formula.gt("x", 0.5))

# Boolean monitor (strict mode)
monitor = ostl.Monitor(phi, semantics="boolean")

for t in range(10):
    result = monitor.update("x", 0.8, float(t))
    if result['verdicts']:
        for v in result['verdicts']:
            print(f"t={v['timestamp']}: {v['value']}")  # True or False
```

### Example 2: Quantitative Robustness

```python
import ostl_python.ostl_python as ostl

# Formula: Eventually[0,3](x < 0.2)
phi = ostl.Formula.eventually(0, 3, ostl.Formula.lt("x", 0.2))

# Quantitative monitor
monitor = ostl.Monitor(phi, semantics="quantitative")

for t in range(10):
    result = monitor.update("x", 0.1, float(t))
    if result['verdicts']:
        for v in result['verdicts']:
            rho = v['value']
            status = "✓" if rho > 0 else "✗"
            print(f"t={v['timestamp']}: {status} ρ={rho:.3f}")
```

### Example 3: Robustness Interval (RoSI)

```python
import ostl_python.ostl_python as ostl

# Formula: G[0,2](x > 0.5) AND F[0,3](y < 0.8)
phi = ostl.Formula.and_(
    ostl.Formula.always(0, 2, ostl.Formula.gt("x", 0.5)),
    ostl.Formula.eventually(0, 3, ostl.Formula.lt("y", 0.8))
)

# RoSI monitor with eager evaluation
monitor = ostl.Monitor(phi, semantics="robustness", mode="eager")

for t in range(10):
    # Update with x signal
    result_x = monitor.update("x", 0.6, float(t))
    
    # Update with y signal
    result_y = monitor.update("y", 0.7, float(t))
    
    # Print verdicts
    for result in [result_x, result_y]:
        if result['verdicts']:
            for v in result['verdicts']:
                rho_min, rho_max = v['value']
                print(f"t={v['timestamp']}: ρ∈[{rho_min:+.3f}, {rho_max:+.3f}]")
```

### Example 4: Complex Multi-Signal Formula

```python
import ostl_python.ostl_python as ostl

# Formula: (x > 2.0) → F[0,5](y < 1.0)
# "If x exceeds 2.0, then y must drop below 1.0 within 5 seconds"
phi = ostl.Formula.implies(
    ostl.Formula.gt("x", 2.0),
    ostl.Formula.eventually(0, 5, ostl.Formula.lt("y", 1.0))
)

monitor = ostl.Monitor(phi, semantics="robustness")

# Feed interleaved signals
for t in range(20):
    if t % 2 == 0:
        result = monitor.update("x", 2.5, float(t))
    else:
        result = monitor.update("y", 0.5, float(t))
    
    if result['verdicts']:
        print(f"t={t}: {len(result['verdicts'])} verdicts")
```

## API Reference

### Formula Class

Static methods for constructing STL formulas:

- `Formula.gt(signal: str, value: float) -> Formula`
- `Formula.lt(signal: str, value: float) -> Formula`
- `Formula.true_() -> Formula`
- `Formula.false_() -> Formula`
- `Formula.and_(left: Formula, right: Formula) -> Formula`
- `Formula.or_(left: Formula, right: Formula) -> Formula`
- `Formula.not_(child: Formula) -> Formula`
- `Formula.implies(left: Formula, right: Formula) -> Formula`
- `Formula.always(start: float, end: float, child: Formula) -> Formula`
- `Formula.eventually(start: float, end: float, child: Formula) -> Formula`
- `Formula.until(start: float, end: float, left: Formula, right: Formula) -> Formula`

### Monitor Class

```python
Monitor(
    formula: Formula,
    semantics: str = "boolean",  # "boolean", "quantitative", or "robustness"
    strategy: str = "incremental",  # "incremental" or "naive"
    mode: str = None  # "eager" or "strict" (auto-selected if None)
) -> Monitor
```

Methods:

- `update(signal: str, value: float, timestamp: float) -> dict`

Returns a dictionary with:

```python
{
    'input_signal': str,      # Signal name that was updated
    'input_timestamp': float, # Timestamp of the input
    'verdicts': [             # List of verdicts produced
        {
            'timestamp': float,  # When this verdict is for
            'value': ...         # bool, float, or (float, float) depending on semantics
        },
        ...
    ]
}
```
