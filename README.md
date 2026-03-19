# CANARIFY

[![Rust CI](https://github.com/Voggo/STL-Rust/workflows/Rust%20CI/badge.svg)](https://github.com/Voggo/STL-Rust/actions/workflows/rust.yml)
[![Python Tests](https://github.com/Voggo/STL-Rust/workflows/Python%20Tests/badge.svg)](https://github.com/Voggo/STL-Rust/actions/workflows/python-tests.yml)
[![codecov](https://codecov.io/gh/Voggo/STL-Rust/branch/main/graph/badge.svg)](https://codecov.io/gh/Voggo/STL-Rust)

CANARIFY is a Rust Library for online monitoring of Signal Temporal Logic (STL) specifications. It is designed for high performance and low memory usage, making it suitable for real-time applications. The library also has Python bindings, allowing for easy integration with Python-based workflows.

- [CANARIFY](#canarify)
  - [About](#about)
  - [Theory](#theory)
    - [Signal Temporal Logic (STL)](#signal-temporal-logic-stl)
    - [Evaluation Semantics](#evaluation-semantics)
  - [Installation](#installation)
    - [Rust](#rust)
    - [Python](#python)
  - [Usage](#usage)
    - [Rust Usage](#rust-usage)
    - [Python Usage](#python-usage)
  - [References](#references)

## About

Cyber-Physical Systems (CPSs) increasingly rely on real-time fault detection and runtime monitoring to ensure safe operation. CANARIFY provides a unified monitoring interface that addresses the stringent performance requirements of these systems. Key features include:

- **Embedded DSL:** A macro-based DSL (`stl!`) allows specifications to be embedded and syntax-checked directly at compile time in Rust.
- **Unified Semantics Interface:** Supports multiple online evaluation modes (Qualitative, Quantitative, Eager, and RoSI) in a single framework.
- **Python Bindings:** Exposed via the `ostl_python` (or `CANARIFY`) package to enable interactive workflows in environments like Jupyter Notebooks.
- **High Performance:** Benchmarks demonstrate throughput exceeding existing state-of-the-art tools, with native optimizations running blazingly fast across modern architectures.

## Theory

### Signal Temporal Logic (STL)

Signal Temporal Logic (STL) is a formalism for specifying properties of real-valued signals that evolve over time, providing a compact language to describe the desired behaviors of dynamic systems. STL evaluates properties over signals, which are defined as functions mapping a time domain (such as nonnegative real numbers, $\mathbb{R}_{\ge0}$) to a value domain.

CANARIFY focuses on bounded STL, meaning all temporal operators are constrained by finite time intervals of the form $[a, b]$, where $0 \le a < t$.

The core syntax of STL is built from a minimal set of primitive operators:

- **True ($\top$)**: The Boolean constant True.

- **Atomic Predicates ($\mu(x) < c$)**: Evaluates to True if the function over the signal is less than a constant $c$.

- **Negation ($\neg\phi$)**: The logical NOT of a formula.

- **Conjunction ($\phi \wedge \psi$)**: The logical AND of two formulas.

- **Until ($\phi \mathcal{U}_{[a,b]} \psi$)**: States that $\phi$ must hold continuously until $\psi$ becomes true within the time interval $[a, b]$.

From these primitives, the library derives other highly useful operators to simplify specifications:

- **Disjunction (OR)**: $\phi \vee \psi$

- **Implication**: $\phi \rightarrow \psi$

- **Eventually (Future)**: $\diamondsuit_{[a,b]}\phi$

- **Globally (Always)**: $\Box_{[a,b]}\phi$

### Evaluation Semantics

An online monitor observes a system's behavior incrementally as discrete samples arrive. CANARIFY provides a unified interface supporting four distinct monitoring semantics, allowing users to trade off between expressiveness and verdict latency:

- **Delayed Qualitative:** Computes standard Boolean satisfaction. This mode requires the signal to be fully resolved up to the formula's maximum temporal horizon before emitting a strict true/false verdict.

- **Delayed Quantitative:** Computes a real-valued robustness score indicating the precise degree of satisfaction or violation. Similar to the qualitative mode, it requires full signal availability up to the temporal depth.

- **Eager Qualitative:** Leverages the monotonicity in Boolean and temporal logic to emit early verdicts over partial traces. For example, a violation of a "globally" ($\Box$) property immediately yields a false verdict without waiting for the full interval to elapse.

- **Robust Satisfaction Intervals (RoSI):** Provides quantitative reasoning over partial traces. Instead of a single robustness value, the monitor computes an interval $[\rho_{min}, \rho_{max}]$ that encloses all possible future robustness values. A formula is definitively satisfied when $\rho_{min} > 0$ and definitively violated when $\rho_{max} < 0$.

To compute these semantics efficiently, CANARIFY uses a bottom-up dynamic programming approach. For sliding window operations (like *eventually* and *globally*), the library incorporates Lemire's algorithm to aggressively reduce cache footprints and computation time.

## Installation

### Rust

Add CANARIFY to your `Cargo.toml`:

```toml
[dependencies]
ostl = "0.1.0"

```

### Python

Install the Python bindings via pip:

```bash
pip install ostl_python

```

## Usage

### Rust Usage

For more examples, see the [`ostl/examples`](./ostl/examples) directory.
The following snippet demonstrates how to create a monitor for the STL formula $\Box_{[0, 2]}(x > 5)$ using the embedded DSL and process incoming signal data.

CANARIFY utilizes the Builder pattern to configure the monitor's formula, semantics, and algorithm before processing the data stream.

```rust
use ostl::ring_buffer::Step;
use ostl::stl::monitor::{Algorithm, DelayedQuantitative, StlMonitor};
use std::time::Duration;

// Define a formula using the embedded DSL
let formula = ostl::stl!(G[0, 2](x > 5.0));

// Build the monitor
let mut monitor = StlMonitor::builder()
    .formula(formula)
    .algorithm(Algorithm::Incremental)
    .semantics(DelayedQuantitative)
    .build()
    .expect("Failed to build monitor");

// Feed data steps to the monitor
let out1 = monitor.update(&Step::new("x", 7.0, Duration::from_secs(0)));
let out2 = monitor.update(&Step::new("x", 6.0, Duration::from_secs(1)));

// Process the finalized verdicts
for verdict in out2.verdicts() {
    println!("t={:?}: {:?}", verdict.timestamp, verdict.value);
}

```

### Python Usage
For more Python examples, see the [`ostl-python/examples`](./ostl-python/examples) directory.
The Python API wraps the core Rust engine, offering comparable performance via an intuitive Pythonic interface.

```python
import ostl_python as ostl

# Parse formula using the DSL syntax
phi = ostl.parse_formula("G[0, 10](x > 5)")

# Create a monitor using the selected semantics
monitor = ostl.Monitor(phi, semantics="DelayedQuantitative")

# Update the monitor with streaming data (signal_name, value, timestamp)
output = monitor.update("x", 6.0, 0.5)

# Print formatted verdicts or extract structured data
print(f"Verdicts: {output}")
print(output.to_dict())

```

## References

1. Deshmukh, J.V., et al. "Robust Online Monitoring of Signal Temporal Logic." *arXiv preprint arXiv:1506.08234* (2015).
2. Lemire, D. "Streaming Maximum-Minimum Filter Using No More than Three Comparisons per Element." *arXiv preprint arXiv:cs/0610046* (2007).
3. Maler, O., & Nickovic, D. "Monitoring Temporal Properties of Continuous Signals." *Formal Techniques, Modelling and Analysis of Timed and Fault-Tolerant Systems* (2004).
