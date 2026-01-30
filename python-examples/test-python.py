import json
from ostl_python import ostl_python as ostl
import numpy as np

phi = ostl.parse_formula("G[0,4](x > 0.5) U[0,3] (y < -0.5)")

print(f"Monitoring Formula: {phi}")

# 2. Create the Monitor (using Robustness Semantics)
monitor = ostl.Monitor(phi, semantics="Rosi")
# Get signal identifiers used in the formula
signal_ids = monitor.get_signal_identifiers()
print(f"Signal Identifiers in the formula: {signal_ids}")

vals = [
    ("x", 0.0, 0.0),
    ("y", 0.0, 0.0),
    ("x", 1.0, 0.6),
    ("x", 2.0, 0.7),
    ("x", 3.0, 0.4),
    ("x", 4.0, 0.8),
    ("x", 5.0, 0.9),
    ("y", 6.0, -0.6),
    ("y", 7.0, -0.4),
    ("y", 8.0, -0.7),
]

for var, t, val in vals:
    # 3. Feed input to the Monitor
    result = monitor.update(var, val, t)

    # 4. Print results
    print(f"Input t={t:.1f}, {var}={val:.2f}")
    finalized = result.finalize()

    # finalized is a list of tuples, convert to np array for easier handling
    finalized_array = np.array(finalized, dtype=object)
    print(finalized_array)
