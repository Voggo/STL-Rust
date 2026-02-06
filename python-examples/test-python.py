import ostl_python as ostl

vars = ostl.Variables()
vars.set("threshold", 0.5)
phi = ostl.parse_formula("G[0,4](x > $threshold) U[0,3] (y < $threshold)")

print(f"Monitoring Formula: {phi}")

# 2. Create the Monitor (using Robustness Semantics)
monitor = ostl.Monitor(
    phi, semantics="Rosi", synchronization="ZeroOrderHold", variables=vars
)

print(monitor)

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

    if t == 3.0:
        # Update threshold at runtime
        vars.set("threshold", 0.6)
        print(f"Updated threshold to {vars.get('threshold')} at time {t}")

    # Feed input to the Monitor
    result = monitor.update(var, val, t)

    # Print results
    print(f"At time {t}, after feeding {var}={val}:")
    print(result)
    print("-" * 40)


# Batch update example
print("\n" + "=" * 50)
print("Batch Update Example")
print("=" * 50)

# Create a simple monitor for batch processing
phi_batch = ostl.parse_formula("x > 10.0")
batch_monitor = ostl.Monitor(phi_batch, semantics="Robustness")

# Prepare batch data: dict mapping signal names to lists of (value, timestamp) tuples
batch_steps = {
    "x": [
        (5.0, 0.0),  # x=5 at t=0 (robustness: 5-10 = -5)
        (15.0, 1.0),  # x=15 at t=1 (robustness: 15-10 = 5)
        (8.0, 2.0),  # x=8 at t=2 (robustness: 8-10 = -2)
        (12.0, 3.0),  # x=12 at t=3 (robustness: 12-10 = 2)
    ]
}

# Process all steps at once
output = batch_monitor.update_batch(batch_steps)

print("Batch Update Results:")
print(output)
