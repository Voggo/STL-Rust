import json
from ostl_python import ostl_python as ostl

phi = ostl.Formula.and_(
    ostl.Formula.always(start=0.0, end=5.0, child=ostl.Formula.gt("x", 0.5)),
    ostl.Formula.eventually(start=2.0, end=7.0, child=ostl.Formula.lt("y", -0.5)),
)

print(f"Monitoring Formula: {phi}")

# 2. Create the Monitor (using Robustness Semantics)
monitor = ostl.Monitor(phi, semantics="rosi", mode="eager")

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
    print(f"Input t={t:.1f}, {var}={val:.2f} -> Produced Verdicts:")
    print(json.dumps(result["evaluations"], indent=4))
