import math
from ostl_python import ostl_python as ostl

# 1. Define a Formula: Globally[0, 5] (x > 0.5)
#    This means "For the next 5 seconds, x must be greater than 0.5"
# phi = ostl_python.Formula.always(
#     start=0.0, end=5.0, child=ostl_python.Formula.gt("x", 0.5)
# )


phi = ostl.Formula.and_(
    ostl.Formula.always(start=0.0, end=5.0, child=ostl.Formula.gt("x", 0.5)),
    ostl.Formula.eventually(start=2.0, end=7.0, child=ostl.Formula.lt("y", -0.5)),
)

print(f"Monitoring Formula: {phi}")

# 2. Create the Monitor (using Robustness Semantics)
# NEW API: You can now specify semantics, strategy, and mode explicitly
monitor = ostl.Monitor(phi, semantics="robustness", mode="eager")


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
    import json
    result = monitor.update(var, val, t)

    # 4. Print results
    # verdicts = result["verdicts"]
    print(f"Input t={t:.1f}, {var}={val:.2f} -> Produced Verdicts:")
        # for v in verdicts:
            # For robustness, value is a tuple (min, max)
            # print(f"\tVerdict at t={v['timestamp']:.1f}: Robustness={v['value']}")
    print(json.dumps(result['evaluations'], indent=4))
    # else:
    #     print(f"Input t={t:.1f}, {var}={val:.2f} -> (no verdict)")

