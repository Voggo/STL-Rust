import math
from ostl_python import ostl_python as ostl

print("=" * 80)
print("OSTL Python Interface - Comprehensive Example")
print("=" * 80)

# Define a formula: Always[0, 5](x > 0.5)
phi = ostl.Formula.always(0.0, 5.0, ostl.Formula.gt("x", 0.5))
print(f"\nFormula: {phi}")

# -----------------------------------------------------------------------------
# Example 1: Boolean Semantics (Classic True/False)
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Example 1: Boolean Semantics")
print("=" * 80)

monitor_bool = ostl.Monitor(phi, semantics="boolean", strategy="incremental", mode="strict")
print(f"Monitor: {monitor_bool}")

print("\nFeeding sine wave data...")
for i in range(15):
    t = float(i)
    val = math.sin(t / 2.0)
    result = monitor_bool.update("x", val, t)
    
    if result['verdicts']:
        print(f"t={t:4.1f}, x={val:+.2f} -> Verdicts:")
        for v in result['verdicts']:
            verdict_val = "✓ True" if v['value'] else "✗ False"
            print(f"  └─ t={v['timestamp']:4.1f}: {verdict_val}")
    else:
        print(f"t={t:4.1f}, x={val:+.2f} -> (buffered)")

# -----------------------------------------------------------------------------
# Example 2: Quantitative Semantics (Robustness as float)
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Example 2: Quantitative Semantics")
print("=" * 80)

monitor_quant = ostl.Monitor(phi, semantics="quantitative", strategy="incremental")
print(f"Monitor: {monitor_quant}")

print("\nFeeding sine wave data...")
for i in range(15):
    t = float(i)
    val = math.sin(t / 2.0)
    result = monitor_quant.update("x", val, t)
    
    if result['verdicts']:
        print(f"t={t:4.1f}, x={val:+.2f} -> Verdicts:")
        for v in result['verdicts']:
            robustness = v['value']
            status = "✓" if robustness > 0 else "✗"
            print(f"  └─ t={v['timestamp']:4.1f}: {status} ρ={robustness:+.3f}")
    else:
        print(f"t={t:4.1f}, x={val:+.2f} -> (buffered)")

# -----------------------------------------------------------------------------
# Example 3: Robustness Interval Semantics (RoSI)
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Example 3: Robustness Interval Semantics (RoSI)")
print("=" * 80)

monitor_rosi = ostl.Monitor(phi, semantics="robustness", strategy="incremental", mode="eager")
print(f"Monitor: {monitor_rosi}")

print("\nFeeding sine wave data...")
for i in range(15):
    t = float(i)
    val = math.sin(t / 2.0)
    result = monitor_rosi.update("x", val, t)
    
    if result['verdicts']:
        print(f"t={t:4.1f}, x={val:+.2f} -> Verdicts:")
        for v in result['verdicts']:
            rho_min, rho_max = v['value']
            status = "✓" if rho_min > 0 else ("?" if rho_max > 0 else "✗")
            print(f"  └─ t={v['timestamp']:4.1f}: {status} ρ∈[{rho_min:+.3f}, {rho_max:+.3f}]")
    else:
        print(f"t={t:4.1f}, x={val:+.2f} -> (buffered)")

# -----------------------------------------------------------------------------
# Example 4: Complex Formula with All Operators
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Example 4: Complex Formula")
print("=" * 80)

# Formula: (x > 0.3) AND Eventually[0,3](y < 0.8)
phi_complex = ostl.Formula.and_(
    ostl.Formula.gt("x", 0.3),
    ostl.Formula.eventually(0.0, 3.0, ostl.Formula.lt("y", 0.8))
)
print(f"Formula: {phi_complex}")

monitor_complex = ostl.Monitor(phi_complex, semantics="robustness")

print("\nFeeding two-signal data...")
for i in range(10):
    t = float(i)
    x_val = math.sin(t / 2.0)
    y_val = math.cos(t / 2.0)
    
    # Update with x signal
    result_x = monitor_complex.update("x", x_val, t)
    if result_x['verdicts']:
        print(f"t={t:4.1f}, x={x_val:+.2f} -> {len(result_x['verdicts'])} verdict(s)")
        for v in result_x['verdicts']:
            rho_min, rho_max = v['value']
            print(f"  └─ t={v['timestamp']:4.1f}: ρ∈[{rho_min:+.3f}, {rho_max:+.3f}]")
    else:
        print(f"t={t:4.1f}, x={x_val:+.2f} -> (buffered)")
    
    # Update with y signal  
    result_y = monitor_complex.update("y", y_val, t)
    if result_y['verdicts']:
        print(f"t={t:4.1f}, y={y_val:+.2f} -> {len(result_y['verdicts'])} verdict(s)")
        for v in result_y['verdicts']:
            rho_min, rho_max = v['value']
            print(f"  └─ t={v['timestamp']:4.1f}: ρ∈[{rho_min:+.3f}, {rho_max:+.3f}]")
    else:
        print(f"t={t:4.1f}, y={y_val:+.2f} -> (buffered)")

# -----------------------------------------------------------------------------
# Example 5: Using True and False constants
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Example 5: True and False Constants")
print("=" * 80)

# Formula: Always[0,2](True) - should always be satisfied
phi_true = ostl.Formula.always(0.0, 2.0, ostl.Formula.true_())
print(f"Formula (always true): {phi_true}")

monitor_true = ostl.Monitor(phi_true, semantics="boolean")
for i in range(5):
    t = float(i)
    result = monitor_true.update("x", 0.0, t)
    if result['verdicts']:
        print(f"t={t}: {result['verdicts'][0]['value']}")

print("\n" + "=" * 80)
print("All examples completed successfully!")
print("=" * 80)
