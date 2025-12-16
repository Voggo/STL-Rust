| Formula | Total | Violations | Rate (%) | Contradiction | Breach Decided | RoSI Decided | Breach Tighter | RoSI Tighter | Mismatch |
|---------|-------|------------|----------|---------------|----------------|--------------|----------------|--------------|----------|
| F1      | 5000  | 1          | 0.02     | 0             | 0              | 1            | 0              | 0            | 0        |
| F2      | 5000  | 1          | 0.02     | 0             | 0              | 1            | 0              | 0            | 0        |
| F3      | 5000  | 1          | 0.02     | 1             | 0              | 0            | 0              | 0            | 0        |
| F4      | 5000  | 10         | 0.20     | 0             | 1              | 0            | 9              | 0            | 0        |
| F5      | 5000  | 100        | 2.00     | 0             | 86             | 0            | 14             | 0            | 0        |
| F6      | 5000  | 785        | 15.70    | 0             | 86             | 0            | 635            | 64           | 0        |
| F7      | 5000  | 10         | 0.20     | 0             | 0              | 1            | 0              | 9            | 0        |
| F8      | 5000  | 100        | 2.00     | 0             | 0              | 15           | 0              | 85           | 0        |
| F9      | 5000  | 268        | 5.36     | 0             | 0              | 2            | 262            | 4            | 0        |


### Violation Types Legend

- **Contradiction**: The intervals from Breach and RoSI are disjoint (e.g., one says satisfied, the other violated).
- **Breach Decided**: Breach provides a definitive verdict (Sat/Unsat), while RoSI returns Unknown (interval contains 0).
- **RoSI Decided**: RoSI provides a definitive verdict, while Breach returns Unknown.
- **Breach Tighter**: Breach's interval is strictly contained within RoSI's interval (Breach is more precise).
- **RoSI Tighter**: RoSI's interval is strictly contained within Breach's interval (RoSI is more precise).
- **Mismatch**: Intervals overlap and agree on the verdict (or both unknown), but the bounds differ significantly.
