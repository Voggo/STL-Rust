| Formula | Total | Violations | Rate (%) | Contradiction | Breach Decided | RoSI Decided | Breach Tighter | RoSI Tighter | Mismatch |
|---------|-------|------------|----------|---------------|----------------|--------------|----------------|--------------|----------|
| F1      | 5000  | 0          | 0.00     | 0             | 0              | 0            | 0              | 0            | 0        |
| F2      | 5000  | 0          | 0.00     | 0             | 0              | 0            | 0              | 0            | 0        |
| F3      | 5000  | 0          | 0.00     | 0             | 0              | 0            | 0              | 0            | 0        |
| F4      | 5000  | 10         | 0.20     | 10            | 0              | 0            | 0              | 0            | 0        |
| F5      | 5000  | 100        | 2.00     | 100           | 0              | 0            | 0              | 0            | 0        |
| F6      | 5000  | 1000       | 20.00    | 1000          | 0              | 0            | 0              | 0            | 0        |
| F7      | 5000  | 0          | 0.00     | 0             | 0              | 0            | 0              | 0            | 0        |
| F8      | 5000  | 0          | 0.00     | 0             | 0              | 0            | 0              | 0            | 0        |
| F9      | 5000  | 0          | 0.00     | 0             | 0              | 0            | 0              | 0            | 0        |
| F13     | 5000  | 0          | 0.00     | 0             | 0              | 0            | 0              | 0            | 0        |
| F14     | 5000  | 0          | 0.00     | 0             | 0              | 0            | 0              | 0            | 0        |
| F15     | 5000  | 0          | 0.00     | 0             | 0              | 0            | 0              | 0            | 0        |
| F16     | 5000  | 450        | 8.63     | 420           | 0              | 0            | 10             | 0            | 0        |
| F17     | 5000  | 470        | 8.85     | 430           | 0              | 0            | 10             | 0            | 0        |
| F18     | 5000  | 510        | 9.29     | 450           | 0              | 0            | 10             | 0            | 0        |


### Violation Types Legend

- **Contradiction**: The intervals from Breach and RoSI are disjoint (e.g., one says satisfied, the other violated).
- **Breach Decided**: Breach provides a definitive verdict (Sat/Unsat), while RoSI returns Unknown (interval contains 0).
- **RoSI Decided**: RoSI provides a definitive verdict, while Breach returns Unknown.
- **Breach Tighter**: Breach's interval is strictly contained within RoSI's interval (Breach is more precise).
- **RoSI Tighter**: RoSI's interval is strictly contained within Breach's interval (RoSI is more precise).
- **Mismatch**: Intervals overlap and agree on the verdict (or both unknown), but the bounds differ significantly.
