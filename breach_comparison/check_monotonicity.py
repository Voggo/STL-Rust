import json
import csv
import glob
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class Interval:
    lower: float
    upper: float

    def __str__(self):
        return f"[{self.lower}, {self.upper}]"

def parse_json_list_with_inf(json_str: str) -> List[Any]:
    try:
        json_str = json_str.strip()
        json_str = json_str.replace("-inf", "NEG_INF")
        json_str = json_str.replace("inf", "POS_INF")
        json_str = json_str.replace("NEG_INF", '"-inf"')
        json_str = json_str.replace("POS_INF", '"inf"')
        return json.loads(json_str)
    except Exception:
        return []

def check_file(filepath: str):
    print(f"Checking {os.path.basename(filepath)}...")
    
    # Map time_step -> last_seen_interval
    current_intervals: Dict[int, Interval] = {}
    violations = 0
    row_count = 0
    
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                row_count += 1
                times_str = row.get("output_times", "[]")
                values_str = row.get("outputs_values", "[]") 
                
                times = parse_json_list_with_inf(times_str)
                values = parse_json_list_with_inf(values_str)
                
                if not times or not values or len(times) != len(values):
                    continue
                    
                for t, v in zip(times, values):
                    if not isinstance(v, list) or len(v) < 2:
                        continue
                        
                    def to_float(val):
                        if isinstance(val, str):
                            if val == "-inf": return float("-inf")
                            if val == "inf": return float("inf")
                        return float(val)
                        
                    new_lower = to_float(v[0])
                    new_upper = to_float(v[1])
                    t_int = int(t)
                    
                    if t_int in current_intervals:
                        old_int = current_intervals[t_int]
                        
                        # Check monotonicity
                        # Lower bound should be non-decreasing: new >= old
                        if new_lower < old_int.lower - 1e-9:
                            print(f"  Violation at time {t_int} (Row {row_idx}): Lower bound decreased! {old_int.lower} -> {new_lower}")
                            violations += 1
                            
                        # Upper bound should be non-increasing: new <= old
                        if new_upper > old_int.upper + 1e-9:
                            print(f"  Violation at time {t_int} (Row {row_idx}): Upper bound increased! {old_int.upper} -> {new_upper}")
                            violations += 1
                            
                    current_intervals[t_int] = Interval(new_lower, new_upper)
                    
    except Exception as e:
        print(f"Error reading file: {e}")
        
    if violations == 0:
        print(f"  ✓ OK ({row_count} rows processed)")
    else:
        print(f"  ❌ Found {violations} violations")

def main():
    files = glob.glob("results/breach/eval_results/breach_eval_results_*.csv")
    files.sort()
    
    if not files:
        print("No files found.")
        return

    for f in files:
        check_file(f)

if __name__ == "__main__":
    main()
