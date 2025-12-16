# Refactored Comparison Module Structure

## Overview
The comparison logic has been refactored into a modular package structure for better maintainability and separation of concerns.

## Module Organization

```
comparison/
├── __init__.py              # Package initialization, exports public API
├── helpers.py               # Common utilities and helper functions
├── rosi_comparison.py       # RoSI interval comparison logic
├── strict_comparison.py     # Strict value comparison logic
└── table_writer.py          # Summary table writing utilities
```

## Module Descriptions

### `helpers.py`
Common utility functions used across all comparison modules:
- `parse_json_list_with_inf()` - Parse JSON with infinity values
- `to_float()` - Convert values to float, handling special cases
- `extract_formula_number()` - Extract formula ID from filename
- `find_matching_files()` - Find files matching a glob pattern

### `rosi_comparison.py`
RoSI interval comparison logic:
- `ViolationType` enum - Types of interval comparison violations
- `Interval` dataclass - Represents robustness intervals
- `ComparisonResult` - Single comparison result between two intervals
- `FormulaComparison` - Aggregated results for a formula
- `read_results()` - Read interval evaluation results from CSV

### `strict_comparison.py`
Strict value comparison logic:
- `StrictViolationType` enum - Types of value comparison violations
- `StrictComparisonResult` - Single comparison result between two values
- `StrictFormulaComparison` - Aggregated results for a formula
- `read_strict_results()` - Read strict evaluation results from CSV

### `table_writer.py`
Summary table generation:
- `write_summary_table()` - Write RoSI comparison summary to markdown
- `write_strict_summary_table()` - Write Strict comparison summary to markdown

### `compare_eval_results.py` (Main Script)
Main execution script with enhanced features:
- `process_rosi_comparisons()` - Compare RoSI intervals with progress bar
- `process_strict_comparisons()` - Compare strict values with progress bar
- `print_violations_summary()` - Display violation summaries
- `write_detailed_log()` - Write detailed comparison logs
- **Uses tqdm for progress tracking** on all major loops

## Key Improvements

1. **Modular Design**: Each comparison type (RoSI, Strict) has its own module
2. **Separation of Concerns**: Helpers, comparison logic, and output writing are separated
3. **Progress Bars**: tqdm integration for better UX during long processing
4. **Cleaner Imports**: Main script only imports what it needs from the package
5. **Reusability**: Modules can be imported and used independently
6. **Maintainability**: Easier to locate and modify specific comparison logic

## Usage

Run the main comparison script:
```bash
python compare_eval_results.py
```

The script will:
1. Process RoSI interval comparisons with progress bar
2. Process Strict value comparisons with progress bar
3. Print summary statistics
4. Generate detailed violation reports
5. Write markdown summary tables
6. Write detailed log file

## Imports Example

You can also use the modules independently:
```python
from comparison import FormulaComparison, read_results, write_summary_table
from comparison.rosi_comparison import ViolationType, Interval

# Your custom logic here
```
