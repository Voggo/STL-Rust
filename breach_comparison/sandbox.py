import json
import os

# load 'merged_results.json' and print number of records
def load_and_print_merged_results(file_path='merged_results.json'):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Loaded {len(data)} records from '{file_path}'.")
    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"ERROR: Failed to decode JSON from '{file_path}'.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")  
load_and_print_merged_results()
