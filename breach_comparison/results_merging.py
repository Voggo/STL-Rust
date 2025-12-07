import json
import os


def merge_benchmark_results(input_files, output_file):
    # Dictionary to store merged results, keyed by the benchmark 'id' to detect duplicates
    merged_data_map = {}

    # List to maintain the order or store the final result
    final_results = []

    print(f"Starting merge of {len(input_files)} files...\n")

    for filename in input_files:
        print(f"Taking results from '{filename}'")

        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)

                # Check if the file contains a list of records
                if isinstance(data, list):
                    for record in data:
                        # Extract the unique ID
                        record_id = record.get("id")

                        if record_id:
                            if record_id in merged_data_map:
                                print(
                                    f"WARNING: Duplicate ID found: '{record_id}'. Skipping entry from '{filename}'."
                                )
                            else:
                                merged_data_map[record_id] = record
                                final_results.append(record)
                        else:
                            print(
                                f"WARNING: Record found without an 'id' field in '{filename}'. Skipping."
                            )
                else:
                    print(f"WARNING: Content of '{filename}' is not a list. Skipping.")

        except FileNotFoundError:
            print(
                f"ERROR: File '{filename}' not found. Please ensure it is in the current directory."
            )
        except json.JSONDecodeError:
            print(f"ERROR: Failed to decode JSON from '{filename}'.")
        except Exception as e:
            print(f"ERROR: An unexpected error occurred processing '{filename}': {e}")

    # Optional: Sort the results by the integer value at the start of the ID (e.g., "1: ...")
    # If your IDs don't always start with a number, you can remove this sort.
    try:
        final_results.sort(key=lambda x: int(x["id"].split(":")[0]))
        print("\nSorted merged results by ID number.")
    except (ValueError, IndexError, AttributeError):
        print(
            "\nCould not auto-sort by ID number (IDs might not start with 'Number:'). Keeping insertion order."
        )

    # Write the merged data to the output file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=4, ensure_ascii=False)
        print(
            f"\nSuccessfully merged {len(final_results)} records into '{output_file}'."
        )
    except Exception as e:
        print(f"ERROR: Could not write output file: {e}")


if __name__ == "__main__":
    # Define the files to merge
    files_to_merge = [
        "results/own/results_1-12.json",
        "results/own/results_12.json",
        "results/own/results_13-21.json",
    ]

    output_filename = "merged_results.json"

    merge_benchmark_results(files_to_merge, output_filename)
