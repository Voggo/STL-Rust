import json
import pandas as pd
import os


def convert_merged_json_to_csv(
    input_file="merged_results.json",
    formulas_file="formulas_own.csv",
    output_file="benchmark_results_own.csv",
):
    # 1. Load Formula Mapping
    formulas_map = {}
    if os.path.exists(formulas_file):
        try:
            df_formulas = pd.read_csv(formulas_file)
            for _, row in df_formulas.iterrows():
                # Ensure we map integer IDs to formula strings
                formulas_map[int(row["id"])] = row["formula"]
            print(f"Loaded {len(formulas_map)} formulas from {formulas_file}")
        except Exception as e:
            print(f"Warning: Could not load {formulas_file}: {e}")
    else:
        print(f"Warning: {formulas_file} not found. Formulas will be 'Unknown'.")

    # 2. Load the Merged JSON Data
    if not os.path.exists(input_file):
        print(
            f"Error: Input file '{input_file}' not found. Please run the merge script first."
        )
        return

    print(f"Processing results from '{input_file}'...")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    csv_rows = []

    # 3. Process Each Record
    for record in data:
        # We prioritize report_directory for metadata extraction
        report_dir = record.get("report_directory", "")

        # Initialize variables with defaults
        bench_num = -1
        formula = "Unknown Formula"
        approach = "Unknown"
        out_type = "Unknown"
        eval_mode = "Unknown"
        sizeN = "Unknown"

        # --- Parse Metadata from Report Directory ---
        # Example Path: ".../reports/21_ (((...)))/Incremental_bool_Eager/10000"
        if report_dir:
            try:
                # Normalize path separators to forward slashes
                norm_path = report_dir.replace("\\", "/")
                # Remove trailing slash if present, then split
                path_parts = norm_path.strip("/").split("/")

                # Check if path has enough depth to extract info
                if len(path_parts) >= 3:
                    # A. Extract Size (Last path component)
                    size_str = path_parts[-1]
                    try:
                        sizeN = int(size_str)
                    except ValueError:
                        sizeN = size_str

                    # B. Extract Configuration (Second to last component)
                    # Expected format: Approach_OutType_EvalMode (e.g., Incremental_bool_Eager)
                    config_str = path_parts[-2]
                    config_parts = config_str.split("_")

                    if len(config_parts) >= 3:
                        approach = config_parts[0]  # e.g., Incremental
                        out_type_raw = config_parts[1]  # e.g., bool
                        eval_mode = config_parts[2]  # e.g., Eager

                        # Normalize 'rosi' variants
                        if out_type_raw.lower() in ["rosi", "robustnessinterval"]:
                            out_type = "rosi"
                        else:
                            out_type = out_type_raw
                    else:
                        # Fallback if underscores are missing
                        approach = config_str

                    # C. Extract ID (Third to last component)
                    # Expected format: "21_ (formula...)"
                    id_folder_str = path_parts[-3]
                    # We assume the ID is the integer before the first underscore
                    if "_" in id_folder_str:
                        id_part = id_folder_str.split("_")[0]
                        try:
                            bench_num = int(id_part)
                            formula = formulas_map.get(bench_num, "Unknown Formula")
                        except ValueError:
                            pass
            except Exception as e:
                print(f"Error parsing directory '{report_dir}': {e}")

        # --- Fallback: Parse 'id' field if ID wasn't found in directory ---
        if bench_num == -1 and "id" in record:
            rid = record["id"]
            # Example: "1: (x < 0.5)..."
            if ":" in rid:
                try:
                    bench_num = int(rid.split(":")[0])
                    if formula == "Unknown Formula":
                        formula = formulas_map.get(bench_num, "Unknown Formula")
                except ValueError:
                    pass

        # --- Extract Statistics ---
        stats_str = ""
        throughput_val = 0.0

        if "mean" in record:
            # Time values are in nanoseconds (ns)
            mean_est = record["mean"]["estimate"]
            mean_lb = record["mean"]["lower_bound"]
            mean_ub = record["mean"]["upper_bound"]

            # Format statistics into a single string column
            stats_str = f"mean={mean_est:.2f}, lb={mean_lb:.2f}, ub={mean_ub:.2f}"

            # Calculate Throughput (elements/second)
            # throughput = items_per_iter / (mean_time_seconds)
            if (
                "throughput" in record
                and isinstance(record["throughput"], list)
                and len(record["throughput"]) > 0
            ):
                per_iter = record["throughput"][0]["per_iteration"]
                if mean_est > 0:
                    throughput_val = per_iter / (mean_est * 1e-9)

        # --- Calculate All Means ---
        # Calculate sample means from measured values and iteration counts
        all_means_list = []
        if "measured_values" in record and "iteration_count" in record:
            m_vals = record["measured_values"]
            counts = record["iteration_count"]

            if len(m_vals) == len(counts):
                all_means_list = [m / c for m, c in zip(m_vals, counts)]

        # --- Append Row ---
        csv_rows.append(
            {
                "sizeN": sizeN,
                "approach": approach,
                "eval_mode": eval_mode,
                "out_type": out_type,
                "statistics": stats_str,
                "throughput": throughput_val,
                "formula": formula,
                "all_means": str(
                    all_means_list
                ),  # Convert list to string for CSV compatibility
            }
        )

    # 4. Create DataFrame and Save
    df = pd.DataFrame(csv_rows)

    # Ensure columns are in the requested order
    cols = [
        "sizeN",
        "approach",
        "eval_mode",
        "out_type",
        "statistics",
        "throughput",
        "formula",
        "all_means",
    ]
    df = df[cols]

    try:
        df.to_csv(output_file, index=False)
        print(f"\nSuccessfully converted {len(df)} records to '{output_file}'.")
        print("First 5 rows:")
        print(df.head())
    except Exception as e:
        print(f"Error saving CSV file: {e}")


if __name__ == "__main__":
    convert_merged_json_to_csv()
