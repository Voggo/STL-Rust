import pandas as pd
import os


def update_formulas_in_csv(input_csv_path, output_csv_path="updated_results.csv"):
    # 1. Load the Formula Definitions
    print("Loading formula definitions...")

    # Load Breach formulas (source)
    # Assuming no header, or if the first line is data. Based on inspection, it has no header.
    try:
        df_breach = pd.read_csv("results/breach/formulas_breach.csv", header=None)
        breach_formulas = df_breach[0].tolist()
    except Exception as e:
        print(f"Error loading formulas_breach.csv: {e}")
        return

    # Load Own formulas (target)
    try:
        df_own = pd.read_csv("results/own/formulas_own.csv")
        own_formulas = df_own["formula"].tolist()
    except Exception as e:
        print(f"Error loading formulas_own.csv: {e}")
        return

    # Check for length mismatch
    if len(breach_formulas) != len(own_formulas):
        print(
            f"Error: Formula lists have different lengths ({len(breach_formulas)} vs {len(own_formulas)}). Cannot map 1-to-1."
        )
        return

    # Create Mapping Dictionary (stripping whitespace for safer matching)
    # Map: Normalized Breach String -> Own String
    formula_map = {b.strip(): o for b, o in zip(breach_formulas, own_formulas)}

    print(f"Created mapping for {len(formula_map)} formulas.")

    # 2. Load the Data CSV
    if not os.path.exists(input_csv_path):
        print(f"Error: Input data file '{input_csv_path}' not found.")
        return

    print(f"Reading data from '{input_csv_path}'...")
    try:
        df_data = pd.read_csv(input_csv_path)

        # Verify required column exists
        if "formula" not in df_data.columns:
            print("Error: Input CSV does not have a 'formula' column.")
            return

        # 3. Update Formulas
        # We apply the map. If a formula isn't in the map, we keep the original.
        def map_formula(f):
            if pd.isna(f):
                return f
            return formula_map.get(str(f).strip(), f)

        # Count changes
        original_formulas = df_data["formula"].tolist()
        # df_data["formula"] = df_data["formula"].apply(map_formula)
        for row in range(len(df_data)):
            original = df_data.at[row, "formula"]
            mapped = formula_map.get(str(original).strip(), original)
            print(f"Row {row}: '{original}' -> '{mapped}'")
            df_data.at[row, "formula"] = mapped

        # Check how many were updated
        changed_count = sum(
            1 for o, n in zip(original_formulas, df_data["formula"]) if o != n
        )
        print(f"Updated {changed_count} rows out of {len(df_data)}.")

        # 4. Save Output
        # Ensure the headers match the request: timestamp,sizeN,M,approach,mean_elapsed_s,std_elapsed_s,formula,all_means
        # (The input file should ideally already have these columns, we just write them back)
        df_data.to_csv(output_csv_path, index=False)
        print(f"Saved updated data to '{output_csv_path}'.")

        # Optional: Show first few rows
        print("\nFirst 5 rows of updated data:")
        print(df_data[["formula", "approach", "mean_elapsed_s"]].head())

    except Exception as e:
        print(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    # Replace 'your_input_file.csv' with the actual name of your data file
    # Example usage:
    input_filename = (
        "results/breach/breach_results_all.csv"  # Change this to your actual filename
    )

    # Check if a file was provided via command line or just run default
    # For now, we just call the function.
    if os.path.exists(input_filename):
        update_formulas_in_csv(input_filename)
    else:
        print(
            f"Please ensure your input CSV ('{input_filename}') matches the filename in the script."
        )
