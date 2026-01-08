#!/usr/bin/env python3
"""
Consolidate L/NL accuracy files into standardized format
"""
import pandas as pd
import glob
import os

def consolidate_l_nl_accuracies(dataset_name):
    """Consolidate L/NL accuracy files for a specific dataset."""
    print(f"Consolidating L/NL accuracies for {dataset_name}...")

    # Find all L/NL accuracy files for this dataset
    pattern = f"../../data/analysis/accuracies/l_nl_{dataset_name}_*.csv"
    files = glob.glob(pattern)

    if not files:
        print(f"No L/NL accuracy files found for {dataset_name}")
        return

    consolidated_data = []

    for file_path in sorted(files):
        try:
            # Extract model name from filename
            filename = os.path.basename(file_path)
            model_name = filename.replace(f"l_nl_{dataset_name}_", "").replace(".csv", "")

            # Read the CSV file
            df = pd.read_csv(file_path)
            if len(df) > 0:
                row = df.iloc[0]
                consolidated_data.append({
                    'filename': model_name,
                    'l_acc': row['l_acc'],
                    'nl_acc': row['nl_acc']
                })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if consolidated_data:
        # Create consolidated DataFrame
        df_consolidated = pd.DataFrame(consolidated_data)

        # Sort by filename for consistency
        df_consolidated = df_consolidated.sort_values('filename')

        # Save consolidated file
        output_file = f"../../data/analysis/accuracies/l_nl_accuracies_{dataset_name}.csv"
        df_consolidated.to_csv(output_file, index=False)

        print(f"Consolidated {len(consolidated_data)} models into {output_file}")
        print(f"Columns: {list(df_consolidated.columns)}")
        print(f"Sample data:\n{df_consolidated.head()}")

    else:
        print(f"No valid data found for {dataset_name}")

def main():
    """Consolidate L/NL accuracies for all datasets."""
    datasets = ["vanilla", "independent_feature"]

    for dataset in datasets:
        consolidate_l_nl_accuracies(dataset)
        print()

    print("L/NL accuracy consolidation complete!")

if __name__ == "__main__":
    main()


