#!/usr/bin/env python3

import os
import pandas as pd
from pathlib import Path
import sys

def calculate_accuracy(prediction_file):
    """Calculate accuracy from a prediction TSV file."""
    try:
        # Read the TSV file (tab-separated)
        df = pd.read_csv(prediction_file, sep='\t')

        # Check if required columns exist
        if 'prediction' not in df.columns or 'target' not in df.columns:
            print(f"Warning: Missing required columns in {prediction_file}")
            return None

        # Calculate accuracy as exact string matches, ignoring stress markers
        correct = (df['prediction'].str.replace('ˈ', '') == df['target'].str.replace('ˈ', '')).sum()
        total = len(df)
        accuracy = (correct / total) * 100 if total > 0 else 0

        return accuracy

    except Exception as e:
        print(f"Error processing {prediction_file}: {e}")
        return None

def calculate_accuracy_sep_char(prediction_file, target_file):
    """Calculate accuracy from a prediction file (CSV format: index,prediction) and target file."""
    try:
        # Read prediction file (CSV format: index,prediction)
        df_pred = pd.read_csv(prediction_file, header=None, names=['index', 'prediction'])
        
        # Sort by index in ascending order
        df_pred = df_pred.sort_values('index', ascending=True).reset_index(drop=True)
        
        # Read target file (space-separated characters, one per line)
        with open(target_file, 'r', encoding='utf-8') as f:
            targets = [line.strip() for line in f.readlines()]
        
        # Check if lengths match
        if len(df_pred) != len(targets):
            print(f"Warning: Length mismatch - predictions: {len(df_pred)}, targets: {len(targets)}")
            min_len = min(len(df_pred), len(targets))
            df_pred = df_pred.iloc[:min_len]
            targets = targets[:min_len]
        
        # Normalize predictions and targets (remove spaces and stress markers)
        # Only use the 'prediction' column (index and comma already removed by pandas)
        preds_normalized = df_pred['prediction'].astype(str).str.replace(' ', '').str.replace('ˈ', '').str.strip()
        targets_normalized = pd.Series(targets).str.replace(' ', '').str.replace('ˈ', '').str.strip()
        
        # Calculate accuracy
        correct = (preds_normalized == targets_normalized).sum()
        total = len(df_pred)
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        return accuracy
    
    except Exception as e:
        print(f"Error processing {prediction_file}: {e}")
        return None

def process_directory(predictions_dir, output_filename):
    """Process a predictions directory and save results to CSV."""
    if not predictions_dir.exists():
        print(f"Error: Predictions directory not found: {predictions_dir}")
        return

    # List to store filename and accuracy pairs
    results = []

    # Process all TSV files
    tsv_files = list(predictions_dir.glob("*.tsv"))
    # Sort files by filename for consistent ordering
    tsv_files.sort(key=lambda f: f.name)
    print(f"Found {len(tsv_files)} TSV files to process in {predictions_dir.name}")

    for tsv_file in tsv_files:
        filename = tsv_file.name

        # Extract the first part of filename based on directory type
        if predictions_dir.name == "predictions_feature_invariant":
            short_filename = filename.split('.decode')[0]
        elif predictions_dir.name == "predictions_independent_feature":
            short_filename = filename.split('.decode')[0]
        elif predictions_dir.name == "predictions_binaryfeature":
            short_filename = filename.split('.decode')[0]
        else:  # predictions_dual_source
            short_filename = filename.split('.nll')[0]

        print(f"Processing {filename}...")
        accuracy = calculate_accuracy(tsv_file)

        if accuracy is not None:
            results.append({
                'filename': short_filename,
                'accuracy': round(accuracy, 4)  # Keep more decimal places for accuracy
            })
            print(".4f")
        else:
            print(f"  Failed to calculate accuracy for {filename}")

    # Create DataFrame
    df_output = pd.DataFrame(results)

    # Ensure output directory exists
    output_dir = Path("../../data/analysis/accuracies")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    output_file = output_dir / output_filename
    df_output.to_csv(output_file, index=False)

    print(f"\nAccuracy calculation complete!")
    print(f"Results saved to: {output_file}")
    print(f"Processed {len(results)} files successfully")
    print()

def process_directory_sep_char(predictions_dir, output_filename, base_path):
    """Process a predictions directory with CSV format (index,prediction) and save results to CSV."""
    if not predictions_dir.exists():
        print(f"Error: Predictions directory not found: {predictions_dir}")
        return

    # List to store filename and accuracy pairs
    results = []

    # Process all .txt files
    txt_files = list(predictions_dir.glob("*.txt"))
    # Sort files by filename for consistent ordering
    txt_files.sort(key=lambda f: f.name)
    print(f"Found {len(txt_files)} TXT files to process in {predictions_dir.name}")

    for txt_file in txt_files:
        filename = txt_file.name
        # Remove .txt extension to get model name (e.g., 10L_90NL_1_1)
        short_filename = filename.replace('.txt', '')
        
        # Parse model name to get condition and run
        # Format: {condition}_{run}_{split} (e.g., 10L_90NL_1_1)
        parts = short_filename.split('_')
        if len(parts) >= 4:
            condition = f"{parts[0]}_{parts[1]}"  # e.g., 10L_90NL
            run = parts[2]  # e.g., 1
            model = short_filename  # e.g., 10L_90NL_1_1
            
            # Find target file (targets are in data/{condition}/test/)
            data_root = Path("../../data")
            target_file = data_root / condition / "test" / f"run{run}" / f"test.{model}.tgt"
            
            if not target_file.exists():
                print(f"  Warning: Target file not found: {target_file}")
                continue
            
            print(f"Processing {filename}...")
            accuracy = calculate_accuracy_sep_char(txt_file, target_file)
            
            if accuracy is not None:
                results.append({
                    'filename': short_filename,
                    'accuracy': round(accuracy, 4)
                })
                print(f"  Accuracy: {accuracy:.4f}%")
            else:
                print(f"  Failed to calculate accuracy for {filename}")
        else:
            print(f"  Warning: Could not parse filename {filename}, skipping...")

    # Create DataFrame
    df_output = pd.DataFrame(results)

    # Ensure output directory exists
    output_dir = Path("../../data/analysis/accuracies")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    output_file = output_dir / output_filename
    df_output.to_csv(output_file, index=False)

    print(f"\nAccuracy calculation complete!")
    print(f"Results saved to: {output_file}")
    print(f"Processed {len(results)} files successfully")
    print()

def main():
    # Process both directories
    base_path = Path("../../data/predictions")

    print("=== Processing Dual Source Predictions ===")
    dual_source_dir = base_path / "predictions_dualsource"
    process_directory(dual_source_dir, "overall_accuracies_dual_source.csv")

    print("=== Processing Feature Invariant Predictions ===")
    feature_invariant_dir = base_path / "predictions_feature_invariant"
    process_directory(feature_invariant_dir, "overall_accuracies_feature_invariant.csv")

    print("=== Processing Independent Feature Predictions ===")
    independent_feature_dir = base_path / "predictions_independent_feature"
    process_directory(independent_feature_dir, "overall_accuracies_independent_feature.csv")

    print("=== Processing Separated Character Predictions ===")
    sep_char_dir = base_path / "processed_predictions_sep_char"
    process_directory_sep_char(sep_char_dir, "overall_accuracies_sep_char.csv", base_path)

    print("=== Processing Binary Feature (Feature-Geometric) Predictions ===")
    binaryfeature_dir = base_path / "predictions_binaryfeature"
    process_directory(binaryfeature_dir, "overall_accuracies_binaryfeature.csv")

    print("All processing complete!")

if __name__ == "__main__":
    main()
