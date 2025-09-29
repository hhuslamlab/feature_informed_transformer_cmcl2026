#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_spanish_accuracies(file_path):
    """Parse the Spanish accuracy CSV file and extract accuracies by group."""
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Dictionary to store accuracies by group
    accuracies = defaultdict(list)
    
    # Process each row
    for _, row in df.iterrows():
        filename = row['filename']
        accuracy = row['accuracy']
        
        # Extract group name from filename (e.g., "10L_90NL_1_1" -> "10L_90NL")
        if '_' in filename:
            group_name = '_'.join(filename.split('_')[:2])
            accuracies[group_name].append(accuracy)
    
    return accuracies

def create_box_plot(accuracies):
    """Create a box plot of accuracies by group."""
    
    # Prepare data for plotting
    groups = []
    data = []
    
    for group_name in ['10L_90NL', '50L_50NL', '90L_10NL']:
        if group_name in accuracies:
            groups.append(group_name)
            data.append(accuracies[group_name])
    
    # Create the box plot
    plt.figure(figsize=(10, 6))
    box_plot = plt.boxplot(data, labels=groups, patch_artist=True)
    
    # Customize colors
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add labels
    plt.xlabel('Conditions', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Set y-axis range from 0 to 100
    plt.ylim(0, 100)
    
    # Add statistics text
    for i, (group, values) in enumerate(zip(groups, data)):
        mean_val = np.mean(values)
        std_val = np.std(values)
        plt.text(i + 1, max(values) + 2, f'Mean: {mean_val:.1f}%\nStd: {std_val:.1f}%', 
                ha='center', va='bottom', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('/home/akhilesh/projects/research/feature_invariant_transformer/analysis/accuracies/spanish_accuracy_boxplot.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("Spanish Model Summary Statistics:")
    print("=" * 50)
    for group_name in ['10L_90NL', '50L_50NL', '90L_10NL']:
        if group_name in accuracies:
            values = accuracies[group_name]
            print(f"{group_name}:")
            print(f"  Count: {len(values)}")
            print(f"  Mean: {np.mean(values):.2f}%")
            print(f"  Std: {np.std(values):.2f}%")
            print(f"  Min: {np.min(values):.2f}%")
            print(f"  Max: {np.max(values):.2f}%")
            print(f"  Values: {[f'{v:.1f}%' for v in values]}")
            print()

def remove_duplicates_and_analyze(file_path):
    """Remove duplicate entries and analyze the data."""
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Remove duplicates based on filename
    df_unique = df.drop_duplicates(subset=['filename'], keep='first')
    
    print("Original data points:", len(df))
    print("Unique data points:", len(df_unique))
    print()
    
    return df_unique

if __name__ == "__main__":
    # File path for Spanish data
    file_path = '/home/akhilesh/projects/research/modeling_spanish_naacl25/data/analysis/accuracies/combine.csv'
    
    # Remove duplicates and analyze
    df_unique = remove_duplicates_and_analyze(file_path)
    
    # Parse the accuracy data
    accuracies = parse_spanish_accuracies(file_path)
    
    # Create and display the box plot
    create_box_plot(accuracies)
