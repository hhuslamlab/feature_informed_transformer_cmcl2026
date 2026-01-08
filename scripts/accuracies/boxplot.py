#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def create_accuracy_boxplot():
    """Create boxplot comparing accuracies between vanilla, dual source, feature invariant, and character separated transformers"""

    # Paths to accuracy files
    vanilla_file = Path("../../data/analysis/accuracies/overall_accuracies.csv")
    independent_feature_file = Path("../../data/analysis/accuracies/overall_accuracies_independent_feature.csv")
    feature_invariant_file = Path("../../data/analysis/accuracies/overall_accuracies_feature_invariant.csv")
    sep_char_file = Path("../../data/analysis/accuracies/overall_accuracies_sep_char.csv")

    if not vanilla_file.exists() or not independent_feature_file.exists() or not feature_invariant_file.exists() or not sep_char_file.exists():
        print(f"Error: Accuracy files not found")
        return

    # Read all CSV files
    df_vanilla = pd.read_csv(vanilla_file)
    df_dual = pd.read_csv(independent_feature_file)
    df_feature = pd.read_csv(feature_invariant_file)
    df_sep_char = pd.read_csv(sep_char_file)

    # Check if required columns exist
    for df, name in [(df_vanilla, 'vanilla'), (df_dual, 'dual_source'), (df_feature, 'feature_invariant'), (df_sep_char, 'sep_char')]:
        if 'filename' not in df.columns or 'accuracy' not in df.columns:
            print(f"Error: {name} CSV must have 'filename' and 'accuracy' columns")
            return

    # Add dataset labels
    df_vanilla['dataset'] = 'Vanilla'
    df_dual['dataset'] = 'Dual Source'
    df_feature['dataset'] = 'Feature Invariant'
    df_sep_char['dataset'] = 'Character separated'

    # Combine datasets in desired order: Vanilla, Feature Invariant, Dual Source, Character separated
    df_combined = pd.concat([df_vanilla, df_feature, df_dual, df_sep_char], ignore_index=True)
    
    # Set categorical order to ensure correct ordering in plot
    df_combined['dataset'] = pd.Categorical(df_combined['dataset'], 
                                           categories=['Vanilla', 'Feature Invariant', 'Dual Source', 'Character separated'],
                                           ordered=True)

    # Extract condition from filename (first part before first underscore)
    df_combined['condition'] = df_combined['filename'].str.extract(r'^([^_]+_[^_]+)')

    # Filter to only include the three main conditions
    valid_conditions = ['10L_90NL', '50L_50NL', '90L_10NL']
    df_filtered = df_combined[df_combined['condition'].isin(valid_conditions)]

    if len(df_filtered) == 0:
        print("Error: No valid conditions found in the data")
        return

    # Create the plots directory if it doesn't exist
    plots_dir = Path("../../data/analysis/accuracies/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create the boxplot - optimized for 2-column LaTeX layout
    # Wider figure size for better readability when scaled to \columnwidth
    plt.figure(figsize=(8, 5))  # Optimized for \columnwidth in 2-column layout, wider for 4 datasets

    # Create boxplot with hue for dataset comparison
    # Palette order: Vanilla, Feature Invariant, Dual Source, Character separated
    ax = sns.boxplot(data=df_filtered, x='condition', y='accuracy', hue='dataset',
                     palette=['#D55E00', '#0072B2', '#CC79A7', '#009E73'])  # Color-blind friendly: Vermillion, Blue, Reddish purple, Bluish green

    # Add strip plot for individual points (without legend)
    sns.stripplot(data=df_filtered, x='condition', y='accuracy', hue='dataset',
                  dodge=True, color='black', alpha=0.6, size=2.5, jitter=True,
                  legend=False)


    # Customize the plot - larger fonts for 2-column layout
    plt.xlabel('Condition', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=16, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(title='', loc='lower right', framealpha=0.9, fontsize=16)
    
    # Set custom x-axis labels
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['10%L-90%NL', '50%L-50%NL', '90%L-10%NL'])
    
    # Increase tick label sizes
    ax.tick_params(labelsize=16)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    output_file = plots_dir / "accuracy_comparison_boxplot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Boxplot saved to: {output_file}")

    # Also save as PDF for publication quality
    pdf_file = plots_dir / "accuracy_comparison_boxplot.pdf"
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"PDF version saved to: {pdf_file}")

    # Show statistics
    print("\nAccuracy Statistics by Condition and Dataset:")
    print("=" * 50)
    for dataset in ['Vanilla', 'Feature Invariant', 'Dual Source', 'Character separated']:
        print(f"\n{dataset}:")
        dataset_data = df_filtered[df_filtered['dataset'] == dataset]
        for condition in valid_conditions:
            condition_data = dataset_data[dataset_data['condition'] == condition]['accuracy']
            if len(condition_data) > 0:
                print(f"  {condition}:")
                print(f"    Count: {len(condition_data)}")
                print(f"    Mean: {condition_data.mean():.1f}")
                print(f"    Median: {condition_data.median():.1f}")
                print(f"    Min: {condition_data.min():.1f}")
                print(f"    Max: {condition_data.max():.1f}")

    plt.show()

if __name__ == "__main__":
    create_accuracy_boxplot()
