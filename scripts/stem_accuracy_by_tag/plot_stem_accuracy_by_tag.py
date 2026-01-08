"""
Plot stem accuracies grouped by target tag
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "l_nl_accuracies")))

from config import condition_10L_90NL, condition_50L_50NL, condition_90L_10NL


def plot_accuracy_by_tag(dataset="vanilla"):
    """Create plots showing stem accuracy by target tag for a given dataset."""
    
    data_dir = f"../../data/analysis/stem_accuracy_by_tag/{dataset}/"
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    # Read the combined results
    combined_file = f"{data_dir}stem_acc_by_tag_all_models.csv"
    if not os.path.exists(combined_file):
        print(f"Combined results file not found: {combined_file}")
        return
    
    df = pd.read_csv(combined_file)
    
    # Create output directory for plots
    plot_dir = f"../../data/analysis/plots/stem_accuracy_by_tag/{dataset}/"
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Plot average accuracy by tag across all models
    print(f"\nGenerating plots for {dataset}...")
    
    tag_summary = df.groupby('target_tag').agg({
        'correct': 'sum',
        'total': 'sum',
        'accuracy': 'mean'
    }).reset_index()
    tag_summary['overall_accuracy'] = (tag_summary['correct'] / tag_summary['total'] * 100).round(2)
    tag_summary = tag_summary.sort_values('overall_accuracy', ascending=True)
    
    # Create horizontal bar plot for overall accuracy by tag
    plt.figure(figsize=(10, max(6, len(tag_summary) * 0.3)))
    bars = plt.barh(range(len(tag_summary)), tag_summary['overall_accuracy'])
    
    # Color bars by accuracy level
    colors = ['#d62728' if acc < 50 else '#ff7f0e' if acc < 75 else '#2ca02c' 
              for acc in tag_summary['overall_accuracy']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.yticks(range(len(tag_summary)), tag_summary['target_tag'])
    plt.xlabel('Stem Accuracy (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Target Tag', fontsize=12, fontweight='bold')
    plt.title(f'Stem Accuracy by Target Tag - {dataset.replace("_", " ").title()}', 
              fontsize=14, fontweight='bold')
    plt.xlim(0, 100)
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(tag_summary.iterrows()):
        plt.text(row['overall_accuracy'] + 1, i, f"{row['overall_accuracy']:.1f}%", 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}stem_accuracy_by_tag_overall.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{plot_dir}stem_accuracy_by_tag_overall.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_dir}stem_accuracy_by_tag_overall.png")
    
    # 2. Plot accuracy by tag for each condition
    conditions = [
        ('10L_90NL', condition_10L_90NL, '10%L-90%NL'),
        ('50L_50NL', condition_50L_50NL, '50%L-50%NL'),
        ('90L_10NL', condition_90L_10NL, '90%L-10%NL')
    ]
    
    for cond_name, cond_models, cond_label in conditions:
        # Filter data for this condition
        cond_df = df[df['model'].isin(cond_models)]
        
        if len(cond_df) == 0:
            continue
        
        # Calculate average accuracy by tag for this condition
        cond_summary = cond_df.groupby('target_tag').agg({
            'correct': 'sum',
            'total': 'sum',
            'accuracy': 'mean'
        }).reset_index()
        cond_summary['overall_accuracy'] = (cond_summary['correct'] / cond_summary['total'] * 100).round(2)
        cond_summary = cond_summary.sort_values('overall_accuracy', ascending=True)
        
        plt.figure(figsize=(10, max(6, len(cond_summary) * 0.3)))
        bars = plt.barh(range(len(cond_summary)), cond_summary['overall_accuracy'])
        
        # Color bars by accuracy level
        colors = ['#d62728' if acc < 50 else '#ff7f0e' if acc < 75 else '#2ca02c' 
                  for acc in cond_summary['overall_accuracy']]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.yticks(range(len(cond_summary)), cond_summary['target_tag'])
        plt.xlabel('Stem Accuracy (%)', fontsize=12, fontweight='bold')
        plt.ylabel('Target Tag', fontsize=12, fontweight='bold')
        plt.title(f'Stem Accuracy by Target Tag - {cond_label} - {dataset.replace("_", " ").title()}', 
                  fontsize=14, fontweight='bold')
        plt.xlim(0, 100)
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(cond_summary.iterrows()):
            plt.text(row['overall_accuracy'] + 1, i, f"{row['overall_accuracy']:.1f}%", 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{plot_dir}stem_accuracy_by_tag_{cond_name}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{plot_dir}stem_accuracy_by_tag_{cond_name}.pdf", bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plot_dir}stem_accuracy_by_tag_{cond_name}.png")
    
    # 3. Create a heatmap showing accuracy by tag and condition
    pivot_data = []
    for cond_name, cond_models, cond_label in conditions:
        cond_df = df[df['model'].isin(cond_models)]
        if len(cond_df) == 0:
            continue
        cond_summary = cond_df.groupby('target_tag').agg({
            'correct': 'sum',
            'total': 'sum'
        }).reset_index()
        cond_summary['accuracy'] = (cond_summary['correct'] / cond_summary['total'] * 100).round(2)
        cond_summary['condition'] = cond_label
        pivot_data.append(cond_summary[['target_tag', 'condition', 'accuracy']])
    
    if pivot_data:
        pivot_df = pd.concat(pivot_data, ignore_index=True)
        pivot_table = pivot_df.pivot(index='target_tag', columns='condition', values='accuracy')
        
        plt.figure(figsize=(10, max(6, len(pivot_table) * 0.4)))
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', 
                   vmin=0, vmax=100, cbar_kws={'label': 'Stem Accuracy (%)'})
        plt.title(f'Stem Accuracy by Target Tag and Condition - {dataset.replace("_", " ").title()}', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Condition', fontsize=12, fontweight='bold')
        plt.ylabel('Target Tag', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{plot_dir}stem_accuracy_by_tag_heatmap.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{plot_dir}stem_accuracy_by_tag_heatmap.pdf", bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plot_dir}stem_accuracy_by_tag_heatmap.png")


def create_comparison_plot():
    """Create a comparison plot across all datasets."""
    
    datasets = ["vanilla", "feature_invariant", "independent_feature", "character_separated"]
    dataset_titles = ["Vanilla", "Feature Invariant", "Dual Source", "Character Separated"]
    
    all_data = []
    
    for dataset in datasets:
        data_dir = f"../../data/analysis/stem_accuracy_by_tag/{dataset}/"
        summary_file = f"{data_dir}stem_acc_by_tag_summary.csv"
        
        if os.path.exists(summary_file):
            df = pd.read_csv(summary_file)
            df['dataset'] = dataset
            all_data.append(df)
    
    if not all_data:
        print("No data found for comparison plot")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Get top 12 most common tags (by total occurrences)
    tag_totals = combined_df.groupby('target_tag')['total'].sum().sort_values(ascending=False)
    top_tags = tag_totals.head(12).index.tolist()
    
    # Filter to top tags
    plot_df = combined_df[combined_df['target_tag'].isin(top_tags)]
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(top_tags))
    width = 0.2
    
    colors = ['#D55E00', '#0072B2', '#CC79A7', '#009E73']
    
    for i, (dataset, title) in enumerate(zip(datasets, dataset_titles)):
        dataset_data = plot_df[plot_df['dataset'] == dataset]
        accuracies = [dataset_data[dataset_data['target_tag'] == tag]['overall_accuracy'].values[0] 
                     if len(dataset_data[dataset_data['target_tag'] == tag]) > 0 else 0 
                     for tag in top_tags]
        ax.bar(x + i * width, accuracies, width, label=title, color=colors[i])
    
    ax.set_xlabel('Target Tag', fontsize=12, fontweight='bold')
    ax.set_ylabel('Stem Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Stem Accuracy by Target Tag - Model Comparison (Top 12 Tags)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(top_tags, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    plot_dir = "../../data/analysis/plots/stem_accuracy_by_tag/"
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.savefig(f"{plot_dir}stem_accuracy_by_tag_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{plot_dir}stem_accuracy_by_tag_comparison.pdf", bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparison plot: {plot_dir}stem_accuracy_by_tag_comparison.png")


if __name__ == "__main__":
    # Generate plots for each dataset
    datasets = ["vanilla", "feature_invariant", "independent_feature", "character_separated"]
    
    for dataset in datasets:
        plot_accuracy_by_tag(dataset)
    
    # Create comparison plot
    create_comparison_plot()
    
    print("\n" + "="*80)
    print("All plots generated successfully!")
    print("="*80)







