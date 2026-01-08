"""
L-Shape Focused Statistical Analysis (Lemma-Specific & Condition-Specific)
Analyzes the relationship between 1SG Indicative (the "L-base") and Subjunctive forms.
Calculates predictiveness PER LEMMA and PER CONDITION across models.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_data(dataset="vanilla"):
    """Load the lemma-tag accuracy data."""
    data_path = f"../../data/analysis/stem_accuracy_by_lemma_tag/{dataset}/stem_acc_by_lemma_tag_all_models.csv"

    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return None

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} data points from {dataset} dataset")
    return df

def analyze_condition_specific_l_shape(df):
    """
    Calculate L-shape correlation separately for each condition (10L_90NL, etc.)
    """
    print("\n" + "="*60)
    print("ANALYSIS: LEMMA-SPECIFIC L-SHAPE CORRELATIONS BY CONDITION")
    print("="*60)

    conditions = df['condition'].unique()
    
    all_results = []
    
    for condition in conditions:
        print(f"\nProcessing Condition: {condition}")
        
        # Filter data for this condition
        cond_df = df[df['condition'] == condition]
        
        # Pivot
        df_pivot = cond_df.pivot_table(
            index=['lemma', 'model'],
            columns='target_tag',
            values='accuracy'
        ).reset_index()

        base_tag = "V;IND;PRS;1;SG"
        sbjv_tags = [col for col in df_pivot.columns if "V;SBJV" in col]
        
        if base_tag not in df_pivot.columns:
            print(f"  Error: Base tag {base_tag} not found in data.")
            continue

        # Calculate mean SBJV accuracy
        df_pivot['mean_sbjv_acc'] = df_pivot[sbjv_tags].mean(axis=1)
        
        lemmas = df_pivot['lemma'].unique()
        lemma_correlations = []
        
        for lemma in lemmas:
            lemma_data = df_pivot[df_pivot['lemma'] == lemma]
            
            # Need variance to calculate correlation
            if len(lemma_data) > 2 and lemma_data[base_tag].std() > 0 and lemma_data['mean_sbjv_acc'].std() > 0:
                corr, p_val = stats.pearsonr(lemma_data[base_tag], lemma_data['mean_sbjv_acc'])
                lemma_correlations.append({
                    'condition': condition,
                    'lemma': lemma,
                    'correlation': corr,
                    'base_mean': lemma_data[base_tag].mean(),
                    'sbjv_mean': lemma_data['mean_sbjv_acc'].mean()
                })
            else:
                # Handle zero variance cases
                base_mean = lemma_data[base_tag].mean()
                sbjv_mean = lemma_data['mean_sbjv_acc'].mean()
                is_consistent = (base_mean == 100 and sbjv_mean == 100) or (base_mean == 0 and sbjv_mean == 0)
                
                lemma_correlations.append({
                    'condition': condition,
                    'lemma': lemma,
                    'correlation': 1.0 if is_consistent else 0.0, 
                    'base_mean': base_mean,
                    'sbjv_mean': sbjv_mean,
                    'note': 'Zero variance'
                })

        results_df = pd.DataFrame(lemma_correlations)
        all_results.append(results_df)
        
        # Statistics for this condition
        mean_corr = results_df['correlation'].mean()
        median_corr = results_df['correlation'].median()
        strong_lemmas = len(results_df[results_df['correlation'] > 0.8])
        total_lemmas = len(results_df)
        
        print(f"  Mean Correlation: {mean_corr:.4f}")
        print(f"  Median Correlation: {median_corr:.4f}")
        print(f"  Strong L-Shape Lemmas (>0.8): {strong_lemmas}/{total_lemmas} ({strong_lemmas/total_lemmas*100:.1f}%)")

    # Combine all results for plotting comparison
    if not all_results:
        return None
        
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Visualization: Boxplot comparison
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='condition', y='correlation', data=combined_results, order=sorted(conditions))
    plt.title('L-Shape Correlation Strength by Condition')
    plt.ylabel('Pearson Correlation (Lemma-Specific)')
    plt.xlabel('Data Condition')
    plt.tight_layout()
    plt.savefig('../../data/analysis/plots/comprehensive_analysis/l_shape_by_condition_boxplot.png', dpi=300)
    plt.close()
    
    # Visualization: Distribution comparison (KDE)
    plt.figure(figsize=(10, 6))
    for condition in sorted(conditions):
        subset = combined_results[combined_results['condition'] == condition]
        sns.kdeplot(subset['correlation'], label=condition, clip=(-1, 1), bw_adjust=0.6)
    
    plt.title('Distribution of L-Shape Correlations by Condition')
    plt.xlabel('Correlation Coefficient')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../../data/analysis/plots/comprehensive_analysis/l_shape_by_condition_dist.png', dpi=300)
    plt.close()

    return combined_results

def main():
    dataset = "vanilla"
    print(f"Running Condition-Specific L-Shape analysis for {dataset} dataset...")
    
    os.makedirs('../../data/analysis/plots/comprehensive_analysis', exist_ok=True)
    
    df = load_data(dataset)
    if df is None:
        return

    analyze_condition_specific_l_shape(df)

    print("\nAnalysis complete! Results saved to ../../data/analysis/plots/comprehensive_analysis/")

if __name__ == "__main__":
    main()
