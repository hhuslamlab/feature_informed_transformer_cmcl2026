"""
L-Shape Focused Statistical Analysis (Lemma-Specific & Condition-Specific)
Analyzes the relationship between 1SG Indicative (the "L-base") and INDIVIDUAL Subjunctive forms.
Avoids averaging Subjunctive forms to prevent masking variance.
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
    data_path = f"../../data/accuracies/stem_accuracy_by_lemma_tag/{dataset}/stem_acc_by_lemma_tag_all_models.csv"

    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return None

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} data points from {dataset} dataset")
    return df

def analyze_condition_specific_l_shape_granular(df):
    """
    Calculate L-shape correlation separately for each condition.
    Correlates IND;1;SG with EACH Subjunctive form independently.
    """
    print("\n" + "="*60)
    print("ANALYSIS: GRANULAR L-SHAPE CORRELATIONS (IND;1;SG vs EACH SBJV TAG)")
    print("="*60)

    conditions = df['condition'].unique()
    base_tag = "V;IND;PRS;1;SG"
    
    all_results = []
    
    for condition in conditions:
        print(f"\nProcessing Condition: {condition}")
        
        # Filter data for this condition
        cond_df = df[df['condition'] == condition]
        
        # Pivot to get tags as columns
        df_pivot = cond_df.pivot_table(
            index=['lemma', 'model'],
            columns='target_tag',
            values='accuracy'
        ).reset_index()

        if base_tag not in df_pivot.columns:
            print(f"  Error: Base tag {base_tag} not found in data.")
            continue

        sbjv_tags = [col for col in df_pivot.columns if "V;SBJV" in col]
        
        lemmas = df_pivot['lemma'].unique()
        lemma_correlations = []
        
        for lemma in lemmas:
            lemma_data = df_pivot[df_pivot['lemma'] == lemma]
            
            # Calculate correlation for EACH subjunctive tag separately
            for sbjv_tag in sbjv_tags:
                # Need variance to calculate correlation
                if len(lemma_data) > 2 and lemma_data[base_tag].std() > 0 and lemma_data[sbjv_tag].std() > 0:
                    corr, p_val = stats.pearsonr(lemma_data[base_tag], lemma_data[sbjv_tag])
                    lemma_correlations.append({
                        'condition': condition,
                        'lemma': lemma,
                        'sbjv_tag': sbjv_tag,
                        'correlation': corr
                    })
                else:
                    # Handle zero variance cases (consistency check)
                    base_mean = lemma_data[base_tag].mean()
                    sbjv_mean = lemma_data[sbjv_tag].mean()
                    
                    # Consistent if both are 100% or both are 0%
                    # Or if they are remarkably close (e.g. consistent errors)
                    is_consistent = (base_mean == sbjv_mean) or \
                                  (base_mean > 99 and sbjv_mean > 99) or \
                                  (base_mean < 1 and sbjv_mean < 1)
                    
                    lemma_correlations.append({
                        'condition': condition,
                        'lemma': lemma,
                        'sbjv_tag': sbjv_tag,
                        'correlation': 1.0 if is_consistent else 0.0,
                        'note': 'Zero variance'
                    })

        results_df = pd.DataFrame(lemma_correlations)
        all_results.append(results_df)
        
        # Statistics for this condition (aggregated across all SBJV tags)
        mean_corr = results_df['correlation'].mean()
        median_corr = results_df['correlation'].median()
        
        # Strong adherence: correlation > 0.8 across the majority of SBJV forms
        # We group by lemma first to see how many lemmas are "L-shape consistent"
        lemma_scores = results_df.groupby('lemma')['correlation'].mean()
        strong_lemmas = len(lemma_scores[lemma_scores > 0.8])
        total_lemmas = len(lemma_scores)
        
        print(f"  Mean Correlation (Granular): {mean_corr:.4f}")
        print(f"  Median Correlation (Granular): {median_corr:.4f}")
        print(f"  Strong L-Shape Lemmas (>0.8 avg corr): {strong_lemmas}/{total_lemmas} ({strong_lemmas/total_lemmas*100:.1f}%)")

    if not all_results:
        return None
        
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='condition', y='correlation', hue='sbjv_tag', data=combined_results, order=sorted(conditions))
    plt.title('L-Shape Correlation Strength by Condition and Subjunctive Tag')
    plt.ylabel('Pearson Correlation')
    plt.xlabel('Condition')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('../../data/accuracies/plots/comprehensive_analysis/l_shape_granular_boxplot.png', dpi=300)
    plt.close()

    return combined_results

def main():
    dataset = "vanilla"
    print(f"Running Granular L-Shape analysis for {dataset} dataset...")
    
    os.makedirs('../../data/accuracies/plots/comprehensive_analysis', exist_ok=True)
    
    df = load_data(dataset)
    if df is None:
        return

    analyze_condition_specific_l_shape_granular(df)

    print("\nAnalysis complete! Results saved to ../../data/accuracies/plots/comprehensive_analysis/")

if __name__ == "__main__":
    main()






