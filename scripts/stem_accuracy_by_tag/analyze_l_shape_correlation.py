"""
Analyze the relationship between V;IND;PRS;1;SG stem accuracy and V;SBJV* stem accuracies.
This tests the L-shape hypothesis: does knowing the 1SG Indicative stem predict Subjunctive performance?
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_l_shape_correlation(dataset="vanilla"):
    # Load the combined data
    data_path = f"../../data/analysis/stem_accuracy_by_lemma_tag/{dataset}/stem_acc_by_lemma_tag_all_models.csv"

    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    df = pd.read_csv(data_path)

    # Filter for relevant tags
    ind_1sg_tag = "V;IND;PRS;1;SG"
    sbjv_tags = [tag for tag in df['target_tag'].unique() if "V;SBJV" in tag]

    print(f"Analyzing relationship between {ind_1sg_tag} and {len(sbjv_tags)} Subjunctive tags.")

    # Pivot table: Rows=Lemma+Model+Run, Cols=Tag, Values=Accuracy
    # We group by model/run/lemma to see if the pattern holds across different training instances
    df_pivot = df.pivot_table(
        index=['model', 'lemma'],
        columns='target_tag',
        values='accuracy'
    ).reset_index()

    # Calculate average Subjunctive accuracy for each lemma/model
    df_pivot['avg_sbjv_accuracy'] = df_pivot[sbjv_tags].mean(axis=1)

    # Drop rows where we might be missing data
    df_analysis = df_pivot.dropna(subset=[ind_1sg_tag, 'avg_sbjv_accuracy'])

    # 1. Correlation Analysis
    pearson_corr, p_value = stats.pearsonr(df_analysis[ind_1sg_tag], df_analysis['avg_sbjv_accuracy'])
    spearman_corr, s_p_value = stats.spearmanr(df_analysis[ind_1sg_tag], df_analysis['avg_sbjv_accuracy'])

    print("\nStatistical Analysis:")
    print(f"Number of data points: {len(df_analysis)}")
    print(f"Pearson Correlation (r): {pearson_corr:.4f} (p={p_value:.4e})")
    print(f"Spearman Correlation (rho): {spearman_corr:.4f} (p={s_p_value:.4e})")

    # 2. Regression Analysis (Linear Model)
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df_analysis[ind_1sg_tag], df_analysis['avg_sbjv_accuracy']
    )
    print(f"\nLinear Regression Model: y = {slope:.4f}x + {intercept:.4f}")
    print(f"R-squared: {r_value**2:.4f}")

    # 3. Visualization
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_analysis, x=ind_1sg_tag, y='avg_sbjv_accuracy', alpha=0.6)

    # Add regression line
    x_vals = np.array([df_analysis[ind_1sg_tag].min(), df_analysis[ind_1sg_tag].max()])
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, color='red', linestyle='--', label=f'Fit: y={slope:.2f}x + {intercept:.2f}')

    plt.title(f'Correlation: 1SG Indicative vs. Avg Subjunctive Accuracy\n({dataset} dataset)', fontsize=14)
    plt.xlabel('Accuracy: V;IND;PRS;1;SG (%)', fontsize=12)
    plt.ylabel('Average Accuracy: V;SBJV;* (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    output_dir = f"../../data/analysis/plots/l_shape_analysis/{dataset}/"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}ind_vs_sbjv_correlation.png", dpi=300)
    plt.savefig(f"{output_dir}ind_vs_sbjv_correlation.pdf")
    print(f"\nPlot saved to {output_dir}ind_vs_sbjv_correlation.png")

    # 4. Detailed Lemma Breakdown (Top outliers/perfect matches)
    df_analysis['diff'] = df_analysis['avg_sbjv_accuracy'] - df_analysis[ind_1sg_tag]

    print("\nLemmas with largest discrepancy (Avg SBJV - IND 1SG):")
    print(df_analysis[['lemma', 'model', ind_1sg_tag, 'avg_sbjv_accuracy', 'diff']]
          .sort_values('diff', ascending=False).head(10))

    print("\nLemmas with smallest discrepancy (Best L-shape predictors):")
    df_analysis['abs_diff'] = df_analysis['diff'].abs()
    print(df_analysis[['lemma', 'model', ind_1sg_tag, 'avg_sbjv_accuracy', 'abs_diff']]
          .sort_values('abs_diff').head(10))

if __name__ == "__main__":
    # You can change this to loop over all datasets
    analyze_l_shape_correlation("vanilla")






