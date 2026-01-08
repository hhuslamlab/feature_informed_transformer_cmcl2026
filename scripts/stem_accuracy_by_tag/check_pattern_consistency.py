"""
Check if V;IND;PRS;1;SG is consistently the best and V;IND;PRS;1;PL is consistently the worst
across all model files.
"""
import pandas as pd
import os
from glob import glob

def analyze_all_files():
    data_dir = "../../data/analysis/stem_accuracy_by_lemma_tag/vanilla"
    
    # Get all individual model files (exclude summary and all_models)
    files = glob(os.path.join(data_dir, "stem_acc_by_lemma_tag_*.csv"))
    files = [f for f in files if "summary" not in f and "all_models" not in f]
    
    print(f"Analyzing {len(files)} model files...\n")
    
    all_results = []
    
    for file_path in sorted(files):
        filename = os.path.basename(file_path)
        df = pd.read_csv(file_path)
        
        # Calculate average accuracy per target_tag
        tag_stats = df.groupby('target_tag').agg({
            'correct': 'sum',
            'total': 'sum'
        }).reset_index()
        tag_stats['accuracy'] = (tag_stats['correct'] / tag_stats['total']) * 100
        
        # Find best and worst
        best_tag = tag_stats.loc[tag_stats['accuracy'].idxmax()]
        worst_tag = tag_stats.loc[tag_stats['accuracy'].idxmin()]
        
        all_results.append({
            'file': filename,
            'best_tag': best_tag['target_tag'],
            'best_acc': best_tag['accuracy'],
            'worst_tag': worst_tag['target_tag'],
            'worst_acc': worst_tag['accuracy']
        })
    
    results_df = pd.DataFrame(all_results)
    
    # Count how many times each tag is best/worst
    best_counts = results_df['best_tag'].value_counts()
    worst_counts = results_df['worst_tag'].value_counts()
    
    print("="*60)
    print("BEST PERFORMING TAGS (Frequency)")
    print("="*60)
    for tag, count in best_counts.items():
        pct = (count / len(results_df)) * 100
        print(f"{tag}: {count}/{len(results_df)} files ({pct:.1f}%)")
    
    print("\n" + "="*60)
    print("WORST PERFORMING TAGS (Frequency)")
    print("="*60)
    for tag, count in worst_counts.items():
        pct = (count / len(results_df)) * 100
        print(f"{tag}: {count}/{len(results_df)} files ({pct:.1f}%)")
    
    # Check if pattern holds
    ind_1sg_best = (results_df['best_tag'] == 'V;IND;PRS;1;SG').sum()
    ind_1pl_worst = (results_df['worst_tag'] == 'V;IND;PRS;1;PL').sum()
    
    print("\n" + "="*60)
    print("PATTERN CHECK")
    print("="*60)
    print(f"V;IND;PRS;1;SG is best in: {ind_1sg_best}/{len(results_df)} files ({(ind_1sg_best/len(results_df)*100):.1f}%)")
    print(f"V;IND;PRS;1;PL is worst in: {ind_1pl_worst}/{len(results_df)} files ({(ind_1pl_worst/len(results_df)*100):.1f}%)")
    
    # Also show average accuracy per tag across all files
    print("\n" + "="*60)
    print("AVERAGE ACCURACY BY TAG (across all files)")
    print("="*60)
    
    all_tag_accs = []
    for file_path in sorted(files):
        df = pd.read_csv(file_path)
        tag_stats = df.groupby('target_tag').agg({
            'correct': 'sum',
            'total': 'sum'
        }).reset_index()
        tag_stats['accuracy'] = (tag_stats['correct'] / tag_stats['total']) * 100
        all_tag_accs.append(tag_stats[['target_tag', 'accuracy']])
    
    combined = pd.concat(all_tag_accs)
    avg_by_tag = combined.groupby('target_tag')['accuracy'].mean().sort_values(ascending=False)
    
    for tag, acc in avg_by_tag.items():
        print(f"{tag}: {acc:.2f}%")

if __name__ == "__main__":
    analyze_all_files()






