"""
Calculate global stem accuracy per condition for reporting.
"""
import pandas as pd
import os

def report_accuracies():
    data_path = "../../data/accuracies/stem_accuracy_by_lemma_tag/vanilla/stem_acc_by_lemma_tag_all_models.csv"
    if not os.path.exists(data_path):
        print("Data file not found")
        return

    df = pd.read_csv(data_path)
    
    print("\nStem Accuracy by Condition:")
    print("-" * 40)
    
    # Group by condition and calculate weighted average accuracy
    stats = df.groupby('condition').apply(
        lambda x: pd.Series({
            'accuracy': (x['correct'].sum() / x['total'].sum()) * 100,
            'total_items': x['total'].sum()
        })
    )
    
    print(stats)

if __name__ == "__main__":
    report_accuracies()






