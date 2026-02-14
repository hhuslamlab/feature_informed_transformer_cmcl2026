"""
Calculate stem accuracy grouped by target tag (the last tag in the source file)
"""
import sys
import os
import pandas as pd
import re
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "l_nl_accuracies")))

from config import AR_SUFFIX_DICT, ER_SUFFIX_DICT, IR_SUFFIX_DICT, all_models


def get_stem(form, suffixes):
    """
    Extract the stem by removing the suffix from the form
    """
    if form[-4:] in suffixes:
        return form[: len(form) - 4]
    if form[-3:] in suffixes:
        return form[: len(form) - 3]
    if form[-2:] in suffixes:
        return form[: len(form) - 2]
    if form[-1:] in suffixes:
        return form[: len(form) - 1]
    return form


def extract_target_tag(line):
    """
    Extract the last tag (target tag) from a source line.
    Example: "t ɾ a d ˈ u s e n <V;IND;PRS;3;PL> # ... # <V;SBJV;PRS;2;PL>"
    Returns: "V;SBJV;PRS;2;PL"
    """
    # Find all tags in angle brackets
    tags = re.findall(r'<([^>]+)>', line)
    if tags:
        # Return the last tag (without the angle brackets)
        return tags[-1]
    return None


def process_predictions(pred_type, pred_dir_name):
    """Process predictions for a given prediction type and calculate stem accuracy by tag"""
    print(f"\nProcessing {pred_type} predictions...")
    
    # Prepare suffixes
    ar_suffixes = list(AR_SUFFIX_DICT.values())
    er_suffixes = list(ER_SUFFIX_DICT.values())
    ir_suffixes = list(IR_SUFFIX_DICT.values())
    suffixes = ar_suffixes + er_suffixes + ir_suffixes
    suffixes = sorted(list(set(suffixes)), key=len, reverse=True)

    # Store all results for overall analysis
    all_results = []

    for model in all_models:
        print(f"  Processing {model}...")
        condition = model.split("_")[0] + "_" + model.split("_")[1]
        run = model.split("_")[2]

        # Read predictions
        if pred_type == "vanilla" or pred_type == "character_separated":
            pred_file = f"../../data/predictions/{pred_dir_name}/{model}.txt"
            try:
                # TXT format - may have index,prediction format
                pred_data = []
                with open(pred_file, 'r') as f:
                    pred_lines = f.readlines()
                    for line in pred_lines:
                        line = line.strip()
                        if ',' in line:
                            # Format: "index,prediction"
                            parts = line.split(',', 1)  # Split only on first comma
                            idx = int(parts[0])
                            # Normalize: remove spaces, remove stress markers, then strip
                            pred = parts[1].replace(" ", "").replace("ˈ", "").strip()
                            pred_data.append((idx, pred))
                        else:
                            # Direct prediction format (no index)
                            # Normalize: remove spaces, remove stress markers, then strip
                            pred = line.replace(" ", "").replace("ˈ", "").strip()
                            pred_data.append((len(pred_data), pred))
                
                # Sort by index to ensure correct alignment with test data
                pred_data.sort(key=lambda x: x[0])
                preds = [pred for _, pred in pred_data]
            except Exception as e:
                print(f"    Error reading {pred_file}: {e}")
                continue
        else:
            # TSV format for dual_source and feature_invariant
            pred_dir = f"../../data/predictions/{pred_dir_name}/"
            matching_files = [f for f in os.listdir(pred_dir) if f.startswith(model) and f.endswith('.tsv')]
            if matching_files:
                pred_file = f"../../data/predictions/{pred_dir_name}/{matching_files[0]}"
            else:
                print(f"    No prediction file found for {model} in {pred_dir_name}")
                continue
            
            try:
                df_pred = pd.read_csv(pred_file, sep='\t')
                # Normalize: remove spaces, remove stress markers, then strip
                preds = [str(pred).replace(" ", "").replace("ˈ", "").strip() for pred in df_pred['prediction']]
            except Exception as e:
                print(f"    Error reading {pred_file}: {e}")
                continue

        # Read test data (targets)
        test_dir = f"../../data/{condition}/test/run{run}/"
        test_file = f"test.{model}.tgt"
        try:
            with open(test_dir + test_file) as f:
                test_data = f.readlines()
                # Normalize: remove spaces, remove stress markers, then strip
                test_data = [item.replace(" ", "").replace("ˈ", "").strip() for item in test_data]
        except Exception as e:
            print(f"    Error reading test file {test_dir + test_file}: {e}")
            continue

        # Read source file to extract target tags
        src_file = f"test.{model}.src"
        try:
            with open(test_dir + src_file) as f:
                src_lines = f.readlines()
                target_tags = [extract_target_tag(line.strip()) for line in src_lines]
        except Exception as e:
            print(f"    Error reading source file {test_dir + src_file}: {e}")
            continue

        # Validate data alignment
        if not (len(preds) == len(test_data) == len(target_tags)):
            print(f"    Warning: Data length mismatch for {model}")
            print(f"      Predictions: {len(preds)}, Test data: {len(test_data)}, Target tags: {len(target_tags)}")
            min_len = min(len(preds), len(test_data), len(target_tags))
            preds = preds[:min_len]
            test_data = test_data[:min_len]
            target_tags = target_tags[:min_len]

        # Extract stems
        preds_stems = []
        test_stems = []
        
        for pred, test in zip(preds, test_data):
            stem_pred = get_stem(pred, suffixes)
            stem_test = get_stem(test, suffixes)
            preds_stems.append(stem_pred)
            test_stems.append(stem_test)

        # Calculate stem accuracy by target tag
        tag_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for pred_stem, test_stem, tag in zip(preds_stems, test_stems, target_tags):
            if tag is not None:
                tag_stats[tag]['total'] += 1
                if pred_stem == test_stem:
                    tag_stats[tag]['correct'] += 1

        # Create DataFrame for this model
        for tag, stats in tag_stats.items():
            accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            all_results.append({
                'model': model,
                'condition': condition,
                'run': run,
                'target_tag': tag,
                'correct': stats['correct'],
                'total': stats['total'],
                'accuracy': round(accuracy, 2)
            })

        # Save per-model detailed results
        model_df = pd.DataFrame([
            {
                'target_tag': tag,
                'correct': stats['correct'],
                'total': stats['total'],
                'accuracy': round((stats['correct'] / stats['total'] * 100), 2) if stats['total'] > 0 else 0
            }
            for tag, stats in tag_stats.items()
        ])
        
        # Sort by accuracy for easier reading
        model_df = model_df.sort_values('accuracy', ascending=False)
        
        # Create output directory
        output_dir = f"../../data/analysis/stem_accuracy_by_tag/{pred_type}/"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model-specific results
        model_df.to_csv(f"{output_dir}stem_acc_by_tag_{model}.csv", index=False)
        print(f"    Saved results to {output_dir}stem_acc_by_tag_{model}.csv")

    # Save combined results for all models
    if all_results:
        combined_df = pd.DataFrame(all_results)
        combined_df = combined_df.sort_values(['condition', 'run', 'target_tag'])
        
        output_dir = f"../../data/analysis/stem_accuracy_by_tag/{pred_type}/"
        os.makedirs(output_dir, exist_ok=True)
        combined_df.to_csv(f"{output_dir}stem_acc_by_tag_all_models.csv", index=False)
        print(f"\n  Saved combined results to {output_dir}stem_acc_by_tag_all_models.csv")
        
        # Create summary statistics by tag across all models
        summary_by_tag = combined_df.groupby('target_tag').agg({
            'correct': 'sum',
            'total': 'sum',
            'accuracy': 'mean'
        }).reset_index()
        summary_by_tag['overall_accuracy'] = (summary_by_tag['correct'] / summary_by_tag['total'] * 100).round(2)
        summary_by_tag = summary_by_tag.sort_values('overall_accuracy', ascending=False)
        summary_by_tag.to_csv(f"{output_dir}stem_acc_by_tag_summary.csv", index=False)
        print(f"  Saved summary statistics to {output_dir}stem_acc_by_tag_summary.csv")


if __name__ == "__main__":
    # Process prediction types
    prediction_types = [
        ("vanilla", "predictions_vanilla"),
        ("character_separated", "processed_predictions_sep_char"),
        ("independent_feature", "predictions_independent_feature"),
        ("feature_invariant", "predictions_feature_invariant"),
        ("feature_geometric", "predictions_binaryfeature")
    ]

    for pred_type, pred_dir in prediction_types:
        process_predictions(pred_type, pred_dir)

    print("\n" + "="*80)
    print("All stem accuracy by tag calculations completed!")
    print("="*80)







