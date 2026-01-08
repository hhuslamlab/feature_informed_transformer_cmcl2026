"""
Test script to analyze stem accuracy by tag for a single model
This is useful for testing and debugging
"""
import sys
import os
import pandas as pd
import re
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "l_nl_accuracies")))

from config import AR_SUFFIX_DICT, ER_SUFFIX_DICT, IR_SUFFIX_DICT


def get_stem(form, suffixes):
    """Extract the stem by removing the suffix from the form"""
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
    """Extract the last tag (target tag) from a source line"""
    tags = re.findall(r'<([^>]+)>', line)
    if tags:
        return tags[-1]
    return None


def analyze_single_model(model, pred_type="vanilla", pred_dir_name="predictions_vanilla"):
    """Analyze stem accuracy by tag for a single model"""
    print(f"\nAnalyzing {model} ({pred_type})...")
    print("="*60)
    
    # Prepare suffixes
    ar_suffixes = list(AR_SUFFIX_DICT.values())
    er_suffixes = list(ER_SUFFIX_DICT.values())
    ir_suffixes = list(IR_SUFFIX_DICT.values())
    suffixes = ar_suffixes + er_suffixes + ir_suffixes
    suffixes = sorted(list(set(suffixes)), key=len, reverse=True)

    condition = model.split("_")[0] + "_" + model.split("_")[1]
    run = model.split("_")[2]

    # Read predictions
    if pred_type == "vanilla" or pred_type == "character_separated":
        pred_file = f"../../../{pred_dir_name}/{model}.txt"
        try:
            pred_data = []
            with open(pred_file, 'r') as f:
                pred_lines = f.readlines()
                for line in pred_lines:
                    line = line.strip()
                    if ',' in line:
                        parts = line.split(',', 1)
                        idx = int(parts[0])
                        pred = parts[1].replace(" ", "").replace("ˈ", "").strip()
                        pred_data.append((idx, pred))
                    else:
                        pred = line.replace(" ", "").replace("ˈ", "").strip()
                        pred_data.append((len(pred_data), pred))
            
            pred_data.sort(key=lambda x: x[0])
            preds = [pred for _, pred in pred_data]
        except Exception as e:
            print(f"Error reading {pred_file}: {e}")
            return
    else:
        pred_dir = f"../../../{pred_dir_name}/"
        matching_files = [f for f in os.listdir(pred_dir) if f.startswith(model) and f.endswith('.tsv')]
        if matching_files:
            pred_file = f"../../../{pred_dir_name}/{matching_files[0]}"
        else:
            print(f"No prediction file found for {model} in {pred_dir_name}")
            return
        
        try:
            df_pred = pd.read_csv(pred_file, sep='\t')
            preds = [str(pred).replace(" ", "").replace("ˈ", "").strip() for pred in df_pred['prediction']]
        except Exception as e:
            print(f"Error reading {pred_file}: {e}")
            return

    # Read test data
    test_dir = f"../../../analysis/{condition}/test/run{run}/"
    test_file = f"test.{model}.tgt"
    try:
        with open(test_dir + test_file) as f:
            test_data = f.readlines()
            test_data = [item.replace(" ", "").replace("ˈ", "").strip() for item in test_data]
    except Exception as e:
        print(f"Error reading test file {test_dir + test_file}: {e}")
        return

    # Read source file to extract target tags
    src_file = f"test.{model}.src"
    try:
        with open(test_dir + src_file) as f:
            src_lines = f.readlines()
            target_tags = [extract_target_tag(line.strip()) for line in src_lines]
    except Exception as e:
        print(f"Error reading source file {test_dir + src_file}: {e}")
        return

    print(f"\nData loaded:")
    print(f"  Predictions: {len(preds)}")
    print(f"  Test data: {len(test_data)}")
    print(f"  Target tags: {len(target_tags)}")

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

    # Print results
    print(f"\nStem Accuracy by Target Tag:")
    print("-" * 60)
    print(f"{'Tag':<30} {'Correct':<10} {'Total':<10} {'Accuracy'}")
    print("-" * 60)
    
    sorted_tags = sorted(tag_stats.items(), 
                        key=lambda x: (x[1]['correct'] / x[1]['total']) if x[1]['total'] > 0 else 0,
                        reverse=True)
    
    for tag, stats in sorted_tags:
        accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"{tag:<30} {stats['correct']:<10} {stats['total']:<10} {accuracy:.2f}%")
    
    # Overall statistics
    total_correct = sum(stats['correct'] for stats in tag_stats.values())
    total_items = sum(stats['total'] for stats in tag_stats.values())
    overall_accuracy = (total_correct / total_items * 100) if total_items > 0 else 0
    
    print("-" * 60)
    print(f"{'OVERALL':<30} {total_correct:<10} {total_items:<10} {overall_accuracy:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage - test with first model from 10L_90NL condition
    model = "10L_90NL_1_1"
    
    # You can change these to test different prediction types
    pred_type = "vanilla"
    pred_dir = "predictions_vanilla"
    
    # Uncomment to test other prediction types:
    # pred_type = "character_separated"
    # pred_dir = "processed_predictions_sep_char"
    
    # pred_type = "independent_feature"
    # pred_dir = "predictions_independent_feature"
    
    # pred_type = "feature_invariant"
    # pred_dir = "predictions_feature_invariant"
    
    analyze_single_model(model, pred_type, pred_dir)
    
    print("\n\nTo analyze a different model, modify the 'model' variable in this script.")
    print("Available models: 10L_90NL_1_1, 10L_90NL_1_2, ..., 90L_10NL_3_4")







