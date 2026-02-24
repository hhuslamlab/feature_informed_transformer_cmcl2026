"""
Calculate stem accuracy grouped by lemma
Uses ipa_clean_lshaped_dict.json to map target forms to their lemmas
"""
import sys
import os
import pandas as pd
import json
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


def normalize_form(form):
    """Normalize a form for matching (remove spaces and stress markers)"""
    return form.replace(" ", "").replace("ˈ", "").strip()


def build_form_to_lemma_map(lemma_dict):
    """
    Build a reverse mapping from normalized forms to lemmas
    Returns: dict mapping normalized_form -> lemma
    """
    form_to_lemma = {}

    for lemma, inflections in lemma_dict.items():
        for tag, form in inflections.items():
            normalized = normalize_form(form)
            # Store lemma without normalization for output
            form_to_lemma[normalized] = lemma

    return form_to_lemma


def process_predictions(pred_type, pred_dir_name, lemma_dict_path):
    """Process predictions for a given prediction type and calculate stem accuracy by lemma"""
    print(f"\nProcessing {pred_type} predictions...")

    # Load lemma dictionary
    print(f"  Loading lemma dictionary from {lemma_dict_path}...")
    try:
        with open(lemma_dict_path, 'r', encoding='utf-8') as f:
            lemma_dict = json.load(f)
        print(f"  Loaded {len(lemma_dict)} lemmas")
    except Exception as e:
        print(f"  Error loading lemma dictionary: {e}")
        return

    # Build reverse mapping
    form_to_lemma = build_form_to_lemma_map(lemma_dict)
    print(f"  Built mapping for {len(form_to_lemma)} unique forms")

    # Prepare suffixes
    ar_suffixes = list(AR_SUFFIX_DICT.values())
    er_suffixes = list(ER_SUFFIX_DICT.values())
    ir_suffixes = list(IR_SUFFIX_DICT.values())
    suffixes = ar_suffixes + er_suffixes + ir_suffixes
    suffixes = sorted(list(set(suffixes)), key=len, reverse=True)

    # Store all results for overall analysis
    all_results = []
    unmapped_forms = set()

    for model in all_models:
        print(f"  Processing {model}...")
        condition = model.split("_")[0] + "_" + model.split("_")[1]
        run = model.split("_")[2]

        # Read predictions
        if pred_type == "vanilla" or pred_type == "character_separated":
            pred_file = f"../../data/predictions/{pred_dir_name}/{model}.txt"
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
                print(f"    Error reading {pred_file}: {e}")
                continue
        else:
            pred_dir = f"../../data/predictions/{pred_dir_name}/"
            matching_files = [f for f in os.listdir(pred_dir) if f.startswith(model) and f.endswith('.tsv')]
            if matching_files:
                pred_file = f"../../data/predictions/{pred_dir_name}/{matching_files[0]}"
            else:
                print(f"    No prediction file found for {model} in {pred_dir_name}")
                continue

            try:
                df_pred = pd.read_csv(pred_file, sep='\t')
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
                test_data = [item.replace(" ", "").replace("ˈ", "").strip() for item in test_data]
        except Exception as e:
            print(f"    Error reading test file {test_dir + test_file}: {e}")
            continue

        # Validate data alignment
        if not (len(preds) == len(test_data)):
            print(f"    Warning: Data length mismatch for {model}")
            print(f"      Predictions: {len(preds)}, Test data: {len(test_data)}")
            min_len = min(len(preds), len(test_data))
            preds = preds[:min_len]
            test_data = test_data[:min_len]

        # Extract stems and map to lemmas
        preds_stems = []
        test_stems = []
        lemmas = []

        for pred, test in zip(preds, test_data):
            stem_pred = get_stem(pred, suffixes)
            stem_test = get_stem(test, suffixes)
            preds_stems.append(stem_pred)
            test_stems.append(stem_test)

            # Map test form to lemma
            normalized_test = normalize_form(test)
            lemma = form_to_lemma.get(normalized_test, None)

            if lemma is None:
                unmapped_forms.add(test)

            lemmas.append(lemma)

        # Calculate stem accuracy by lemma
        lemma_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

        for pred_stem, test_stem, lemma in zip(preds_stems, test_stems, lemmas):
            if lemma is not None:
                lemma_stats[lemma]['total'] += 1
                if pred_stem == test_stem:
                    lemma_stats[lemma]['correct'] += 1

        # Create DataFrame for this model
        for lemma, stats in lemma_stats.items():
            accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            all_results.append({
                'model': model,
                'condition': condition,
                'run': run,
                'lemma': lemma,
                'correct': stats['correct'],
                'total': stats['total'],
                'accuracy': round(accuracy, 2)
            })

        # Save per-model detailed results
        model_df = pd.DataFrame([
            {
                'lemma': lemma,
                'correct': stats['correct'],
                'total': stats['total'],
                'accuracy': round((stats['correct'] / stats['total'] * 100), 2) if stats['total'] > 0 else 0
            }
            for lemma, stats in lemma_stats.items()
        ])

        # Sort by accuracy for easier reading
        model_df = model_df.sort_values('accuracy', ascending=False)

        # Create output directory
        output_dir = f"../../data/analysis/stem_accuracy_by_lemma/{pred_type}/"
        os.makedirs(output_dir, exist_ok=True)

        # Save model-specific results
        model_df.to_csv(f"{output_dir}stem_acc_by_lemma_{model}.csv", index=False)
        print(f"    Saved results to {output_dir}stem_acc_by_lemma_{model}.csv")

    # Report unmapped forms
    if unmapped_forms:
        print(f"\n  Warning: {len(unmapped_forms)} forms could not be mapped to lemmas")
        print(f"  First 10 unmapped forms: {list(unmapped_forms)[:10]}")

    # Save combined results for all models
    if all_results:
        combined_df = pd.DataFrame(all_results)
        combined_df = combined_df.sort_values(['condition', 'run', 'lemma'])

        output_dir = f"../../data/analysis/stem_accuracy_by_lemma/{pred_type}/"
        os.makedirs(output_dir, exist_ok=True)
        combined_df.to_csv(f"{output_dir}stem_acc_by_lemma_all_models.csv", index=False)
        print(f"\n  Saved combined results to {output_dir}stem_acc_by_lemma_all_models.csv")

        # Create summary statistics by lemma across all models
        summary_by_lemma = combined_df.groupby('lemma').agg({
            'correct': 'sum',
            'total': 'sum',
            'accuracy': 'mean'
        }).reset_index()
        summary_by_lemma['overall_accuracy'] = (summary_by_lemma['correct'] / summary_by_lemma['total'] * 100).round(2)
        summary_by_lemma = summary_by_lemma.sort_values('overall_accuracy', ascending=False)
        summary_by_lemma.to_csv(f"{output_dir}stem_acc_by_lemma_summary.csv", index=False)
        print(f"  Saved summary statistics to {output_dir}stem_acc_by_lemma_summary.csv")

        # Print top and bottom lemmas
        print(f"\n  Top 10 lemmas by accuracy:")
        for idx, row in summary_by_lemma.head(10).iterrows():
            print(f"    {row['lemma']:<30} {row['overall_accuracy']:>6.2f}% ({row['correct']}/{row['total']})")

        print(f"\n  Bottom 10 lemmas by accuracy:")
        for idx, row in summary_by_lemma.tail(10).iterrows():
            print(f"    {row['lemma']:<30} {row['overall_accuracy']:>6.2f}% ({row['correct']}/{row['total']})")


if __name__ == "__main__":
    # Path to lemma dictionary
    lemma_dict_path = "../../data/nevins_data/ipa_clean_lshaped_dict.json"

    # Process prediction types
    prediction_types = [
        ("vanilla", "predictions_vanilla"),
        ("character_separated", "processed_predictions_sep_char"),
        ("independent_feature", "predictions_independent_feature"),
        ("feature_invariant", "predictions_feature_invariant")
    ]

    for pred_type, pred_dir in prediction_types:
        process_predictions(pred_type, pred_dir, lemma_dict_path)

    print("\n" + "="*80)
    print("All stem accuracy by lemma calculations completed!")
    print("="*80)







