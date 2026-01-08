"""
Calculate stem accuracy grouped by (lemma, target tag) pairs.
"""
import sys
import os
import json
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "l_nl_accuracies")))

from config import AR_SUFFIX_DICT, ER_SUFFIX_DICT, IR_SUFFIX_DICT, all_models


def get_stem(form, suffixes):
    """Extract the stem by removing the suffix from the form."""
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
    """Normalize a form for matching (remove spaces and stress markers)."""
    return form.replace(" ", "").replace("ˈ", "").strip()


def extract_target_tag(line):
    """
    Extract the last tag (target tag) from a source line.
    Returns the inner content without angle brackets.
    """
    start = line.rfind("<")
    end = line.rfind(">")
    if start != -1 and end != -1 and end > start:
        return line[start + 1 : end]
    return None


def build_form_to_lemma_map(lemma_dict):
    """Build a reverse mapping from normalized forms to lemmas."""
    mapping = {}
    for lemma, inflections in lemma_dict.items():
        for _, form in inflections.items():
            mapping[normalize_form(form)] = lemma
    return mapping


def process_predictions(pred_type, pred_dir_name, lemma_dict_path):
    """Process predictions for a given prediction type and calculate accuracy by lemma/tag."""
    print(f"\nProcessing {pred_type} predictions...")

    try:
        with open(lemma_dict_path, "r", encoding="utf-8") as f:
            lemma_dict = json.load(f)
        print(f"  Loaded {len(lemma_dict)} lemmas from {lemma_dict_path}")
    except Exception as exc:
        print(f"  Error loading lemma dictionary: {exc}")
        return

    form_to_lemma = build_form_to_lemma_map(lemma_dict)
    print(f"  Reverse map covers {len(form_to_lemma)} unique forms")


    suffixes = list({*AR_SUFFIX_DICT.values(), *ER_SUFFIX_DICT.values(), *IR_SUFFIX_DICT.values()})
    suffixes = sorted(suffixes, key=len, reverse=True)

    all_results = []
    unmapped_forms = set()
    missing_tags = 0

    for model in all_models:
        print(f"  Processing {model}...")
        condition = model.split("_")[0] + "_" + model.split("_")[1]
        run = model.split("_")[2]

        try:
            if pred_type in ("vanilla", "character_separated"):
                pred_file = f"../../../{pred_dir_name}/{model}.txt"
                pred_data = []
                with open(pred_file, "r") as f:
                    for idx, line in enumerate(f):
                        line = line.strip()
                        if "," in line:
                            split_idx, pred = line.split(",", 1)
                            pred_data.append((int(split_idx), normalize_form(pred)))
                        else:
                            pred_data.append((idx, normalize_form(line)))
                pred_data.sort(key=lambda x: x[0])
                preds = [pred for _, pred in pred_data]
            else:
                pred_dir = f"../../../{pred_dir_name}/"
                matching = [f for f in os.listdir(pred_dir) if f.startswith(model) and f.endswith(".tsv")]
                if not matching:
                    print(f"    No prediction file found for {model} in {pred_dir_name}")
                    continue
                pred_file = os.path.join(pred_dir, matching[0])
                df_pred = pd.read_csv(pred_file, sep="\t")
                preds = [normalize_form(str(pred)) for pred in df_pred["prediction"]]
        except Exception as exc:
            print(f"    Error reading predictions for {model}: {exc}")
            continue

        # Read targets
        test_dir = f"../../../analysis/{condition}/test/run{run}/"
        test_file = f"test.{model}.tgt"
        try:
            with open(os.path.join(test_dir, test_file), "r") as f:
                test_data = [normalize_form(line) for line in f]
        except Exception as exc:
            print(f"    Error reading test file {test_dir + test_file}: {exc}")
            continue

        src_file = f"test.{model}.src"
        with open(os.path.join(test_dir, src_file), "r") as f:
                target_tags = [extract_target_tag(line.strip()) for line in f]

        preds_stems = []
        test_stems = []
        lemmas = []
        tags = []

        for pred, test, tag in zip(preds, test_data, target_tags):
            preds_stems.append(get_stem(pred, suffixes))
            test_stems.append(get_stem(test, suffixes))
            normalized_test = normalize_form(test)
            lemma = form_to_lemma.get(normalized_test)
            if lemma is None:
                unmapped_forms.add(test)
            if tag is None:
                missing_tags += 1
            lemmas.append(lemma)
            tags.append(tag)

        combo_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for pred_stem, test_stem, lemma, tag in zip(preds_stems, test_stems, lemmas, tags):
            if lemma is None or tag is None:
                continue
            key = (lemma, tag)
            combo_stats[key]["total"] += 1
            if pred_stem == test_stem:
                combo_stats[key]["correct"] += 1

        # Persist per-model stats
        model_rows = []
        for (lemma, tag), stats in combo_stats.items():
            accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] else 0
            model_rows.append(
                {
                    "lemma": lemma,
                    "target_tag": tag,
                    "correct": stats["correct"],
                    "total": stats["total"],
                    "accuracy": round(accuracy, 2),
                }
            )
            all_results.append(
                {
                    "model": model,
                    "condition": condition,
                    "run": run,
                    "lemma": lemma,
                    "target_tag": tag,
                    "correct": stats["correct"],
                    "total": stats["total"],
                    "accuracy": round(accuracy, 2),
                }
            )

        output_dir = f"../../data/accuracies/stem_accuracy_by_lemma_tag/{pred_type}/"
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(model_rows).sort_values(["lemma", "target_tag"]).to_csv(
            os.path.join(output_dir, f"stem_acc_by_lemma_tag_{model}.csv"), index=False
        )
        print(f"    Saved results to {output_dir}stem_acc_by_lemma_tag_{model}.csv")

    if unmapped_forms:
        print(f"\n  Warning: {len(unmapped_forms)} forms could not be mapped to lemmas (showing up to 10):")
        print(f"    {list(unmapped_forms)[:10]}")
    if missing_tags:
        print(f"  Warning: {missing_tags} entries missing target tags in source files.")

    if all_results:
        combined_df = pd.DataFrame(all_results).sort_values(["condition", "run", "lemma", "target_tag"])
        output_dir = f"../../data/accuracies/stem_accuracy_by_lemma_tag/{pred_type}/"
        os.makedirs(output_dir, exist_ok=True)
        combined_df.to_csv(os.path.join(output_dir, "stem_acc_by_lemma_tag_all_models.csv"), index=False)
        print(f"\n  Saved combined results to {output_dir}stem_acc_by_lemma_tag_all_models.csv")

        summary_df = (
            combined_df.groupby(["lemma", "target_tag"])
            .agg({"correct": "sum", "total": "sum"})
            .reset_index()
        )
        summary_df["overall_accuracy"] = (summary_df["correct"] / summary_df["total"] * 100).round(2)
        summary_df = summary_df.sort_values(["lemma", "target_tag"])
        summary_df.to_csv(os.path.join(output_dir, "stem_acc_by_lemma_tag_summary.csv"), index=False)
        print(f"  Saved summary statistics to {output_dir}stem_acc_by_lemma_tag_summary.csv")


if __name__ == "__main__":
    lemma_dict_path = "../../../ipa_clean_lshaped_dict.json"
    prediction_types = [
        ("vanilla", "predictions_vanilla"),
        ("character_separated", "processed_predictions_sep_char"),
        ("independent_feature", "predictions_independent_feature"),
        ("feature_invariant", "predictions_feature_invariant"),
    ]

    for pred_type, pred_dir in prediction_types:
        process_predictions(pred_type, pred_dir, lemma_dict_path)

    print("\n" + "=" * 80)
    print("All stem accuracy by (lemma, tag) combinations processed!")
    print("=" * 80)







