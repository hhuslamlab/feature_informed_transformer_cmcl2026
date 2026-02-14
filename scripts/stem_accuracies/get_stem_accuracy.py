import sys, os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "l_nl_accuracies")))

from config import AR_SUFFIX_DICT, ER_SUFFIX_DICT, IR_SUFFIX_DICT, all_models


def get_stem(form, suffixes):
    """
    get stem final consonant of the form given a suffix
    """
    if form[-4:] in suffixes:
        form = form[: len(form) - 4]
        return form
    if form[-3:] in suffixes:
        form = form[: len(form) - 3]
        return form
    if form[-2:] in suffixes:
        form = form[: len(form) - 2]
        return form
    if form[-1:] in suffixes:
        form = form[: len(form) - 1]
        return form
    return form


def process_predictions(pred_type, pred_dir_name):
    """Process predictions for a given prediction type"""
    print(f"\nProcessing {pred_type} predictions...")
    
    ar_suffixes = list(AR_SUFFIX_DICT.values())
    er_suffixes = list(ER_SUFFIX_DICT.values())
    ir_suffixes = list(IR_SUFFIX_DICT.values())
    suffixes = ar_suffixes + er_suffixes + ir_suffixes
    suffixes = sorted(list(set(suffixes)), key=len, reverse=True)

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
                            # Normalize: remove spaces, remove stress markers, then strip (matching calc_overall_accuracy.py)
                            pred = parts[1].replace(" ", "").replace("ˈ", "").strip()
                            pred_data.append((idx, pred))
                        else:
                            # Direct prediction format (no index)
                            # Normalize: remove spaces, remove stress markers, then strip (matching calc_overall_accuracy.py)
                            pred = line.replace(" ", "").replace("ˈ", "").strip()
                            pred_data.append((len(pred_data), pred))  # Use line number as index
                
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
                # Normalize: remove spaces, remove stress markers, then strip (matching calc_overall_accuracy.py)
                preds = [str(pred).replace(" ", "").replace("ˈ", "").strip() for pred in df_pred['prediction']]
            except Exception as e:
                print(f"    Error reading {pred_file}: {e}")
                continue

        # Read test data
        test_dir = f"../../data/{condition}/test/run{run}/"
        test_file = f"test.{model}.tgt"
        try:
            with open(test_dir + test_file) as f:
                test_data = f.readlines()
                # Normalize: remove spaces, remove stress markers, then strip (matching calc_overall_accuracy.py)
                test_data = [item.replace(" ", "").replace("ˈ", "").strip() for item in test_data]
        except Exception as e:
            print(f"    Error reading test file {test_dir + test_file}: {e}")
            continue

        # Read shapes
        shape_info_dir = "../shape_info/"
        try:
            with open(shape_info_dir + model) as f:
                shapes = [item.strip() for item in f.readlines()]
        except Exception as e:
            print(f"    Error reading shape file {shape_info_dir + model}: {e}")
            continue

        # Extract stems
        preds_stems = []
        test_stems = []
        preds_forms = []
        tests_forms = []
        shapes_list = []

        for pred, test, shape in zip(preds, test_data, shapes):
            stem_pred = get_stem(pred, suffixes)
            stem_test = get_stem(test, suffixes)
            preds_stems.append(stem_pred)
            test_stems.append(stem_test)
            preds_forms.append(pred)
            tests_forms.append(test)
            shapes_list.append(shape)

        # Save stems file
        stems_df = pd.DataFrame()
        stems_df["test_form"] = tests_forms
        stems_df["preds_form"] = preds_forms
        stems_df["preds_stems"] = preds_stems
        stems_df["test_stems"] = test_stems
        stems_df["shapes"] = shapes_list
        
        # For vanilla, save in existing location for backward compatibility
        if pred_type == "vanilla":
            stems_df.to_csv(f"../../data/analysis/stems/stems_{model}.csv", index=False)
        else:
            stems_dir = f"../../data/analysis/stems/{pred_type}/"
            os.makedirs(stems_dir, exist_ok=True)
            stems_df.to_csv(f"{stems_dir}stems_{model}.csv", index=False)

        # Calculate stem accuracies
        lshape_count = 0
        nlshape_count = 0
        all_lshape_count = len([item for item in shapes_list if item == "L"])
        all_nlshape_count = len([item for item in shapes_list if item == "NL"])

        for pred_stem, test_stem, shape in zip(preds_stems, test_stems, shapes_list):
            if pred_stem == test_stem:
                if shape == "L":
                    lshape_count += 1
                if shape == "NL":
                    nlshape_count += 1

        l_acc = round(lshape_count / all_lshape_count * 100, 2) if all_lshape_count > 0 else 0
        nl_acc = round(nlshape_count / all_nlshape_count * 100, 2) if all_nlshape_count > 0 else 0

        # Save stem accuracy
        acc_df = pd.DataFrame()
        acc_df["filename"] = [model]
        acc_df["l_acc"] = [l_acc]
        acc_df["nl_acc"] = [nl_acc]

        # For vanilla, save in existing location for backward compatibility
        if pred_type == "vanilla":
            acc_df.to_csv(f"../../data/analysis/stem_accuracies/stem_acc_{model}.csv", index=False)
        else:
            acc_dir = f"../../data/analysis/stem_accuracies/{pred_type}/"
            os.makedirs(acc_dir, exist_ok=True)
            acc_df.to_csv(f"{acc_dir}stem_acc_{model}.csv", index=False)


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

    print("\nAll stem accuracies processed!")
