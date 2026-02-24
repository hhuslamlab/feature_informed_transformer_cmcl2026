"""
get L and NL accuracy for all prediction types
"""
import pandas as pd
import os

def process_predictions(pred_type, pred_dir_name):
    """Process predictions for a given prediction type"""
    print(f"\nProcessing {pred_type} predictions...")

    # Get all model names from shape_info files
    shape_info_dir = "../shape_info/"
    all_models = []
    for file in os.listdir(shape_info_dir):
        if os.path.isfile(os.path.join(shape_info_dir, file)):
            all_models.append(file)
    all_models.sort()

    for model in all_models:
        condition = model.split("_")[0] + "_" + model.split("_")[1]
        run = model.split("_")[2]

        # Find the correct test file path
        test_dir = f"../../data/{condition}/test/run{run}/"
        test_file = f"test.{model}.tgt"

        if not os.path.exists(test_dir + test_file):
            print(f"Test file not found: {test_dir + test_file}")
            continue

        with open(test_dir + test_file) as f:
            test_data = f.readlines()
            # Normalize: remove spaces, remove stress markers, then strip (matching calc_overall_accuracy.py)
            test_data = [
                item.replace(" ", "").replace("ˈ", "").strip() for item in test_data
            ]

        with open(shape_info_dir + model) as f:
            shapes = f.readlines()
            shapes = [item.strip() for item in shapes]

        # Find the prediction file (different formats for different model types)
        if pred_type == "vanilla" or pred_type == "character_separated":
            pred_file = f"../../data/predictions/{pred_dir_name}/{model}.txt"
        else:
            # For dual_source and feature_invariant, look for .tsv files
            pred_dir = f"../../data/predictions/{pred_dir_name}/"
            matching_files = [f for f in os.listdir(pred_dir) if f.startswith(model) and f.endswith('.tsv')]
            if matching_files:
                pred_file = f"../../data/predictions/{pred_dir_name}/{matching_files[0]}"
            else:
                print(f"No prediction file found for {model} in {pred_dir_name}")
                continue

        if not os.path.exists(pred_file):
            print(f"No prediction file found for {model}: {pred_file}")
            continue

        # Read the prediction file (different formats for different model types)
        try:
            predictions = []
            if pred_type == "vanilla" or pred_type == "character_separated":
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
                predictions = [pred for _, pred in pred_data]
            else:
                # TSV format for dual_source and feature_invariant
                df = pd.read_csv(pred_file, sep='\t')
                for pred in df['prediction']:
                    # Normalize: remove spaces, remove stress markers, then strip (matching calc_overall_accuracy.py)
                    pred_clean = str(pred).replace(" ", "").replace("ˈ", "").strip()
                    predictions.append(pred_clean)

        except Exception as e:
            print(f"Error reading prediction file {pred_file}: {e}")
            continue

        lshape_count = 0
        nlshape_count = 0
        correct = 0
        all_lshape_count = len([item for item in shapes if item == "L"])
        all_nlshape_count = len([item for item in shapes if item == "NL"])

        for pred, test, shape in zip(predictions, test_data, shapes):
            if pred == test:
                if shape == "L":
                    lshape_count += 1
                if shape == "NL":
                    nlshape_count += 1
                correct +=1

        l_acc = round(lshape_count / all_lshape_count * 100, 2) if all_lshape_count > 0 else 0
        nl_acc = round(nlshape_count / all_nlshape_count * 100, 2) if all_nlshape_count > 0 else 0
        df = pd.DataFrame()
        df["filename"] = [model]
        df["l_acc"] = [l_acc]
        df["nl_acc"] = [nl_acc]

        df.to_csv(
            f"../../data/analysis/accuracies/l_nl_{pred_type.lower().replace(' ', '_')}_{model}.csv",
            index=False,
        )

if __name__ == "__main__":
    # Process prediction types
    prediction_types = [
        ("vanilla", "predictions_vanilla"),
        ("character_separated", "processed_predictions_sep_char"),
        ("feature_invariant", "predictions_feature_invariant"),
        ("independent_feature", "predictions_independent_feature"),
        ("feature_geometric", "predictions_binaryfeature")
    ]

    for pred_type, pred_dir in prediction_types:
        process_predictions(pred_type, pred_dir)

    print("\nAll prediction types processed!")
