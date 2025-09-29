"""
get L and NL accuracy
"""
import pandas as pd
from config import all_models

if __name__ == "__main__":
    for model in all_models:
        condition = model.split("_")[0] + "_" + model.split("_")[1]
        run = model.split("_")[2]
        try:
            with open(
                "../../data/"
                + condition
                + "/dev/run"
                + run
                + "/dev."
                + model
                + ".tgt"
            ) as f:
                test_data = f.readlines()
                test_data = [
                    item.strip().replace(" ", "").replace("ˈ", "") for item in test_data
                ]
            with open("../../data/analysis/shape_info/" + model) as f:
                shapes = f.readlines()
                shapes = [item.strip() for item in shapes]

            with open("../../data/batch_size_32_predictions/processed_predictions_orig/" + model + ".txt") as f:
                predictions = f.readlines()
                ## sort by first column
                predictions = sorted(predictions, key=lambda x: int(x.split(",")[0]), reverse=False)
                predictions = [
                    item.split(",")[1].replace("ˈ", "").strip() for item in predictions
                ]
            lshape_count = 0
            nlshape_count = 0

            all_lshape_count = len([item for item in shapes if item == "L"])
            all_nlshape_count = len([item for item in shapes if item == "NL"])

            for pred, test, shape in zip(predictions, test_data, shapes):
                if pred == test:
                    if shape == "L":
                        lshape_count += 1
                    if shape == "NL":
                        nlshape_count += 1
            l_acc = round(lshape_count / all_lshape_count * 100, 2)
            nl_acc = round(nlshape_count / all_nlshape_count * 100, 2)

            df = pd.DataFrame()
            df["filename"] = [model]
            df["l_acc"] = [l_acc]
            df["nl_acc"] = [nl_acc]

            df.to_csv(
                "../../data/analysis/batch_size_32/l_nl_accuracies/" + model + ".csv",
                index=False,
            )
        except FileNotFoundError:
            print("File not found for model: ", model)
            continue
