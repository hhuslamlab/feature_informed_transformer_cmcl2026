"""
Script to compute the average confusion matrix for each condition.

Usage:
    compute_avg_confusion_matrix.py --condition=<c>

Options:
    --condition=<c>                       Provide condition (10L_90NL, 50L_50NL, 90L_10NL)
"""
from docopt import docopt
import sys, os
import pandas as pd
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    all_models,
    condition_10L_90NL,
    condition_50L_50NL,
    condition_90L_10NL,
)

if __name__ == "__main__":
    args = docopt(__doc__)
    input_condition = args["--condition"]

    if input_condition == "10L_90NL":
        condition = condition_10L_90NL
    elif input_condition == "50L_50NL":
        condition = condition_50L_50NL
    elif input_condition == "90L_10NL":
        condition = condition_90L_10NL

    confusion_matrix = {}
    for model in condition:
        condition = model.split("_")[0] + "_" + model.split("_")[1]
        run = model.split("_")[2]
        with open(
            "../../data/fixed_run/analysis/lemma_test_pred_sf/" + model + ".json"
        ) as f:
            cm = json.load(f)
        cm = {eval(k): eval(v) for k, v in cm.items()}
        for k, v in cm.items():
            if k not in confusion_matrix:
                confusion_matrix[k] = {}
            for k1, v1 in v.items():
                if k1 not in confusion_matrix[k]:
                    confusion_matrix[k][k1] = 0
                confusion_matrix[k][k1] += v1

    # get the average confusion matrix
    for k, v in confusion_matrix.items():
        for k1, v1 in v.items():
            confusion_matrix[k][k1] = round(v1 / 12)
    confusion_matrix_new = {}
    for k, v in confusion_matrix.items():
        confusion_matrix_new[str(k)] = str(v)

    with open(
        "../../data/fixed_run/analysis/lemma_test_pred_sf/"
        + input_condition
        + "/avg/confusion_matrix.json",
        "w",
    ) as f:
        json.dump(confusion_matrix_new, f, indent=4)

    # filter the first 5 keys with the highest values in the confusion matrix
    confusion_matrix = {
        k: v
        for k, v in sorted(
            confusion_matrix.items(),
            key=lambda item: sum(item[1].values()),
            reverse=True,
        )[:5]
    }
    confusion_matrix_filtered = {}
    for k, v in confusion_matrix.items():
        confusion_matrix_filtered[k] = dict(
            sorted(v.items(), key=lambda item: item[1], reverse=True)[:5]
        )

    confusion_matrix_filtered_new = {}
    for k, v in confusion_matrix_filtered.items():
        confusion_matrix_filtered_new[str(k)] = str(v)

    with open(
        "../../data/fixed_run/analysis/lemma_test_pred_sf/"
        + input_condition
        + "/avg/confusion_matrix_filtered.json",
        "w",
    ) as f:
        json.dump(confusion_matrix_filtered_new, f, indent=4)
