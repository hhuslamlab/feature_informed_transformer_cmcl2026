"""
Script to compute overall suffix accuracy for each condition.

Usage:
    compute_overall_suffix_accuracy.py --condition
"""
import sys, os, json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import AR_SUFFIX_DICT, ER_SUFFIX_DICT, IR_SUFFIX_DICT, all_models

if __name__ == "__main__":
    for model in all_models:
        condition = model.split("_")[0] + "_" + model.split("_")[1]
        run = model.split("_")[2]

        overall_suffix_accuracy = pd.read_csv(
            "../../data/fixed_run/analysis/suffix_accuracies/"
            + condition
            + "/overall_accuracies/"
            + "combine.csv"
        )

        avg_accuracy = overall_suffix_accuracy["accuracy"].mean()
        avg_accuracy = round(avg_accuracy, 2)
        avg_accuracy_when_stems_are_correct = overall_suffix_accuracy[
            "accuracy_when_stems_are_correct"
        ].mean()
        avg_accuracy_when_stems_are_correct = round(
            avg_accuracy_when_stems_are_correct, 2
        )
        avg_accuracy_when_stems_are_incorrect = overall_suffix_accuracy[
            "accuracy_when_stems_are_incorrect"
        ].mean()
        avg_accuracy_when_stems_are_incorrect = round(
            avg_accuracy_when_stems_are_incorrect, 2
        )
        overall_df = pd.DataFrame()
        overall_df["accuracy"] = [avg_accuracy]
        overall_df["accuracy_when_stems_are_correct"] = [
            avg_accuracy_when_stems_are_correct
        ]
        overall_df["accuracy_when_stems_are_incorrect"] = [
            avg_accuracy_when_stems_are_incorrect
        ]
        overall_df.to_csv(
            "../../data/fixed_run/analysis/suffix_accuracies/"
            + condition
            + "/overall_accuracies/"
            + "avg_accuracy.csv",
            index=False,
        )

        lshaped_suffix_accuracy = pd.read_csv(
            "../../data/fixed_run/analysis/suffix_accuracies/"
            + condition
            + "/lshaped/overall_accuracies/combine.csv"
        )
        lshape_avg_accuracy = lshaped_suffix_accuracy["lshaped_accuracy"].mean()
        lshape_avg_accuracy = round(lshape_avg_accuracy, 2)
        lshape_overall_df = pd.DataFrame()
        lshape_overall_df["accuracy"] = [lshape_avg_accuracy]
        lshape_overall_df.to_csv(
            "../../data/fixed_run/analysis/suffix_accuracies/"
            + condition
            + "/lshaped/overall_accuracies/"
            + "avg_accuracy.csv",
            index=False,
        )

        nlshape_suffix_accuracy = pd.read_csv(
            "../../data/fixed_run/analysis/suffix_accuracies/"
            + condition
            + "/nlshaped/overall_accuracies/combine.csv"
        )
        nlshape_avg_accuracy = nlshape_suffix_accuracy["nlshaped_accuracy"].mean()
        nlshape_avg_accuracy = round(nlshape_avg_accuracy, 2)
        nlshape_overall_df = pd.DataFrame()
        nlshape_overall_df["accuracy"] = [nlshape_avg_accuracy]
        nlshape_overall_df.to_csv(
            "../../data/fixed_run/analysis/suffix_accuracies/"
            + condition
            + "/nlshaped/overall_accuracies/"
            + "avg_accuracy.csv",
            index=False,
        )
