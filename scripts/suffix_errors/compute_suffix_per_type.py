"""
Script to compute average suffix accuracy per type for each condition.

Usage:
    compute_suffix_per_type.py
"""
import sys, os, json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import AR_SUFFIX_DICT, ER_SUFFIX_DICT, IR_SUFFIX_DICT, all_models

if __name__ == "__main__":
    suffixes = ["er", "ar", "ir"]
    for model in all_models:
        condition = model.split("_")[0] + "_" + model.split("_")[1]
        run = model.split("_")[2]

        ar_df = pd.read_csv(
            "../../data/fixed_run/analysis/suffix_accuracies/"
            + condition
            + "/"
            + suffixes[1]
            + "_suffixes/combine.csv"
        )
        er_df = pd.read_csv(
            "../../data/fixed_run/analysis/suffix_accuracies/"
            + condition
            + "/"
            + suffixes[0]
            + "_suffixes/combine.csv"
        )
        ir_df = pd.read_csv(
            "../../data/fixed_run/analysis/suffix_accuracies/"
            + condition
            + "/"
            + suffixes[2]
            + "_suffixes/combine.csv"
        )

        avg_ar_accuracy = ar_df["accuracy"].mean()
        avg_ar_accuracy_present_in_verb_class = ar_df[
            "accuracy_present_in_verb_class"
        ].mean()
        avg_ar = pd.DataFrame()
        avg_ar["accuracy"] = [avg_ar_accuracy]
        avg_ar["accuracy_present_in_verb_class"] = [
            avg_ar_accuracy_present_in_verb_class
        ]
        avg_ar.to_csv(
            "../../data/fixed_run/analysis/suffix_accuracies/"
            + condition
            + "/"
            + suffixes[1]
            + "_suffixes/avg_accuracy.csv",
            index=False,
        )

        avg_er_accuracy = er_df["accuracy"].mean()
        avg_er_accuracy_present_in_verb_class = er_df[
            "accuracy_present_in_verb_class"
        ].mean()
        avg_er = pd.DataFrame()
        avg_er["accuracy"] = [avg_er_accuracy]
        avg_er["accuracy_present_in_verb_class"] = [
            avg_er_accuracy_present_in_verb_class
        ]
        avg_er.to_csv(
            "../../data/fixed_run/analysis/suffix_accuracies/"
            + condition
            + "/"
            + suffixes[0]
            + "_suffixes/avg_accuracy.csv",
            index=False,
        )

        avg_ir_accuracy = ir_df["accuracy"].mean()
        avg_ir_accuracy_present_in_verb_class = ir_df[
            "accuracy_present_in_verb_class"
        ].mean()
        avg_ir = pd.DataFrame()
        avg_ir["accuracy"] = [avg_ir_accuracy]
        avg_ir["accuracy_present_in_verb_class"] = [
            avg_ir_accuracy_present_in_verb_class
        ]
        avg_ir.to_csv(
            "../../data/fixed_run/analysis/suffix_accuracies/"
            + condition
            + "/"
            + suffixes[2]
            + "_suffixes/avg_accuracy.csv",
            index=False,
        )
