"""
Script to check the errors for each suffix type and check if the wrongly predicted suffix lies in the verb class.

Usage:
    suffix_error_verb_type.py
"""
from typing import List
import sys, os, json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import AR_SUFFIX_DICT, ER_SUFFIX_DICT, IR_SUFFIX_DICT, all_models


if __name__ == "__main__":
    suffixes = ["er", "ar", "ir"]

    ar_suffixes = list(AR_SUFFIX_DICT.values())
    er_suffixes = list(ER_SUFFIX_DICT.values())
    ir_suffixes = list(IR_SUFFIX_DICT.values())
    suffix_dict = {"er": er_suffixes, "ar": ar_suffixes, "ir": ir_suffixes}

    for model in all_models:
        condition = model.split("_")[0] + "_" + model.split("_")[1]
        run = model.split("_")[2]

        with open(
            "../../data/fixed_run/"
            + condition
            + "/test/run"
            + run
            + "/lshaped_lemmas.txt"
        ) as f:
            lshaped_lemmas = f.readlines()
            lshaped_lemmas = [
                lemma.replace(" ", "").strip() for lemma in lshaped_lemmas
            ]

        er_df = pd.read_csv(
            "../../data/fixed_run/analysis/suffix_errors/"
            + condition
            + "/"
            + suffixes[0]
            + "_suffixes/"
            + model
            + ".csv",
        )

        ar_df = pd.read_csv(
            "../../data/fixed_run/analysis/suffix_errors/"
            + condition
            + "/"
            + suffixes[1]
            + "_suffixes/"
            + model
            + ".csv",
        )

        ir_df = pd.read_csv(
            "../../data/fixed_run/analysis/suffix_errors/"
            + condition
            + "/"
            + suffixes[2]
            + "_suffixes/"
            + model
            + ".csv",
        )
        ar_pred_stems = ar_df["pred_stems"].tolist()
        ar_pred_suffixes = ar_df["pred_suffixes"].tolist()
        ar_true_suffixes = ar_df["test_suffixes"].tolist()
        ar_test_stems = ar_df["test_stems"].tolist()
        ar_lemmas = ar_df["lemmas"].tolist()

        er_pred_stems = er_df["pred_stems"].tolist()
        er_pred_suffixes = er_df["pred_suffixes"].tolist()
        er_true_suffixes = er_df["test_suffixes"].tolist()
        er_test_stems = er_df["test_stems"].tolist()
        er_lemmas = er_df["lemmas"].tolist()

        ir_pred_stems = ir_df["pred_stems"].tolist()
        ir_pred_suffixes = ir_df["pred_suffixes"].tolist()
        ir_true_suffixes = ir_df["test_suffixes"].tolist()
        ir_test_stems = ir_df["test_stems"].tolist()
        ir_lemmas = ir_df["lemmas"].tolist()

        pred_suffixes = [ar_pred_suffixes, er_pred_suffixes, ir_pred_suffixes]
        true_suffixes = [ar_true_suffixes, er_true_suffixes, ir_true_suffixes]

        lshaped_lemmas_all = [ar_lemmas, er_lemmas, ir_lemmas]

        pred_stems = [ar_pred_stems, er_pred_stems, ir_pred_stems]
        test_stems = [ar_test_stems, er_test_stems, ir_test_stems]

        overall_errors = 0
        overall_correct = 0
        suffix_errors_when_stems_are_correct = 0
        suffix_errors_when_stems_are_incorrect = 0

        for i in range(3):
            for j in range(len(pred_suffixes[i])):
                if pred_suffixes[i][j] != true_suffixes[i][j]:
                    overall_errors += 1
                    if pred_stems[i][j] == test_stems[i][j]:
                        suffix_errors_when_stems_are_correct += 1
                    if pred_stems[i][j] != test_stems[i][j]:
                        suffix_errors_when_stems_are_incorrect += 1
                else:
                    overall_correct += 1

        overall_df = pd.DataFrame()
        overall_df["model"] = [model]
        overall_df["overall_errors"] = [overall_errors]
        overall_df["overall_correct"] = [overall_correct]
        overall_df["suffix_errors_when_stems_are_correct"] = [
            suffix_errors_when_stems_are_correct
        ]
        overall_df["suffix_errors_when_stems_are_incorrect"] = [
            suffix_errors_when_stems_are_incorrect
        ]
        overall_df["total_predictions"] = [
            len(ar_pred_stems) + len(er_pred_stems) + len(ir_pred_stems)
        ]
        overall_df["error_rate"] = [
            round(
                overall_errors
                / (len(ar_pred_stems) + len(er_pred_stems) + len(ir_pred_stems))
                * 100,
                2,
            )
        ]
        overall_df["suffix_error_rate_when_stems_are_correct"] = [
            round(
                suffix_errors_when_stems_are_correct
                / (len(ar_pred_stems) + len(er_pred_stems) + len(ir_pred_stems))
                * 100,
                2,
            )
        ]
        overall_df["suffix_error_rate_when_stems_are_incorrect"] = [
            round(
                suffix_errors_when_stems_are_incorrect
                / (len(ar_pred_stems) + len(er_pred_stems) + len(ir_pred_stems))
                * 100,
                2,
            )
        ]
        overall_df["accuracy"] = [
            round(
                overall_correct
                / (len(ar_pred_stems) + len(er_pred_stems) + len(ir_pred_stems))
                * 100,
                2,
            )
        ]
        overall_df["accuracy_when_stems_are_correct"] = [
            round(
                (overall_correct - suffix_errors_when_stems_are_correct)
                / overall_errors)
                * 100,
                2,
            )
        ]
        overall_df["accuracy_when_stems_are_incorrect"] = [
            round(
                (overall_correct - suffix_errors_when_stems_are_incorrect)
                / overall_errors)
                * 100,
                2,
            )
        ]
        overall_df.to_csv(
            "../../data/fixed_run/analysis/suffix_accuracies/"
            + condition
            + "/overall_accuracies/"
            + model
            + ".csv",
            index=False,
        )

        ### for lshaped lemmas
        lshaped_overall_errors = 0
        lshaped_overall_correct = 0
        lshaped_preds = 0
        nlshaped_overall_errors = 0
        nlshaped_overall_correct = 0
        nlshaped_preds = 0

        for i in range(3):
            for j in range(len(pred_suffixes[i])):
                if lshaped_lemmas_all[i][j] in lshaped_lemmas:
                    lshaped_preds += 1
                    if pred_suffixes[i][j] != true_suffixes[i][j]:
                        lshaped_overall_errors += 1
                    else:
                        lshaped_overall_correct += 1
                if lshaped_lemmas_all[i][j] not in lshaped_lemmas:
                    nlshaped_preds += 1
                    if pred_suffixes[i][j] != true_suffixes[i][j]:
                        nlshaped_overall_errors += 1
                    else:
                        nlshaped_overall_correct += 1

        lshaped_overall_df = pd.DataFrame()
        lshaped_overall_df["model"] = [model]
        lshaped_overall_df["lshaped_overall_errors"] = [lshaped_overall_errors]
        lshaped_overall_df["lshaped_overall_correct"] = [lshaped_overall_correct]
        lshaped_overall_df["total_lshaped_predictions"] = [len(lshaped_lemmas)]
        lshaped_overall_df["lshaped_error_rate"] = [
            round(lshaped_overall_errors / lshaped_preds * 100, 2)
        ]
        lshaped_overall_df["lshaped_accuracy"] = [
            round(lshaped_overall_correct / lshaped_preds * 100, 2)
        ]
        lshaped_overall_df.to_csv(
            "../../data/fixed_run/analysis/suffix_accuracies/"
            + condition
            + "/lshaped/overall_accuracies/"
            + model
            + ".csv",
            index=False,
        )

        nlshaped_overall_df = pd.DataFrame()
        nlshaped_overall_df["model"] = [model]
        nlshaped_overall_df["nlshaped_overall_errors"] = [nlshaped_overall_errors]
        nlshaped_overall_df["nlshaped_overall_correct"] = [nlshaped_overall_correct]
        nlshaped_overall_df["total_nlshaped_predictions"] = [len(lshaped_lemmas)]
        nlshaped_overall_df["nlshaped_error_rate"] = [
            round(nlshaped_overall_errors / nlshaped_preds * 100, 2)
        ]
        nlshaped_overall_df["nlshaped_accuracy"] = [
            round(nlshaped_overall_correct / nlshaped_preds * 100, 2)
        ]
        nlshaped_overall_df.to_csv(
            "../../data/fixed_run/analysis/suffix_accuracies/"
            + condition
            + "/nlshaped/overall_accuracies/"
            + model
            + ".csv",
            index=False,
        )

        num_ar_errors = 0
        num_ar_correct = 0
        ar_present_in_verb_class = 0
        ar_not_present_in_verb_class = 0

        for i in range(len(ar_pred_stems)):
            if ar_pred_suffixes[i] != ar_true_suffixes[i]:
                num_ar_errors += 1
                if ar_pred_suffixes[i] in ar_suffixes:
                    ar_present_in_verb_class += 1
                if ar_pred_suffixes[i] not in ar_suffixes:
                    ar_not_present_in_verb_class += 1
            else:
                num_ar_correct += 1

        num_ar_df = pd.DataFrame()
        num_ar_df["model"] = [model]
        num_ar_df["num_ar_errors"] = [num_ar_errors]
        num_ar_df["num_ar_correct"] = [num_ar_correct]
        num_ar_df["total_ar_predictions"] = [len(ar_pred_stems)]
        num_ar_df["ar_present_in_verb_class"] = [ar_present_in_verb_class]
        num_ar_df["ar_not_present_in_verb_class"] = [ar_not_present_in_verb_class]
        num_ar_df["ar_error_rate"] = [
            round(num_ar_errors / len(ar_pred_stems) * 100, 2)
        ]
        num_ar_df["accuracy"] = [round(num_ar_correct / len(ar_pred_stems) * 100, 2)]
        num_ar_df["accuracy_present_in_verb_class"] = [
            round(ar_present_in_verb_class / num_ar_errors * 100, 2)
        ]
        num_ar_df.to_csv(
            "../../data/fixed_run/analysis/suffix_accuracies/"
            + condition
            + "/"
            + suffixes[1]
            + "_suffixes/"
            + model
            + ".csv",
            index=False,
        )

        num_er_errors = 0
        num_er_correct = 0
        er_present_in_verb_class = 0
        er_not_present_in_verb_class = 0

        for i in range(len(er_pred_stems)):
            if er_pred_suffixes[i] != er_true_suffixes[i]:
                num_er_errors += 1
                if er_pred_suffixes[i] in er_suffixes:
                    er_present_in_verb_class += 1

                if er_pred_suffixes[i] not in er_suffixes:
                    er_not_present_in_verb_class += 1
            else:
                num_er_correct += 1

        num_er_df = pd.DataFrame()
        num_er_df["model"] = [model]
        num_er_df["num_er_errors"] = [num_er_errors]
        num_er_df["num_er_correct"] = [num_er_correct]
        num_er_df["total_er_predictions"] = [len(er_pred_stems)]
        num_er_df["er_present_in_verb_class"] = [er_present_in_verb_class]
        num_er_df["er_not_present_in_verb_class"] = [er_not_present_in_verb_class]
        num_er_df["er_error_rate"] = [
            round(num_er_errors / len(er_pred_stems) * 100, 2)
        ]
        num_er_df["accuracy"] = [round(num_er_correct / len(er_pred_stems) * 100, 2)]
        num_er_df["accuracy_present_in_verb_class"] = [
            round(er_present_in_verb_class / num_er_errors * 100, 2)
        ]
        num_er_df.to_csv(
            "../../data/fixed_run/analysis/suffix_accuracies/"
            + condition
            + "/"
            + suffixes[0]
            + "_suffixes/"
            + model
            + ".csv",
            index=False,
        )

        num_ir_errors = 0
        num_ir_correct = 0
        ir_present_in_verb_class = 0
        ir_not_present_in_verb_class = 0

        for i in range(len(ir_pred_stems)):
            if ir_pred_suffixes[i] != ir_true_suffixes[i]:
                num_ir_errors += 1
                if ir_pred_suffixes[i] in ir_suffixes:
                    ir_present_in_verb_class += 1
                if ir_pred_suffixes[i] not in ir_suffixes:
                    ir_not_present_in_verb_class += 1
            else:
                num_ir_correct += 1

        num_ir_df = pd.DataFrame()
        num_ir_df["model"] = [model]
        num_ir_df["num_ir_errors"] = [num_ir_errors]
        num_ir_df["num_ir_correct"] = [num_ir_correct]
        num_ir_df["total_ir_predictions"] = [len(ir_pred_stems)]
        num_ir_df["ir_present_in_verb_class"] = [ir_present_in_verb_class]
        num_ir_df["ir_not_present_in_verb_class"] = [ir_not_present_in_verb_class]
        num_ir_df["ir_error_rate"] = [
            round(num_ir_errors / len(ir_pred_stems) * 100, 2)
        ]
        num_ir_df["accuracy"] = [round(num_ir_correct / len(ir_pred_stems) * 100, 2)]
        num_ir_df["accuracy_present_in_verb_class"] = [
            round(ir_present_in_verb_class / num_ir_errors * 100, 2)
        ]
        num_ir_df.to_csv(
            "../../data/fixed_run/analysis/suffix_accuracies/"
            + condition
            + "/"
            + suffixes[2]
            + "_suffixes/"
            + model
            + ".csv",
            index=False,
        )
