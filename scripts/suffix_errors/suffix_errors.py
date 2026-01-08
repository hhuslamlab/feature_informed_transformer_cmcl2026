"""
Script to get suffix errors for eɾ, aɾ, iɾ type lemmas.

Usage:
    suffix_errors.py
"""
from typing import List
import sys, os, json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import AR_SUFFIX_DICT, ER_SUFFIX_DICT, IR_SUFFIX_DICT, all_models

from consonant_pair.get_stems import *


def get_stem_and_suffix(items: List[str], form_suffixes: List[str]):
    item_stems = []
    item_suffixes = []

    for item in items:
        stem = get_stem(item, form_suffixes)
        if not stem:
            item_stems.append("")
            item_suffixes.append("")
        else:
            item_stems.append(stem)
            suffix = item[len(stem) :]
            item_suffixes.append(suffix)

    return item_stems, item_suffixes


if __name__ == "__main__":
    suffixes = ["eɾ", "aɾ", "iɾ"]
    ar_suffixes = list(AR_SUFFIX_DICT.values())

    er_suffixes = list(ER_SUFFIX_DICT.values())

    ir_suffixes = list(IR_SUFFIX_DICT.values())

    form_suffixes = ar_suffixes + er_suffixes + ir_suffixes

    form_suffixes = sorted(list(set(form_suffixes)), key=len, reverse=True)

    for model in all_models:
        condition = model.split("_")[0] + "_" + model.split("_")[1]
        run = model.split("_")[2]

        lemmas_df = pd.read_csv(
            "../../data/fixed_run/analysis/lemmas_sf/"
            + "/test/run"
            + run
            + "/"
            + model
            + ".csv"
        )

        lemmas = lemmas_df["lemma"].tolist()
        idxs = lemmas_df["idx"].tolist()

        lemma_type, er_lemmas, ar_lemmas, ir_lemmas = [], [], [], []
        er_lemmas_idx, ar_lemmas_idx, ir_lemmas_idx = [], [], []

        for idx, lemma in zip(idxs, lemmas):
            if lemma.endswith(suffixes[0]):
                er_lemmas.append(lemma)
                er_lemmas_idx.append(idx)
                lemma_type.append("er")

            if lemma.endswith(suffixes[2]):
                ir_lemmas.append(lemma)
                ir_lemmas_idx.append(idx)
                lemma_type.append("ir")

            if lemma.endswith(suffixes[1]):
                ar_lemmas.append(lemma)
                ar_lemmas_idx.append(idx)
                lemma_type.append("ar")

        pred_df = pd.read_csv(
            "../../data/fixed_run/analysis/pred_sf/" + model + ".csv"
        ).fillna("")

        preds = pred_df["pred"].tolist()
        tests = pred_df["test"].tolist()

        ir_preds = [preds[item] for item in ir_lemmas_idx]
        ar_preds = [preds[item] for item in ar_lemmas_idx]
        er_preds = [preds[item] for item in er_lemmas_idx]

        ir_tests = [tests[item] for item in ir_lemmas_idx]
        ar_tests = [tests[item] for item in ar_lemmas_idx]
        er_tests = [tests[item] for item in er_lemmas_idx]

        ir_pred_stem, ir_pred_suffixes = get_stem_and_suffix(ir_preds, form_suffixes)
        er_pred_stem, er_pred_suffixes = get_stem_and_suffix(er_preds, form_suffixes)
        ar_pred_stem, ar_pred_suffixes = get_stem_and_suffix(ar_preds, form_suffixes)

        ir_test_stem, ir_test_suffixes = get_stem_and_suffix(ir_tests, form_suffixes)
        er_test_stem, er_test_suffixes = get_stem_and_suffix(er_tests, form_suffixes)
        ar_test_stem, ar_test_suffixes = get_stem_and_suffix(ar_tests, form_suffixes)

        ar_df = pd.DataFrame()
        ar_df["lemmas"] = ar_lemmas
        ar_df["pred_stems"] = ar_pred_stem
        ar_df["pred_suffixes"] = ar_pred_suffixes
        ar_df["test_stems"] = ar_test_stem
        ar_df["test_suffixes"] = ar_test_suffixes
        ar_df.to_csv(
            "../../data/fixed_run/analysis/suffix_errors/"
            + condition
            + "/"
            + "ar_suffixes/"
            + model
            + ".csv",
            index=False,
        )

        er_df = pd.DataFrame()
        er_df["lemmas"] = er_lemmas
        er_df["pred_stems"] = er_pred_stem
        er_df["pred_suffixes"] = er_pred_suffixes
        er_df["test_stems"] = er_test_stem
        er_df["test_suffixes"] = er_test_suffixes
        er_df.to_csv(
            "../../data/fixed_run/analysis/suffix_errors/"
            + condition
            + "/"
            + "er_suffixes/"
            + model
            + ".csv",
            index=False,
        )

        ir_df = pd.DataFrame()
        ir_df["lemmas"] = ir_lemmas
        ir_df["pred_stems"] = ir_pred_stem
        ir_df["pred_suffixes"] = ir_pred_suffixes
        ir_df["test_stems"] = ir_test_stem
        ir_df["test_suffixes"] = ir_test_suffixes
        ir_df.to_csv(
            "../../data/fixed_run/analysis/suffix_errors/"
            + condition
            + "/"
            + "ir_suffixes/"
            + model
            + ".csv",
            index=False,
        )
