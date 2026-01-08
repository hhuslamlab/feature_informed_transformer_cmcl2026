"""
Script to get stems.
"""
import sys, os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import AR_SUFFIX_DICT, ER_SUFFIX_DICT, IR_SUFFIX_DICT, all_models
import pandas as pd


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


if __name__ == "__main__":
    ar_suffixes = list(AR_SUFFIX_DICT.values())

    er_suffixes = list(ER_SUFFIX_DICT.values())

    ir_suffixes = list(IR_SUFFIX_DICT.values())

    suffixes = ar_suffixes + er_suffixes + ir_suffixes

    suffixes = sorted(list(set(suffixes)), key=len, reverse=True)

    for model in all_models:
        print(model)
        condition = model.split("_")[0] + "_" + model.split("_")[1]
        run = model.split("_")[2]

        with open("../../../predictions_vanilla/" + model + ".txt") as f:
            preds = f.readlines()
            preds = [
                item.split(",")[1].strip().replace(" ", "").replace("ˈ", "")
                for item in preds
            ]
        with open(
            "../../data/"
            + condition
            + "/test/run"
            + run
            + "/test."
            + model
            + ".tgt"
        ) as f:
            test_data = f.readlines()
            test_data = [
                item.strip().replace(" ", "").replace("ˈ", "") for item in test_data
            ]

        with open("../../data/shape_info/" + model) as f:
            shapes = f.readlines()
            shapes = [item.strip() for item in shapes]

        preds_stems = []
        test_stems = []
        preds_ = []
        tests_ = []
        shapes_ = []
        df = pd.DataFrame()

        for pred, test, shape in zip(preds, test_data, shapes):
            stem_pred = get_stem(pred, suffixes)
            stem_test = get_stem(test, suffixes)
            preds_stems.append(stem_pred)
            test_stems.append(stem_test)
            preds_.append(pred)
            tests_.append(test)
            shapes_.append(shape)

        df["test_form"] = tests_
        df["preds_form"] = preds_
        df["preds_stems"] = preds_stems
        df["test_stems"] = test_stems
        df["shapes"] = shapes_

        df.to_csv("../../data/stems/" + model + ".csv", index=False)
