"""
Script to get stem-final consonants for predictions.
"""
import sys, os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
from config import AR_SUFFIX_DICT, ER_SUFFIX_DICT, IR_SUFFIX_DICT, all_models
import re

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
        df = pd.read_csv(
            "../../data/fixed_run/analysis/stems/" + model + ".csv"
        ).fillna("")
        preds = df["preds_stems"].tolist()
        test_forms = df["test_form"].tolist()
        preds_forms = df["preds_form"].tolist()

        pred_sf = []
        all_preds = []
        for pred in preds:
            print(pred)
            pred_stem_final_consonant = re.search(r"([^aeiou]*)$", pred).group(1)
            pred_sf.append(pred_stem_final_consonant.replace("ˈ", ""))
            all_preds.append(pred)
        df = pd.DataFrame()
        df["test"] = test_forms
        df["pred"] = preds_forms
        df["pred_sf"] = pred_sf

        df.to_csv(
            "../../data/fixed_run/analysis/pred_sf/" + model + ".csv", index=False
        )
