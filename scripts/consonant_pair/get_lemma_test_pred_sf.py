"""
Script to get the lemma test sf and lemma prediction sf.
"""
import sys, os
import pandas as pd
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import all_models

if __name__ == "__main__":
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

        lemma_df = pd.read_csv(
            "../../data/fixed_run/analysis/lemmas_sf/test/run"
            + run
            + "/"
            + model
            + ".csv"
        )

        pred_df = pd.read_csv("../../data/fixed_run/analysis/pred_sf/" + model + ".csv")
        # loop through the lemma_df and check if lemma is in lshaped_lemmas. If it is, append the lemma_sf and tgt_sf to a list. Do the same for the predictions.
        lemma_sf = []
        tgt_sf = []
        pred_sf = []

        filtered_lemmas = []
        for i in range(len(lemma_df)):
            if lemma_df["lemma"].iloc[i] in lshaped_lemmas:
                lemma_sf.append(lemma_df["lemma_sf"].iloc[i])
                tgt_sf.append(lemma_df["tgt_sf"].iloc[i])
                filtered_lemmas.append(lemma_df["lemma"].iloc[i])
                pred_sf.append(pred_df["pred_sf"].iloc[i])

        lemma_tgt_sf = list(zip(lemma_sf, tgt_sf))
        lemma_pred_sf = list(zip(lemma_sf, pred_sf))

        confusion_matrix = {}
        for i in range(len(lemma_tgt_sf)):
            if lemma_tgt_sf[i] not in confusion_matrix:
                confusion_matrix[lemma_tgt_sf[i]] = {}
            if lemma_pred_sf[i] not in confusion_matrix[lemma_tgt_sf[i]]:
                confusion_matrix[lemma_tgt_sf[i]][lemma_pred_sf[i]] = 0
            confusion_matrix[lemma_tgt_sf[i]][lemma_pred_sf[i]] += 1

        confusion_matrix_new = {}
        for k, v in confusion_matrix.items():
            confusion_matrix_new[str(k)] = str(v)

        # replace nan values with ''
        for k, v in confusion_matrix_new.items():
            confusion_matrix_new[k] = v.replace("nan", "")
        with open(
            "../../data/fixed_run/analysis/lemma_test_pred_sf/" + model + ".json", "w"
        ) as f:
            json.dump(confusion_matrix_new, f, indent=4)

        # filter the first 5 keys with the highest values in the confusion matrix
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
            + model
            + "_filtered.json",
            "w",
        ) as f:
            json.dump(confusion_matrix_filtered_new, f, indent=4)
