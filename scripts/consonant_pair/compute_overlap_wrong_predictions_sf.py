"""
here we compute the overlap of stem final consonants across predictions set and train set for wrong predictions.
"""

import pandas as pd
from config import all_models


if __name__ == "__main__":
    for model in all_models:
        print(model)
        train_df = pd.read_csv("../data/fixed_run/analysis/sfs/train/" + model + ".csv")
        train_src1 = train_df["src1_sf"].to_list()
        train_src2 = train_df["src2_sf"].to_list()
        train_tgt = train_df["tgt_sf"].to_list()

        pred_df = pd.read_csv(
            "../data/fixed_run/analysis/sfs/l_shape/pred/" + model + ".csv"
        )
        pred_src1 = pred_df["src1_sf"].to_list()
        pred_src2 = pred_df["src2_sf"].to_list()
        pred_tgt = pred_df["tgt_sf"].to_list()

        stem_sfs = pd.read_csv("../data/fixed_run/analysis/stems/" + model + ".csv")

        predictions = stem_sfs["preds_form"].tolist()
        true_tests = stem_sfs["test_form"].tolist()

        train_triples = list(zip(train_src1, train_src2, train_tgt))
        pred_triples = list(zip(pred_src1, pred_src2, pred_tgt))

        sfs_not_seen_in_train = []
        for pred_triple, pred, test in zip(pred_triples, predictions, true_tests):
            if pred != test:
                if pred_triple not in train_triples:
                    sfs_not_seen_in_train.append(pred_triple)

        print(len(sfs_not_seen_in_train) / len(pred_triples) * 100)
