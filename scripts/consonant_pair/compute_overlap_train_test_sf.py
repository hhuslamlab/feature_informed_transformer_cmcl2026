"""
Script to compute the overlap of stem final consonants across test set and train set
"""
import sys, os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import AR_SUFFIX_DICT, ER_SUFFIX_DICT, IR_SUFFIX_DICT, all_models


if __name__ == "__main__":
    for model in all_models:
        condition = model.split("_")[0] + "_" + model.split("_")[1]
        train_df = pd.read_csv(
            "../../data/fixed_run/analysis/src_sf/train/" + model + ".csv"
        )
        train_src1 = train_df["src1_sf"].to_list()
        train_src2 = train_df["src2_sf"].to_list()
        train_tgt_df = pd.read_csv(
            "../../data/fixed_run/analysis/tgt_sf/train/" + model + ".csv"
        )

        train_tgt = train_tgt_df["tgt_sf"].to_list()

        test_df = pd.read_csv(
            "../../data/fixed_run/analysis/src_sf/test/" + model + ".csv"
        )
        test_src1 = test_df["src1_sf"].to_list()
        test_src2 = test_df["src2_sf"].to_list()

        test_tgt_df = pd.read_csv(
            "../../data/fixed_run/analysis/tgt_sf/test/" + model + ".csv"
        )
        test_tgt = test_tgt_df["tgt_sf"].to_list()

        train = list(zip(train_src1, train_src2, train_tgt))
        test = list(zip(test_src1, test_src2, test_tgt))

        sfs_not_seen_in_train = []
        for item in test:
            if item not in train:
                sfs_not_seen_in_train.append(item)
        per_unseen = round(len(sfs_not_seen_in_train) / len(test) * 100, 2)
        per_seen = round(100 - per_unseen, 2)

        df = pd.DataFrame()
        df["model"] = [model]
        df["per_seen_triples"] = [per_seen]
        df["per_unseen_triples"] = [per_unseen]
        df.to_csv(
            "../../data/fixed_run/analysis/compute_overlap_train_test_sf/all/"
            + condition
            + "/"
            + model
            + ".csv",
            index=False,
        )
