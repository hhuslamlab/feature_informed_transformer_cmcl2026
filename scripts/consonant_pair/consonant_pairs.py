"""
get consonant pairs
"""

import json
import re
import pandas as pd
from config import AR_SUFFIX_DICT, ER_SUFFIX_DICT, IR_SUFFIX_DICT, all_models
from get_stems import get_stem


if __name__ == "__main__":
    for model in all_models:
        print(model)
        condition = model.split("_")[0] + "_" + model.split("_")[1]
        run = model.split("_")[2]

        train_srcs_df = pd.read_csv(
            "../data/fixed_run/analysis/src_sf/train/" + model + ".csv"
        )
        tgt_srcs_df = pd.read_csv(
            "../data/fixed_run/analysis/tgt_sf/train/" + model + ".csv"
        )

        test_srcs_df = pd.read_csv(
            "../data/fixed_run/analysis/src_sf/test/" + model + ".csv"
        )
        test_tgt_df = pd.read_csv(
            "../data/fixed_run/analysis/tgt_sf/test/" + model + ".csv"
        )

        pred_tgt_df = pd.read_csv(
            "../data/fixed_run/analysis/pred_sf/" + model + ".csv"
        )

        idx_train_src = train_srcs_df["idx"].tolist()

        idx_test_src = test_srcs_df["idx"].tolist()

        idx_test_tgt = test_tgt_df["idx"].tolist()

        idx_srcs_tgt = test_tgt_df["idx"].tolist()
