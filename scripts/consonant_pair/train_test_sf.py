"""
Script to get stem-final consonants of the lemma and its corresponding form in both train and test in L/NL shaped.

This is done by taking the lemma_sf and getting the lemmas of the L/NL lemmas.

Usage:
    train_test_sf.py --split=<sp>

Options:
    --split=<sp>                        Provide split (train, test)
"""

import sys, os
from docopt import docopt
import json
import re
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import AR_SUFFIX_DICT, ER_SUFFIX_DICT, IR_SUFFIX_DICT, all_models
from get_stems import get_stem

if __name__ == "__main__":
    args = docopt(__doc__)

    split = args["--split"]

    suffixes = ["eɾ", "aɾ", "iɾ"]

    for model in all_models:
        condition = model.split("_")[0] + "_" + model.split("_")[1]
        run = model.split("_")[2]

        data = pd.read_csv(
            "../../data/fixed_run/analysis/lemmas_sf/"
            + split
            + "/run"
            + run
            + "/"
            + model
            + ".csv",
        ).fillna("")

        lemma_sfs = data["lemma_sf"].tolist()
        tgt_sfs = data["tgt_sf"].tolist()
        lemmas = data["lemma"].tolist()

        with open(
            "../../data/fixed_run/"
            + condition
            + "/"
            + split
            + "/run"
            + run
            + "/lshaped_lemmas.txt"
        ) as f:
            l_shaped_lemmas = f.readlines()
            l_shaped_lemmas = [
                item.strip().replace(" ", "") for item in l_shaped_lemmas
            ]

        lemma_sf_dict = {}
        l_shaped_lemma_sf_dict = {}

        for lemma, lemma_sf, tgt_sf in list(set(zip(lemmas, lemma_sfs, tgt_sfs))):
            lemma_sf_dict[lemma] = {"lemma_sf": lemma_sf, "tgt_sf": tgt_sf}
            if lemma_sf != tgt_sf:
                if lemma in l_shaped_lemmas:
                    l_shaped_lemma_sf_dict[lemma] = {
                        "lemma_sf": lemma_sf,
                        "tgt_sf": tgt_sf,
                    }
                else:
                    print("exceptions: ", lemma)
        print(model)
        data = pd.DataFrame()
        set_lemmas = []
        set_lemmas_sf = []
        set_tgt_sf = []

        set_l_shaped_lemmas = []
        set_l_shaped_lemmas_sf = []
        set_l_shaped_tgt_sf = []
        print(l_shaped_lemma_sf_dict)
        for item in list(set(lemmas)):
            if item in l_shaped_lemmas:
                try:
                    set_l_shaped_lemmas_sf.append(
                        l_shaped_lemma_sf_dict[item]["lemma_sf"]
                    )
                    set_l_shaped_tgt_sf.append(l_shaped_lemma_sf_dict[item]["tgt_sf"])
                    set_l_shaped_lemmas.append(item)
                except:
                    pass
            set_lemmas_sf.append(lemma_sf_dict[item]["lemma_sf"])
            set_tgt_sf.append(lemma_sf_dict[item]["tgt_sf"])
            set_lemmas.append(item)

        lemma_sf_df = pd.DataFrame()
        lemma_sf_df["lemmas"] = set_lemmas
        lemma_sf_df["lemmas_sf"] = set_lemmas_sf
        lemma_sf_df["tgt_sf"] = set_tgt_sf

        lemma_l_shaped_sf_df = pd.DataFrame()
        lemma_l_shaped_sf_df["lemmas"] = set_l_shaped_lemmas
        lemma_l_shaped_sf_df["lemmas_sf"] = set_l_shaped_lemmas_sf
        lemma_l_shaped_sf_df["tgt_sf"] = set_l_shaped_tgt_sf

        lemma_sf_df.to_csv(
            "../../data/fixed_run/analysis/lemma_train_test_sf/all/"
            + split
            + "/run"
            + run
            + "/"
            + model
            + ".csv",
            index=False,
        )

        lemma_l_shaped_sf_df.to_csv(
            "../../data/fixed_run/analysis/lemma_train_test_sf/l_shaped/"
            + split
            + "/run"
            + run
            + "/"
            + model
            + ".csv",
            index=False,
        )
