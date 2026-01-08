"""
Script to get stem-final consonants for targets in both train and test. Just change the split to train/test.

Usage:
    tgt_sf.py --split=<s>

Options:
   --split=<s>                       Provide split (train, test)
"""

import sys, os
from docopt import docopt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from tqdm import tqdm
from config import AR_SUFFIX_DICT, ER_SUFFIX_DICT, IR_SUFFIX_DICT, all_models
from get_stems import get_stem
import re

if __name__ == "__main__":
    args = docopt(__doc__)

    split = args["--split"]

    ar_suffixes = list(AR_SUFFIX_DICT.values())

    er_suffixes = list(ER_SUFFIX_DICT.values())

    ir_suffixes = list(IR_SUFFIX_DICT.values())

    suffixes = ar_suffixes + er_suffixes + ir_suffixes

    suffixes = sorted(list(set(suffixes)), key=len, reverse=True)

    for model in all_models:
        condition = model.split("_")[0] + "_" + model.split("_")[1]
        run = model.split("_")[2]
        with open(
            "../../data/fixed_run/"
            + condition
            + "/"
            + split
            + "/run"
            + run
            + "/"
            + split
            + "."
            + model
            + ".tgt"
        ) as f:
            tgt_data = f.readlines()
            tgt_data = [item.strip().replace(" ", "") for item in tgt_data]

        tgt_sf = []
        all_tgt = []
        idxs = []
        for idx, tgt in enumerate(tgt_data):
            tgt_stem = get_stem(tgt, suffixes)
            if tgt_stem:
                tgt_stem_final_consonant = re.search(r"([^aeiou]*)$", tgt_stem).group(1)
                tgt_sf.append(
                    tgt_stem_final_consonant.replace("ˈ", "").replace(" ", "")
                )
                all_tgt.append(tgt.replace(" ", ""))
                idxs.append(idx)
        df = pd.DataFrame()
        df["idx"] = idxs
        df["tgt"] = all_tgt
        df["tgt_sf"] = tgt_sf

        df.to_csv(
            "../../data/fixed_run/analysis/tgt_sf/" + split + "/" + model + ".csv",
            index=False,
        )
