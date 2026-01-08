"""
Script to get stem-final consonants for lemmas in both train and test. Just change the split to train/test.
Usage:
    lemma_sf.py --split=<s>

Options:
   --split=<s>                       Provide split (train, test)
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
    ar_suffixes = list(AR_SUFFIX_DICT.values())

    er_suffixes = list(ER_SUFFIX_DICT.values())

    ir_suffixes = list(IR_SUFFIX_DICT.values())

    form_suffixes = ar_suffixes + er_suffixes + ir_suffixes

    form_suffixes = sorted(list(set(form_suffixes)), key=len, reverse=True)

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
            + "/lemma_form.json"
        ) as f:
            lemma_form = json.load(f)

        lemma_form_new = {}
        for item in lemma_form:
            for k, v in item.items():
                for tag, form in v.items():
                    form = form.replace(" ", "")
                    k = k.replace(" ", "")
                    lemma_form_new[form] = k

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
            tgts = f.readlines()
            tgts = [item.strip().replace(" ", "") for item in tgts]
        lemmas = []
        lemmas_test_sf = []
        tgts_ = []
        idxs = []
        tgt_sf = []
        df = pd.DataFrame()

        for idx, tgt in enumerate(tgts):
            try:
                lemma = lemma_form_new[tgt]
                lemma_stem = get_stem(lemma, suffixes)
                lemma_stem_final_consonant = re.search(
                    r"([^aeiou]*)$", lemma_stem
                ).group(1)
                lemmas_test_sf.append(
                    lemma_stem_final_consonant.replace("ˈ", "").replace(" ", "")
                )
                lemmas.append(lemma)
                tgts_.append(tgt)
                idxs.append(idx)
            except:
                pass

        for tgt in tgts_:
            try:
                tgt_stem = get_stem(tgt, form_suffixes)
                tgt_stem_final_consonant = re.search(r"([^aeiou]*)$", tgt_stem).group(1)
                tgt_sf.append(
                    tgt_stem_final_consonant.replace("ˈ", "").replace(" ", "")
                )
            except:
                tgt_sf.append("")
        df["idx"] = idxs
        df["tgt"] = tgts_
        df["lemma"] = lemmas
        df["lemma_sf"] = lemmas_test_sf
        df["tgt_sf"] = tgt_sf
        df.to_csv(
            "../../data/fixed_run/analysis/lemmas_sf/"
            + split
            + "/run"
            + run
            + "/"
            + model
            + ".csv",
            index=False,
        )
