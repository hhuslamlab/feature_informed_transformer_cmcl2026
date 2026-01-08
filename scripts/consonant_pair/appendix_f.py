"""
number of occurences stem final consonants of L-shaped verbs across all conditions.
"""

import json
import re
import pandas as pd
from config import AR_SUFFIX_DICT, ER_SUFFIX_DICT, IR_SUFFIX_DICT, all_models
from get_stems import get_stem
from collections import Counter

if __name__ == "__main__":
    ar_suffixes = list(AR_SUFFIX_DICT.values())

    er_suffixes = list(ER_SUFFIX_DICT.values())

    ir_suffixes = list(IR_SUFFIX_DICT.values())

    suffixes = ar_suffixes + er_suffixes + ir_suffixes

    suffixes = sorted(list(set(suffixes)), key=len, reverse=True)

    with open("../data/ipa_clean_lshaped_dict.json") as f:
        lshaped_dict = json.load(f)

    for model in all_models:
        print(model)
        condition = model.split("_")[0] + "_" + model.split("_")[1]
        run = model.split("_")[2]

        df = pd.read_csv(
            "../data/fixed_run/analysis/l_shaped/lemma_sf/pred/"
            + condition
            + "/run"
            + run
            + "/combine.csv",
            header=None,
        )

        lemmas = df[1].tolist()
        lemmas_sfs = df[2].tolist()
        forms = df[0].tolist()
        forms_sfs = []

        for form in forms:
            form_stem = get_stem(form, suffixes)
            form_stem_final_consonant = re.search(r"([^aeiou]*)$", form_stem).group(1)
            forms_sfs.append(
                form_stem_final_consonant.replace("ˈ", "").replace(" ", "")
            )

        consonant_pairs = list(set(zip(lemmas, lemmas_sfs, forms_sfs)))

        sfs_ = []
        for item in consonant_pairs:
            lemma_sf_ = item[1]
            forms_sf_ = item[2]

            sfs_.append((lemma_sf_, forms_sf_))

        pd.DataFrame(
            Counter(sfs_).most_common(), columns=["consonant_pairs", "count"]
        ).to_csv(
            "../data/fixed_run/analysis/l_shaped/lemma_sf/pred/"
            + condition
            + "/figure_3/run"
            + run
            + "/combine.csv",
            index=False,
        )
