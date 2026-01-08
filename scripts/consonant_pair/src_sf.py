"""
Script to get stem-final consonants for sources in both train and test. Just change the split to train/test.

Usage:
    src_sf.py --split=<s>

Options:
   --split=<s>                       Provide split (train, test)
"""
import sys, os
from docopt import docopt
import re
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import AR_SUFFIX_DICT, ER_SUFFIX_DICT, IR_SUFFIX_DICT, all_models
from get_stems import get_stem


def get_srcs(input_line):
    """
    get srcs
    """
    tags = re.findall("\<.*?\>", input_line)
    src1: str = input_line.split(tags[0])[0]

    src2: str = input_line[len(src1 + tags[0]) : len(input_line.split(tags[1])[0])]
    src2: str = src2.replace("#", "").replace(" ", "")

    src1: str = src1.replace(" ", "")

    return src1, src2


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
            + ".src"
        ) as f:
            src_data = f.readlines()
            src_data = [item.strip() for item in src_data]

        src1_sf = []
        src2_sf = []
        all_src1 = []
        all_src2 = []
        idxs = []
        for idx, src in enumerate(src_data):
            src1, src2 = get_srcs(src)
            src1_stem = get_stem(src1, suffixes)
            src2_stem = get_stem(src2, suffixes)
            if src1_stem and src2_stem:
                src1_stem_final_consonant = re.search(r"([^aeiou]*)$", src1_stem).group(
                    1
                )
                src2_stem_final_consonant = re.search(r"([^aeiou]*)$", src2_stem).group(
                    1
                )
                src1_sf.append(src1_stem_final_consonant.replace("ˈ", ""))
                src2_sf.append(src2_stem_final_consonant.replace("ˈ", ""))
                all_src1.append(src1)
                all_src2.append(src2)
                idxs.append(idx)
        df = pd.DataFrame()
        df["idx"] = idxs
        df["src1"] = all_src1
        df["src2"] = all_src2
        df["src1_sf"] = src1_sf
        df["src2_sf"] = src2_sf

        df.to_csv(
            "../../data/fixed_run/analysis/src_sf/" + split + "/" + model + ".csv",
            index=False,
        )
