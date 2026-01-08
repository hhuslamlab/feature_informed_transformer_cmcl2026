"""
Here we get the sfs counts of src1, src2 and tgt/pred in test and the training sets.

Usage:
    overlap_sf_all_tokens.py --split=<s>

Options:
   --split=<s>                       Provide split (train, test, pred)
"""
import sys, os
from docopt import docopt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from config import all_models

if __name__ == "__main__":
    args = docopt(__doc__)

    split = args["--split"]

    for model in all_models:
        print(model)

        ## we will be getting src1_sf and src2_sf from here:
        src_sf_df = pd.read_csv(
            "../data/fixed_run/analysis/src_sf/" + split + "/" + model + ".csv"
        )
        src1_sfs = src_sf_df["src1_sf"].tolist()
        src2_sfs = src_sf_df["src2_sf"].tolist()

        ## we will be getting tgt_sf from here:

        tgt_sf_df = pd.read_csv("../data/fixed_run/analysis/pred_sf/" + model + ".csv")

        tgt_sfs = tgt_sf_df["pred_sf"].tolist()

        df = pd.DataFrame(
            list(zip(src1_sfs, src2_sfs, tgt_sfs)),
            columns=["src1_sf", "src2_sf", "tgt_sf"],
        )

        df.to_csv(
            "../data/fixed_run/analysis/sfs/pred/" + "/" + model + ".csv", index=False
        )
