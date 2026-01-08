"""
Here we check the overlap between sfs of src1, src2 and tgt/pred in test and the training sets of all L-shape forms.
"""

import pandas as pd
from config import all_models

if __name__ == "__main__":
    split = "test"
    for model in all_models:
        print(model)

        ## we will be getting src1_sf and src2_sf from here:
        src_sf_df = pd.read_csv(
            "../data/fixed_run/analysis/src_sf/" + split + "/" + model + ".csv"
        )
        src1_sfs = src_sf_df["src1_sf"].tolist()
        src2_sfs = src_sf_df["src2_sf"].tolist()

        ## we will be getting the shape information from here:
        stem_sfs = pd.read_csv("../data/fixed_run/analysis/stems/" + model + ".csv")

        ## we will be getting tgt_sf from here:
        tgt_sf_df = pd.read_csv("../data/fixed_run/analysis/tgt_sf/" + model + ".csv")

        tgt_sfs = tgt_sf_df["pred_sf"].tolist()

        df = pd.DataFrame(
            list(zip(src1_sfs, src2_sfs, tgt_sfs)),
            columns=["src1_sf", "src2_sf", "tgt_sf"],
        )

        df.to_csv(
            "../data/fixed_run/analysis/sfs/pred/lshaped/" + model + ".csv", index=False
        )
