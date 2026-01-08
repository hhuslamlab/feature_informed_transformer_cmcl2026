"""
we filter the sfs for forms of L-shaped verbs. we first take the index of the test items from the fixed_run/analysis/stems/
"""

import pandas as pd
from config import all_models


if __name__ == "__main__":
    for model in all_models:
        print(model)
        ## get predictions from the predictions sf
        preds_df = pd.read_csv("../data/fixed_run/analysis/pred_sf/" + model + ".csv")

        ## get shape information from stems/ dir.
        shapes_df = pd.read_csv("../data/fixed_run/analysis/stems/" + model + ".csv")

        ## get srcs from the src_sf of the test set
        srcs_df = pd.read_csv(
            "../data/fixed_run/analysis/src_sf/test/" + model + ".csv"
        )
        ## we are first verifying if the prediction in preds_sfs are same as that in the shapes_df. If its the same, then we get the corresponding the sf from pred_sfs

        pred_from_pred_df = preds_df["pred"].tolist()

        pred_from_shapes_df = shapes_df["preds_form"].tolist()
        all_shapes = shapes_df["shapes"].tolist()

        true_test = preds_df["test"].tolist()

        src1_sf = srcs_df["src1_sf"].tolist()

        src2_sf = srcs_df["src2_sf"].tolist()

        pred_sf = preds_df["pred_sf"].tolist()

        triplets = []
        for pp, ps, src1, src2, pr_sf, shape in zip(
            pred_from_pred_df,
            pred_from_shapes_df,
            src1_sf,
            src2_sf,
            pred_sf,
            all_shapes,
        ):
            if pp == ps and shape == "L":
                triplets.append((src1, src2, pr_sf))

        pd.DataFrame(triplets, columns=["src1_sf", "src2_sf", "tgt_sf"]).to_csv(
            "../data/fixed_run/analysis/sfs/l_shape/pred/" + model + ".csv", index=False
        )
