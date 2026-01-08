"""
Computing percentage of misclassified lemmas at the combination level
"""

import json
import pandas as pd
from config import all_models


if __name__ == "__main__":
    with open("../data/lshaped_lemmas_exceptions.json") as f:
        lemmas = json.load(f)

    for model in all_models:
        print(model)
        condition = model.split("_")[0] + "_" + model.split("_")[1]
        run = model.split("_")[2]
        split = "train"
        cls = "tgt"
        if cls == "src":
            srcs_df = pd.read_csv(
                "../data/fixed_run/analysis/src_sf/" + split + "/" + model + ".csv"
            )
            srcs1 = srcs_df["src1"].tolist()
            srcs2 = srcs_df["src2"].tolist()
            exception_forms = []

            for item in lemmas:
                for k, v in item.items():
                    exception_forms.append(v.replace(" ", ""))

            count_src1 = 0
            count_src2 = 0
            for src1 in srcs1:
                if src1 in list(set(exception_forms)):
                    count_src1 += 1

            for src2 in srcs2:
                if src2 in list(set(exception_forms)):
                    count_src2 += 1

            per_compute_src1 = round(count_src1 / len(srcs1) * 100, 2)
            per_compute_src2 = round(count_src2 / len(srcs1) * 100, 2)

            df = pd.DataFrame()
            df["total_count"] = [len(srcs1)]
            df["per_compute_src1"] = [per_compute_src1]
            df["per_compute_src2"] = [per_compute_src2]

        if cls == "tgt":
            tgt_df = pd.read_csv(
                "../data/fixed_run/analysis/tgt_sf/" + split + "/" + model + ".csv"
            )
            tgts = tgt_df["tgt"].tolist()
            exception_forms = []

            for item in lemmas:
                for k, v in item.items():
                    exception_forms.append(v.replace(" ", ""))

            count_tgt = 0
            for tgt in tgts:
                if tgt in list(set(exception_forms)):
                    count_tgt += 1

            per_compute_tgt = round(count_tgt / len(tgts) * 100, 2)

            df = pd.DataFrame()
            df["total_count"] = [len(tgts)]
            df["per_compute_tgt"] = [per_compute_tgt]

            df.to_csv(
                "../data/fixed_run/analysis/misclassification/"
                + condition
                + "/"
                + cls
                + "/"
                + split
                + "/"
                + model
                + ".csv",
                index=False,
            )
