import json
import re
import pandas as pd
from config import AR_SUFFIX_DICT, ER_SUFFIX_DICT, IR_SUFFIX_DICT, all_models


if __name__ == "__main__":
    suffixes = ["eɾ", "aɾ", "iɾ"]

    with open("../data/ipa_clean_lshaped_dict.json") as f:
        lshaped_dict = json.load(f)

    for model in all_models:
        print(model)
        condition = model.split("_")[0] + "_" + model.split("_")[1]
        run = model.split("_")[2]

        with open(
            "../data/fixed_run/" + condition + "/test/run" + run + "/lshaped_lemmas.txt"
        ) as f:
            lshaped_lemmas = f.readlines()
            lshaped_lemmas = [item.strip() for item in lshaped_lemmas]
        lemma_form = {}

        for lemma in lshaped_lemmas:
            lf = lshaped_dict[lemma]
            for tag, form in lf.items():
                form = form.replace(" ", "").replace("ˈ", "")
                lemma = lemma.replace(" ", "")
                lemma_form[form] = lemma
        ## combine forms in src and tgt and have a set of forms.
        preds_df = pd.read_csv("../data/fixed_run/analysis/pred_sf/" + model + ".csv")
        tgt = preds_df["test"].tolist()
        preds = preds_df["pred"].tolist()

        df = pd.DataFrame()

        l_shape_lemmas = []
        lemmas_sf = []
        preds_ = []
        for t, p in zip(tgt, preds):
            try:
                lemma_ = lemma_form[t]
                l_shape_lemmas.append(lemma_)
                lemma_stem = lemma_[:-2].replace("ˈ", "")
                lemma_stem_final_consonant = re.search(
                    r"([^aeiou]*)$", lemma_stem
                ).group(1)
                lemmas_sf.append(
                    lemma_stem_final_consonant.replace("ˈ", "").replace(" ", "")
                )
                preds_.append(p)
            except:
                pass
        df["pred"] = preds_
        df["lemma"] = l_shape_lemmas
        df["lemma_sf"] = lemmas_sf

        df.to_csv(
            "../data/fixed_run/analysis/l_shaped/lemma_sf/pred/all_models/"
            + model
            + ".csv",
            index=False,
        )
