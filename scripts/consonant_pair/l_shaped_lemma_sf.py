import json
import re
import pandas as pd
from config import AR_SUFFIX_DICT, ER_SUFFIX_DICT, IR_SUFFIX_DICT, all_models
from get_stems import get_stem


if __name__ == "__main__":
    suffixes = ["eɾ", "aɾ", "iɾ"]

    with open("../data/ipa_clean_lshaped_dict.json") as f:
        lshaped_dict = json.load(f)

    for model in all_models:
        print(model)
        condition = model.split("_")[0] + "_" + model.split("_")[1]
        run = model.split("_")[2]

        with open(
            "../data/fixed_run/"
            + condition
            + "/train/run"
            + run
            + "/lshaped_lemmas.txt"
        ) as f:
            lshaped_lemmas = f.readlines()
            lshaped_lemmas = [item.strip() for item in lshaped_lemmas]
        lemma_form = {}

        for lemma in lshaped_lemmas:
            lf = lshaped_dict[lemma]
            for tag, form in lf.items():
                form = form.replace(" ", "")
                lemma = lemma.replace(" ", "")
                lemma_form[form] = lemma
        ## combine forms in src and tgt and have a set of forms.

        srcs = pd.read_csv("../data/fixed_run/analysis/src_sf/train/" + model + ".csv")
        src1 = srcs["src1"].tolist()
        src2 = srcs["src2"].tolist()
        tgts = pd.read_csv("../data/fixed_run/analysis/tgt_sf/train/" + model + ".csv")
        tgt = tgts["tgt"].tolist()

        unique_forms = list(set(src1) | set(src2) | set(tgt))
        print(len(unique_forms))
        df = pd.DataFrame()

        lemmas_ = []
        lemmas_sf = []
        forms_ = []
        for fo in unique_forms:
            try:
                lemma = lemma_form[fo]
                lemma_stem = get_stem(lemma, suffixes)
                lemma_stem_final_consonant = re.search(
                    r"([^aeiou]*)$", lemma_stem
                ).group(1)
                lemmas_sf.append(
                    lemma_stem_final_consonant.replace("ˈ", "").replace(" ", "")
                )
                lemmas_.append(lemma)
                forms_.append(fo)
            except:
                pass
        df["form"] = forms_
        df["lemma"] = lemmas_
        df["lemma_sf"] = lemmas_sf

        df.to_csv(
            "../data/fixed_run/analysis/l_shaped/lemma_sf/train/all_models/"
            + model
            + ".csv",
            index=False,
        )
