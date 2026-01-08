"""
TODO: refactor!!
Script to get stem-final consonants overlap between test and train.

Usage:
    freq_train_test_sf.py
"""
import os, sys
import pandas as pd
from collections import Counter


def get_l_shaped_test_count(l_shaped_test):
    l_shaped_test_lemma_sf = l_shaped_test["lemmas_sf"]
    l_shaped_test_tgt_sf = l_shaped_test["tgt_sf"]
    l_shaped_test_zip = list(zip(l_shaped_test_lemma_sf, l_shaped_test_tgt_sf))
    counter_lshaped_test = Counter(l_shaped_test_zip)

    return counter_lshaped_test


def get_l_shaped_train_count(l_shaped_train):
    l_shaped_train_lemma_sf = l_shaped_train["lemmas_sf"]
    l_shaped_train_tgt_sf = l_shaped_train["tgt_sf"]
    l_shaped_train_zip = list(zip(l_shaped_train_lemma_sf, l_shaped_train_tgt_sf))
    counter_lshaped_train = Counter(l_shaped_train_zip)

    return counter_lshaped_train


if __name__ == "__main__":
    all_models = [
        "10L_90NL_1_1",
        "10L_90NL_2_1",
        "10L_90NL_3_1",
        "50L_50NL_1_1",
        "50L_50NL_2_1",
        "50L_50NL_3_1",
        "90L_10NL_1_1",
        "90L_10NL_2_1",
        "90L_10NL_3_1",
    ]
    for model in all_models:
        condition = model.split("_")[0] + "_" + model.split("_")[1]
        run = model.split("_")[2]

        l_shaped_test = pd.read_csv(
            "../../data/fixed_run/analysis/lemma_train_test_sf/l_shaped/test"
            + "/run"
            + run
            + "/"
            + model
            + ".csv"
        )

        l_shaped_train = pd.read_csv(
            "../../data/fixed_run/analysis/lemma_train_test_sf/l_shaped/train"
            + "/run"
            + run
            + "/"
            + model
            + ".csv"
        )

        if model.startswith("90L_"):
            if run == "3":
                print(model)
                counter_lshaped_test = get_l_shaped_test_count(l_shaped_test)
                counter_lshaped_train = get_l_shaped_train_count(l_shaped_train)

                common_elements = counter_lshaped_test & counter_lshaped_train
                for element, count in counter_lshaped_test.items():
                    if element in common_elements:
                        print(f"Element: {element}")
                        print(
                            f"Count in l_shaped_test: {counter_lshaped_test[element]}"
                        )
                        print(
                            f"Count in l_shaped_train: {counter_lshaped_train[element]}"
                        )
                        print()
                    else:
                        print(f"Element: {element}")
                        print(
                            f"Count in l_shaped_test: {counter_lshaped_test[element]}"
                        )
                        print(f"Count in l_shaped_train: 0")
                        print()
