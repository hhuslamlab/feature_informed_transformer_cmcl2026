"""
get shape information of the form
"""

from typing import List
import json
from config import all_models


def get_lshaped_forms() -> List[str]:
    with open("../data/ipa_clean_lshaped_dict.json") as f:
        lshaped_dict = json.load(f)

    forms = [
        form.replace(" ", "") for k, v in lshaped_dict.items() for _, form in v.items()
    ]
    return forms


def get_shape(form: str, lshaped_forms: List[str]) -> str:
    if form in lshaped_forms:
        return "L"

    else:
        return "NL"


if __name__ == "__main__":
    lshaped_forms = get_lshaped_forms()
    for model in all_models:
        condition = model.split("_")[0] + "_" + model.split("_")[1]
        run = model.split("_")[2]
        with open(
            "../data/fixed_run/"
            + condition
            + "/test/run"
            + run
            + "/test."
            + model
            + ".tgt"
        ) as f:
            test_data = f.readlines()
            test_data = [item.strip().replace(" ", "") for item in test_data]
        with open("../data/fixed_run/analysis/shape_info/" + model, "w+") as f:
            for item in test_data:
                f.write(get_shape(item, lshaped_forms) + "\n")
