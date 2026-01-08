"""
compute metrics related to section 6.3
"""

import pandas as pd
import statistics

if __name__ == "__main__":
    conditions = ["10L_90NL", "50L_50NL", "90L_10NL"]
    for cond in conditions:
        data = pd.read_csv(
            "../data/fixed_run/analysis/section_6_3/overlap_sf_all_conditions/"
            + cond
            + ".csv"
        )
        sf_percentage = data["sf_percentage"].tolist()

        mean = statistics.mean(sf_percentage)
        sd = statistics.stdev(sf_percentage)

        print(cond, mean, sd)
