#! /bin/bash
awk 'FNR > 1' ../../data/fixed_run/analysis/l_nl_accuracies/*.csv > ../../data/fixed_run/analysis/l_nl_accuracies/combine.csv
