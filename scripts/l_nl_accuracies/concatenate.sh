#! /bin/bash
awk 'FNR > 1' ../../data/analysis/l_nl_accuracies/*.csv > ../../data/analysis/l_nl_accuracies/combine.csv
