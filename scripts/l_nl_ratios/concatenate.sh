#! /bin/bash
awk 'FNR > 1' ../../data/analysis/batch_size_32/l_nl_accuracies/*.csv > ../../data/analysis/batch_size_32/l_nl_accuracies/combine.csv
