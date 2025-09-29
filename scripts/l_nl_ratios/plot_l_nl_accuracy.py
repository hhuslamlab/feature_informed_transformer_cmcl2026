"""
plot L and NL accuracies
"""
import tikzplotlib
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from math import sqrt
from config import condition_10L_90NL, condition_50L_50NL, condition_90L_10NL

# condition_10L_90NL = [item for item in condition_10L_90NL if item != "10L_90NL_2_2" and item != "10L_90NL_3_1"]
# condition_90L_10NL = [item for item in condition_10L_90NL if item != "90L_10NL_2_3"]

def plot_confidence_interval(
    x, values, point_color, label, z=1.96, color="#2187bb", horizontal_line_width=0.25
):
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    confidence_interval = z * stdev / sqrt(len(values))
    print(confidence_interval)
    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    plt.plot(x, mean, "o", color=point_color, label=label)

    return mean, confidence_interval



if __name__ == "__main__":
    data = pd.read_csv(
        "../../data/analysis/naacl25/l_nl_accuracies/combine.csv"
    )

    res_10L_90NL = data[data.apply(lambda r: r.str.contains('10L_90NL').any(), axis=1)]
    res_50L_50NL = data[data.apply(lambda r: r.str.contains('50L_50NL').any(), axis=1)]
    res_90L_10NL = data[data.apply(lambda r: r.str.contains('90L_10NL').any(), axis=1)]


    l_acc_10L_90NL = res_10L_90NL["l_acc"].tolist()
    l_acc_50L_50NL = res_50L_50NL["l_acc"].tolist()
    l_acc_90L_10NL = res_90L_10NL["l_acc"].tolist()

    print(statistics.mean(l_acc_10L_90NL))
    print(statistics.mean(l_acc_50L_50NL))
    print(statistics.mean(l_acc_90L_10NL))

    nl_acc_10L_90NL = res_10L_90NL["nl_acc"].tolist()
    nl_acc_50L_50NL = res_50L_50NL["nl_acc"].tolist()
    nl_acc_90L_10NL = res_90L_10NL["nl_acc"].tolist()

    print(statistics.mean(nl_acc_10L_90NL))
    print(statistics.mean(nl_acc_50L_50NL))
    print(statistics.mean(nl_acc_90L_10NL))

    plt.title("")
    plot_confidence_interval(1, l_acc_10L_90NL, "#FFA500", "L-shape")
    plot_confidence_interval(1, nl_acc_10L_90NL, "#808080", "NL-shape")
    plot_confidence_interval(2, l_acc_50L_50NL, "#FFA500", "")
    plot_confidence_interval(2, nl_acc_50L_50NL, "#808080", "")
    plot_confidence_interval(3, l_acc_90L_10NL, "#FFA500", "")
    plot_confidence_interval(3, nl_acc_90L_10NL, "#808080", "")
    barWidth = 1

    plt.xticks(
        [r + barWidth for r in range(3)], ["10%L-90%NL", "50%L-50%NL", "90%L-10%NL"]
    )
    plt.ylabel("Accuracy (in %)")
    # plt.xlabel('Condition (L%-NL%)')
    plt.ylim(bottom=0)
    # plt.legend(('L-shape', 'NL-shape'))
    plt.title("")
    tikzplotlib.save(
        "../../data/analysis/plots/naacl25_l_vs_nl_accuracy_without_stress.tex"
    )
