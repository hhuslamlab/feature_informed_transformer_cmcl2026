import pandas as pd
import statistics

if __name__ == "__main__":
    df = pd.read_csv("../../data/fixed_run/analysis/stem_accuracies/combine.csv")

    models = df["filename"].tolist()
    l_accs = df["l_acc"].tolist()
    nl_accs = df["nl_acc"].tolist()

    l_acc_10L = []
    l_acc_50L = []
    l_acc_90L = []
    for model, acc in zip(models, l_accs):
        if model.startswith("10L"):
            l_acc_10L.append(acc)

        if model.startswith("50L"):
            l_acc_50L.append(acc)

        if model.startswith("90L"):
            l_acc_90L.append(acc)

    l_mean_10L = round(statistics.mean(l_acc_10L), 2)
    l_mean_50L = round(statistics.mean(l_acc_50L), 2)
    l_mean_90L = round(statistics.mean(l_acc_90L), 2)

    df = pd.DataFrame()
    df["l_10L_90NL"] = [l_mean_10L]
    df["l_50L_50NL"] = [l_mean_50L]
    df["l_90L_10NL"] = [l_mean_90L]

    nl_acc_10L = []
    nl_acc_50L = []
    nl_acc_90L = []
    for model, acc in zip(models, nl_accs):
        if model.startswith("10L"):
            nl_acc_10L.append(acc)

        if model.startswith("50L"):
            nl_acc_50L.append(acc)

        if model.startswith("90L"):
            nl_acc_90L.append(acc)

    nl_mean_10L = round(statistics.mean(nl_acc_10L), 2)
    nl_mean_50L = round(statistics.mean(nl_acc_50L), 2)
    nl_mean_90L = round(statistics.mean(nl_acc_90L), 2)

    df["nl_10L_90NL"] = [nl_mean_10L]
    df["nl_50L_50NL"] = [nl_mean_50L]
    df["nl_90L_10NL"] = [nl_mean_90L]

    df.to_csv(
        "../../data/fixed_run/analysis/stem_accuracies/mean_accuracies.csv", index=False
    )
    print(f"l mean 10L: {l_mean_10L}")
    print(f"l mean 50L: {l_mean_50L}")
    print(f"l mean 90L: {l_mean_90L}")
    print(f"nl mean 10L: {nl_mean_10L}")
    print(f"nl mean 50L: {nl_mean_50L}")
    print(f"nl mean 90L: {nl_mean_90L}")
