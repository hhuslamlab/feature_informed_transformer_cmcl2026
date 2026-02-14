"""
plot L and NL accuracies
"""
import tikzplotlib
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import os
from math import sqrt

# Define the model lists for each condition
condition_10L_90NL = [
    "10L_90NL_1_1", "10L_90NL_1_2", "10L_90NL_1_3", "10L_90NL_1_4",
    "10L_90NL_2_1", "10L_90NL_2_3", "10L_90NL_2_4",
    "10L_90NL_3_1", "10L_90NL_3_2", "10L_90NL_3_3", "10L_90NL_3_4"
]

condition_50L_50NL = [
    "50L_50NL_1_1", "50L_50NL_1_2", "50L_50NL_2_3", "50L_50NL_2_4",
    "50L_50NL_3_2", "50L_50NL_3_3", "50L_50NL_3_4"
]

condition_90L_10NL = [
    "90L_10NL_1_1", "90L_10NL_1_2", "90L_10NL_1_3", "90L_10NL_1_4",
    "90L_10NL_2_1", "90L_10NL_2_2", "90L_10NL_2_3", "90L_10NL_2_4",
    "90L_10NL_3_1", "90L_10NL_3_2", "90L_10NL_3_3", "90L_10NL_3_4"
]


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


def final(filename, dataset="vanilla"):
    filepath = f"../../data/analysis/accuracies/l_nl_{dataset}_" + filename + ".csv"
    if not os.path.exists(filepath):
        return None
    data = pd.read_csv(filepath)
    return {
        "l_acc": round(data["l_acc"].tolist()[0]),
        "nl_acc": round(data["nl_acc"].tolist()[0]),
    }


def create_combined_l_nl_plot():
    """Create single combined L/NL accuracy plot for all datasets."""
    # Color-blind friendly color scheme (Okabe-Ito palette inspired)
    # Order: Vanilla, Char. Sep., Feat. Inv., Feat.-Onehot, Feat.-Geom.
    model_colors = ['#D55E00', '#009E73', '#0072B2', '#CC79A7', '#F0E442']  # Vermillion, Bluish green, Blue, Reddish purple, Yellow
    datasets = ["vanilla", "character_separated", "feature_invariant", "independent_feature", "feature_geometric"]
    dataset_titles = ["Vanilla", "Char. Sep.", "Feat. Inv.", "Feat.-Onehot", "Feat.-Geom."]

    # Sized for full-width figure* (\textwidth ≈ 6.3")
    fig, ax = plt.subplots(figsize=(6.3, 3.2))

    # Position offsets for different models at each condition
    x_positions = [1, 2, 3]  # Base positions for conditions
    model_offsets = [-0.34, -0.17, 0.0, 0.17, 0.34]  # Small offsets to avoid overlap (5 models)

    legend_elements = []

    for idx, (dataset_name, color, title) in enumerate(zip(datasets, model_colors, dataset_titles)):
        try:
            res_10L_90NL = [final(item, dataset_name) for item in condition_10L_90NL]
            res_50L_50NL = [final(item, dataset_name) for item in condition_50L_50NL]
            res_90L_10NL = [final(item, dataset_name) for item in condition_90L_10NL]
            
            # Filter out None values (missing files)
            res_10L_90NL = [r for r in res_10L_90NL if r is not None]
            res_50L_50NL = [r for r in res_50L_50NL if r is not None]
            res_90L_10NL = [r for r in res_90L_10NL if r is not None]

            l_acc_10L_90NL = [item["l_acc"] for item in res_10L_90NL]
            l_acc_50L_50NL = [item["l_acc"] for item in res_50L_50NL]
            l_acc_90L_10NL = [item["l_acc"] for item in res_90L_10NL]

            nl_acc_10L_90NL = [item["nl_acc"] for item in res_10L_90NL]
            nl_acc_50L_50NL = [item["nl_acc"] for item in res_50L_50NL]
            nl_acc_90L_10NL = [item["nl_acc"] for item in res_90L_10NL]

            print(f"\n=== {dataset_name.upper()} Dataset Statistics ===")
            print(f"10L_90NL NL mean: {statistics.mean(nl_acc_10L_90NL):.2f}")
            print(f"50L_50NL NL mean: {statistics.mean(nl_acc_50L_50NL):.2f}")
            print(f"90L_10NL NL mean: {statistics.mean(nl_acc_90L_10NL):.2f}")

            # Plot L-shape accuracies (circles)
            l_means = []
            for cond_idx, (cond_name, l_acc) in enumerate([("10L_90NL", l_acc_10L_90NL),
                                                          ("50L_50NL", l_acc_50L_50NL),
                                                          ("90L_10NL", l_acc_90L_10NL)]):
                mean_val, _ = plot_single_point_with_ci(x_positions[cond_idx] + model_offsets[idx],
                                                       l_acc, color, 'o', 6)
                l_means.append(mean_val)

            # Plot NL-shape accuracies (squares)
            nl_means = []
            for cond_idx, (cond_name, nl_acc) in enumerate([("10L_90NL", nl_acc_10L_90NL),
                                                           ("50L_50NL", nl_acc_50L_50NL),
                                                           ("90L_10NL", nl_acc_90L_10NL)]):
                mean_val, _ = plot_single_point_with_ci(x_positions[cond_idx] + model_offsets[idx],
                                                       nl_acc, color, 's', 6)
                nl_means.append(mean_val)

            # Add to legend (only once per model)
            legend_elements.append(plt.Line2D([0], [0], marker='o', color=color, markerfacecolor=color,
                                            markersize=6, label=f'{title} L'))
            legend_elements.append(plt.Line2D([0], [0], marker='s', color=color, markerfacecolor=color,
                                            markersize=6, label=f'{title} NL'))

        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")

    # Customize the plot
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["10L-90NL", "50L-50NL", "90L-10NL"], fontsize=10)
    ax.set_xlabel("Condition", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(bottom=0, top=100)
    ax.grid(True, alpha=0.3, linewidth=0.3)
    ax.tick_params(labelsize=10)

    # Legend outside plot area
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.22),
              ncol=5, framealpha=0.9, fontsize=8, columnspacing=0.8, handletextpad=0.3,
              borderpad=0.4, labelspacing=0.4)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.32)

    # Save combined plots
    plt.savefig("../../data/analysis/accuracies/plots/l_vs_nl_accuracy_combined.png", dpi=300, bbox_inches='tight')
    plt.savefig("../../data/analysis/accuracies/plots/l_vs_nl_accuracy_combined.pdf", bbox_inches='tight')
    tikzplotlib.save("../../data/analysis/accuracies/plots/l_vs_nl_accuracy_combined.tex")

    print("Combined L/NL accuracy plot saved")
    plt.close()


def plot_single_point_with_ci(x, values, point_color, marker, markersize, z=1.96, horizontal_line_width=0.08):
    """Plot a single point with confidence interval."""
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    confidence_interval = z * stdev / sqrt(len(values))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval

    plt.plot([x, x], [top, bottom], color=point_color, linewidth=1.2)
    plt.plot([left, right], [top, top], color=point_color, linewidth=1.2)
    plt.plot([left, right], [bottom, bottom], color=point_color, linewidth=1.2)
    plt.plot(x, mean, marker, color=point_color, markersize=markersize, markeredgecolor='black', markeredgewidth=0.5)

    return mean, confidence_interval


def plot_confidence_interval_subplot(ax, x, values, point_color, label, z=1.96, color="#2187bb", horizontal_line_width=0.25):
    """Plot confidence interval on a subplot axis."""
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    confidence_interval = z * stdev / sqrt(len(values))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval

    ax.plot([x, x], [top, bottom], color=color)
    ax.plot([left, right], [top, top], color=color)
    ax.plot([left, right], [bottom, bottom], color=color)
    ax.plot(x, mean, "o", color=point_color, label=label)

    return mean, confidence_interval


if __name__ == "__main__":
    # Create combined plot
    create_combined_l_nl_plot()

    print("\nCombined L/NL accuracy plot generated!")
