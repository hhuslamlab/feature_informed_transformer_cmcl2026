"""
plot stem accuracies for all model types
"""
import os, sys
# import tikzplotlib  # Commented out due to matplotlib compatibility issues
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from math import sqrt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "l_nl_accuracies")))

from config import condition_10L_90NL, condition_50L_50NL, condition_90L_10NL


def plot_single_point_with_ci(x, values, point_color, marker, markersize, z=1.96, horizontal_line_width=0.15):
    """Plot a single point with confidence interval."""
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    confidence_interval = z * stdev / sqrt(len(values))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval

    plt.plot([x, x], [top, bottom], color=point_color, linewidth=2)
    plt.plot([left, right], [top, top], color=point_color, linewidth=2)
    plt.plot([left, right], [bottom, bottom], color=point_color, linewidth=2)
    plt.plot(x, mean, marker, color=point_color, markersize=markersize, markeredgecolor='black', markeredgewidth=1)

    return mean, confidence_interval


def final(filename, dataset="vanilla"):
    """Read stem accuracy data for a given model and dataset type"""
    if dataset == "vanilla":
        filepath = f"../../data/accuracies/stem_accuracies/stem_acc_{filename}.csv"
    else:
        filepath = f"../../data/accuracies/stem_accuracies/{dataset}/stem_acc_{filename}.csv"
    
    if not os.path.exists(filepath):
        return None
    
    data = pd.read_csv(filepath)
    return {
        "l_acc": round(data["l_acc"].tolist()[0]),
        "nl_acc": round(data["nl_acc"].tolist()[0]),
    }


def create_combined_stem_plot():
    """Create single combined stem L/NL accuracy plot for all datasets."""
    # Color-blind friendly color scheme (Okabe-Ito palette inspired)
    model_colors = ['#D55E00', '#0072B2', '#CC79A7', '#009E73']  # Vermillion, Blue, Reddish purple, Bluish green
    datasets = ["vanilla", "feature_invariant", "independent_feature", "character_separated"]
    dataset_titles = ["Vanilla", "Feature Invariant", "Dual Source", "Character Separated"]

    # Optimized for 2-column LaTeX layout - matching boxplot.py
    plt.figure(figsize=(7, 5))

    # Position offsets for different models at each condition
    x_positions = [1, 2, 3]  # Base positions for conditions
    model_offsets = [-0.3, -0.1, 0.1, 0.3]  # Small offsets to avoid overlap (4 models)

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
            for cond_idx, (cond_name, l_acc) in enumerate([("10L_90NL", l_acc_10L_90NL),
                                                          ("50L_50NL", l_acc_50L_50NL),
                                                          ("90L_10NL", l_acc_90L_10NL)]):
                plot_single_point_with_ci(x_positions[cond_idx] + model_offsets[idx],
                                         l_acc, color, 'o', 8)

            # Plot NL-shape accuracies (squares)
            for cond_idx, (cond_name, nl_acc) in enumerate([("10L_90NL", nl_acc_10L_90NL),
                                                           ("50L_50NL", nl_acc_50L_50NL),
                                                           ("90L_10NL", nl_acc_90L_10NL)]):
                plot_single_point_with_ci(x_positions[cond_idx] + model_offsets[idx],
                                         nl_acc, color, 's', 8)

            # Add to legend (only once per model)
            legend_elements.append(plt.Line2D([0], [0], marker='o', color=color, markerfacecolor=color,
                                            markersize=8, label=f'{title} L-shape'))
            legend_elements.append(plt.Line2D([0], [0], marker='s', color=color, markerfacecolor=color,
                                            markersize=8, label=f'{title} NL-shape'))

        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")

    # Customize the plot
    plt.xticks(x_positions, ["10%L-90%NL", "50%L-50%NL", "90%L-10%NL"], fontsize=12)
    plt.xlabel("Condition", fontsize=12, fontweight='bold')
    plt.ylabel("Stem Accuracy (%)", fontsize=12, fontweight='bold')
    plt.ylim(bottom=0, top=100)
    plt.grid(True, alpha=0.3)
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), framealpha=0.9, fontsize=8)
    
    # Set tick label sizes
    ax = plt.gca()
    ax.tick_params(labelsize=12)

    plt.tight_layout()

    # Save combined plots
    os.makedirs("../../data/accuracies/plots", exist_ok=True)
    plt.savefig("../../data/accuracies/plots/stem_accuracies_combined.png", dpi=300, bbox_inches='tight')
    plt.savefig("../../data/accuracies/plots/stem_accuracies_combined.pdf", bbox_inches='tight')
    # tikzplotlib.save("../../data/accuracies/plots/stem_accuracies_combined.tex")  # Commented out due to matplotlib compatibility issues

    print("Combined stem accuracy plot saved")
    plt.close()


if __name__ == "__main__":
    # Create combined plot
    create_combined_stem_plot()

    print("\nCombined stem accuracy plot generated!")
