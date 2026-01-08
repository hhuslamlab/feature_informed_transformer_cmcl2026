"""
Analyze tag-level accuracies to test the L-shape hypothesis via clustering.

Steps:
1. Load lemma-tag accuracy data and aggregate accuracy per target tag.
2. Transform accuracies into L-shape scores:
   - Keep scores for V;IND;PRS;1;SG (the L-base) and all subjunctive tags.
   - For all other tags, use (1 - accuracy) so that 0 == purely non-L, 1 == purely L.
3. Build a 12x12 distance matrix with absolute differences between tag scores.
4. Run a 2-cluster solution (KMeans) on the scores.
5. Visualize the clusters in a paradigm grid (IND vs SBJV by person/number).
"""
import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans


DATA_ROOT = "../../data/accuracies/stem_accuracy_by_lemma_tag"
OUTPUT_ROOT = "../../data/accuracies/plots/tag_clustering"

BASE_TAG = "V;IND;PRS;1;SG"
SBJV_PREFIX = "V;SBJV"
TAG_ORDER = [
    "V;IND;PRS;1;SG",
    "V;IND;PRS;2;SG",
    "V;IND;PRS;3;SG",
    "V;IND;PRS;1;PL",
    "V;IND;PRS;2;PL",
    "V;IND;PRS;3;PL",
    "V;SBJV;PRS;1;SG",
    "V;SBJV;PRS;2;SG",
    "V;SBJV;PRS;3;SG",
    "V;SBJV;PRS;1;PL",
    "V;SBJV;PRS;2;PL",
    "V;SBJV;PRS;3;PL",
]
PARADIGM_ROWS = ["IND", "SBJV"]
PARADIGM_COLS = ["1;SG", "2;SG", "3;SG", "1;PL", "2;PL", "3;PL"]


def load_tag_accuracies(dataset: str) -> pd.DataFrame:
    """Return tag-level accuracies aggregated across models and conditions."""
    data_path = os.path.join(DATA_ROOT, dataset, "stem_acc_by_lemma_tag_all_models.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    if "condition" not in df.columns:
        df["condition"] = "vanilla"

    tag_stats = (
        df.groupby(["condition", "target_tag"])[["correct", "total"]]
        .sum()
        .reset_index()
    )
    tag_stats["accuracy"] = tag_stats["correct"] / tag_stats["total"]
    return tag_stats


def transform_scores(tag_stats: pd.DataFrame) -> pd.DataFrame:
    """Apply the L-shape score transformation."""
    def to_score(row: pd.Series) -> float:
        accuracy = row["accuracy"]
        tag = row["target_tag"]
        if tag == BASE_TAG or tag.startswith(SBJV_PREFIX):
            return accuracy
        return 1.0 - accuracy

    tag_stats["l_shape_score"] = tag_stats.apply(to_score, axis=1)
    return tag_stats


def build_distance_matrix(tag_stats: pd.DataFrame) -> pd.DataFrame:
    """Return 12x12 distance matrix from L-shape scores."""
    scores = tag_stats.set_index("target_tag").loc[TAG_ORDER]["l_shape_score"].values
    diff = scores[:, None] - scores[None, :]
    distance = np.abs(diff)
    return pd.DataFrame(distance, index=TAG_ORDER, columns=TAG_ORDER)


def cluster_tags(distance_df: pd.DataFrame, score_series: pd.Series) -> pd.DataFrame:
    """Cluster tags using KMeans on distance-based vectors."""
    vectors = distance_df.loc[TAG_ORDER, TAG_ORDER].values
    model = KMeans(n_clusters=2, random_state=42, n_init=20)
    labels = model.fit_predict(vectors)

    cluster_df = pd.DataFrame(
        {
            "target_tag": TAG_ORDER,
            "l_shape_score": score_series.loc[TAG_ORDER].values,
            "cluster": labels,
        }
    )
    return cluster_df


def build_paradigm_grid(cluster_df: pd.DataFrame) -> pd.DataFrame:
    """Create a mood x person-number grid with cluster labels."""
    tag_to_cluster = dict(zip(cluster_df["target_tag"], cluster_df["cluster"]))
    grid = pd.DataFrame(np.nan, index=PARADIGM_ROWS, columns=PARADIGM_COLS)

    for mood in PARADIGM_ROWS:
        for col in PARADIGM_COLS:
            tag = f"V;{mood};PRS;{col}"
            if tag in tag_to_cluster:
                grid.loc[mood, col] = tag_to_cluster[tag]

    return grid


def visualize_paradigm(grid: pd.DataFrame, dataset: str, condition: str) -> None:
    """Visualize the paradigm grid with cluster colors (axes swapped)."""
    heatmap_data = grid.loc[PARADIGM_ROWS, PARADIGM_COLS].T
    cmap = ListedColormap(["#3182bd", "#e6550d"])
    plt.figure(figsize=(5, 4))
    ax = sns.heatmap(
        heatmap_data,
        cmap=cmap,
        cbar=False,
        linewidths=1,
        linecolor="white",
        annot=True,
        fmt=".0f",
        annot_kws={"color": "white", "weight": "bold"},
    )
    ax.set_xticks(np.arange(len(PARADIGM_ROWS)) + 0.5)
    ax.set_xticklabels(PARADIGM_ROWS, rotation=0)
    ax.set_yticks(np.arange(len(PARADIGM_COLS)) + 0.5)
    ax.set_yticklabels(PARADIGM_COLS, rotation=0)
    ax.set_xlabel("Mood")
    ax.set_ylabel("Person;Number")
    plt.tight_layout()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    filename = f"paradigm_clusters_{dataset}_{condition}.png"
    plt.savefig(os.path.join(OUTPUT_ROOT, filename), dpi=300)
    plt.close()


def save_outputs(distance: pd.DataFrame, cluster_df: pd.DataFrame, dataset: str, condition: str) -> None:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    distance.to_csv(os.path.join(OUTPUT_ROOT, f"tag_distance_matrix_{dataset}_{condition}.csv"))
    cluster_df.to_csv(os.path.join(OUTPUT_ROOT, f"tag_clusters_{dataset}_{condition}.csv"), index=False)


def report(cluster_df: pd.DataFrame, condition: str) -> None:
    """Print a quick summary for the console."""
    print(f"Tag Scores and Cluster Assignments ({condition}):")
    print(cluster_df.to_string(index=False, formatters={"l_shape_score": "{:.3f}".format}))


def main(dataset: str, conditions: List[str] | None) -> None:
    tag_stats = load_tag_accuracies(dataset)
    tag_stats = transform_scores(tag_stats)

    if not conditions:
        conditions = sorted(tag_stats["condition"].unique())

    for condition in conditions:
        subset = tag_stats[tag_stats["condition"] == condition]
        if subset.empty:
            print(f"Skipping condition {condition}: no data found.")
            continue

        distance = build_distance_matrix(subset)
        score_series = subset.set_index("target_tag")["l_shape_score"]
        cluster_df = cluster_tags(distance, score_series)
        grid = build_paradigm_grid(cluster_df)

        save_outputs(distance, cluster_df, dataset, condition)
        visualize_paradigm(grid, dataset, condition)
        report(cluster_df, condition)
        print(f"Outputs saved to {OUTPUT_ROOT} for condition {condition}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster paradigm tags based on L-shape scores.")
    parser.add_argument("--dataset", default="vanilla", help="Dataset name under the accuracies directory.")
    parser.add_argument(
        "--conditions",
        nargs="*",
        help="Optional list of conditions to analyze (default: all present).",
    )
    args = parser.parse_args()
    main(args.dataset, args.conditions)

