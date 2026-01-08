"""
Analyze tag-level accuracies per lemma to test the L-shape hypothesis via clustering.

For each lemma under 10L_90NL condition:
1. Aggregate accuracy per target tag across all models.
2. Transform accuracies into L-shape scores:
   - Keep scores for V;IND;PRS;1;SG (the L-base) and all subjunctive tags.
   - For all other tags, use (1 - accuracy) so that 0 == purely non-L, 1 == purely L.
3. Build a 12x12 distance matrix with absolute differences between tag scores.
4. Run a 2-cluster solution (KMeans) on the distance matrix rows.
5. Save cluster assignments and distance matrices.
"""
import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


DATA_ROOT = "../../data/analysis/stem_accuracy_by_lemma_tag"
OUTPUT_ROOT = "../../data/analysis/plots/lemma_clustering"

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


def load_lemma_tag_accuracies(dataset: str, condition: str) -> pd.DataFrame:
    """Load lemma-tag accuracy data and filter for specific condition."""
    data_path = os.path.join(DATA_ROOT, dataset, "stem_acc_by_lemma_tag_all_models.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    if "condition" not in df.columns:
        raise ValueError("Data file missing 'condition' column")
    
    # Filter for the specified condition
    df = df[df["condition"] == condition].copy()
    if df.empty:
        raise ValueError(f"No data found for condition: {condition}")
    
    return df


def aggregate_lemma_tag_accuracies(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate accuracy per (lemma, tag) across all models."""
    agg_df = (
        df.groupby(["lemma", "target_tag"])[["correct", "total"]]
        .sum()
        .reset_index()
    )
    agg_df["accuracy"] = agg_df["correct"] / agg_df["total"]
    return agg_df


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
    # Ensure all tags are present, fill missing with NaN
    score_dict = dict(zip(tag_stats["target_tag"], tag_stats["l_shape_score"]))
    scores = np.array([score_dict.get(tag, np.nan) for tag in TAG_ORDER])
    
    # Check if we have all required tags
    if np.any(np.isnan(scores)):
        missing = [tag for i, tag in enumerate(TAG_ORDER) if np.isnan(scores[i])]
        raise ValueError(f"Missing tags for distance matrix: {missing}")
    
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


def save_lemma_outputs(
    distance: pd.DataFrame,
    cluster_df: pd.DataFrame,
    lemma: str,
    dataset: str,
    condition: str,
) -> None:
    """Save distance matrix and cluster assignments for a lemma."""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Sanitize lemma for filename (replace spaces and special chars)
    lemma_safe = lemma.replace(" ", "_").replace("/", "_").replace("\\", "_")
    
    distance.to_csv(
        os.path.join(OUTPUT_ROOT, f"lemma_distance_matrix_{dataset}_{condition}_{lemma_safe}.csv")
    )
    cluster_df.to_csv(
        os.path.join(OUTPUT_ROOT, f"lemma_clusters_{dataset}_{condition}_{lemma_safe}.csv"),
        index=False,
    )


def process_lemma(
    lemma: str,
    lemma_data: pd.DataFrame,
    dataset: str,
    condition: str,
) -> None:
    """Process clustering for a single lemma."""
    # Aggregate across models for this lemma
    tag_stats = aggregate_lemma_tag_accuracies(lemma_data)
    
    # Check if we have all 12 tags
    present_tags = set(tag_stats["target_tag"].unique())
    required_tags = set(TAG_ORDER)
    missing_tags = required_tags - present_tags
    
    if missing_tags:
        print(f"  Lemma {lemma}: Missing tags {missing_tags}, skipping...")
        return
    
    # Transform to L-shape scores
    tag_stats = transform_scores(tag_stats)
    
    # Build distance matrix
    try:
        distance = build_distance_matrix(tag_stats)
    except ValueError as e:
        print(f"  Lemma {lemma}: {e}, skipping...")
        return
    
    # Cluster
    score_series = tag_stats.set_index("target_tag")["l_shape_score"]
    cluster_df = cluster_tags(distance, score_series)
    
    # Save outputs
    save_lemma_outputs(distance, cluster_df, lemma, dataset, condition)
    
    print(f"  Processed {lemma}: {len(cluster_df[cluster_df['cluster'] == 0])} tags in cluster 0, "
          f"{len(cluster_df[cluster_df['cluster'] == 1])} tags in cluster 1")


def main(dataset: str = "vanilla", condition: str = "10L_90NL") -> None:
    """Main function to process all lemmas for the specified condition."""
    print(f"Loading lemma-tag accuracy data for {dataset}/{condition}...")
    df = load_lemma_tag_accuracies(dataset, condition)
    
    print(f"Found {len(df)} rows across {df['lemma'].nunique()} lemmas")
    
    # Get unique lemmas
    lemmas = sorted(df["lemma"].unique())
    print(f"Processing {len(lemmas)} lemmas...\n")
    
    processed = 0
    skipped = 0
    
    for lemma in lemmas:
        lemma_data = df[df["lemma"] == lemma].copy()
        try:
            process_lemma(lemma, lemma_data, dataset, condition)
            processed += 1
        except Exception as e:
            print(f"  Error processing {lemma}: {e}")
            skipped += 1
    
    print(f"\nCompleted: {processed} lemmas processed, {skipped} skipped")
    print(f"Outputs saved to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()





