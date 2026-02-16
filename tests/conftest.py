import pathlib
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

CONDITIONS = ["10L_90NL", "50L_50NL", "90L_10NL"]
RUNS = [1, 2, 3]
SEEDS = [1, 2, 3, 4]
ALL_MODELS = [f"{c}_{r}_{s}" for c in CONDITIONS for r in RUNS for s in SEEDS]

# Architecture -> (prediction directory name, file glob pattern)
PREDICTION_DIRS = {
    "vanilla": ("predictions_vanilla", "*.txt"),
    "character_separated": ("processed_predictions_sep_char", "[0-9]*.txt"),
    "feature_invariant": ("predictions_feature_invariant", "*.decode.test.tsv"),
    "independent_feature": ("predictions_independent_feature", "*.decode.test.tsv"),
    "feature_geometric": ("predictions_binaryfeature", "*.decode.test.tsv"),
}


@pytest.fixture
def project_root():
    return ROOT
