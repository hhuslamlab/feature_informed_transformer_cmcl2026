"""Data integrity tests for CMCL 2026 reproducibility."""

import re
import pytest
from conftest import ROOT, CONDITIONS, RUNS, SEEDS, ALL_MODELS, PREDICTION_DIRS


@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("run", RUNS)
def test_data_splits_exist(condition, run):
    """Each condition/run must have train, dev, and test directories with .src/.tgt pairs."""
    for split in ("train", "dev", "test"):
        split_dir = ROOT / "data" / condition / split / f"run{run}"
        assert split_dir.is_dir(), f"Missing directory: {split_dir}"
        src_files = list(split_dir.glob("*.src"))
        tgt_files = list(split_dir.glob("*.tgt"))
        assert len(src_files) >= 1, f"No .src files in {split_dir}"
        assert len(tgt_files) >= 1, f"No .tgt files in {split_dir}"


@pytest.mark.parametrize("arch", PREDICTION_DIRS.keys())
def test_prediction_files_exist(arch):
    """Each architecture must have prediction files covering all 36 model conditions."""
    pred_dir_name, glob_pattern = PREDICTION_DIRS[arch]
    pred_dir = ROOT / "data" / "predictions" / pred_dir_name
    assert pred_dir.is_dir(), f"Missing prediction directory: {pred_dir}"

    pred_files = list(pred_dir.glob(glob_pattern))
    assert len(pred_files) >= 36, (
        f"{arch}: expected >= 36 prediction files, found {len(pred_files)} "
        f"in {pred_dir} with pattern '{glob_pattern}'"
    )

    found_models = set()
    for f in pred_files:
        match = re.search(r"(\d+L_\d+NL_\d+_\d+)", f.name)
        if match:
            found_models.add(match.group(1))

    missing = set(ALL_MODELS) - found_models
    assert not missing, f"{arch}: missing model predictions for {sorted(missing)}"


@pytest.mark.parametrize("model", ALL_MODELS)
def test_shape_info_files_exist(model):
    """Each model must have a shape_info file with only 'L' or 'NL' labels."""
    shape_file = ROOT / "scripts" / "shape_info" / model
    assert shape_file.is_file(), f"Missing shape_info file: {shape_file}"

    lines = shape_file.read_text().strip().splitlines()
    assert len(lines) > 0, f"Empty shape_info file: {shape_file}"
    invalid = [line for line in lines if line.strip() not in ("L", "NL")]
    assert not invalid, f"Invalid labels in {shape_file}: {invalid[:5]}"


def test_nevins_data_exists():
    """Nonce verb test data must be present."""
    nevins_dir = ROOT / "data" / "nevins_data"
    assert nevins_dir.is_dir(), f"Missing directory: {nevins_dir}"

    required_files = ["nevins_test.src", "full_paradigm.src", "full_paradigm.tgt"]
    for fname in required_files:
        fpath = nevins_dir / fname
        assert fpath.is_file(), f"Missing Nevins data file: {fpath}"


@pytest.mark.parametrize(
    "condition,expected_l_low,expected_l_high",
    [
        ("10L_90NL", 0.05, 0.25),
        ("50L_50NL", 0.35, 0.65),
        ("90L_10NL", 0.75, 1.00),
    ],
)
def test_l_nl_ratios(condition, expected_l_low, expected_l_high):
    """Average L-ratio across runs/seeds must fall within expected range."""
    l_ratios = []
    for run in RUNS:
        for seed in SEEDS:
            model = f"{condition}_{run}_{seed}"
            shape_file = ROOT / "scripts" / "shape_info" / model
            if not shape_file.is_file():
                continue
            lines = shape_file.read_text().strip().splitlines()
            n_total = len(lines)
            if n_total == 0:
                continue
            n_l = sum(1 for line in lines if line.strip() == "L")
            l_ratios.append(n_l / n_total)

    assert len(l_ratios) > 0, f"No shape_info files found for {condition}"
    avg_l = sum(l_ratios) / len(l_ratios)
    assert expected_l_low <= avg_l <= expected_l_high, (
        f"{condition}: average L-ratio = {avg_l:.3f}, "
        f"expected [{expected_l_low}, {expected_l_high}]"
    )
