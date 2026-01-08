# Stem Accuracy Analysis

This folder contains scripts to calculate and visualize stem accuracy grouped by:
1. **Target tags** - the last tag in the source file enclosed in `<>`
2. **Lemmas** - the base form of the verb

## Overview

These scripts analyze how well different models predict stems for different morphological categories and lemmas.

For example, in the source line:
```
s u p e ɾ b e n ɡ ˈ a m o s <V;SBJV;PRS;1;PL> # s u p e ɾ b ˈ e n ɡ a n <V;SBJV;PRS;3;PL> # <V;SBJV;PRS;2;PL>
```
The target tag is `V;SBJV;PRS;2;PL` (the rightmost tag).

## Scripts

### 1. `calculate_stem_accuracy_by_tag.py`

Calculates stem accuracy grouped by target tags for all models and prediction types.

**Usage:**
```bash
cd /home/akhilesh/projects/research/neural-transducer/analysis/scripts/stem_accuracy_by_tag
python3 calculate_stem_accuracy_by_tag.py
```

**Quick Run:**
```bash
./run_analysis.sh
```
This will run both calculation and plotting scripts automatically.

**Output:**
- Creates CSV files in `../../data/accuracies/stem_accuracy_by_tag/{dataset}/`
- For each model: `stem_acc_by_tag_{model}.csv` - detailed breakdown by tag
- `stem_acc_by_tag_all_models.csv` - combined results for all models
- `stem_acc_by_tag_summary.csv` - summary statistics by tag across all models

**Output Format:**

*Per-model files* contain:
- `target_tag`: The morphological tag
- `correct`: Number of correct stem predictions
- `total`: Total number of instances
- `accuracy`: Stem accuracy percentage for this tag

*Combined file* contains:
- `model`: Model name (e.g., "10L_90NL_1_1")
- `condition`: Condition (e.g., "10L_90NL")
- `run`: Run number
- `target_tag`: The morphological tag
- `correct`: Number of correct stem predictions
- `total`: Total number of instances
- `accuracy`: Stem accuracy percentage

*Summary file* contains:
- `target_tag`: The morphological tag
- `correct`: Total correct predictions across all models
- `total`: Total instances across all models
- `accuracy`: Average accuracy across models
- `overall_accuracy`: Overall accuracy (total correct / total instances)

---

## Scripts for Lemma-Based Analysis

### 3. `calculate_stem_accuracy_by_lemma.py`

Calculates stem accuracy grouped by lemma for all models and prediction types. Uses the `ipa_clean_lshaped_dict.json` file to map target forms to their lemmas.

**Usage:**
```bash
cd /home/akhilesh/projects/research/neural-transducer/analysis/scripts/stem_accuracy_by_tag
python3 calculate_stem_accuracy_by_lemma.py
```

**How it works:**
1. Loads the lemma dictionary (`ipa_clean_lshaped_dict.json`)
2. Builds a reverse mapping from inflected forms to lemmas
3. For each target form, looks up its lemma
4. Calculates stem accuracy grouped by lemma

**Output:**
- Creates CSV files in `../../data/accuracies/stem_accuracy_by_lemma/{dataset}/`
- For each model: `stem_acc_by_lemma_{model}.csv` - detailed breakdown by lemma
- `stem_acc_by_lemma_all_models.csv` - combined results for all models
- `stem_acc_by_lemma_summary.csv` - summary statistics by lemma across all models

**Output Format:**

*Per-model files* contain:
- `lemma`: The lemma (in IPA format with spaces)
- `correct`: Number of correct stem predictions
- `total`: Total number of instances
- `accuracy`: Stem accuracy percentage for this lemma

*Combined file* contains:
- `model`: Model name (e.g., "10L_90NL_1_1")
- `condition`: Condition (e.g., "10L_90NL")
- `run`: Run number
- `lemma`: The lemma
- `correct`: Number of correct stem predictions
- `total`: Total number of instances
- `accuracy`: Stem accuracy percentage

*Summary file* contains:
- `lemma`: The lemma
- `correct`: Total correct predictions across all models
- `total`: Total instances across all models
- `accuracy`: Average accuracy across models
- `overall_accuracy`: Overall accuracy (total correct / total instances)

**Note:** Only forms that exist in `ipa_clean_lshaped_dict.json` will be mapped to lemmas. Forms not in the dictionary will be excluded from the analysis.

### 4. `calculate_stem_accuracy_by_lemma_tag.py`

Calculates stem accuracy grouped jointly by lemma and target tag.

**Usage:**
```bash
cd /home/akhilesh/projects/research/neural-transducer/analysis/scripts/stem_accuracy_by_tag
python3 calculate_stem_accuracy_by_lemma_tag.py
```

**Output:**
- Per-model files: `stem_acc_by_lemma_tag_{model}.csv`
  - Columns: `lemma`, `target_tag`, `correct`, `total`, `accuracy`
- Combined across all models: `stem_acc_by_lemma_tag_all_models.csv`
  - Adds `model`, `condition`, `run`
- Summary: `stem_acc_by_lemma_tag_summary.csv`
  - Aggregated totals and `overall_accuracy` for each lemma/tag pair

**Notes:**
- Lemmas are sourced from `ipa_clean_lshaped_dict.json`
- Tags are extracted from the source `.src` files (last `<...>` token per line)
- Entries missing lemma or tag are skipped with a warning

### 5. `test_single_model_lemma.py`

Quick test script for analyzing a single model's stem accuracy grouped by lemma.

**Usage:**
```bash
cd /home/akhilesh/projects/research/neural-transducer/analysis/scripts/stem_accuracy_by_tag
python3 test_single_model_lemma.py
```

Shows the top 20 and bottom 20 lemmas by accuracy, along with unmapped form count.

---

## Scripts for Tag-Based Analysis

### 2. `plot_stem_accuracy_by_tag.py`

Creates visualizations of stem accuracy by target tag.

**Usage:**
```bash
cd /home/akhilesh/projects/research/neural-transducer/analysis/scripts/stem_accuracy_by_tag
python3 plot_stem_accuracy_by_tag.py
```

**Output:**
Creates plots in `../../data/accuracies/plots/stem_accuracy_by_tag/`:

For each dataset (vanilla, feature_invariant, character_separated, independent_feature):
- `stem_accuracy_by_tag_overall.png/pdf` - Overall accuracy by tag across all models
- `stem_accuracy_by_tag_10L_90NL.png/pdf` - Accuracy by tag for 10L_90NL condition
- `stem_accuracy_by_tag_50L_50NL.png/pdf` - Accuracy by tag for 50L_50NL condition
- `stem_accuracy_by_tag_90L_10NL.png/pdf` - Accuracy by tag for 90L_10NL condition
- `stem_accuracy_by_tag_heatmap.png/pdf` - Heatmap showing accuracy by tag and condition

Additionally:
- `stem_accuracy_by_tag_comparison.png/pdf` - Comparison across all datasets for top 12 tags

## Data Flow

1. **Input Files:**
   - Source files: `analysis/{condition}/test/run{run}/test.{model}.src`
   - Target files: `analysis/{condition}/test/run{run}/test.{model}.tgt`
   - Prediction files: `predictions_{dataset}/{model}.txt` or `.tsv`

2. **Processing:**
   - Extract target tag from each source line (last tag in angle brackets)
   - Load predictions and ground truth targets
   - Remove suffixes to extract stems
   - Calculate accuracy grouped by target tag

3. **Output:**
   - CSV files with detailed accuracy metrics
   - PNG and PDF plots for visualization

## Quick Start

### Tag-Based Analysis
```bash
# Run full analysis (calculation + plotting)
./run_analysis.sh

# Or run individually:
python3 calculate_stem_accuracy_by_tag.py
python3 plot_stem_accuracy_by_tag.py

# Test with single model
python3 test_single_model.py
```

### Lemma-Based Analysis
```bash
# Run full analysis
./run_lemma_analysis.sh

# Or run directly:
python3 calculate_stem_accuracy_by_lemma.py
python3 calculate_stem_accuracy_by_lemma_tag.py

# Test with single model
python3 test_single_model_lemma.py
```

## Requirements

- pandas
- matplotlib
- seaborn
- numpy
- json

## Files in This Directory

| File | Purpose |
|------|---------|
| `calculate_stem_accuracy_by_tag.py` | Calculate accuracy by target tag |
| `calculate_stem_accuracy_by_lemma.py` | Calculate accuracy by lemma |
| `calculate_stem_accuracy_by_lemma_tag.py` | Calculate accuracy by lemma/tag pair |
| `plot_stem_accuracy_by_tag.py` | Generate visualizations for tag-based analysis |
| `test_single_model.py` | Quick test for tag-based analysis on one model |
| `test_single_model_lemma.py` | Quick test for lemma-based analysis on one model |
| `run_analysis.sh` | Run complete tag-based analysis |
| `run_lemma_analysis.sh` | Run complete lemma-based analysis |
| `README.md` | This documentation |
| `EXAMPLE_OUTPUT.md` | Example output from test runs |

## Notes

- The scripts process four prediction types: vanilla, feature_invariant, character_separated, and independent_feature
- Stems are extracted by removing Spanish verb suffixes (AR, ER, IR conjugations)
- All text is normalized by removing spaces and stress markers (ˈ)
- Results are sorted by accuracy for easier interpretation
- Lemma-based analysis only includes forms found in `ipa_clean_lshaped_dict.json`
- Tag-based analysis includes all forms in the test set

