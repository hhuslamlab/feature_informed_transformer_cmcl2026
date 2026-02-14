# Feature-Informed Morphological Inflection

Transformer-based models for Spanish verb morphological inflection with different morphological feature representations. This repository accompanies the LREC 2026 paper.

## Overview

We investigate how different representations of morphological features affect the performance of sequence-to-sequence models on morphological inflection. The models take a lemma and a set of morphological features as input and produce the inflected form.

### Model Variants

| Architecture | `--arch` flag | Description |
|---|---|---|
| Vanilla | `vanilla` | Standard transformer with features as a tag string |
| Character-separated | `charsep` | Features separated by individual characters |
| Feature-invariant | `featureinvariant` | Shared feature embedding across all feature types |
| Independent-feature | `independentfeature` | Separate embedding per feature type |
| Feature-geometric | `featuregeometric` | Feature geometry-aware encoding |
| Binary-feature | `binaryfeature` | Binary feature vector representation |
| Dual-source | `dualsource` | Separate encoder for lemma and features |

## Data

The data covers Spanish verb inflection with paradigm splits based on L-shaped (L) and non-L-shaped (NL) paradigm structure:

- **10L/90NL** &mdash; 10% L-shaped, 90% non-L-shaped training lemmas
- **50L/50NL** &mdash; 50% L-shaped, 50% non-L-shaped training lemmas
- **90L/10NL** &mdash; 90% L-shaped, 10% non-L-shaped training lemmas

Each condition has 3 independent data splits (runs), each trained with 4 random seeds.

Data files use `.src` (source: lemma + features) and `.tgt` (target: inflected form) format, organized under `data/{condition}/`.

## Repository Structure

```
feature_informed/
├── scripts/
│   ├── train.py                  # Training script
│   ├── test.py                   # Testing / decoding script
│   ├── transformer.py            # Base Transformer architecture
│   ├── independent_feature_transformer.py
│   ├── binary_feature_transformer.py
│   ├── dual_source_transformer.py
│   ├── dataloader.py             # Data loading and vocabulary
│   ├── trainer.py                # Training loop
│   ├── decoding.py               # Beam search / greedy decoding
│   ├── util.py                   # Utilities and argument parsing
│   ├── model.py                  # Transducer models
│   ├── accuracies/               # Overall accuracy computation
│   ├── l_nl_accuracies/          # L-shaped vs non-L-shaped accuracy
│   ├── stem_accuracies/          # Stem-level accuracy analysis
│   └── stem_accuracy_by_tag/     # Per-tag stem accuracy analysis
├── data/
│   ├── 10L_90NL/                 # Train/dev/test splits
│   ├── 50L_50NL/
│   ├── 90L_10NL/
│   ├── predictions/              # Model predictions
│   ├── analysis/                 # Accuracy CSVs and analysis results
│   └── plots/                    # Generated figures
├── train_*.sh                    # Training shell scripts per condition/run
├── test_*.sh                     # Testing shell scripts per condition/run
├── requirements.txt
└── README.md
```

## Usage

### Training

```bash
python scripts/train.py \
    --arch independentfeature \
    --dataset taginbrackets \
    --train data/10L_90NL/train/run1/train.10L_90NL_1_1.src \
            data/10L_90NL/train/run1/train.10L_90NL_1_1.tgt \
    --dev   data/10L_90NL/dev/run1/dev.10L_90NL_1_1.src \
            data/10L_90NL/dev/run1/dev.10L_90NL_1_1.tgt \
    --model checkpoints/10L_90NL_1_1 \
    --embed_dim 256 --src_hs 1024 --trg_hs 1024 \
    --src_layer 4 --trg_layer 4 --nb_heads 4 \
    --dropout 0.3 --bs 400 --max_steps 10000 \
    --warmup_steps 4000 --lr 0.001 --label_smooth 0.1
```

Batch training scripts are provided for each condition and run (e.g., `train_10L_run1.sh`).

### Testing

```bash
python scripts/test.py \
    --arch independentfeature \
    --dataset taginbrackets \
    --train data/10L_90NL/train/run1/train.10L_90NL_1_1.src \
            data/10L_90NL/train/run1/train.10L_90NL_1_1.tgt \
    --dev   data/10L_90NL/dev/run1/dev.10L_90NL_1_1.src \
            data/10L_90NL/dev/run1/dev.10L_90NL_1_1.tgt \
    --test  data/10L_90NL/test/run1/test.10L_90NL_1_1.src \
            data/10L_90NL/test/run1/test.10L_90NL_1_1.tgt \
    --model checkpoints/10L_90NL_1_1 \
    --load  checkpoints/10L_90NL_1_1.epoch_104 \
    --eval_test
```

Batch testing scripts are provided (e.g., `test_10L_run1.sh`).

### Analysis

Analysis scripts under `scripts/` compute accuracies at different granularities:

- `scripts/accuracies/` &mdash; Overall accuracy and box plots
- `scripts/l_nl_accuracies/` &mdash; Accuracy by L-shaped vs non-L-shaped paradigm membership
- `scripts/stem_accuracies/` &mdash; Stem-level accuracy
- `scripts/stem_accuracy_by_tag/` &mdash; Per-tag and per-lemma accuracy with clustering

## Requirements

See `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

## Citation

```bibtex
@inproceedings{Wiemerslage2026feature,
    title     = {Feature-Informed Morphological Inflection},
    author    = {Wiemerslage, Adam and Akhilesh},
    booktitle = {Proceedings of the 2026 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2026)},
    year      = {2026}
}
```
