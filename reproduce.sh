#!/bin/bash
# Reproduction script for:
# "Character-aware Transformers Learn an Irregular Morphological Pattern Yet None Generalize Like Humans"
# CMCL 2026
#
# Usage:
#   bash reproduce.sh           # Run analysis only (uses pre-computed predictions)
#   bash reproduce.sh --train   # Full pipeline: train all models + analysis
#
# Prerequisites:
#   - Python 3.10+ with dependencies installed (uv sync)
#   - GPU recommended for training
#   - tikzplotlib for LaTeX-format plot export

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
CONDITIONS=("10L_90NL" "50L_50NL" "90L_10NL")
RUNS=(1 2 3)
SEEDS=(1 2 3 4)


# Training pipeline (optional, skip with pre-computed predictions)


if [ "$1" = "--train" ]; then
    echo "Training all models"
    echo "NOTE: This trains 4 architecture types x 3 conditions x 3 runs x 4 seeds = 144 models"
    echo ""

    # Architecture flags:
    #   transformer        -> Vanilla AND Character-separated
    #                         (same model class, difference is in input data format)
    #   tagtransformer     -> Feature-invariant (tag tokens get position encoding 0)
    #   independentfeature -> Feature-onehot
    #   binaryfeature      -> Feature-geometric
    ARCHS=("transformer" "tagtransformer" "independentfeature" "binaryfeature")

    # Hyperparameters (shared across all architectures)
    TRAIN_ARGS="--dataset taginbrackets \
        --embed_dim 256 \
        --src_hs 1024 --trg_hs 1024 \
        --src_layer 4 --trg_layer 4 \
        --nb_heads 4 \
        --dropout 0.3 \
        --bs 400 \
        --max_steps 10000 \
        --warmup_steps 4000 \
        --lr 0.001 \
        --label_smooth 0.1 \
        --skip_dev_eval \
        --save_last_only"

    TEST_ARGS="--dataset taginbrackets \
        --embed_dim 256 \
        --src_hs 1024 --trg_hs 1024 \
        --src_layer 4 --trg_layer 4 \
        --nb_heads 4 \
        --dropout 0.3 \
        --bs 16 \
        --decode beam \
        --decode_beam_size 5 \
        --eval_test"

    cd "${ROOT}"
    export PYTHONPATH="${ROOT}/scripts:${PYTHONPATH}"

    for arch in "${ARCHS[@]}"; do
        echo "Training arch: ${arch}"
        for cond in "${CONDITIONS[@]}"; do
            for run in "${RUNS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    model_id="${cond}_${run}_${seed}"
                    echo "Training ${arch} / ${model_id}"

                    python scripts/train.py \
                        --arch "${arch}" \
                        ${TRAIN_ARGS} \
                        --train "data/${cond}/train/run${run}/train.${model_id}.src" \
                                "data/${cond}/train/run${run}/train.${model_id}.tgt" \
                        --dev   "data/${cond}/dev/run${run}/dev.${model_id}.src" \
                                "data/${cond}/dev/run${run}/dev.${model_id}.tgt" \
                        --model "checkpoints/${arch}/${model_id}"

                    echo "Testing ${arch} / ${model_id}"

                    # Find the last checkpoint
                    CHECKPOINT=$(find "checkpoints/${arch}" -name "${model_id}*epoch_*" ! -name "*.log" -type f 2>/dev/null | sort | tail -1)

                    if [ -z "$CHECKPOINT" ]; then
                        echo "WARNING: No checkpoint found for ${arch}/${model_id}, skipping test"
                        continue
                    fi

                    python scripts/test.py \
                        --arch "${arch}" \
                        ${TEST_ARGS} \
                        --model "checkpoints/${arch}/${model_id}" \
                        --load "${CHECKPOINT}" \
                        --test  "data/${cond}/test/run${run}/test.${model_id}.src" \
                                "data/${cond}/test/run${run}/test.${model_id}.tgt" \
                        --train "data/${cond}/train/run${run}/train.${model_id}.src" \
                                "data/${cond}/train/run${run}/train.${model_id}.tgt" \
                        --dev   "data/${cond}/dev/run${run}/dev.${model_id}.src" \
                                "data/${cond}/dev/run${run}/dev.${model_id}.tgt"
                done
            done
        done
    done

    echo "Training complete"
fi


# Analysis pipeline (Section 4)


echo "Running analysis pipeline"

#Section 4.1.1: Sequence accuracy (overall) ---
echo "Section 4.1.1: Overall sequence accuracy"
cd "${ROOT}/scripts/accuracies"
python calc_overall_accuracy.py
python boxplot.py

#Section 4.1.1: L vs NL accuracy ---
echo "Section 4.1.1: L vs NL accuracy"
cd "${ROOT}/scripts/l_nl_accuracies"
python get_accuracy.py
python plot_l_nl_accuracy.py

#Section 4.1.2: Stem accuracy ---
echo "Section 4.1.2: Stem accuracy"
cd "${ROOT}/scripts/stem_accuracies"
python get_stem_accuracy.py
python plot_stem_accuracy.py

#Section 4.1.3: Paradigm shape analysis ---
echo "Section 4.1.3: Paradigm shape (per-tag stem accuracy)"
cd "${ROOT}/scripts/stem_accuracy_by_tag"
bash run_analysis.sh
python tag_cluster_paradigm.py
bash run_lemma_analysis.sh
python lemma_cluster_paradigm.py
python lemma_shape_analysis.py

#Section 4.2.1: Nonce verb stem accuracy ---
echo "Section 4.2.1: Nonce verb accuracy"
cd "${ROOT}/scripts/wug_test"
python calculate_wug_accuracies.py
python calculate_nevins_test_accuracies.py
python calculate_sep_char_wug_accuracies.py
python calculate_vanilla_wug_accuracies.py

cd "${ROOT}/scripts/wug_test/nevins"
python model.py
python model_sep_char.py
python model_vanilla.py
python model_stem_accuracy.py
python model_sep_char_stem_accuracy.py
python model_vanilla_stem_accuracy.py

# Human wug test data depends on external sibling repo
HUMAN_STEM_DATA="${ROOT}/../cognitive_modeling_aaacl/data/analysis/accuracies/stem_participants_accuracy.csv"
if [ -f "${HUMAN_STEM_DATA}" ]; then
    python human_stem_accuracy.py
else
    echo "WARNING: Human data not found at ${HUMAN_STEM_DATA}, skipping human_stem_accuracy.py"
fi

python plot_accuracies.py
python plot_stem_accuracy_boxplot.py


cd "${ROOT}"
echo "Analysis complete"
echo "Results are in data/analysis/"
echo "Plots are in data/analysis/plots/ and data/plots/"
