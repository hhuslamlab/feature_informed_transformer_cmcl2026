#!/bin/bash

# Base paths - set BASE_DIR environment variable or modify this default
BASE_DIR="${BASE_DIR:-$(pwd)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${BASE_DIR}/example/transformer/checkpoints/transformer/dualsource_clean}"
DATA_DIR="${DATA_DIR:-${BASE_DIR}/example/transformer}"
TEST_SRC="${TEST_SRC:-${BASE_DIR}/full_paradigm.src}"
TEST_TGT="${TEST_TGT:-${BASE_DIR}/full_paradigm.tgt}"
OUTPUT_DIR="${OUTPUT_DIR:-predictions_nevins/dualsource}"

# Common args
COMMON_ARGS="--eval_test --arch dualsource --dataset taginbrackets --bs 16 --decode beam --decode_beam_size 5"

# Add src to PYTHONPATH for imports to work
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Loop through all conditions
for condition in 10L_90NL 50L_50NL 90L_10NL; do
    echo "============================================"
    echo "Processing condition: ${condition}"
    echo "============================================"

    for run in 1 2 3; do
        for seed in 1 2 3 4; do
            MODEL_NAME="${condition}_${run}_${seed}"

            # Find checkpoint with epoch_103 or epoch_104 (exclude .log files)
            CHECKPOINT=$(find "${CHECKPOINT_DIR}" \( -name "*${MODEL_NAME}*epoch_103*" -o -name "*${MODEL_NAME}*epoch_104*" \) ! -name "*.log" -type f 2>/dev/null | head -1)

            if [ -z "$CHECKPOINT" ]; then
                echo "WARNING: No checkpoint found for ${MODEL_NAME}, skipping..."
                continue
            fi

            # Check if prediction file already exists
            PRED_FILE="${OUTPUT_DIR}/${MODEL_NAME}.decode.test.tsv"
            if [ -f "$PRED_FILE" ]; then
                echo "Skipping ${MODEL_NAME}: prediction file already exists"
                continue
            fi

            echo "=========================================="
            echo "Running test for ${MODEL_NAME}"
            echo "Checkpoint: ${CHECKPOINT}"
            echo "=========================================="

            python src/test.py \
                ${COMMON_ARGS} \
                --model "${OUTPUT_DIR}/${MODEL_NAME}" \
                --load "${CHECKPOINT}" \
                --test "${TEST_SRC}" "${TEST_TGT}" \
                --train "${DATA_DIR}/${condition}/train/run${run}/train.${MODEL_NAME}.src" \
                        "${DATA_DIR}/${condition}/train/run${run}/train.${MODEL_NAME}.tgt" \
                --dev "${DATA_DIR}/${condition}/dev/run${run}/dev.${MODEL_NAME}.src" \
                      "${DATA_DIR}/${condition}/dev/run${run}/dev.${MODEL_NAME}.tgt"

            echo ""
        done
    done
done

echo "All tests completed!"
