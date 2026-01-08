#!/bin/bash

# Script to run stem accuracy by lemma analysis

echo "=================================="
echo "Stem Accuracy by Lemma Analysis"
echo "=================================="
echo ""

# Navigate to the script directory
cd "$(dirname "$0")"

echo "Calculating stem accuracy by lemma..."
python3 calculate_stem_accuracy_by_lemma.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "Analysis complete!"
    echo "=================================="
    echo ""
    echo "Results saved to:"
    echo "  - Data: ../../data/accuracies/stem_accuracy_by_lemma/"
else
    echo ""
    echo "Error: Calculation failed."
    exit 1
fi







