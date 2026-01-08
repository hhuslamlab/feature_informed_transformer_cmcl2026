#!/bin/bash

# Script to run stem accuracy by tag analysis

echo "=================================="
echo "Stem Accuracy by Target Tag Analysis"
echo "=================================="
echo ""

# Navigate to the script directory
cd "$(dirname "$0")"

echo "Step 1: Calculating stem accuracy by target tag..."
python3 calculate_stem_accuracy_by_tag.py

if [ $? -eq 0 ]; then
    echo ""
    echo "Step 2: Generating plots..."
    python3 plot_stem_accuracy_by_tag.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=================================="
        echo "Analysis complete!"
        echo "=================================="
        echo ""
        echo "Results saved to:"
        echo "  - Data: ../../data/accuracies/stem_accuracy_by_tag/"
        echo "  - Plots: ../../data/accuracies/plots/stem_accuracy_by_tag/"
    else
        echo ""
        echo "Error: Plotting failed."
        exit 1
    fi
else
    echo ""
    echo "Error: Calculation failed."
    exit 1
fi

