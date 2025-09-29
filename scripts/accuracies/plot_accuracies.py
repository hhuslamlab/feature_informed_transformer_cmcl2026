#!/usr/bin/env python3

import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_accuracies(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    accuracies = defaultdict(list)
    sections = re.split(r'\n(10L_90NL|50L_50NL|90L_10NL)_\d+_\d+', content)
    
    for i in range(1, len(sections), 2):
        if i + 1 < len(sections):
            group_name = sections[i]
            section_content = sections[i + 1]
            
            accuracy_match = re.search(r'Estimated accuracy: ([\d.]+)', section_content)
            if accuracy_match:
                accuracy = float(accuracy_match.group(1))
                accuracies[group_name].append(accuracy * 100)
    
    return accuracies

def create_box_plot(accuracies):
    groups = []
    data = []
    
    for group_name in ['10L_90NL', '50L_50NL', '90L_10NL']:
        if group_name in accuracies:
            groups.append(group_name)
            data.append(accuracies[group_name])
    
    plt.figure(figsize=(10, 6))
    box_plot = plt.boxplot(data, labels=groups, patch_artist=True)
    
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.xlabel('Conditions', fontsize=12)
    plt.ylabel('Estimated Accuracy (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    for i, (group, values) in enumerate(zip(groups, data)):
        mean_val = np.mean(values)
        std_val = np.std(values)
        plt.text(i + 1, max(values) + 1, f'Mean: {mean_val:.1f}%\nStd: {std_val:.1f}%', 
                ha='center', va='bottom', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('../../analysis/accuracies/accuracy_boxplot.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Summary Statistics:")
    print("=" * 50)
    for group_name in ['10L_90NL', '50L_50NL', '90L_10NL']:
        if group_name in accuracies:
            values = accuracies[group_name]
            print(f"{group_name}:")
            print(f"  Count: {len(values)}")
            print(f"  Mean: {np.mean(values):.2f}%")
            print(f"  Std: {np.std(values):.2f}%")
            print(f"  Min: {np.min(values):.2f}%")
            print(f"  Max: {np.max(values):.2f}%")
            print(f"  Values: {[f'{v:.2f}%' for v in values]}")
            print()

if __name__ == "__main__":
    file_path = '../../analysis/accuracies/all_accuracies.txt'
    accuracies = parse_accuracies(file_path)
    create_box_plot(accuracies)
