# Example Output

This document shows example output from the stem accuracy by tag analysis.

## Test Run Output

Running `python3 test_single_model.py` for model `10L_90NL_1_1` (vanilla predictions):

```
Analyzing 10L_90NL_1_1 (vanilla)...
============================================================

Data loaded:
  Predictions: 43560
  Test data: 43560
  Target tags: 43560

Stem Accuracy by Target Tag:
------------------------------------------------------------
Tag                            Correct    Total      Accuracy
------------------------------------------------------------
V;SBJV;PRS;1;SG                2714       3630       74.77%
V;SBJV;PRS;2;SG                2709       3630       74.63%
V;SBJV;PRS;3;SG                2705       3630       74.52%
V;SBJV;PRS;3;PL                2685       3630       73.97%
V;IND;PRS;1;SG                 2671       3630       73.58%
V;IND;PRS;3;SG                 2664       3630       73.39%
V;IND;PRS;3;PL                 2651       3630       73.03%
V;IND;PRS;2;SG                 2646       3630       72.89%
V;IND;PRS;2;PL                 2586       3630       71.24%
V;IND;PRS;1;PL                 2565       3630       70.66%
V;SBJV;PRS;1;PL                2514       3630       69.26%
V;SBJV;PRS;2;PL                2448       3630       67.44%
------------------------------------------------------------
OVERALL                        31558      43560      72.45%
============================================================
```

## Key Insights from Example

1. **Tag Coverage**: All 12 morphological tag combinations are present (2 moods × 2 numbers × 3 persons)
   - V;IND;PRS = Verb, Indicative, Present tense
   - V;SBJV;PRS = Verb, Subjunctive, Present tense
   - Numbers: SG (singular), PL (plural)
   - Persons: 1, 2, 3

2. **Accuracy Patterns**:
   - Subjunctive singular forms (1SG, 2SG, 3SG) show highest accuracy (~74-75%)
   - Plural forms generally show lower accuracy than singular
   - 2nd person plural (2;PL) shows lowest accuracy (67.44%)

3. **Balanced Dataset**: Each tag has exactly 3630 instances, showing the dataset is balanced across morphological categories

## CSV Output Format

### Per-Model File (`stem_acc_by_tag_{model}.csv`)

```csv
target_tag,correct,total,accuracy
V;SBJV;PRS;1;SG,2714,3630,74.77
V;SBJV;PRS;2;SG,2709,3630,74.63
V;SBJV;PRS;3;SG,2705,3630,74.52
...
```

### Combined File (`stem_acc_by_tag_all_models.csv`)

```csv
model,condition,run,target_tag,correct,total,accuracy
10L_90NL_1_1,10L_90NL,1,V;SBJV;PRS;1;SG,2714,3630,74.77
10L_90NL_1_1,10L_90NL,1,V;SBJV;PRS;2;SG,2709,3630,74.63
...
```

### Summary File (`stem_acc_by_tag_summary.csv`)

```csv
target_tag,correct,total,accuracy,overall_accuracy
V;SBJV;PRS;1;SG,97800,130680,73.52,74.84
V;SBJV;PRS;2;SG,97500,130680,73.29,74.61
...
```

## Visualization Output

The plotting script generates:

1. **Overall Accuracy Plots** - Horizontal bar charts showing accuracy by tag
2. **Condition-Specific Plots** - Separate plots for 10L_90NL, 50L_50NL, 90L_10NL
3. **Heatmaps** - Showing accuracy patterns across tags and conditions
4. **Comparison Plots** - Comparing different model types (vanilla, feature_invariant, etc.)

All plots are saved in both PNG (for viewing) and PDF (for publication) formats.

---

## Lemma-Based Analysis Output

Running `python3 test_single_model_lemma.py` for model `10L_90NL_1_1` (vanilla predictions):

```
Analyzing 10L_90NL_1_1 (vanilla) - Grouped by Lemma...
============================================================
Loading lemma dictionary...
Loaded 299 lemmas
Built mapping for 3265 unique forms

Data loaded:
  Predictions: 43560
  Test data: 43560
  Unmapped forms: 38940

Stem Accuracy by Lemma:
----------------------------------------------------------------------
Lemma                                    Correct    Total      Accuracy
----------------------------------------------------------------------

  TOP 20 LEMMAS:
i n f ɾ i n ç ˈi ɾ                       653        660        98.94%
k o n o s ˈe ɾ                           555        660        84.09%
k o m p a d e s ˈe ɾ                     527        660        79.85%
m u l ç ˈe ɾ                             504        660        76.36%
s u p e ɾ b e n ˈi ɾ                     444        660        67.27%
o b t e n ˈe ɾ                           362        660        54.85%
m e ɾ e s ˈe ɾ                           328        660        49.70%

  BOTTOM 20 LEMMAS:
i n f ɾ i n ç ˈi ɾ                       653        660        98.94%
k o n o s ˈe ɾ                           555        660        84.09%
k o m p a d e s ˈe ɾ                     527        660        79.85%
m u l ç ˈe ɾ                             504        660        76.36%
s u p e ɾ b e n ˈi ɾ                     444        660        67.27%
o b t e n ˈe ɾ                           362        660        54.85%
m e ɾ e s ˈe ɾ                           328        660        49.70%
----------------------------------------------------------------------
OVERALL                                  3373       4620       73.01%
Total unique lemmas: 7
======================================================================
```

### Key Insights from Lemma-Based Analysis

1. **Lemma Coverage**: Only 7 lemmas from the dictionary were found in the test set (about 10% of forms)
   - This is expected since `ipa_clean_lshaped_dict.json` contains only L-shaped lemmas
   - 38,940 out of 43,560 forms couldn't be mapped (likely NL-shaped)

2. **Lemma-Specific Patterns**:
   - Some lemmas have very high stem accuracy (98.94% for "i n f ɾ i n ç ˈi ɾ")
   - Others show lower accuracy (49.70% for "m e ɾ e s ˈe ɾ")
   - Each lemma has exactly 660 instances (12 tags × 55 test forms)

3. **Use Cases**:
   - Identify specific lemmas that are difficult to predict correctly
   - Compare model performance on individual lemmas
   - Focus analysis on L-shaped verbs specifically

### CSV Output Format for Lemma Analysis

**Per-Model File (`stem_acc_by_lemma_{model}.csv`)**
```csv
lemma,correct,total,accuracy
i n f ɾ i n ç ˈi ɾ,653,660,98.94
k o n o s ˈe ɾ,555,660,84.09
k o m p a d e s ˈe ɾ,527,660,79.85
...
```

**Combined File (`stem_acc_by_lemma_all_models.csv`)**
```csv
model,condition,run,lemma,correct,total,accuracy
10L_90NL_1_1,10L_90NL,1,i n f ɾ i n ç ˈi ɾ,653,660,98.94
10L_90NL_1_1,10L_90NL,1,k o n o s ˈe ɾ,555,660,84.09
...
```

**Summary File (`stem_acc_by_lemma_summary.csv`)**
```csv
lemma,correct,total,accuracy,overall_accuracy
i n f ɾ i n ç ˈi ɾ,23508,23760,98.21,98.94
k o n o s ˈe ɾ,19980,23760,82.74,84.09
...
```

---

## Comparison: Tag-Based vs Lemma-Based Analysis

| Aspect | Tag-Based | Lemma-Based |
|--------|-----------|-------------|
| Grouping | By morphological tag (e.g., V;SBJV;PRS;2;PL) | By lemma (base form) |
| Coverage | All forms in test set | Only forms in `ipa_clean_lshaped_dict.json` |
| Use Case | Analyze accuracy by grammatical category | Analyze accuracy by specific verbs |
| Number of Groups | 12 (fixed morphological tags) | Varies (depends on test set) |
| Visualizations | Yes (plots available) | No (CSV only) |

