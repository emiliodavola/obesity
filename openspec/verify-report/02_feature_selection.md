# Verification Report: 02_feature_selection

## Status
**FAIL (CRITICAL)**

## Executive Summary
The `02_feature_selection` experiment failed to meet the stability requirements. While the Accuracy showed a slight improvement, the **Logloss catastrophically increased** (from 0.5978 in baseline to 9.2205 in the pruned model), indicating a total loss of probabilistic calibration. The **ROC-AUC** also showed a measurable decrease (from 0.7604 to 0.7479). This suggests that the "useless" and "redundant" features identified were actually essential for the model's probability estimation, even if their direct permutation importance appeared low.

## Validation Results

### 1. Feature Integrity (TC_01)
*   **Status**: `passed`
*   **Evidence**: All features in `FEATURES_TO_DROP` were successfully removed from `X_train` and `X_test`.

### 2. Stability (ROC-AUC)
*   **Status**: `FAILED`
*   **Baseline (01_baseline)**: 0.760417
*   **Pruned (02_feature_selection)**: 0.747866
*   **Delta**: -0.012551 (Decrease)
*   **Observation**: The decrease exceeds the stability threshold for meaningful predictive performance.

### 3. Stability (Logloss)
*   **Status**: `FAILED (CRITICAL)`
*   **Baseline (01_baseline)**: 0.597797
*   **Pruned (02_feature_selection)**: 9.220469
*   **Delta**: +8.622672 (Catastrophic Increase)
*   **Observation**: The Logloss collapse indicates the model is no longer providing reliable probability estimates, rendering the model useless for probabilistic tasks.

## Conclusion
The pruning strategy is fundamentally flawed for this specific dataset/model combination. The features identified as "useless" by permutation importance were likely acting as critical calibrators or providing non-linear signal that `skrub` failed to capture in the baseline audit.

**Recommendation**: Revert to baseline and investigate the interaction between the removed features and the TabPFN's calibration logic. Do NOT proceed with further pruning.
