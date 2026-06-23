# Design: 03_medical_complexity

## Architecture
This experiment is a "Differential Isolation" study. We deviate from `01_baseline` by applying a specific filter to the feature set, purposely leaving "noise" features intact to prove they are not the source of the failure.

## Implementation Strategy
1.  **Load Data**: Read `data/obesity_dataset.csv`.
2.  **Feature Filtering**:
    - Identify `COMPLEXITY_FEATURES` (the list provided in the task).
    - Create `df_complexity_reduced` by dropping ONLY `COMPLEXITY_FEATURES`.
3.  **Preprocessing**: 
    - Use the preprocessor configuration from `01_baseline` (preserving categorical features like `RACE` and `ETHNICITY`).
4.  **Modeling**: 
    - Use `TabPFN` (v3) with the specified checkpoint.
    - Standard pipeline: `fit(X_train, y_train)` $\to$ `predict(X_test)`.
5.  **Evaluation**:
    - Compute ROC-AUC and Logloss on `X_test`.
    - Compare against `01_baseline` (ROC-AUC: 0.7604, Logloss: 0.5978).

## Risk Assessment
- **Feature Mismatch**: Ensure names in the `COMPLEXITY_FEATURES` list match exactly with the CSV header.
- **Preprocessing Drift**: Ensure the preprocessor still receives all relevant categorical columns that weren't dropped.
