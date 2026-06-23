# 01_baseline

## Goal
Establish a performance baseline using a simple TabPFN pipeline for predicting `obese`.

## Proposed Design
- **Data Source**: `data/obesity_dataset.csv`
- **Target**: `obese`
- **Features**: `RACE`, `ETHNICITY`, `has_sinusitis`, `has_anemia`, `has_diabetes`, `has_hypertension`, `has_asthma`, `has_depression`, `has_allergies`, `has_careplans`, `has_immunizations`
- **Model**: `TabPFNClassifier`
- **Preprocessing**: `OrdinalEncoder` on categorical features.
- **Split**: `train_test_split` (70/30)

## Acceptance Criteria
- `skore` report `01_baseline` is generated.
- ROC AUC and Accuracy are calculated.
- Smoke test passes (row count match).
