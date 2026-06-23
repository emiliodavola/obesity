# Spec: 03_medical_complexity

## Objective
Isolate the impact of removing medical complexity features on model calibration (Logloss), distinguishing between demographic noise and functional complexity.

## Inputs
- **Dataset**: `data/obesity_dataset.csv`
- **Target**: `obese`
- **Base Model**: `checkpoints/tabpfn-v3-classifier-v3_20260417_binary.ckpt`

## Experiment Configuration
- **Baseline Features (All)**: All columns from `data/obesity_dataset.csv`.
- **Removed Features (Targeted)**:
    - `n_conditions`
    - `n_emergency`
    - `n_inpatient`
    - `n_outpatient`
    - `n_unique_conditions`
    - `n_unique_procedures`
    - `n_wellness`
    - `total_claim_cost`
    - `total_payer_coverage`
    - `n_immunizations`
    - `n_ambulatory`
    - `n_allergies`
- **Retained Features**: All other features, including demographic/noisy features (`is_female`, `RACE`, `ETHNICITY`, etc.).

## Metrics (Strict)
- **ROC-AUC**: To check predictive power.
- **Logloss**: Primary metric to detect calibration collapse.

## Success Criteria
- **Result (Observed)**: 
    - Logloss $\Delta$: -0.2122 (Improvement: 0.5978 $\to$ 0.3856)
    - ROC-AUC $\Delta$: +0.1553 (Improvement: 0.7604 $\to$ 0.9157)
- **Conclusion**: Hypothesis refuted. Medical complexity features acted as information noise, not as calibration stabilizers.
