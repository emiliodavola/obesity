# 04_ultimate_lean_model Specification

## Purpose

Establish the final, high-performance pipeline for obesity classification by using a parsimonious feature set that maximizes both predictive discrimination (ROC-AUC) and probabilistic calibration (Logloss). This model addresses the finding that high-density medical features introduced calibration noise.

## Requirements

### Requirement: Feature Parsimony

The system MUST use a highly selective feature set, excluding both redundant and high-noise complexity features, to ensure high signal-to-noise ratio for the TabPFN.

#### Scenario: Feature Set Integrity
- GIVEN the original feature set from `01_baseline`.
- WHEN the "Complexity" and "Redundant" features are removed.
- THEN the feature set MUST consist only of:
    - `is_female`
    - `RACE`
    - `ETHNICITY`
    - `has_anemia`
    - `has_asthma`
    - `has_depression`
    - `has_diabetes`
    - `has_hypertension`
    - `has_sinusitis`
    - `is_dead`
    - `n_encounters`
    - `n_unique_medications`

### Requirement: Performance Excellence

The model MUST achieve superior metrics compared to the baseline, proving that parsimony enhances both discrimination and calibration.

#### Scenario: Peak Performance
- GIVEN the `01_baseline` metrics (AUC: 0.7604, Logloss: 0.5978).
- WHEN the `04_ultimate_lean_model` is executed.
- THEN the ROC-AUC MUST be $\ge$ 0.91.
- AND the Logloss MUST be $\le$ 0.40.

## Implementation Details

### Data Selection
- **Dataset**: `data/obesity_dataset.csv`
- **Target**: `obese`
- **Excluded Features**: 
    - `n_conditions`, `n_emergency`, `n_inpatient`, `n_outpatient`, `n_unique_conditions`, `n_unique_procedures`, `n_wellness`, `total_claim_cost`, `total_payer_coverage`, `n_immunizations`, `n_ambulatory`, `n_allergies`.
    - `n_careplans`, `n_procedures`, `n_medications`, `total_med_cost` (Redundancy).

### Workflow
1. **Data Load**: Load `data/obesity_dataset.csv`.
2. **Feature Pruning**: Explicitly drop the excluded list.
3. **Model Pipeline**: Build a pipeline using `TabPFN` on the pruned dataset.
4. **Evaluation**: Run `skore.evaluate` to validate performance against the established thresholds.

## Acceptance Criteria

| Criterion | Requirement | Threshold |
| :--- | :--- | :--- |
| **C-1: AUC Integrity** | ROC-AUC $\ge$ 0.91 | PASS if $\ge$ 0.91 |
| **C-2: Calibration Stability** | Logloss $\le$ 0.40 | PASS if $\le$ 0.40 |
| **C-3: Parsimony Compliance** | Feature count $< 15$ | PASS if count $< 15$ |
