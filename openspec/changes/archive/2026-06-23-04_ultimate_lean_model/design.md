# Design: 04_ultimate_lean_model

## Purpose
Achieve maximum parsimony by combining the successful findings of `03_medical_complexity` (removing complexity noise) and `01_baseline` (retaining demographic signals) while eliminating highly correlated redundant features. The goal is the most robust and efficient model.

## Architecture
A "Minimalist Calibration" architecture. We reduce the input dimensionality to only the features that provide non-redundant, high-signal information for both predictive accuracy (AUC) and probabilistic calibration (Logloss).

## Implementation Strategy

### 1. Feature Set Definition
The feature set will be constructed by applying a double-filter to the original dataset:

*   **DROP (Noise/Complexity)**: 
    *   `n_conditions`, `n_emergency`, `n_inpatient`, `n_outpatient`, `n_unique_conditions`, `n_unique_procedures`, `n_wellness`, `total_claim_cost`, `total_payer_coverage`, `n_immunizations`, `n_ambulatory`, `n_allergies`
*   **DROP (Redundancy)**:
    *   `n_careplans` (redundant with `n_immunizations`)
    *   `n_procedures` (redundant with `n_unique_procedures`)
    *   `n_medications` (redundant with `n_unique_medications`)
    *   `total_med_cost` (redundant with `total_claim_cost`)
*   **RETAIN (Signal/Demographic)**:
    *   `is_female`, `RACE`, `ETHNICITY`, `has_anemia`, `has_asthma`, `has_depression`, `has_diabetes`, `has_hypertension`, `has_sinusitis`, `is_dead`, `n_emergency`, `n_encounters`, `n_immunizations`, `n_inpatient`, `n_outpatient`, `n_unique_conditions`, `n_unique_procedures`, `n_wellness`, `total_claim_cost`, `total_payer_coverage`, `n_ambulatory`, `n_allergies`... (Wait, let's re-evaluate based on `03_medical_complexity` success).

**Correction based on `03_medical_complexity` finding:** 
The experiment `03_medical_complexity` proved that `n_conditions`, `n_emergency`, etc., were **NOISE**. Therefore, they MUST be dropped. 

**Final Selection Logic:**
*   **Include ONLY**: The features that were NOT in the `COMPLEXITY_FEATURES` list of `03_medical_complexity` AND NOT in the `REDUNDANT_FEATURES` list.
*   **Specifically**: `is_female`, `RACE`, `ETHNICITY`, `has_anemia`, `has_asthma`, `has_depression`, `has_diabetes`, `has_hypertension`, `has_sinusitis`, `is_dead`, `n_encounters`, `n_unique_medications`, `total_payer_coverage` (Wait, `total_payer_coverage` was in complexity? No, `total_claim_cost` and `total_payer_coverage` were in the complexity list in the prompt. Let me re-read).

**Re-reading `03_medical_complexity` task for `COMPLEXITY_FEATURES`:**
`n_conditions`, `n_emergency`, `n_inpatient`, `n_outpatient`, `n_unique_conditions`, `n_unique_procedures`, `n_wellness`, `total_claim_cost`, `total_payer_coverage`, `n_immunizations`, `n_ambulatory`, `n_allergies`.

**Re-reading `REDUNDANT_FEATURES`:**
`n_careplans` (vs `n_immunizations`), `n_procedures` (vs `n_unique_procedures`), `n_medications` (vs `n_unique_medications`), `total_med_cost` (vs `total_claim_cost`).

**Final List for `04_ultimate_lean_model` (The "Gold Standard"):**
1.  **Demographics**: `is_female`, `RACE`, `ETHNICITY`
2.  **Clinical/Status (Non-Complexity)**: `has_anemia`, `has_asthma`, `has_depression`, `has_diabetes`, `has_hypertension`, `has_sinusitis`, `is_dead`
3.  **Structural/Administrative (Non-Complexity/Non-Redundant)**: `n_encounters`, `n_unique_medications`, `n_unique_procedures` (Wait, `n_unique_procedures` was in complexity list? Yes. `n_unique_medications` was NOT in complexity list).

**Refined Feature List for implementation:**
`is_female`, `RACE`, `ETHNICITY`, `has_anemia`, `has_asthma`, `has_depression`, `has_diabetes`, `has_hypertension`, `has_sinusitis`, `is_dead`, `n_encounters`, `n_unique_medications`.

### 2. Execution Pipeline
1.  **Load**: `df = load_data("data/obesity_dataset.csv")`
2.  **Filter**: `df_lean = df.drop(columns=DROPPED_LIST)`
3.  **Model**: `clf_pipeline.fit(X_train_lean, y_train)`
4.  **Evaluate**: `skore.evaluate(clf_pipeline, X_test_lean, y_test)`

## Acceptance Criteria

| Criterion | Requirement | Threshold |
| :--- | :--- | :--- |
| **C-1: Performance** | ROC-AUC vs `03_medical_complexity` | $\ge$ 0.91 |
| **C-2: Calibration** | Logloss vs `03_medical_complexity` | $\le$ 0.40 |
| **C-3: Parsimony** | Feature Count | $< 15$ |

## Manual Notes
- The `03_medical_complexity` experiment was a breakthrough. It revealed that medical complexity features were not providing calibration, but actually interfering with the TabPFN's probabilistic density estimation.
- The `04_ultimate_lean_model` focuses on the intersection of "signal-only" features.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Design defined based on empirical findings from 03_medical_complexity and 01_baseline."
    }
  ],
  "changedFiles": [
    "openspec/designs/04_ultimate_lean_model.md"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [],
  "validationOutput": [],
  "residualRisks": [
    "Extreme sensitivity to the specific subset of features chosen; any additional feature might re-introduce calibration noise."
  ],
  "noStagedFiles": true,
  "notes": "The Lean Model design is finalized. Ready for implementation phase."
}
```