# Feature Selection Specification

## Purpose

The purpose of this specification is to define the requirements for the `02_feature_selection` experiment. The goal is to stabilize the model's generalization (ROC-AUC and Logloss) by eliminating redundant (highly correlated) and non-informative (low permutation importance) features identified during the `01_baseline` audit.

## Requirements

### Requirement: Feature Pruning (Redundancy and Noise)

The system MUST remove the following features from the input feature set to minimize dimensionality and noise:

**Redundant Features (High Correlation > 0.9):**
- `n_careplans` (Correlated with `n_immunizations`)
- `n_procedures` (Correlated with `n_unique_procedures`)
- `n_medications` (Correlated with `n_unique_medications`)
- `total_med_cost` (Correlated with `total_claim_cost`)

**Non-Informative Features (Zero Permutation Importance):**
- `ETHNICITY`
- `HEALTHCARE_COVERAGE`
- `HEALTHCARE_EXPENSES`
- `LAT`
- `LON`
- `RACE`
- `has_anemia`
- `has_asthma`
- `has_depression`
- `has_diabetes`
- `has_hypertension`
- `has_sinusitis`
- `is_dead`
- `is_female`
- `n_allergies`
- `n_ambulatory`
- `n_conditions`
- `n_emergency`
- `n_encounters`
- `n_immunizations`
- `n_inpatient`
- `n_outpatient`
- `n_unique_conditions`
- `n_unique_procedures`
- `n_wellness`
- `total_claim_cost`
- `total_payer_coverage`

#### Scenario: Validated Dimensionality

- GIVEN a dataset containing the full set of features from `01_baseline`.
- WHEN the pruning process is applied according to this requirement.
- THEN the resulting feature set MUST NOT contain any of the features listed above.

### Requirement: Metric Stability

The system MUST ensure that the reduction in dimensionality does not compromise the predictive stability of the primary metrics.

#### Scenario: Metric Stability

- GIVEN the pruned feature set.
- WHEN the model is evaluated on the test set.
- THEN the **ROC-AUC** and **Logloss** MUST exhibit minimal variance and a reduced gap (train vs test) compared to the results recorded in `01_baseline`.

## Risks

- **Information Loss:** The removal of highly correlated features assumes the information is redundant; however, if the correlation is a proxy for a latent non-linear relationship, performance may decrease.
- **Feature Drift:** The removal of "useless" features may inadvertently remove variables that only become significant when combined with others in a different model architecture.
