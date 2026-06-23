# Code Context

## Files Retrieved
1. `experiments/01_baseline.py` (Lines 1-55) - Baseline configuration and execution logic.
2. `experiments/02_feature_selection.py` (Lines 1-85) - The failed pruning experiment script and feature list.
3. `openspec/verify-report/02_feature_selection.md` (Lines 1-55) - Detailed report of the Logloss collapse.
4. `data/obesity_dataset.csv` (Lines 1-10) - Sample data to verify feature existence.
5. `scratch/scout_perturbation_test.py` (Lines 1-100) - Result of my targeted perturbation tests.

## Key Code
The `NOISE_FEATURES` list in `experiments/02_feature_selection.py` includes:
`"ETHNICITY", "HEALTHCARE_COVERAGE", "HEALTHCARE_EXPENSES", "LAT", "LON", "RACE", "has_anemia", "has_asthma", "has_depression", "has_diabetes", "has_hypertension", "has_sinusitis", "is_dead", "is_female", "n_allergies", "n_ambulatory", "n_conditions", "n_emergency", "n_encounters", "n_immunizations", "n_inpatient", "n_outpatient", "n_unique_conditions", "n_unique_medications", "n_unique_procedures", "n_wellness", "total_claim_cost", "total_payer_coverage"`

## Architecture
The experiment `02_feature_selection` was designed to remove both "redundant" (highly correlated) and "noisy" (low permutation importance) features. 
My `scout` investigation performed a controlled isolation:
*   **Isolation Test**: Removing only `is_female` and `RACE` resulted in **negligible** change in Logloss (0.4844 $\to$ 0.4853).
*   **Root Cause Hypothesis**: The Logloss collapse observed in the full `02_feature_selection` run (0.5978 $\to$ 9.2205) is not caused by the "noisy" demographic features, but rather by the removal of the **medical complexity/interaction features** (e.g., `n_conditions`, `n_emergency`, `n_inpatient`, `total_claim_cost`). These features appear to provide the critical density/context required for the TabPFN to calibrate its probabilistic output.

## Start Here
The next step is to perform a more granular investigation into the **medical complexity features** (`n_conditions`, `n_emergency`, etc.) vs. Logloss. The agent should design an experiment (`sdd-design`) to test the impact of removing these specific features.