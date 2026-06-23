# Design: 02_feature_selection

## Implementation Sequence

### Phase 1: Data Preparation and Filtering
1. **Load Raw Data**: Read `data/obesity_dataset.csv`.
2. **Feature Selection Logic**:
    *   Define a list `REDUNDANT_FEATURES` (based on correlation > 0.9).
    *   Define a list `NOISE_FEATURES` (based on zero permutation importance).
    *   Combine into `FEATURES_TO_DROP`.
3. **Transformation**: Apply `df.drop(columns=FEATURES_TO_DROP)` to create `df_pruned`.

### Phase 2: Pipeline Construction
1. **Update Preprocessor**: Modify `obesity/features.py` to accommodate the reduced feature set.
    *   *Note*: The `get_preprocessor` must only be configured with the surviving categorical variables.
2. **Pipeline Assembly**: Re-instantiate `create_pipeline` using the pruned feature set and `TabPFNClassifier`.
3. **Model Training**: Run `clf_pipeline.fit(X_train_pruned, y_train)`.

### Phase 3: Evaluation and Validation
1. **Metric Extraction**: Execute `skore.evaluate(clf_pipeline, X_test_pruned, y_test)` to obtain `EstimatorReport`.
2. **Stability Check (Primary)**:
    *   Compare `ROC-AUC` and `Logloss` of the baseline vs. the pruned model.
    *   Measure the delta between Training and Test metrics for both.
3. **Integrity Check (Secondary)**:
    *   Verify that none of the dropped features are present in the final `X_test_pruned` matrix.

## Code Structure (Planned implementation for `experiments/02_feature_selection.py`)

```python
# Planned implementation structure
import os
import skore
from obesity.data import load_data, split_data
from obesity.features import get_preprocessor
from obesity.pipeline import create_pipeline
from obesity.utils import ensure_model_checkpoint, get_device

# Configuration
CSV_PATH = "data/obesity_dataset.csv"
TARGET_COL = "obese"
REDUNDANT_FEATURES = ["n_careplans", "n_procedures", "n_medications", "total_med_cost"]
NOISE_FEATURES = ["ETHNICITY", "HEALTHCARE_COVERAGE", "HEALTHCARE_EXPENSES", "LAT", "LON", "RACE", 
                 "has_anemia", "has_asthma", "has_depression", "has_diabetes", "has_hypertension", 
                 "has_sinusitis", "is_dead", "is_female", "n_allergies", "n_ambulatory", 
                 "n_conditions", "n_emergency", "n_encounters", "n_immunizations", "n_inpatient", 
                 "n_outpatient", "n_unique_conditions", "n_unique_procedures", "n_wellness", 
                 "total_claim_cost", "total_payer_coverage"]
FEATURES_TO_DROP = REDUNDANT_FEATURES + NOISE_FEATURES

# Execution
df = load_data(CSV_PATH)
df_pruned = df.drop(columns=FEATURES_TO_DROP)
X_train, X_test, y_train, y_test = split_data(df_pruned, TARGET_COL)

# ... (Rest of pipeline logic)
```

## Testing Structure

| Test Case | Target | Expected Result |
| :--- | :--- | :--- |
| **TC_01_Dimensionality** | Feature List | No features from `FEATURES_TO_DROP` present in `X_test_pruned`. |
| **TC_02_Stability** | ROC-AUC | $\text{Gap}_{test-train} \le \text{Baseline Gap}$ |
| **TC_03_Stability** | Logloss | $\text{Gap}_{test-train} \le \text{Baseline Gap}$ |
