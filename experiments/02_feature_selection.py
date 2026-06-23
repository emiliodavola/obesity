import os
import pandas as pd
import numpy as np
import logging
from typing import Any, Tuple

# We'll use scikit-learn for metrics if skore is unavailable in this env
try:
    import skore
    HAS_SKORE = True
except ImportError:
    HAS_SKORE = False

from obesity.data import load_data, split_data
from obesity.features import get_preprocessor
from obesity.pipeline import create_pipeline
from obesity.utils import ensure_model_checkpoint, get_device
from sklearn.metrics import roc_auc_score, accuracy_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ObesityFeatureSelection")

# Configuration
CSV_PATH = "data/obesity_dataset.csv"
TARGET_COL = "obese"
REDUNDANT_FEATURES = ["n_careplans", "n_procedures", "n_medications", "total_med_cost"]
NOISE_FEATURES = ["ETHNICITY", "HEALTHCARE_COVERAGE", "HEALTHCARE_EXPENSES", "LAT", "LON", "RACE", 
                 "has_anemia", "has_asthma", "has_depression", "has_diabetes", "has_hypertension", 
                 "has_sinusitis", "is_dead", "is_female", "n_allergies", "n_ambulatory", 
                 "n_conditions", "n_emergency", "n_encounters", "n_immunizations", "n_inpatient", 
                 "n_outpatient", "n_unique_conditions", "n_unique_medications", "n_unique_procedures", 
                 "n_wellness", "total_claim_cost", "total_payer_coverage"]
FEATURES_TO_DROP = REDUNDANT_FEATURES + NOISE_FEATURES

def run_experiment() -> Tuple[float, float]:
    """Execute the feature selection experiment."""
    device = get_device()
    model_path = ensure_model_checkpoint("checkpoints/tabpfn-v3-classifier-v3_20260417_binary.ckpt", "Prior-Labs/tabpfn_3", "tabpfn-v3-classifier-v3_20260417_binary.ckpt")

    print(f"Loading raw data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    print(f"Dropping features: {FEATURES_TO_DROP}")
    df_pruned = df.drop(columns=FEATURES_TO_DROP)
    
    # Check dimensionality requirement (TC_01)
    assert all(f not in df_pruned.columns for f in FEATURES_TO_DROP), "Dimensionality check failed!"

    print("Splitting pruned data...")
    X_train, X_test, y_train, y_test = split_data(df_pruned, TARGET_COL)

    # Update preprocessor configuration (Phase 2)
    # The design says: "Modify obesity/features.py to accommodate the reduced feature set."
    # In the pipeline, the preprocessor is created via get_preprocessor(cat_cols).
    # For the experiment, we'll use the surviving categorical columns.
    surviving_cat_cols = [c for c in df.columns if c in ["RACE", "ETHNICITY"] or (c in df_pruned.columns and is_categorical_candidate(df_pruned, c))]
    # Actually, based on the spec, RACE and ETHNICITY are non-informative and should be dropped.
    # So surviving_cat_cols will be empty.
    surviving_cat_cols = [c for c in df_pruned.columns if c in ["RACE", "ETHNICITY"]] # This will be empty.
    
    # Let's be safer: just pass what's left from the original cat_cols that are in df_pruned
    surviving_cat_cols = [c for c in ["RACE", "ETHNICITY"] if c in df_pruned.columns]

    print(f"Using surviving categorical columns: {surviving_cat_cols}")
    preprocessor = get_preprocessor(surviving_cat_cols)
    clf_pipeline = create_pipeline(preprocessor, model_path, device)

    print("Fitting pipeline...")
    clf_pipeline.fit(X_train, y_train)

    print("Evaluating pipeline...")
    
    # Requirement: Metric Stability (Phase 3)
    # We'll compute ROC-AUC and Accuracy.
    preds = clf_pipeline.predict(X_test)
    roc_auc = roc_auc_score(y_test, preds)
    acc = accuracy_score(y_test, preds)

    print(f"ROC AUC: {roc_auc:.6f}")
    print(f"Accuracy: {acc:.6f}")
    
    return roc_auc, acc

def is_categorical_candidate(df: pd.DataFrame, col: str) -> bool:
    # Simple heuristic: if it's in the original cat_cols and not dropped
    return col in ["RACE", "ETHNICITY"]

if __name__ == "__main__":
    roc, acc = run_experiment()
    print(f"FINAL_RESULT_ROC_AUC:{roc}")
    print(f"FINAL_RESULT_ACC:{acc}")
