import os
import pandas as pd
import skore
from obesity.data import load_data, split_data
from obesity.features import get_preprocessor
from obesity.pipeline import create_pipeline
from obesity.utils import ensure_model_checkpoint, get_device

# Configuration
CSV_PATH = "data/obesity_dataset.csv"
TARGET_COL = "obese"
MODEL_CKPT = "checkpoints/tabpfn-v3-classifier-v3_20260417_binary.ckpt"
HF_REPO = ""
HF_FILENAME = ""

# Complexity Features (to DROP)
COMPLEXITY_FEATURES = [
    "n_conditions", "n_emergency", "n_inpatient", "n_outpatient", 
    "n_unique_conditions", "n_unique_procedures", "n_wellness", 
    "total_claim_cost", "total_payer_coverage", "n_immunizations", 
    "n_ambulatory", "n_allergies"
]

# Implementation Logic
def run_experiment():
    device = get_device()
    model_path = ensure_model_checkpoint(MODEL_CKPT, HF_REPO, HF_FILENAME)
    
    # 1. Load and Filter
    df = load_data(CSV_PATH)
    df_filtered = df.drop(columns=COMPLEXITY_FEATURES)
    
    # 2. Split
    X_train, X_test, y_train, y_test = split_data(df_filtered, TARGET_COL)
    
    # 3. Pipeline
    cat_cols = ["is_female", "RACE", "ETHNICITY", "has_anemia", "has_asthma", "has_depression", "has_diabetes", "has_hypertension", "has_sinusitis", "is_dead"]
    preprocessor = get_preprocessor(cat_cols)
    clf_pipeline = create_pipeline(preprocessor, model_path, device)
    clf_pipeline.fit(X_train, y_train)
    
    # 4. Evaluate
    report = skore.evaluate(clf_pipeline, X_test, y_test)
    return report, X_test.columns.tolist()

if __name__ == "__main__":
    report, columns = run_experiment()
    print(f"Final Feature Set: {columns}")
    print("\n--- Metrics Summary ---")
    print(report.metrics.summarize().frame())
    print("\n--- Checks Summary ---")
    print(report.checks.summarize().frame())
