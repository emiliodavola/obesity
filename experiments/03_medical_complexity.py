import os
import skore
import pandas as pd

from obesity.data import load_data, split_data
from obesity.utils import ensure_model_checkpoint, get_device
from obesity.features import get_preprocessor
from obesity.pipeline import create_pipeline

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
project = skore.Project(name="obesity", workspace=PROJECT_ROOT, mode="local")

# %%
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

# Baseline Categorical Columns (to KEEP for preprocessor)
CAT_COLS_ORIGINAL = ["RACE", "ETHNICITY", "has_sinusitis", "has_anemia", "has_diabetes", 
                     "has_hypertension", "has_asthma", "has_depression", "n_allergies", 
                     "n_careplans", "n_immunizations"]

# %%
device = get_device()
model_path = ensure_model_checkpoint(MODEL_CKPT, HF_REPO, HF_FILENAME)

# %%
df = load_data(CSV_PATH)
df_filtered = df.drop(columns=COMPLEXITY_FEATURES)

# Update categorical columns for the preprocessor (only those that remain)
CAT_COLS_FILTERED = [c for c in CAT_COLS_ORIGINAL if c in df_filtered.columns]
print(f"Filtering complete. Remaining categorical columns: {CAT_COLS_FILTERED}")

# %%
X_train, X_test, y_train, y_test = split_data(df_filtered, TARGET_COL)

# %%
preprocessor = get_preprocessor(CAT_COLS_FILTERED)
clf_pipeline = create_pipeline(preprocessor, model_path, device)

# %%
clf_pipeline.fit(X_train, y_train)

# %%
report = skore.evaluate(clf_pipeline, X=X_test, y=y_test)
print("Evaluation complete.")
print(report.metrics.summarize().frame())
print(report.metrics.summarize().frame())

# %%
project.put("03_medical_complexity", report)
project
