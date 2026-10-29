"""Evaluate whether medical complexity features improve obesity classification."""

import os
from typing import Any

import pandas as pd
import skore
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from obesity.data import load_data, split_data
from obesity.features import get_preprocessor
from obesity.pipeline import create_pipeline
from obesity.utils import ensure_model_checkpoint, get_device

PROJECT_ROOT: str = os.path.abspath(os.path.dirname(__file__))
project: skore.Project = skore.Project(
    name="obesity", workspace=PROJECT_ROOT, mode="local"
)

# %%
# Configuration
CSV_PATH: str = "data/obesity_dataset.csv"
TARGET_COL: str = "obese"
MODEL_CKPT: str = "checkpoints/tabpfn-v3-classifier-v3_20260417_binary.ckpt"
HF_REPO: str = ""
HF_FILENAME: str = ""

# Complexity Features (to DROP)
COMPLEXITY_FEATURES: list[str] = [
    "n_conditions",
    "n_emergency",
    "n_inpatient",
    "n_outpatient",
    "n_unique_conditions",
    "n_unique_procedures",
    "n_wellness",
    "total_claim_cost",
    "total_payer_coverage",
    "n_immunizations",
    "n_ambulatory",
    "n_allergies",
]

# Baseline Categorical Columns (to KEEP for preprocessor)
CAT_COLS_ORIGINAL: list[str] = [
    "RACE",
    "ETHNICITY",
    "has_sinusitis",
    "has_anemia",
    "has_diabetes",
    "has_hypertension",
    "has_asthma",
    "has_depression",
    "n_allergies",
    "n_careplans",
    "n_immunizations",
]

# %%
device: str = get_device()
model_path: str = ensure_model_checkpoint(MODEL_CKPT, HF_REPO, HF_FILENAME)

# %%
df: pd.DataFrame = load_data(CSV_PATH)
df_filtered: pd.DataFrame = df.drop(columns=COMPLEXITY_FEATURES)

# Update categorical columns for the preprocessor (only those that remain)
CAT_COLS_FILTERED: list[str] = [
    c for c in CAT_COLS_ORIGINAL if c in df_filtered.columns
]
print(f"Filtering complete. Remaining categorical columns: {CAT_COLS_FILTERED}")

# %%
split: list[Any] = split_data(df_filtered, TARGET_COL)
X_train: pd.DataFrame = split[0]
X_test: pd.DataFrame = split[1]
y_train: pd.Series = split[2]
y_test: pd.Series = split[3]

# %%
preprocessor: ColumnTransformer = get_preprocessor(CAT_COLS_FILTERED)
clf_pipeline: Pipeline = create_pipeline(preprocessor, model_path, device)

# %%
clf_pipeline.fit(X_train, y_train)

# %%
report: Any = skore.evaluate(clf_pipeline, X=X_test, y=y_test)
print("Evaluation complete.")
print(report.metrics.summarize().frame())
print(report.metrics.summarize().frame())

# %%
project.put("03_medical_complexity", report)
project
