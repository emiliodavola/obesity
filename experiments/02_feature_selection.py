"""Evaluate obesity classification after removing redundant and noisy features."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

from obesity.data import split_data
from obesity.features import get_preprocessor
from obesity.pipeline import create_pipeline
from obesity.utils import ensure_model_checkpoint, get_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger: logging.Logger = logging.getLogger("ObesityFeatureSelection")

# Configuration
CSV_PATH: str = "data/obesity_dataset.csv"
TARGET_COL: str = "obese"
REDUNDANT_FEATURES: list[str] = [
    "n_careplans",
    "n_procedures",
    "n_medications",
    "total_med_cost",
]
NOISE_FEATURES: list[str] = [
    "ETHNICITY",
    "HEALTHCARE_COVERAGE",
    "HEALTHCARE_EXPENSES",
    "LAT",
    "LON",
    "RACE",
    "has_anemia",
    "has_asthma",
    "has_depression",
    "has_diabetes",
    "has_hypertension",
    "has_sinusitis",
    "is_dead",
    "is_female",
    "n_allergies",
    "n_ambulatory",
    "n_conditions",
    "n_emergency",
    "n_encounters",
    "n_immunizations",
    "n_inpatient",
    "n_outpatient",
    "n_unique_conditions",
    "n_unique_medications",
    "n_unique_procedures",
    "n_wellness",
    "total_claim_cost",
    "total_payer_coverage",
]
FEATURES_TO_DROP: list[str] = REDUNDANT_FEATURES + NOISE_FEATURES


def run_experiment() -> tuple[float, float]:
    """Measure classifier metrics after removing redundant and noisy features."""
    device: str = get_device()
    model_path: str = ensure_model_checkpoint(
        "checkpoints/tabpfn-v3-classifier-v3_20260417_binary.ckpt",
        "Prior-Labs/tabpfn_3",
        "tabpfn-v3-classifier-v3_20260417_binary.ckpt",
    )

    print(f"Loading raw data from {CSV_PATH}...")
    df: pd.DataFrame = pd.read_csv(CSV_PATH)

    print(f"Dropping features: {FEATURES_TO_DROP}")
    df_pruned: pd.DataFrame = df.drop(columns=FEATURES_TO_DROP)

    assert all(f not in df_pruned.columns for f in FEATURES_TO_DROP), (
        "Dimensionality check failed!"
    )

    print("Splitting pruned data...")
    split: list[Any] = split_data(df_pruned, TARGET_COL)
    X_train: pd.DataFrame = split[0]
    X_test: pd.DataFrame = split[1]
    y_train: pd.Series = split[2]
    y_test: pd.Series = split[3]

    # Consider categorical features in the source dataset and pruned result.
    surviving_cat_cols: list[str] = [
        column
        for column in df.columns
        if column in ["RACE", "ETHNICITY"]
        or (column in df_pruned.columns and is_categorical_candidate(df_pruned, column))
    ]
    # Keep only categorical features that remain after pruning.
    surviving_cat_cols = [
        column for column in df_pruned.columns if column in ["RACE", "ETHNICITY"]
    ]
    # Preserve the configured categorical feature order in the preprocessor.
    surviving_cat_cols = [
        column for column in ["RACE", "ETHNICITY"] if column in df_pruned.columns
    ]

    print(f"Using surviving categorical columns: {surviving_cat_cols}")
    preprocessor: ColumnTransformer = get_preprocessor(surviving_cat_cols)
    clf_pipeline: Pipeline = create_pipeline(preprocessor, model_path, device)

    print("Fitting pipeline...")
    clf_pipeline.fit(X_train, y_train)

    print("Evaluating pipeline...")

    preds: np.ndarray = clf_pipeline.predict(X_test)
    roc_auc: float = roc_auc_score(y_test, preds)
    acc: float = accuracy_score(y_test, preds)

    print(f"ROC AUC: {roc_auc:.6f}")
    print(f"Accuracy: {acc:.6f}")

    return roc_auc, acc


def is_categorical_candidate(df: pd.DataFrame, col: str) -> bool:
    """Return whether a feature is one of the configured categorical columns."""
    return col in ["RACE", "ETHNICITY"]


if __name__ == "__main__":
    metrics: tuple[float, float] = run_experiment()
    roc: float = metrics[0]
    acc: float = metrics[1]
    print(f"FINAL_RESULT_ROC_AUC:{roc}")
    print(f"FINAL_RESULT_ACC:{acc}")
