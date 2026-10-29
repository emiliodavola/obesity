"""Evaluate a reduced feature set for the obesity classification pipeline."""

from typing import Any

import pandas as pd
import skore
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from obesity.data import load_data, split_data
from obesity.features import get_preprocessor
from obesity.pipeline import create_pipeline
from obesity.utils import ensure_model_checkpoint, get_device

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


# Implementation Logic
def run_experiment() -> tuple[
    skore.EstimatorReport | skore.CrossValidationReport | skore.ComparisonReport,
    list[str],
]:
    """Evaluate the classifier using the reduced medical feature set."""
    device: str = get_device()
    model_path: str = ensure_model_checkpoint(MODEL_CKPT, HF_REPO, HF_FILENAME)

    # 1. Load and Filter
    df: pd.DataFrame = load_data(CSV_PATH)
    df_filtered: pd.DataFrame = df.drop(columns=COMPLEXITY_FEATURES)

    # 2. Split
    split: list[Any] = split_data(df_filtered, TARGET_COL)
    X_train: pd.DataFrame = split[0]
    X_test: pd.DataFrame = split[1]
    y_train: pd.Series = split[2]
    y_test: pd.Series = split[3]

    # 3. Pipeline
    cat_cols: list[str] = [
        "is_female",
        "RACE",
        "ETHNICITY",
        "has_anemia",
        "has_asthma",
        "has_depression",
        "has_diabetes",
        "has_hypertension",
        "has_sinusitis",
        "is_dead",
    ]
    preprocessor: ColumnTransformer = get_preprocessor(cat_cols)
    clf_pipeline: Pipeline = create_pipeline(preprocessor, model_path, device)
    clf_pipeline.fit(X_train, y_train)

    # 4. Evaluate
    report: (
        skore.EstimatorReport | skore.CrossValidationReport | skore.ComparisonReport
    ) = skore.evaluate(clf_pipeline, X_test, y_test)
    return report, X_test.columns.tolist()


if __name__ == "__main__":
    result: tuple[
        skore.EstimatorReport | skore.CrossValidationReport | skore.ComparisonReport,
        list[str],
    ] = run_experiment()
    report: (
        skore.EstimatorReport | skore.CrossValidationReport | skore.ComparisonReport
    ) = result[0]
    columns: list[str] = result[1]
    print(f"Final Feature Set: {columns}")
    print("\n--- Metrics Summary ---")
    print(report.metrics.summarize().frame())
    print("\n--- Checks Summary ---")
    print(report.checks.summarize().frame())
