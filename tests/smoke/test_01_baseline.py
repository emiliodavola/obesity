"""Smoke-test row preservation in the baseline obesity classifier."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from obesity.features import get_preprocessor
from obesity.pipeline import create_pipeline


def test_01_baseline_smoke() -> None:
    """Verify predictions preserve the number of requested rows."""
    # Setup data
    csv_path: str = "data/obesity_dataset.csv"
    target_col: str = "obese"
    cat_cols: list[str] = [
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
    model_path: str = "checkpoints/tabpfn-v2-classifier.ckpt"
    device: str = "cpu"

    df: pd.DataFrame = pd.read_csv(csv_path)
    split: list[Any] = train_test_split(
        df.drop(target_col, axis=1),
        df[target_col],
        test_size=0.3,
        random_state=42,
    )
    X_train: pd.DataFrame = split[0]
    X_test: pd.DataFrame = split[1]
    y_train: pd.Series = split[2]
    _y_test: pd.Series = split[3]

    preprocessor: ColumnTransformer = get_preprocessor(cat_cols)
    clf_pipeline: Pipeline = create_pipeline(preprocessor, model_path, device)

    clf_pipeline.fit(X_train, y_train)

    # Predict on a disjoint subset of X_test
    predict_grid: pd.DataFrame = X_test.iloc[:10, :]
    preds: np.ndarray = clf_pipeline.predict(predict_grid)

    # Hard assertion: Row count match
    assert len(preds) == len(predict_grid)


if __name__ == "__main__":
    pytest.main([__file__])
