import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from obesity.data import load_data, split_data
from obesity.features import get_preprocessor
from obesity.pipeline import create_pipeline


def test_01_baseline_smoke():
    """Smoke test for 01_baseline: row count verification."""
    # Setup data
    csv_path = "data/obesity_dataset.csv"
    target_col = "obese"
    cat_cols = [
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
    model_path = "checkpoints/tabpfn-v2-classifier.ckpt"
    device = "cpu"

    df = pd.read_csv(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target_col, axis=1), df[target_col], test_size=0.3, random_state=42
    )

    preprocessor = get_preprocessor(cat_cols)
    clf_pipeline = create_pipeline(preprocessor, model_path, device)

    clf_pipeline.fit(X_train, y_train)

    # Predict on a disjoint subset of X_test
    predict_grid = X_test.iloc[:10, :]
    preds = clf_pipeline.predict(predict_grid)

    # Hard assertion: Row count match
    assert len(preds) == len(predict_grid)

if __name__ == "__main__":
    pytest.main([__file__])
