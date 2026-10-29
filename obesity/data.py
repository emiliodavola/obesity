"""Data loading and splitting utilities."""

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(csv_path: str) -> pd.DataFrame:
    """Load raw data from a CSV file."""
    return pd.read_csv(csv_path)


def split_data(
    df: pd.DataFrame, target_col: str, test_size: float = 0.33, random_state: int = 42
) -> list[Any]:
    """Split data into train and test sets."""
    X: pd.DataFrame = df.drop(target_col, axis=1)
    y: pd.Series = df[target_col]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        # shuffle=True,
        stratify=y,
    )
