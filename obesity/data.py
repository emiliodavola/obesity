"""Data loading and splitting utilities."""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(csv_path: str) -> pd.DataFrame:
    """Load raw data from a CSV file."""
    return pd.read_csv(csv_path)


def split_data(
    df: pd.DataFrame, target_col: str, test_size: float = 0.33, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        # shuffle=True,
        stratify=y
        )
