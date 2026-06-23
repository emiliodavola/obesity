"""
Preprocessing and transformation logic.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def get_preprocessor(cat_cols: list[str]) -> ColumnTransformer:
    """Create a preprocessing ColumnTransformer."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             cat_cols)
             ],
        remainder="passthrough"
    )
