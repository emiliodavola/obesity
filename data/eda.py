"""Explore obesity dataset balance, feature distributions, and correlations."""

# %%
import os

import pandas as pd
import skore

from obesity.data import load_data

PROJECT_ROOT: str = os.path.abspath(os.path.dirname(__file__))
project: skore.Project = skore.Project(
    name="obesity",
    workspace=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    mode="local",
)

# %%
# EDA Configuration
CSV_PATH: str = "data/obesity_dataset.csv"
TARGET_COL: str = "obese"

# %%
df: pd.DataFrame = load_data(CSV_PATH)
df
# %%
import skrub

# %%
report: skrub.TableReport = skrub.TableReport(df)
report
