# %%
import os

import skore

from obesity.data import load_data, split_data
from obesity.utils import ensure_model_checkpoint, get_device
from obesity.evaluate import evaluate_model
from obesity.features import get_preprocessor
from obesity.pipeline import create_pipeline

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
project = skore.Project(name="obesity", workspace=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), mode="local")

# %%
# EDA Configuration
CSV_PATH = "data/obesity_dataset.csv"
TARGET_COL = "obese"

# %%
df = load_data(CSV_PATH)
df
# %%
import skrub
import pandas as pd

# %%
report = skrub.TableReport(df)
report
# %%
df[TARGET_COL].value_counts(normalize=True)
# %%
df.corr(numeric_only=True).unstack().sort_values(ascending=False).head(20)
