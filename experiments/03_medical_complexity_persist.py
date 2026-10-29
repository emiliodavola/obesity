"""Test persistence of a small obesity classifier report in Skore."""

import os
from typing import Any

import numpy as np
import pandas as pd
import skore
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT: str = os.path.abspath(os.path.dirname(__file__))
project: skore.Project = skore.Project(
    name="obesity", workspace=PROJECT_ROOT, mode="local"
)

# Create dummy data for report persistence
df: pd.DataFrame = pd.DataFrame(
    {"feat": np.random.randn(100), "obese": np.random.randint(0, 2, 100)}
)
split: list[Any] = train_test_split(df[["feat"]], df["obese"], test_size=0.2)
X_train: pd.DataFrame = split[0]
X_test: pd.DataFrame = split[1]
y_train: pd.Series = split[2]
y_test: pd.Series = split[3]

model: RandomForestClassifier = RandomForestClassifier()
pipeline: Pipeline = Pipeline([("scaler", StandardScaler()), ("clf", model)])
pipeline.fit(X_train, y_train)

report: Any = skore.evaluate(pipeline, X=X_test, y=y_test)
project.put("03_medical_complexity", report)
