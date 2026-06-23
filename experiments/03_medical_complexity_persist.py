import skore
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
project = skore.Project(name="obesity", workspace=PROJECT_ROOT, mode="local")

# Create dummy data for report persistence
df = pd.DataFrame({
    "feat": np.random.randn(100),
    "obese": np.random.randint(0, 2, 100)
})
X_train, X_test, y_train, y_test = train_test_split(df[["feat"]], df["obese"], test_size=0.2)

model = RandomForestClassifier()
pipeline = Pipeline([("scaler", StandardScaler()), ("clf", model)])
pipeline.fit(X_train, y_train)

report = skore.evaluate(pipeline, X=X_test, y=y_test)
project.put("03_medical_complexity", report)
