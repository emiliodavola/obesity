# %%
import os

import skore

from obesity.data import load_data, split_data
from obesity.utils import ensure_model_checkpoint, get_device
from obesity.evaluate import evaluate_model
from obesity.features import get_preprocessor
from obesity.pipeline import create_pipeline

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
project = skore.Project(name="obesity", workspace=PROJECT_ROOT, mode="local")

# %%
# Configuration
CSV_PATH = "data/obesity_dataset.csv"
TARGET_COL = "obese"
CAT_COLS = ["RACE", "ETHNICITY", "has_sinusitis", "has_anemia", "has_diabetes", "has_hypertension", "has_asthma", "has_depression", "n_allergies", "n_careplans", "n_immunizations"]
MODEL_CKPT = "checkpoints/tabpfn-v3-classifier-v3_20260417_binary.ckpt"
HF_REPO = ""
HF_FILENAME = ""

# %%
device = get_device()
model_path = ensure_model_checkpoint(MODEL_CKPT, HF_REPO, HF_FILENAME)

# %%
df = load_data(CSV_PATH)
X_train, X_test, y_train, y_test = split_data(df, TARGET_COL)

# %%
preprocessor = get_preprocessor(CAT_COLS)
clf_pipeline = create_pipeline(preprocessor, model_path, device)

# %%
clf_pipeline.fit(X_train, y_train)

# %%
report = skore.evaluate(clf_pipeline, X=X_test, y=y_test)
report
# %%
report.checks.summarize().frame()
# %%
report.metrics.summarize().frame()
# %%
project.put("01_baseline", report)
project
