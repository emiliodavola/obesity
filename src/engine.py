import warnings
import os
import shutil
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tabpfn import TabPFNClassifier
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore", category=UserWarning)


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def ensure_model_checkpoint(local_path, hf_repo, hf_filename):
    """Checks if model exists locally, otherwise downloads from Hugging Face."""
    if os.path.exists(local_path):
        return local_path

    print(f"Model not found at {local_path}. Downloading latest version from HF...")
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        cached_path = hf_hub_download(repo_id=hf_repo, filename=hf_filename)
        shutil.copy(cached_path, local_path)
        print(f"Model successfully downloaded to {local_path}")
    except Exception as e:
        print(f"Failed to download model from HF ({hf_repo}/{hf_filename}): {e}")
        raise e

    return local_path


def create_obesity_pipeline(model_path, device, cat_cols):
    preprocessor = ColumnTransformer(
        transformers=[("cat", OrdinalEncoder(), cat_cols)], remainder="passthrough"
    )
    model = TabPFNClassifier(model_path=model_path, device=device)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])


def run_experiment(
    csv_path,
    target_col,
    cat_cols,
    model_ckpt,
    hf_repo,
    hf_filename,
    test_size=0.33,
    random_state=42,
):
    device = get_device()

    model_path = ensure_model_checkpoint(model_ckpt, hf_repo, hf_filename)

    print(f"Loading raw data from {csv_path}...")
    df = pd.read_csv(csv_path)
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    clf_pipeline = create_obesity_pipeline(model_path, device, cat_cols)

    print("Fitting pipeline...")
    clf_pipeline.fit(X_train, y_train)

    print("Evaluating pipeline...")
    probs = clf_pipeline.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, probs[:, 1])

    preds = clf_pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"ROC AUC: {roc_auc:.6f}")
    print(f"Accuracy: {acc:.6f}")
    return roc_auc, acc
