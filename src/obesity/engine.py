"""
Core engine for the obesity classification pipeline.

Includes model management and execution.
"""

import os
import shutil
import warnings

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from tabpfn import TabPFNClassifier

warnings.filterwarnings("ignore", category=UserWarning)


def get_device():
    """Return the available computation device (CUDA or CPU).

    Returns
    -------
    str
        The device identifier.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def ensure_model_checkpoint(local_path, hf_repo, hf_filename):
    """Verify if model exists locally, otherwise downloads from Hugging Face.

    Parameters
    ----------
    local_path : str
        The target directory for the model file.
    hf_repo : str
        Hugging Face repository ID.
    hf_filename : str
        The specific filename to download.

    Returns
    -------
    str
        The path to the model checkpoint.
    """
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
    """Construct a Scikit-learn pipeline for obesity classification.

    Parameters
    ----------
    model_path : str
        Path to the loaded TabPFN model.
    device : str
        The device to use for inference (e.g., 'cuda').
    cat_cols : list[str]
        List of columns to be encoded via OrdinalEncoder.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The assembled preprocessing and classification pipeline.
    """
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
    """Execute the classification experiment from raw data to final metrics.

    Parameters
    ----------
    csv_path : str
        Path to the raw dataset CSV.
    target_col : str
        The name of the target variable to predict.
    cat_cols : list[str]
        Names of categorical columns for preprocessing.
    model_ckpt : str
        Target path for the model checkpoint.
    hf_repo : str
        Hugging Face repository ID.
    hf_filename : str
        The specific filename to download.
    test_size : float, default=0.33
        Proportion of data held out for testing.
    random_state : int, default=42
        Random seed for reproducible splitting.

    Returns
    -------
    tuple[float, float]
        A tuple containing (ROC AUC, Accuracy).
    """
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
