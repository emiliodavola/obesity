"""
Utility functions for the obesity classification pipeline.
"""

import os
import shutil
import torch
from huggingface_hub import hf_hub_download


def get_device() -> str:
    """Return the available computation device (CUDA or CPU).

    Returns
    -------
    str
        The device identifier.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def ensure_model_checkpoint(local_path: str, hf_repo: str, hf_filename: str) -> str:
    """Verify if model exists locally, otherwise downloads from Hugging Face.

    Parameters
    ----------
    local_path : str
        The target directory for the model file.
    hf_repo : str
        The Hugging Face repository ID.
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
