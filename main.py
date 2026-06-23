"""Orchestrates the obesity classification pipeline using Hydra configuration."""

import logging
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd

from obesity.data import load_data, split_data
from obesity.features import get_preprocessor
from obesity.pipeline import create_pipeline
from obesity.evaluate import evaluate_model
from obesity.utils import get_device, ensure_model_checkpoint

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ObesityHydraPipeline")


def run_experiment(
    csv_path: str,
    target_col: str,
    cat_cols: list[str],
    model_ckpt: str,
    hf_repo: str,
    hf_filename: str,
    test_size: float = 0.33,
    random_state: int = 42,
) -> tuple[float, float]:
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
    X_train, X_test, y_train, y_test = split_data(df, target_col, test_size=test_size, random_state=random_state)

    preprocessor = get_preprocessor(cat_cols)
    clf_pipeline = create_pipeline(preprocessor, model_path, device)

    print("Fitting pipeline...")
    clf_pipeline.fit(X_train, y_train)

    print("Evaluating pipeline...")
    roc_auc, acc = evaluate_model(clf_pipeline, X_test, y_test)

    print(f"ROC AUC: {roc_auc:.6f}")
    print(f"Accuracy: {acc:.6f}")
    return roc_auc, acc


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Execute the obesity classification pipeline from configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object containing data and model parameters.

    Returns
    -------
    tuple[float, float]
        The ROC AUC and Accuracy of the experiment.
    """
    try:
        config_dict = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))

        roc, acc = run_experiment(
            csv_path=config_dict["data"]["path"],
            target_col=config_dict["data"]["target_col"],
            cat_cols=config_dict["data"]["cat_cols"],
            model_ckpt=config_dict["model"]["ckpt"],
            hf_repo=config_dict["model"]["hf_repo"],
            hf_filename=config_dict["model"]["hf_filename"],
            test_size=config_dict["params"]["test_size"],
            random_state=config_dict["params"]["random_state"],
        )

        logger.info(f"ROC AUC: {roc:.4f}, Accuracy: {acc:.4f}")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
