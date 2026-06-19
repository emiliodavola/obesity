"""Orchestrates the obesity classification pipeline using Hydra configuration."""

import logging
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from src.obesity.engine import run_experiment

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ObesityHydraPipeline")


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
