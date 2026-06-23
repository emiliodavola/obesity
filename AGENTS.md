# PROJECT KNOWLEDGE BASE

**Generated:** 2026-06-18

## OVERVIEW

Project: **Obesity Classification Pipeline**
Stack: **Python 3.12**, **pixi** (Environment/Package Manager), **Hydra** (Config Management), **TabPFN** (Model), **Scikit-learn** (Pipeline/Metrics), **Pandas** (Data Manipulation), **PyTorch** (Backend), **HuggingFace Hub** (Model Retrieval)

## STRUCTURE

* `main.py`: Orchestration and entry point (Hydra-driven).
* `obesity/`: Core logic and data processing.
* `conf/`: YAML configuration files for experiment parameters.
* `data/`: Raw dataset files.
* `tests/`: Unit tests for core functions.
* `checkpoints/`: Intended storage for downloaded model weights.
* `outputs/`: Intended storage for experiment results.

## COMMANDS

| Action | Command |
|--------|---------|
| Install| `pixi install` |
| Test   | `pixi run python -m pytest` |
| Run    | `pixi run python main.py` |
| Build  | `pixi install` (Environment setup) |

## CODING STANDARDS

* **Language**: Python 3.12 (Strictly typed via `mypy`)
* **Style**: Modular architecture, explicit logging, docstrings, and type casting for Hydra/OmegaConf.
* **Rules**: Linting and formatting via `ruff`. Security via `bandit`. Dependency audit via `pip-audit`.

## VERSIONING POLICY

To ensure reproducibility and minimize repository bloat, the following rules apply:

* **Versioned (The Truths):**
    - `obesity/`, `main.py`, `tests/`, `conf/`, `journal/`, `openspec/`, `experiments/`, `pixi.toml`, and `pixi.lock`.
* **Not Versioned (The Noise):**
    - `data/` (Raw datasets), `checkpoints/` (Model weights), `outputs/` (Results/Logs), and `scratch/` (Ephemeral probes).
* **Rule of Thumb:** If it's a binary, a large file, or a transient execution trace, it stays in the local environment, not in Git.

## WHERE TO LOOK

* **Source**: `obesity/`
* **Tests**: `tests/`
* **Docs/Config**: `conf/config.yaml`

## NOTES

* **Hydra Integration**: Most hyperparameters and data paths are managed via `conf/config.yaml`. You can override parameters directly via CLI (e.g., `pixi run python main.py data.target_col=label`).
* **Model Lifecycle**: Handled by `obesity/utils.py` and orchestrated by `main.py`.
* **Environment**: This project strictly uses `pixi`.
