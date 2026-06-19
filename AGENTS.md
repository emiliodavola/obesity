# PROJECT KNOWLEDGE BASE

**Generated:** 2026-06-18

## OVERVIEW

Project: **Obesity Classification Pipeline**
Stack: **Python 3.12**, **pixi** (Environment/Package Manager), **Hydra** (Config Management), **TabPFN** (Model), **Scikit-learn** (Pipeline/Metrics), **Pandas** (Data Manipulation), **PyTorch** (Backend), **HuggingFace Hub** (Model Retrieval)

## STRUCTURE

* `main.py`: Orchestration and entry point (Hydra-driven).
* `src/`: Core engine logic and data processing.
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

## WHERE TO LOOK

* **Source**: `src/engine.py`
* **Tests**: `tests/`
* **Docs/Config**: `conf/config.yaml`

## NOTES

* **Hydra Integration**: Most hyperparameters and data paths are managed via `conf/config.yaml`. You can override parameters directly via CLI (e.g., `pixi run python main.py data.target_col=label`).
* **Model Lifecycle**: The engine automatically downloads and caches TabPFN checkpoints from Hugging Face Hub to `checkpoints/` if they are missing.
* **Environment**: This project strictly uses `pixi` for dependency management. Do not use `pip install` directly.
