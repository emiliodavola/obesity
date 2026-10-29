# Obesity Classification Pipeline

A high-fidelity machine learning research framework for obesity classification, built on a **Skrub DataOps** architecture. It leverages the **TabPFN** (Tabular Prior-Data Fitted Network) model within a strictly typed, modular pipeline.

## Core Architecture

* **Skrub DataOps Graph**: The pipeline is constructed as a graph of stateless functions and stateful estimators, ensuring metadata (like feature names) propagates through the entire chain.
* **TabPFN Integration**: We use a specialized `TabPFNWrapper` to ensure the TabPFN model complies with the `scikit-learn` and `skrub` API contracts, enabling seamless feature name propagation.
* **Hydra Orchestration**: All experiment parameters, data paths, and model settings are managed via `conf/config.yaml`.

## Project Structure

* `main.py`: The orchestration entry point (Hydra-driven).
* `obesity/`: The core logic (Skrub DataOps modules):
  - `data.py`: Data loading and splitting.
  - `features.py`: Feature engineering and transformations.
  - `pipeline.py`: The classifier assembly (Skrub DataOps graph).
  - `evaluate.py`: Scoring and metric evaluation.
  - `utils.py`: Low-level utilities (device management, model retrieval).
* `conf/`: Configuration management (Hydra).
* `data/`: Raw dataset files.
* `tests/`: Structural and smoke tests (`tests/smoke/`).
* `audit/`: Experiment audit digests (Markdown-based summaries).
* `experiments/`: Individual experiment scripts (`# %%` jupytext format).
* `journal/`: The project's chronological record (`JOURNAL.md`).
* `openspec/`: Project change records and verification reports.

## Standards & Compliance

* **Language**: Python 3.12 (Strictly typed via `mypy`).
* **Code Quality**: Linting and formatting via `ruff`. Security via `bandit`.
* **Environment**: Managed with `uv` and locked in `uv.lock`.
* **Reproducibility**: Every experiment is accompanied by a `journal` entry and a `verify-report`.

## Environment

```bash
uv sync --all-groups
uv run pre-commit install
uv run python -m pytest
PYTHONPATH=. uv run coverage run --source=obesity -m pytest
uv run coverage report --show-missing
uv run python main.py
```

Run `uv run pre-commit install` once after cloning to enable the repository hooks for future commits.

The coverage report shows the percentage and uncovered lines for each module in `obesity/`.
