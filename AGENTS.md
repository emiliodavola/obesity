# Project KNOWLEDGE BASE

**Generated:** 2026-06-18

## OVERVIEW

Project: **Obesity Classification Pipeline**
Stack: **Python 3.12**, **pixi** (Environment/Package Manager), **Hydra** (Config Management), **TabPFN** (Model), **Scikit-learn** (Pipeline/Metrics), **Pandas** (Data Manipulation), **PyTorch** (Backend), **HuggingFace Hub** (Model Retrieval)

## STRUCTURE

* `main.py`: The orchestration entry point (Hydra-driven).
* `obesity/`: Core logic modules (Skrub DataOps):
  - `data.py`: Data loading and splitting.
  - `features.py`: Feature engineering and transformations.
  - `pipeline.py`: The classifier assembly (Skrub DataOps graph).
  - `evaluate.py`: Scoring and metric evaluation.
  - `utils.py`: Low-level utilities (device management, model retrieval).

* `conf/`: Configuration management (Hydra).
* `data/`: Dataset files.
* `tests/`: Structural and smoke tests (`tests/smoke/`).
* `audit/`: Experiment audit digests (Markdown-based summaries).
* `experiments/`: Individual experiment scripts (`# %%` jupytext format).
* `journal/`: The project's chronological record (`JOURNAL.md`).
* `openspec/`: The SDD artifacts (Proposals, Specs, Designs, Tasks, and Reports).

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
    - `data/obesity_dataset.csv`
* **Not Versioned (The Noise):**
    - `checkpoints/`, `outputs/`, `scratch/`, `*.ckpt`, `*.log`, `*.h5`, `*.pt`, `*.pth`, `*.db`, `*.val`

## WHERE TO LOOK

* **Source**: `obesity/`
* `Tests`: `tests/`
* `Docs/Config`: `conf/config.yaml`
