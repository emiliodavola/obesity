# Project KNOWLEDGE BASE

**Generated:** 2026-06-18

## OVERVIEW

Project: **Obesity Classification Pipeline**
Stack: **Python 3.12**, **uv** (Environment/Package Manager), **Hydra** (Config Management), **TabPFN** (Model), **Scikit-learn** (Pipeline/Metrics), **Pandas** (Data Manipulation), **PyTorch** (Backend), **HuggingFace Hub** (Model Retrieval)

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
| Install| `uv sync --all-groups` |
| Install commit hooks | `uv run pre-commit install` (once after cloning) |
| Test   | `uv run python -m pytest` |
| Coverage | `PYTHONPATH=. uv run coverage run --source=obesity -m pytest && uv run coverage report --show-missing` |
| Run    | `uv run python main.py` |
| Build  | `uv sync --all-groups` (Environment setup) |

## CODING STANDARDS

* **Language**: Python 3.12 (Strictly typed via `mypy`)
* **Style**: Modular architecture, explicit logging, docstrings, and type casting for Hydra/OmegaConf.
* **Rules**: Linting and formatting via `ruff`. Security via `bandit`. Dependency audit via `pip-audit`.
* **Docstrings**: Every Python module, class, and function, including test functions, must have a docstring.
* **Type annotations**: Annotate function parameters and return values; declare variables with `name: Type = value`. Reassignments inherit the declared type. Destructuring, loop/comprehension, and exception targets rely on types from their source expressions because Python does not allow inline annotations in those binding positions.
* **Test coverage**: Line coverage from tests must not be below 90% for any `.py` module under `obesity/`. Review each file individually with the coverage command above; a high overall percentage does not replace this requirement.

## DEVELOPMENT WORKFLOW

* Follow Organic Driven Development (ODD) for project changes: inspect the current behaviour, work in small scoped steps, keep relevant tests with the change, and report observed verification results.
* After a fresh clone, run `uv run pre-commit install` once before creating commits so the repository hooks run automatically.

## BRANCHING AND INTEGRATION

* Work on a dedicated feature branch; do not commit or push directly to `main`.
* Open pull requests targeting `main`. The `dev` branch is retired and must not be used.
* Keep CI running on both pull requests to `main` and pushes to `main`; the push trigger validates the merged state, including pull request merges.

## VERSIONING POLICY

To ensure reproducibility and minimize repository bloat, the following rules apply:

* **Versioned (The Truths):**
    - `obesity/`, `main.py`, `tests/`, `conf/`, `journal/`, `openspec/`, `experiments/`, `pyproject.toml`, and `uv.lock`.
    - `data/obesity_dataset.csv`
* **Not Versioned (The Noise):**
    - `checkpoints/`, `outputs/`, `scratch/`, `*.ckpt`, `*.log`, `*.h5`, `*.pt`, `*.pth`, `*.db`, `*.val`

## WHERE TO LOOK

* **Source**: `obesity/`
* `Tests`: `tests/`
* `Docs/Config`: `conf/config.yaml`
