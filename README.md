# Obesity Classification Pipeline

An end-to-end machine learning pipeline for obesity classification leveraging the TabPFN (Tabular Prior-Data Fitted Network) model.

## Overview

This project implements a modular pipeline to predict obesity levels based on tabular data. It uses **Hydra** for flexible configuration and **TabPFN**, a powerful transformer-based model specifically designed for tabular data, which provides high performance with minimal tuning.

## Key Features

- **Modular Configuration**: Powered by Hydra, allowing easy adjustment of hyperparameters, paths, and model settings without touching the code.
- **Automated Model Management**: Integration with Hugging Face Hub to automatically download required TabPFN checkpoints.
- **Robust Preprocessing**: Implements a scikit-learn pipeline using `ColumnTransformer` and `OrdinalEncoder` for handling categorical features.
- **Comprehensive Evaluation**: Tracks performance using ROC AUC and Accuracy metrics.

## Project Structure

- `main.py`: The entry point; orchestrates the experiment via Hydra.
- `obesity/`: Core logic modules:
  - `data.py`: Data loading and splitting.
  - `features.py`: Preprocessing and transformations.
  - `pipeline.py`: The classifier assembly (Skrub DataOps).
  - `evaluate.py`: Scoring and metrics.
  - `utils.py`: Low-level utilities (device, model management).

- `conf/config.yaml`: Configuration file for dataset paths, target columns, and model parameters.
- `data/`: Directory containing the obesity dataset.

## Installation & Usage

### Prerequisites

Ensure you have [pixi](https://pixi.sh) installed.

### Setup

```bash
pixi install
```

### Running the Pipeline

```bash
pixi run python main.py
```

To override configuration parameters via CLI:

```bash
pixi run python main.py params.test_size=0.2 data.path=data/my_dataset.csv
```
