# KAN Lab

This is the KAN lab. An experimentation space for my specialization project "Reconstructing the Cosmic Dawn: Interpretable Machine Learning with Kolmogorov-Arnold Networks" in the MSc in Engineering.

## Setup

This repository works with `uv`. To run the command in the section usage, clone the repo and run `uv sync`locally.

## Usage

The following code show commands which can be used for this lab environment.

```bash
# run default experiment
uv run python train.py

# run with another model, another optimizer
uv run python train.py model=mlp
uv run python train.py model=efficient_kan training=adam

# overwrite parameters
uv run python train.py model.grid=10 training.steps=200

# sweep
uv run python train.py --multirun model.grid=3,5,10,20

# sweep based on experiment
uv run python train.py --multirun experiment=model_comparison

# MLflow UI
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

# notebooks
uv run jupyter notebook notebooks/
```
