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

## Modules

The following KAN variants are submodules in `modules/`, which can be evaluated in this project.

| Module         | Repository                                                                  |
| -------------- | --------------------------------------------------------------------------- |
| `pykan`        | [KindXiaoming/pykan](https://github.com/KindXiaoming/pykan)                 |
| `efficientkan` | [Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan)         |
| `dropkan`      | [Ghaith81/dropkan](https://github.com/Ghaith81/dropkan)                     |
| `gskan`        | [rambamn48/gs-impl](https://github.com/rambamn48/gs-impl)                   |
| `chebykan`     | [SynodicMonth/ChebyKAN](https://github.com/SynodicMonth/ChebyKAN)           |
| `fkan`         | [alirezaafzalaghaei/fKAN](https://github.com/alirezaafzalaghaei/fKAN)       |
| `rkan`         | [alirezaafzalaghaei/rKAN](https://github.com/alirezaafzalaghaei/rKAN)       |
| `fourierkan`   | [GistNoesis/FourierKAN](https://github.com/GistNoesis/FourierKAN)           |
| `sinekan`      | [ereinha/SineKAN](https://github.com/ereinha/SineKAN)                       |
| `wavkan`       | [zavareh1/Wav-KAN](https://github.com/zavareh1/Wav-KAN)                     |
| `fastkan`      | [ZiyaoLi/fast-kan](https://github.com/ZiyaoLi/fast-kan)                     |
| `fasterkan`    | [AthanasiosDelis/faster-kan](https://github.com/AthanasiosDelis/faster-kan) |
| `bsrbfkan`     | [hoangthangta/BSRBF_KAN](https://github.com/hoangthangta/BSRBF_KAN)         |
| `relukan`      | [quiqi/relu_kan](https://github.com/quiqi/relu_kan)                         |
| `afkan`        | [hoangthangta/All-KAN](https://github.com/hoangthangta/All-KAN)             |
| `fckan`        | [hoangthangta/FC_KAN](https://github.com/hoangthangta/FC_KAN)               |
