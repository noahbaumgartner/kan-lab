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

## Datasets

For the datasets I've selected 10 functional, synthetic datasets and 10 regression datasets.

### Functional

Synthetic functions from the foundational KAN paper (https://arxiv.org/abs/2404.19756).

| #   | Name                | Formel                                                     |
| --- | ------------------- | ---------------------------------------------------------- |
| 1   | **ExpSin**          | `f(x,y) = exp(sin(π·x) + y²)`                              |
| 2   | **DeepFormula**     | `f(x1..x4) = exp((sin(π·(x1²+x2²)) + sin(π·(x3²+x4²)))/2)` |
| 3   | **Bessel**          | `f(x,y) = exp(J₀(20x) + y²)`                               |
| 4   | **Multiplication**  | `f(x,y) = x·y`                                             |
| 5   | **LogComposition**  | `f(x,y) = sin(2·(log(x) + log(y)))`                        |
| 6   | **Norm2D**          | `f(x,y) = sqrt(x² + y²)`                                   |
| 7   | **GaussianPeaks**   | `f(x) = Σ exp(-(x - cᵢ)²·300)`, 5 Peaks                    |
| 8   | **Feynman I.37.4**  | `f = I1 + I2 + 2·sqrt(I1·I2)·cos(δ)`                       |
| 9   | **Feynman I.29.16** | `f = sqrt(x1² + x2² - 2·x1·x2·cos(θ1-θ2))`                 |
| 10  | **Feynman I.41.16** | `f = ℏω³/(π²c²·(exp(ℏω/(kb·T))-1))`                        |

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
