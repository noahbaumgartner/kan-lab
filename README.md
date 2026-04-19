# KAN Lab

This is the KAN lab. An experimentation space for my specialization project "Reconstructing the Cosmic Dawn: Interpretable Machine Learning with Kolmogorov-Arnold Networks" in the MSc in Engineering.

## Setup

This repository works with `uv` and uses Git submodules. To set it up locally:

1. Clone this repository with: `git clone --recurse-submodules <repo-url>`
2. Run `uv sync` to install the needed python version and the dependencies

## Usage

The following code show commands which can be used for this lab environment.

```bash
# run default experiment
uv run main.py

# run with another model, another optimizer
uv run main.py model=mlp
uv run main.py model=efficientkan training=adam

# overwrite parameters
uv run main.py model.grid=10 training.steps=200

# sweep
uv run main.py --multirun model.grid=3,5,10,20

# sweep based on experiment
uv run main.py --multirun +experiment=model_comparison

# MLflow UI
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

# notebooks
uv run jupyter notebook notebooks/
```

## Datasets

### Functional Datasets (1-dimensional regression)

These datasets are from the foundational KAN paper ([arXiv:2404.19756](https://arxiv.org/abs/2404.19756)): 5 toy datasets (Section 3.1) and 5 Feynman equations (Section 3.3).

| #   | Name                | Formula                                                 | Variables                         | KAN Shape       |
| --- | ------------------- | ------------------------------------------------------- | --------------------------------- | --------------- |
| 1   | **Bessel**          | `f(x) = J₀(20x)`                                        | x                                 | [1, 1]          |
| 2   | **ExpSin**          | `f(x,y) = exp(sin(πx) + y²)`                            | x, y                              | [2, 1, 1]       |
| 3   | **Multiplication**  | `f(x,y) = xy`                                           | x, y                              | [2, 2, 1]       |
| 4   | **HighDim**         | `f(x₁..x₁₀₀) = exp(1/100 · Σ sin²(πxᵢ/2))`              | x₁..x₁₀₀                          | [100, 1, 1]     |
| 5   | **DeepFormula**     | `f(x₁..x₄) = exp(½(sin(π(x₁²+x₂²)) + sin(π(x₃²+x₄²))))` | x₁..x₄                            | [4, 4, 2, 1]    |
| 6   | **Feynman I.6.2**   | `f(θ,σ) = exp(-θ²/(2σ²)) / √(2πσ²)`                     | θ, σ                              | [2, 2, 1, 1]    |
| 7   | **Feynman I.6.2b**  | `f(θ,θ₁,σ) = exp(-(θ-θ₁)²/(2σ²)) / √(2πσ²)`             | θ, θ₁, σ                          | [3, 2, 2, 1, 1] |
| 8   | **Feynman I.9.18**  | `f = Gm₁m₂ / ((x₂-x₁)²+(y₂-y₁)²+(z₂-z₁)²)`              | G, m₁, m₂, x₁, x₂, y₁, y₂, z₁, z₂ | [6, 4, 2, 1, 1] |
| 9   | **Feynman I.12.11** | `f = q(Ef + Bv·sinθ)`                                   | q, Ef, B, v, θ                    | [2, 2, 2, 1]    |
| 10  | **Feynman I.13.12** | `f = Gm₁m₂(1/r₂ - 1/r₁)`                                | G, m₁, m₂, r₁, r₂                 | [2, 2, 1]       |

### Image / Classification Datasets

| Name              | Task           | Input               | Output     | Description                                           |
| ----------------- | -------------- | ------------------- | ---------- | ----------------------------------------------------- |
| **MNIST**         | Classification | 784 (28x28 flatten) | 10 classes | Handwritten digit classification                      |
| **Gaussian Blob** | Regression     | 100 (10x10 flatten) | 4 values   | Predict center (x, y), width, and amplitude of a blob |

## Models

The following KAN variants are implemented in this project. The module code in `src/modules/` is copied from the respective repositories.

| Model        | Config name     | Repository                                                                  | Paper                                                |
| ------------ | --------------- | --------------------------------------------------------------------------- | ---------------------------------------------------- |
| PyKAN        | `pykan`         | [KindXiaoming/pykan](https://github.com/KindXiaoming/pykan)                 | [arXiv:2404.19756](https://arxiv.org/abs/2404.19756) |
| EfficientKAN | `efficientkan`  | [Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan)         | -                                                    |
| FastKAN      | `fastkan`       | [ZiyaoLi/fast-kan](https://github.com/ZiyaoLi/fast-kan)                     | [arXiv:2405.06721](https://arxiv.org/abs/2405.06721) |
| FasterKAN    | `fasterkan`     | [AthanasiosDelis/faster-kan](https://github.com/AthanasiosDelis/faster-kan) | -                                                    |
| WavKAN       | `wavkan`        | [zavareh1/Wav-KAN](https://github.com/zavareh1/Wav-KAN)                     | [arXiv:2405.12832](https://arxiv.org/abs/2405.12832) |
