#!/usr/bin/env bash
set -euo pipefail

MODELS=(pykan efficient_kan fast_kan faster_kan chebykan relukan wavkan mlp)
DATASETS=(bessel expsin multiplication highdim deepformula)

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    echo "=== Tuning ${model} on ${dataset} ==="
    uv run python train.py --multirun \
      +experiment="tune_${model}" \
      dataset="${dataset}"
  done
done

echo "=== All tuning complete ==="
