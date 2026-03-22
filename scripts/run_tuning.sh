#!/usr/bin/env bash
set -euo pipefail

MODELS=(pykan efficient_kan fast_kan faster_kan chebykan relukan wavkan)
DATASETS=(bessel expsin multiplication highdim deepformula feynman_i_6_2 feynman_i_6_2b feynman_i_9_18 feynman_i_12_11 feynman_i_13_12)

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    echo "=== Tuning ${model} on ${dataset} ==="
    uv run python train.py --multirun \
      +experiment="tune_${model}" \
      dataset="${dataset}"
  done
done

echo "=== All tuning complete ==="
