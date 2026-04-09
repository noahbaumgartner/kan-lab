#!/usr/bin/env bash
set -euo pipefail

DATASETS=(bessel expsin multiplication highdim deepformula feynman_i_6_2 feynman_i_6_2b feynman_i_9_18 feynman_i_12_11 feynman_i_13_12)

for dataset in "${DATASETS[@]}"; do
  echo "=== Tuning efficient_kan on ${dataset} ==="
  uv run python train.py --multirun \
    +experiment=tune_efficient_kan \
    dataset="${dataset}"
done

echo "=== All efficient_kan tuning complete ==="
