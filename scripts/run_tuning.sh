#!/usr/bin/env bash
set -euo pipefail

MODELS=(pykan)
DATASETS=(bessel)

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    echo "=== Tuning ${model} on ${dataset} ==="
    HYDRA_FULL_ERROR=1 uv run main.py --multirun \
      +sweep="tune_${model}" \
      dataset="${dataset}"
  done
done

echo "=== All tuning complete ==="
