#!/usr/bin/env bash
set -euo pipefail

MODELS=(pykan efficientkan fastkan fasterkan wavkan)
DATASETS=(
  bessel
  expsin
  multiplication
  highdim
  deepformula
  feynman_i_6_2
  feynman_i_6_2b
  feynman_i_9_18
  feynman_i_12_11
  feynman_i_13_12
)

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    echo "=== Tuning ${model} on ${dataset} ==="
    HYDRA_FULL_ERROR=1 uv run main.py --multirun \
      hydra/launcher=submitit_slurm \
      hydra.launcher.timeout_min=1440 \
      hydra.launcher.partition=gpu \
      hydra.launcher.gres=gpu:1 \
      hydra.launcher.account=cai_ivs \
      hydra.launcher.mem_gb=32 \
      hydra.launcher.tasks_per_node=1 \
      hydra.launcher.nodes=1 \
      hydra.launcher.array_parallelism=8 \
      +sweep="tune_${model}" \
      dataset="${dataset}"
  done
done

echo "=== All tuning complete ==="
