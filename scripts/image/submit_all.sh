#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <experiment_name>" >&2
  echo "  Submits one SLURM tuning job per KAN model (efficientkan, fastkan," >&2
  echo "  wavkan, kkan), each sweeping over the MNIST, Fashion-MNIST and" >&2
  echo "  Gaussian-Blob datasets. The experiment_name is used as the MLflow" >&2
  echo "  experiment for all runs." >&2
  exit 1
fi

EXPERIMENT="$1"
MODELS=(efficientkan fastkan wavkan kkan)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for model in "${MODELS[@]}"; do
  job="${SCRIPT_DIR}/tune_${model}.submit"
  echo "Submitting ${job} (experiment=${EXPERIMENT})..."
  sbatch --export=ALL,EXPERIMENT="${EXPERIMENT}" "${job}"
done

echo "All ${#MODELS[@]} image tuning jobs submitted. Check with: squeue -u \$USER"
