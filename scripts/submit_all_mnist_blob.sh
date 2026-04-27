#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <experiment_name>" >&2
  echo "  Submits one SLURM tuning job per KAN model (efficientkan, fastkan," >&2
  echo "  wavkan), each sweeping over the MNIST and Gaussian-Blob datasets." >&2
  echo "  The experiment_name is used as the MLflow experiment for all runs." >&2
  exit 1
fi

EXPERIMENT="$1"
MODELS=(efficientkan fastkan wavkan)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for model in "${MODELS[@]}"; do
  job="${SCRIPT_DIR}/tune_${model}_mnist_blob.submit"
  echo "Submitting ${job} (experiment=${EXPERIMENT})..."
  sbatch --export=ALL,EXPERIMENT="${EXPERIMENT}" "${job}"
done

echo "All ${#MODELS[@]} mnist+blob tuning jobs submitted. Check with: squeue -u \$USER"
