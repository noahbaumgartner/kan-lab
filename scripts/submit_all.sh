#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <experiment_name>" >&2
  echo "  Submits one SLURM tuning job per KAN model. The experiment_name is" >&2
  echo "  used as the MLflow experiment for all runs across all models." >&2
  exit 1
fi

EXPERIMENT="$1"
MODELS=(pykan efficientkan fastkan fasterkan wavkan)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for model in "${MODELS[@]}"; do
  job="${SCRIPT_DIR}/tune_${model}.submit"
  echo "Submitting ${job} (experiment=${EXPERIMENT})..."
  sbatch --export=ALL,EXPERIMENT="${EXPERIMENT}" "${job}"
done

echo "All ${#MODELS[@]} tuning jobs submitted. Check with: squeue -u \$USER"
