#!/usr/bin/env bash
# Common logic for per-model GPU tuning jobs. Sources expects $MODEL to be set.
set -euo pipefail

: "${MODEL:?MODEL must be set by the caller (e.g. MODEL=pykan)}"
: "${EXPERIMENT:?EXPERIMENT must be set by the caller (MLflow experiment name)}"

# Optional overrides:
#   SWEEP    — Hydra sweep name including subgroup (default: functional/tune_${MODEL})
#   DATASETS — space-separated list of dataset names
SWEEP="${SWEEP:-functional/tune_${MODEL}}"
if [[ -n "${DATASETS:-}" ]]; then
  read -r -a DATASETS <<< "${DATASETS}"
else
  DATASETS=(
    feynman_i_6_2
    feynman_i_6_2b
    feynman_i_9_18
    feynman_i_12_11
    feynman_i_13_12
  )
fi

export PYTHONUNBUFFERED=1
module load uv/0.10.10
cd /cluster/home/baumgnoa/kan-lab

export UV_PROJECT_ENVIRONMENT=/cluster/home/baumgnoa/kan-lab/.venv
uv sync

# Wait for the mlflow server job to publish its URL.
URL_FILE=/cluster/home/baumgnoa/kan-lab/mlflow/server_url.txt
for i in $(seq 1 60); do
  if [[ -s "${URL_FILE}" ]]; then break; fi
  echo "Waiting for MLflow URL at ${URL_FILE} (${i}/60)..."
  sleep 10
done
if [[ ! -s "${URL_FILE}" ]]; then
  echo "MLflow URL file never appeared; is the mlflow_server job running?" >&2
  exit 1
fi
export MLFLOW_TRACKING_URI=$(cat "${URL_FILE}")
echo "Using MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}"

for dataset in "${DATASETS[@]}"; do
  echo "=== Tuning ${MODEL} on ${dataset} ==="
  HYDRA_FULL_ERROR=1 uv run main.py --multirun \
    +sweep="${SWEEP}" \
    dataset="${dataset}" \
    mlflow_tracking_uri="${MLFLOW_TRACKING_URI}" \
    +experiment="${EXPERIMENT}"
done

echo "=== ${MODEL} tuning complete ==="
