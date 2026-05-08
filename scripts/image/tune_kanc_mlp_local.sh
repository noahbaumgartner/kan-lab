#!/usr/bin/env bash
# Local equivalent of scripts/image/tune_kanc_mlp.submit — sweeps kanc_mlp
# over MNIST and Gaussian Blob using the Optuna sweeper, logging to whichever
# MLflow tracking server is advertised in mlflow/server_url.txt (or
# $MLFLOW_TRACKING_URI if set).
#
# Prereq: start the MLflow server in another terminal first:
#   ./scripts/mlflow_server.sh
#
# Usage:
#   ./scripts/image/tune_kanc_mlp_local.sh <experiment_name>
#   DATASETS="mnist" ./scripts/image/tune_kanc_mlp_local.sh my_experiment

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <experiment_name>" >&2
  exit 1
fi
EXPERIMENT="$1"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL=kanc_mlp
SWEEP=image/tune_kanc_mlp
DATASETS="${DATASETS:-mnist gaussian_blob}"

URL_FILE="${REPO_ROOT}/mlflow/server_url.txt"
if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
  TRACKING_URI="${MLFLOW_TRACKING_URI}"
elif [[ -s "${URL_FILE}" ]]; then
  TRACKING_URI=$(cat "${URL_FILE}")
else
  echo "No MLFLOW_TRACKING_URI set and ${URL_FILE} is missing." >&2
  echo "Start the server first: ./scripts/mlflow_server.sh" >&2
  exit 1
fi
echo "Using MLFLOW_TRACKING_URI=${TRACKING_URI}"

for dataset in ${DATASETS}; do
  echo "=== Tuning ${MODEL} on ${dataset} ==="
  HYDRA_FULL_ERROR=1 uv run python main.py --multirun \
    +sweep="${SWEEP}" \
    dataset="${dataset}" \
    mlflow_tracking_uri="${TRACKING_URI}" \
    +experiment="${EXPERIMENT}"
done

echo "=== ${MODEL} tuning complete ==="
