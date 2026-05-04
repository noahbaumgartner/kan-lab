#!/bin/bash
# Local MLflow tracking server. Mirrors the cluster job in
# scripts/mlflow_server.submit but runs on your workstation.
#
# Usage:
#   ./scripts/mlflow_server.sh                # default port 9299
#   MLFLOW_PORT=5000 ./scripts/mlflow_server.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MLFLOW_HOST="${MLFLOW_HOST:-127.0.0.1}"
MLFLOW_PORT="${MLFLOW_PORT:-9299}"
MLFLOW_DIR="${MLFLOW_DIR:-${REPO_ROOT}/mlflow}"
URL_FILE="${MLFLOW_DIR}/server_url.txt"

mkdir -p "${MLFLOW_DIR}"

TRACKING_URI="http://${MLFLOW_HOST}:${MLFLOW_PORT}"
echo "${TRACKING_URI}" > "${URL_FILE}"
echo "Wrote tracking URI to ${URL_FILE}: ${TRACKING_URI}"

trap 'rm -f "${URL_FILE}"' EXIT

export PYTHONUNBUFFERED=1
export MLFLOW_SERVER_ALLOWED_HOSTS="*"

exec uv run mlflow server \
  --host "${MLFLOW_HOST}" \
  --port "${MLFLOW_PORT}" \
  --backend-store-uri "sqlite:///${MLFLOW_DIR}/mlflow.db" \
  --default-artifact-root "${MLFLOW_DIR}/artifacts"
