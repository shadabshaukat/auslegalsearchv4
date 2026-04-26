#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f .env.production_v2 ]]; then
  echo "[v2-local] ERROR: .env.production_v2 not found in ${ROOT_DIR}"
  exit 1
fi

set -o allexport
source .env.production_v2
set +o allexport

export AUSLEGALSEARCH_V2_ENV_FILE="${ROOT_DIR}/.env.production_v2"
export AUSLEGALSEARCH_EMBED_USE_CUDA="${AUSLEGALSEARCH_EMBED_USE_CUDA:-1}"
export AUSLEGALSEARCH_EMBED_AMP="${AUSLEGALSEARCH_EMBED_AMP:-1}"
export TOKENIZERS_PARALLELISM="false"

API_HOST="${V2_API_HOST:-0.0.0.0}"
API_PORT="${V2_API_PORT:-8010}"
GR_HOST="${V2_GRADIO_HOST:-0.0.0.0}"
GR_PORT="${V2_GRADIO_PORT:-7861}"

mkdir -p logs

echo "[v2-local] Starting FastAPI v2 on ${API_HOST}:${API_PORT}..."
python3 -m uvicorn fastapi_app_v2:app --host "${API_HOST}" --port "${API_PORT}" \
  > logs/v2-fastapi.log 2>&1 &
echo $! > .fastapi_v2_pid

echo "[v2-local] Starting Gradio v2 on ${GR_HOST}:${GR_PORT}..."
python3 gradio_app_v2.py > logs/v2-gradio.log 2>&1 &
echo $! > .gradio_v2_pid

echo "[v2-local] Started experimental v2 stack."
echo "- FastAPI v2: http://localhost:${API_PORT}"
echo "- Gradio  v2: http://localhost:${GR_PORT}"
echo "- Logs: logs/v2-fastapi.log , logs/v2-gradio.log"
echo "Run: bash stop_legalsearch_v2_stack.sh"
