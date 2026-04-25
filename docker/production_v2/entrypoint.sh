#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/app"
cd "${APP_DIR}"

ENV_FILE="${AUSLEGALSEARCH_V2_ENV_FILE:-/app/.env.production_v2}"
if [[ -f "${ENV_FILE}" ]]; then
  echo "[entrypoint] loading env from ${ENV_FILE}"
  set -o allexport
  source "${ENV_FILE}"
  set +o allexport
else
  echo "[entrypoint] WARN: env file not found at ${ENV_FILE}. relying on container env vars."
fi

V2_API_HOST="${V2_API_HOST:-0.0.0.0}"
V2_API_PORT="${V2_API_PORT:-8010}"
V2_GRADIO_HOST="${V2_GRADIO_HOST:-0.0.0.0}"
V2_GRADIO_PORT="${V2_GRADIO_PORT:-7861}"

export AUSLEGALSEARCH_V2_ENV_FILE="${ENV_FILE}"
export V2_API_HOST V2_API_PORT V2_GRADIO_HOST V2_GRADIO_PORT

echo "[entrypoint] starting FastAPI on ${V2_API_HOST}:${V2_API_PORT}"
python -m uvicorn fastapi_app_v2:app --host "${V2_API_HOST}" --port "${V2_API_PORT}" &
API_PID=$!

echo "[entrypoint] starting Gradio on ${V2_GRADIO_HOST}:${V2_GRADIO_PORT}"
python gradio_app_v2.py &
GRADIO_PID=$!

cleanup() {
  echo "[entrypoint] stopping services..."
  if kill -0 "${GRADIO_PID}" 2>/dev/null; then kill "${GRADIO_PID}"; fi
  if kill -0 "${API_PID}" 2>/dev/null; then kill "${API_PID}"; fi
}

trap cleanup SIGINT SIGTERM

# Exit container if either service exits unexpectedly
wait -n "${API_PID}" "${GRADIO_PID}"
STATUS=$?
echo "[entrypoint] a service exited with status ${STATUS}; shutting down"
cleanup
exit ${STATUS}
