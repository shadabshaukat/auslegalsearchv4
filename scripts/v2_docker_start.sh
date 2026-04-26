#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f .env.production_v2 ]]; then
  echo "[v2-docker] ERROR: .env.production_v2 not found."
  echo "[v2-docker] Create .env.production_v2 with your production settings and retry."
  exit 1
fi

set -o allexport
source .env.production_v2
set +o allexport

HOST_INGEST_DIR="${V2_HOST_INGEST_DIR:-./data}"
if [[ ! -d "${HOST_INGEST_DIR}" ]]; then
  echo "[v2-docker] ERROR: V2_HOST_INGEST_DIR does not exist: ${HOST_INGEST_DIR}"
  echo "[v2-docker] Set V2_HOST_INGEST_DIR in .env.production_v2 to a valid host corpus directory."
  exit 1
fi

echo "[v2-docker] Starting production v2 container..."
docker compose --env-file .env.production_v2 -f docker-compose.production_v2.yml up -d

echo "[v2-docker] Started."
echo "API:    http://0.0.0.0:8010"
echo "Gradio: http://0.0.0.0:7861"
