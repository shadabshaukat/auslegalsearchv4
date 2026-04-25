#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f .env.production_v2 ]]; then
  echo "[v2-docker] ERROR: .env.production_v2 not found in ${ROOT_DIR}"
  echo "[v2-docker] Create it from template first: cp .env.production_v2.example .env.production_v2"
  exit 1
fi

echo "[v2-docker] Starting production v2 container..."
docker compose -f docker-compose.production_v2.yml up -d

echo "[v2-docker] Started."
echo "API:    http://0.0.0.0:8010"
echo "Gradio: http://0.0.0.0:7861"
