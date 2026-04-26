#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f .env.production_v2 ]]; then
  echo "[v2-docker] ERROR: .env.production_v2 not found."
  echo "[v2-docker] Create .env.production_v2 with your production settings and retry."
  exit 1
fi

echo "[v2-docker] Building production v2 image..."
docker compose --env-file .env.production_v2 -f docker-compose.production_v2.yml build --pull

echo "[v2-docker] Build complete."
