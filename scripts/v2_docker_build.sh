#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f .env.production_v2 ]]; then
  echo "[v2-docker] .env.production_v2 not found. Creating from template..."
  cp .env.production_v2.example .env.production_v2
fi

echo "[v2-docker] Building production v2 image..."
docker compose --env-file .env.production_v2 -f docker-compose.production_v2.yml build --pull

echo "[v2-docker] Build complete."
