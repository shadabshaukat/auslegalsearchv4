#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[v2-docker] Stopping production v2 container..."
if [[ -f .env.production_v2 ]]; then
  docker compose --env-file .env.production_v2 -f docker-compose.production_v2.yml down
else
  docker compose -f docker-compose.production_v2.yml down
fi

echo "[v2-docker] Stopped."
