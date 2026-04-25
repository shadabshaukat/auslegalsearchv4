#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[v2-docker] Stopping production v2 container..."
docker compose -f docker-compose.production_v2.yml down

echo "[v2-docker] Stopped."
