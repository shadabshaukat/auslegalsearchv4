#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[v2-docker] Building production v2 image..."
docker compose -f docker-compose.production_v2.yml build --pull

echo "[v2-docker] Build complete."
