#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f .env.production_v2 ]]; then
  docker compose --env-file .env.production_v2 -f docker-compose.production_v2.yml logs -f --tail=200
else
  docker compose -f docker-compose.production_v2.yml logs -f --tail=200
fi
