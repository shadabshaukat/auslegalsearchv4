#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f .env.production_v2 ]]; then
  echo "[v2-local-ingest] ERROR: .env.production_v2 not found in ${ROOT_DIR}"
  exit 1
fi

usage() {
  echo "Usage:" >&2
  echo "  $0 <root_dir> [limit_files] [include_html:true|false]" >&2
  echo "  $0 root_dir <root_dir> [limit_files] [include_html:true|false]" >&2
  echo "  $0 --root_dir <root_dir> [--limit_files N] [--include_html true|false]" >&2
  echo "Examples:" >&2
  echo "  $0 /home/ubuntu/auslegalsearchv4/sample-data-austlii-all-file-types 0 true" >&2
  echo "  $0 root_dir /home/ubuntu/auslegalsearchv4/sample-data-austlii-all-file-types" >&2
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

ROOT_DIR_INPUT=""
LIMIT_FILES="0"
INCLUDE_HTML="true"

# Support accidental leading token style: root_dir <path>
if [[ "${1:-}" == "root_dir" ]]; then
  shift
fi

# Support flag-based style
if [[ "${1:-}" == "--root_dir" ]]; then
  shift
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi
  ROOT_DIR_INPUT="$1"
  shift
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --limit_files)
        LIMIT_FILES="${2:-0}"
        shift 2
        ;;
      --include_html)
        INCLUDE_HTML="${2:-true}"
        shift 2
        ;;
      *)
        echo "[v2-local-ingest] ERROR: Unknown argument '$1'" >&2
        usage
        exit 1
        ;;
    esac
  done
else
  # Positional style
  ROOT_DIR_INPUT="${1:-}"
  LIMIT_FILES="${2:-0}"
  INCLUDE_HTML="${3:-true}"
fi

ROOT_DIR_INPUT="$(echo "${ROOT_DIR_INPUT}" | sed -e "s/^'//" -e "s/'$//" -e 's/^"//' -e 's/"$//')"

if [[ ! -d "${ROOT_DIR_INPUT}" ]]; then
  echo "[v2-local-ingest] ERROR: root_dir does not exist: ${ROOT_DIR_INPUT}"
  usage
  exit 1
fi

set -o allexport
source .env.production_v2
set +o allexport

export AUSLEGALSEARCH_V2_ENV_FILE="${ROOT_DIR}/.env.production_v2"
export AUSLEGALSEARCH_EMBED_USE_CUDA="${AUSLEGALSEARCH_EMBED_USE_CUDA:-1}"
export AUSLEGALSEARCH_EMBED_AMP="${AUSLEGALSEARCH_EMBED_AMP:-1}"
export TOKENIZERS_PARALLELISM="false"

API_PORT="${V2_API_PORT:-8010}"
API_USER="${V2_API_USER:-legal_api}"
API_PASS="${V2_API_PASS:-letmein}"

if ! curl -sf -u "${API_USER}:${API_PASS}" "http://localhost:${API_PORT}/health" >/dev/null; then
  echo "[v2-local-ingest] ERROR: API not reachable at http://localhost:${API_PORT}"
  echo "[v2-local-ingest] Start v2 stack first: bash run_legalsearch_v2_stack.sh"
  exit 1
fi

if [[ "${LIMIT_FILES}" == "0" ]]; then
  LIMIT_JSON="null"
else
  LIMIT_JSON="${LIMIT_FILES}"
fi

if [[ "${INCLUDE_HTML,,}" == "true" || "${INCLUDE_HTML}" == "1" ]]; then
  INCLUDE_HTML_JSON="true"
else
  INCLUDE_HTML_JSON="false"
fi

echo "[v2-local-ingest] Submitting async ingestion job..."
curl -sS -u "${API_USER}:${API_PASS}" -X POST "http://localhost:${API_PORT}/v2/ingest/start" \
  -H 'Content-Type: application/json' \
  -d "{\"root_dir\":\"${ROOT_DIR_INPUT}\",\"limit_files\":${LIMIT_JSON},\"include_html\":${INCLUDE_HTML_JSON}}"
echo
