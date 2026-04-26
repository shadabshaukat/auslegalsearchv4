#!/usr/bin/env bash
set -euo pipefail

echo "[v2-local] Stopping experimental AUSLegalSearch v2 services..."

for pair in "fastapi_v2:.fastapi_v2_pid" "gradio_v2:.gradio_v2_pid"; do
  svc="${pair%%:*}"
  pidfile="${pair##*:}"
  if [[ -f "${pidfile}" ]]; then
    pid="$(cat "${pidfile}")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" || true
      sleep 1
      if kill -0 "${pid}" 2>/dev/null; then
        kill -9 "${pid}" || true
      fi
      echo "[v2-local] Stopped ${svc} (PID ${pid})."
    else
      echo "[v2-local] ${svc} not running (stale PID ${pid})."
    fi
    rm -f "${pidfile}"
  else
    echo "[v2-local] ${pidfile} not found."
  fi
done

echo "[v2-local] Experimental v2 services stopped."
