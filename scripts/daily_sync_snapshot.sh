#!/usr/bin/env bash
# Daily unattended pipeline: refresh upstream data, snapshot the dashboard API,
# and publish the capture through funda-api-service. Intended for cron.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" && -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
elif [[ -z "$PYTHON_BIN" && -x "$REPO_ROOT/.venv-macos/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv-macos/bin/python"
elif [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi

cd "$REPO_ROOT"

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "error: no usable Python runtime; set PYTHON_BIN explicitly" >&2
  exit 1
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "error: OPENROUTER_API_KEY is not set" >&2
  exit 1
fi

if [[ -z "${ANTHROPIC_ARR_PRIVATE_ANCHORS_JSON:-}" ]]; then
  echo "error: ANTHROPIC_ARR_PRIVATE_ANCHORS_JSON is not set" >&2
  exit 1
fi

if [[ -z "${FUNDA_API_BASE_URL:-}" ]]; then
  echo "error: FUNDA_API_BASE_URL is not set" >&2
  exit 1
fi

if [[ -z "${FUNDA_ADMIN_API_KEY:-}" ]]; then
  echo "error: FUNDA_ADMIN_API_KEY is not set" >&2
  exit 1
fi

echo "$(date -u +%FT%TZ) daily sync starting"
"$PYTHON_BIN" -m scripts.refresh_snapshot_data

echo "$(date -u +%FT%TZ) snapshot publish starting"
PYTHON_BIN="$PYTHON_BIN" scripts/publish_snapshot.sh
echo "$(date -u +%FT%TZ) daily sync and snapshot publish complete"
