#!/usr/bin/env bash
# Capture the local tracker API and publish it through funda-api-service.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" && -x .venv/bin/python ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
elif [[ -z "$PYTHON_BIN" && -x .venv-macos/bin/python ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv-macos/bin/python"
elif [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

FILE="${1:-$(date -u +%F).json}"
SNAPSHOT_SERVER_PORT="${SNAPSHOT_SERVER_PORT:-18301}"
TRACKER_API_BASE_URL="http://127.0.0.1:${SNAPSHOT_SERVER_PORT}/api/v1"
API_URL="$TRACKER_API_BASE_URL/dashboard"
SERVER_PID=""
SERVER_LOG="$(mktemp)"
cleanup() {
  [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true
  rm -f "$SERVER_LOG"
}
trap cleanup EXIT

if [[ -z "${ANTHROPIC_ARR_PRIVATE_ANCHORS_JSON:-}" ]]; then
  echo "error: ANTHROPIC_ARR_PRIVATE_ANCHORS_JSON is not set" >&2
  exit 1
fi

if curl -sf -o /dev/null "$API_URL" 2>/dev/null; then
  echo "error: snapshot server port $SNAPSHOT_SERVER_PORT is already in use" >&2
  exit 1
fi

echo "starting isolated dashboard API server on :$SNAPSHOT_SERVER_PORT ..." >&2
ANTHROPIC_ARR_PORT="$SNAPSHOT_SERVER_PORT" \
  "$PYTHON_BIN" -m anthropic_arr.api_server >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 30); do
  curl -sf -o /dev/null "$API_URL" 2>/dev/null && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done
if ! curl -sf -o /dev/null "$API_URL" 2>/dev/null; then
  echo "error: dashboard API did not come up on :$SNAPSHOT_SERVER_PORT" >&2
  tail -50 "$SERVER_LOG" >&2
  exit 1
fi

echo "capturing snapshot -> $FILE ..." >&2
TRACKER_API_BASE_URL="$TRACKER_API_BASE_URL" \
  "$PYTHON_BIN" scripts/snapshot.py "$FILE"

echo "validating and publishing $FILE through funda-api-service ..." >&2
"$PYTHON_BIN" scripts/publish_snapshot.py "$FILE"
