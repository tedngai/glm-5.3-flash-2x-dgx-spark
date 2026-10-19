#!/usr/bin/env bash
# Start the tuned GLM-5.3-Flash FP4 recipe on two DGX Sparks.
#
#   ./start.sh              start in the background
#   DAEMON=0 ./start.sh     foreground (launcher log on the terminal)
#   ./start.sh --setup      first run: build image + download/copy model, then start
#
# The upstream launcher is pinned because the B12X recipe and image move
# quickly.  Set SPARK_VLLM_DIR to use an existing checkout instead.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIN="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
LAUNCHER_PIN="0bdc9c9"

if [[ -n "${SPARK_VLLM_DIR:-}" ]]; then
  LAUNCHER_DIR="$SPARK_VLLM_DIR"
elif [[ -x "$ROOT/../spark-vllm-docker/run-recipe.sh" ]]; then
  LAUNCHER_DIR="$ROOT/../spark-vllm-docker"
else
  LAUNCHER_DIR="$ROOT/spark-vllm-docker"
fi

if [[ ! -x "$LAUNCHER_DIR/run-recipe.sh" ]]; then
  echo "cloning spark-vllm-docker at $LAUNCHER_PIN into $LAUNCHER_DIR"
  git clone https://github.com/eugr/spark-vllm-docker.git "$LAUNCHER_DIR"
  git -C "$LAUNCHER_DIR" checkout "$LAUNCHER_PIN"
fi

if curl -sf --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  if [[ "${FORCE:-0}" != "1" ]]; then
    echo "A server is already answering on port 8000." >&2
    echo "Stop it with ./stop.sh first, or set FORCE=1 to let the launcher decide." >&2
    exit 1
  fi
fi

SETUP=()
if [[ "${1:-}" == "--setup" ]]; then
  SETUP=(--setup)
  shift
fi
DAEMON_ARGS=()
if [[ "${DAEMON:-1}" != "0" ]]; then
  DAEMON_ARGS=(--daemon)
fi

echo "recipe: $ROOT/recipe/glm-5.3-flash-tuned.yaml"
echo "launcher: $LAUNCHER_DIR ($(git -C "$LAUNCHER_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown))"
echo "this repository: $PIN"
exec "$LAUNCHER_DIR/run-recipe.sh" \
  "$ROOT/recipe/glm-5.3-flash-tuned.yaml" \
  "${SETUP[@]}" "${DAEMON_ARGS[@]}" \
  -- "$@"
