#!/usr/bin/env bash
# Stop both ranks of the GLM-5.3-Flash FP4 recipe.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SPARK_VLLM_DIR:-}" ]]; then
  LAUNCHER_DIR="$SPARK_VLLM_DIR"
elif [[ -x "$ROOT/../spark-vllm-docker/launch-cluster.sh" ]]; then
  LAUNCHER_DIR="$ROOT/../spark-vllm-docker"
else
  LAUNCHER_DIR="$ROOT/spark-vllm-docker"
fi

if [[ ! -x "$LAUNCHER_DIR/launch-cluster.sh" ]]; then
  echo "spark-vllm-docker not found; set SPARK_VLLM_DIR" >&2
  exit 1
fi
exec "$LAUNCHER_DIR/launch-cluster.sh" stop
