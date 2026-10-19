#!/usr/bin/env bash
# Rebuild the full matrix table from raw events and refresh the stable receipts.
set -euo pipefail
cd "$(dirname "$0")/.."

MATRIX="${1:-results/matrix-20261019T0708Z}"
python3 benchmarks/common_window.py "$MATRIX"
cp "$MATRIX/summary.json" results/local-fp4-mtp1-summary.json
cp "$MATRIX/TABLE.md" results/local-fp4-mtp1-table.md
cp results/launch.json "$MATRIX/launch.json"
python3 benchmarks/audit.py "$MATRIX"
echo "updated $MATRIX and results/local-fp4-mtp1-*"
