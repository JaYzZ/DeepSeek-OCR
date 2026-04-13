#!/bin/bash
# Continuous-thinking OPD launcher for Qwen3-VL.
#
# Usage:
#   bash Qwen/scripts/train_qwen3vl_opd.sh
#   bash Qwen/scripts/train_qwen3vl_opd.sh Qwen/configs/distillation/qwen3vl_opd_continuous.yaml
#   OUTPUT_DIR=/path/to/run bash Qwen/scripts/train_qwen3vl_opd.sh Qwen/configs/distillation/qwen3vl_opd_continuous.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_CONFIG="$REPO_ROOT/Qwen/configs/distillation/qwen3vl_opd_continuous.yaml"

CONFIG_PATH="${1:-$DEFAULT_CONFIG}"
if [ "${1:-}" != "" ]; then
  shift || true
fi

exec bash "$REPO_ROOT/Qwen/scripts/train_qwen3vl_opsd.sh" "$CONFIG_PATH" "$@"
