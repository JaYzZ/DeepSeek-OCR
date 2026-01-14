#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_DIR="$REPO_ROOT/OCRVL/llamafactory/data"
mkdir -p "$DATA_DIR"

# Ensure dataset_info.json is present (tracked in repo).
if [ ! -f "$DATA_DIR/dataset_info.json" ]; then
  echo "ERROR: Missing $DATA_DIR/dataset_info.json" >&2
  exit 1
fi

LLAVA_SRC="${LLAVA_SRC:-/share/project/xiyan/huggingface/liuhaotian/LLaVA-Instruct-150K/llava_v1_5_mix665k.json}"
COT_SRC="${COT_SRC:-/share/project/xiyan/huggingface/Xkev/LLaVA-CoT-100k/train.jsonl}"

if [ ! -f "$LLAVA_SRC" ]; then
  echo "ERROR: LLAVA_SRC not found: $LLAVA_SRC" >&2
  exit 1
fi
if [ ! -f "$COT_SRC" ]; then
  echo "ERROR: COT_SRC not found: $COT_SRC" >&2
  exit 1
fi

ln -sf "$LLAVA_SRC" "$DATA_DIR/llava_v1_5_mix665k.json"
ln -sf "$COT_SRC" "$DATA_DIR/llava_cot_100k_train.jsonl"

echo "✓ Prepared LlamaFactory datasets:"
echo "  - $DATA_DIR/llava_v1_5_mix665k.json -> $LLAVA_SRC"
echo "  - $DATA_DIR/llava_cot_100k_train.jsonl -> $COT_SRC"
echo ""
echo "Use with LlamaFactory configs:"
echo "  dataset_dir: $DATA_DIR"

